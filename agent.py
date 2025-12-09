class BedrockToolAgent:
    """
    Native Bedrock tool-calling using Converse.

    Version B:
    - toolResult payloads are sent as JSON (`content: [{"json": ...}]`).
    - We normalize Pydantic models / lists into JSON-safe objects via
      `_normalize_result_to_json`.
    - We keep the message history minimal per iteration:
        [original_user, latest toolUse, latest toolResult(s)].
    """

    def __init__(self, vco_client: VcoClient):
        self.vco_client = vco_client
        self.model_id = model_or_profile_id
        self.client = bedrock_client
        self.system_prompt = (
            "You are a VeloCloud Orchestrator SD-WAN assistant. "
            "Use tools to answer questions accurately."
        )
        logger.info("Initialized BedrockToolAgent model_id=%s", self.model_id)

    # ------------------------------------------------------------------
    # Helper: resolve edge by name
    # ------------------------------------------------------------------

    def _get_edge_by_name_or_raise(self, edge_name: str) -> VeloEdge:
        """
        Find a single Edge by its display name.

        Raises ValueError if not found or ambiguous.
        """
        edges = self.vco_client.list_edges()
        matches = [e for e in edges if getattr(e, "name", None) == edge_name]

        if not matches:
            raise ValueError(f"No Edge found with name '{edge_name}'")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple Edges found with name '{edge_name}'. "
                "Please specify a unique name or site."
            )
        return matches[0]

    # ------------------------------------------------------------------
    # Tool specifications (exposed to Bedrock Converse)
    # ------------------------------------------------------------------

    def _tool_specs(self) -> List[Dict[str, Any]]:
        """
        Tool definitions, matching AWS's `toolConfig.tools` structure.
        """
        return [
            {
                "toolSpec": {
                    "name": "list_edges",
                    "description": "List all SD-WAN edge devices for this tenant.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        }
                    },
                }
            },
            {
                "toolSpec": {
                    "name": "get_edge_health",
                    "description": "Get health metrics for a single SD-WAN edge.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "logical_id": {"type": "string"},
                                "minutes": {
                                    "type": "integer",
                                    "default": 15,
                                },
                            },
                            "required": ["logical_id"],
                        }
                    },
                }
            },
            {
                "toolSpec": {
                    "name": "Test_Simple_Tool",
                    "description": (
                        "Sanity-check tool that returns a simple string. "
                        "Useful for verifying tool wiring."
                    ),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        }
                    },
                }
            },
            # ---------- NEW: WAN link stats for an edge ----------
            {
                "toolSpec": {
                    "name": "get_wan_link_stats_for_edge",
                    "description": (
                        "Fetch WAN link statistics (interface status, loss, latency, "
                        "and bandwidth utilization) for a specific Edge over a recent "
                        "time window. Use this when the user asks about WAN link health "
                        "or interface utilization on an Edge."
                    ),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "edge_name": {
                                    "type": "string",
                                    "description": "Exact Edge name as shown in VCO.",
                                },
                                "minutes": {
                                    "type": "integer",
                                    "description": (
                                        "How many minutes of recent history to query. "
                                        "Must be between 10 and 1440; default 60."
                                    ),
                                    "minimum": 10,
                                    "maximum": 1440,
                                    "default": 60,
                                },
                            },
                            "required": ["edge_name"],
                        }
                    },
                }
            },
            # ---------- NEW: WAN flow stats for an edge ----------
            {
                "toolSpec": {
                    "name": "get_wan_flow_stats_for_edge",
                    "description": (
                        "Fetch WAN flow statistics (traffic flows grouped by application, "
                        "source/destination, etc.) for a specific Edge over a recent time "
                        "window. Use this for traffic mix, top talkers, or application "
                        "utilization on an Edge."
                    ),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "edge_name": {
                                    "type": "string",
                                    "description": "Exact Edge name as shown in VCO.",
                                },
                                "minutes": {
                                    "type": "integer",
                                    "description": (
                                        "How many minutes of recent history to query. "
                                        "Must be between 10 and 1440; default 60."
                                    ),
                                    "minimum": 10,
                                    "maximum": 1440,
                                    "default": 60,
                                },
                                "group_by": {
                                    "type": "string",
                                    "description": (
                                        "How to group flows: link, application, clientDevice, "
                                        "applicationClass, destFQDN, destIp, destPort, destDomain."
                                    ),
                                    "enum": [
                                        "link",
                                        "application",
                                        "clientDevice",
                                        "applicationClass",
                                        "destFQDN",
                                        "destIp",
                                        "destPort",
                                        "destDomain",
                                    ],
                                    "default": "application",
                                },
                            },
                            "required": ["edge_name"],
                        }
                    },
                }
            },
        ]

    # ------------------------------------------------------------------
    # Normalization: convert tool results into JSON-safe values
    # ------------------------------------------------------------------

    def _normalize_result_to_json(self, raw_result: Any) -> Any:
        """
        Normalize any tool result into a JSON-safe value that can go into `json`.

        Requirements (from Bedrock/LASSO behavior):
        - Top-level must be a JSON object, not an array.
        - Pydantic models (including RootModel) -> model_dump(mode="json")
        - Lists of models -> {"items": [...]}
        - Lists in general -> {"items": [...]}
        - Simple primitives -> {"value": primitive}
        """

        # Pydantic models (BaseModel + RootModel subclasses)
        if isinstance(raw_result, BaseModel):
            dumped = raw_result.model_dump(mode="json")
            # If the model dumps to a list (e.g. RootModel[List[...]]) wrap it
            if isinstance(dumped, list):
                return {"items": dumped}
            return dumped

        # Plain dicts are already JSON objects and safe
        if isinstance(raw_result, dict):
            return raw_result

        # Lists / tuples -> wrap in {"items": [...]}
        if isinstance(raw_result, (list, tuple)):
            items: List[Any] = []
            for item in raw_result:
                if isinstance(item, BaseModel):
                    items.append(item.model_dump(mode="json"))
                else:
                    items.append(item)
            return {"items": items}

        # Simple primitives -> wrap in {"value": ...}
        if isinstance(raw_result, (str, int, float, bool)) or raw_result is None:
            return {"value": raw_result}

        # Fallback: stringify unknown types
        return {"result": str(raw_result)}

    # ------------------------------------------------------------------
    # Main chat loop using Converse + tools
    # ------------------------------------------------------------------

    def chat(self, question: str) -> str:
        """
        Single-turn chat using Converse + tools.

        We intentionally keep the message history minimal:

        - original user question
        - latest assistant toolUse message
        - matching user toolResult message(s)
        """
        original_user_message: Dict[str, Any] = {
            "role": "user",
            "content": [{"text": question}],
        }
        messages: List[Dict[str, Any]] = [original_user_message]

        logger.info("BedrockToolAgent.chat called. question=%s", question)
        logger.debug(
            "Initial messages: %s",
            _pretty_json(messages) if bedrock_debug_logging else f"{len(messages)} messages",
        )

        for step in range(5):
            logger.debug("Tool-calling iteration %d", step + 1)

            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                toolConfig={"tools": self._tool_specs()},
                system=[{"text": self.system_prompt}],
            )

            output_message = response["output"]["message"]
            content_blocks = output_message.get("content", [])
            stop_reason = response.get("stopReason")

            logger.debug("Converse stopReason=%s", stop_reason)

            # Extract any requested tools
            tool_uses = [
                block["toolUse"] for block in content_blocks if "toolUse" in block
            ]
            logger.debug("Found %d toolUse blocks.", len(tool_uses))

            # No tools requested -> final answer
            if not tool_uses:
                text = "\n".join(
                    block["text"] for block in content_blocks if "text" in block
                ).strip()
                logger.info("BedrockToolAgent final answer produced.")
                logger.debug("Final answer text: %s", text)
                return text

            # Execute tool(s) and build toolResult messages
            tool_result_messages: List[Dict[str, Any]] = []
            for tool_use in tool_uses:
                tool_result_message = self._execute_tool(tool_use)
                tool_result_messages.append(tool_result_message)

            # Rebuild messages with:
            #   - original user question
            #   - latest assistant toolUse message
            #   - corresponding user toolResult message(s)
            messages = [original_user_message, output_message]
            messages.extend(tool_result_messages)

            logger.debug("Next-iteration messages count=%d", len(messages))

        logger.error("Exceeded max tool-calling iterations without a final answer.")
        raise RuntimeError("Exceeded max tool-calling iterations without a final answer.")

    # ------------------------------------------------------------------
    # Tool execution: call Python functions and return JSON toolResult
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a toolUse and return a toolResult message.

        Version B: results are returned as JSON in toolResult.content[].json.
        """
        name = tool_use["name"]
        input_data = tool_use.get("input", {}) or {}
        tid = tool_use["toolUseId"]

        logger.info("Executing tool via BedrockToolAgent: %s", name)
        logger.debug("Tool input: %s", _pretty_json(input_data))

        try:
            # ---- Call the actual Python-side tool ----
            if name == "list_edges":
                raw_result = self.vco_client.list_edges()

            elif name == "get_edge_health":
                lid = input_data.get("logical_id")
                minutes = int(input_data.get("minutes", 15))
                if not lid:
                    raise ValueError("Missing logical_id")

                raw_result = self.vco_client.get_edge_health(
                    logical_id=lid,
                    minutes=minutes,
                )

            elif name == "Test_Simple_Tool":
                raw_result = "Simple tool sanity test successful!"

            elif name == "get_wan_link_stats_for_edge":
                edge_name = input_data.get("edge_name")
                if not edge_name:
                    raise ValueError("Missing edge_name")
                minutes = int(input_data.get("minutes", 60))

                edge = self._get_edge_by_name_or_raise(edge_name)
                raw_result = self.vco_client.get_edge_link_stats(
                    logical_id=edge.logical_id,
                    minutes=minutes,
                )

            elif name == "get_wan_flow_stats_for_edge":
                edge_name = input_data.get("edge_name")
                if not edge_name:
                    raise ValueError("Missing edge_name")
                minutes = int(input_data.get("minutes", 60))
                group_by = input_data.get("group_by", "application")

                edge = self._get_edge_by_name_or_raise(edge_name)
                raw_result = self.vco_client.get_edge_flow_stats(
                    logical_id=edge.logical_id,
                    minutes=minutes,
                    group_by=group_by,
                )

            else:
                raise ValueError(f"Unknown tool: {name}")

            payload = self._normalize_result_to_json(raw_result)
            logger.debug("Tool %s normalized payload: %s", name, _pretty_json(payload))

            tool_result = {
                "toolUseId": tid,
                "content": [
                    {
                        "json": payload,
                    }
                ],
            }

            # Per AWS example: role is "user" for toolResult messages.
            return {
                "role": "user",
                "content": [
                    {
                        "toolResult": tool_result,
                    }
                ],
            }

        except Exception as e:
            logger.exception("Tool execution failed for %s", name)
            error_payload = {
                "error": str(e),
                "tool": name,
                "input": input_data,
            }
            tool_result = {
                "toolUseId": tid,
                "content": [
                    {
                        "json": error_payload,
                    }
                ],
                "status": "error",
            }
            return {
                "role": "user",
                "content": [
                    {
                        "toolResult": tool_result,
                    }
                ],
            }
