class BedrockToolAgent:
    """
    Native Bedrock tool-calling using Converse.

    Important design choices for working with LASSO:
    - toolResult payloads are sent as SHORT TEXT summaries, not large JSON.
    - message history is kept minimal per iteration to avoid confusing the
      LASSO normalizer and jailbreak heuristics.
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
                            },
                            "required": ["edge_name"],
                        }
                    },
                }
            },
        ]

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

        This reduces the chance of LASSO jailbreak / normalization errors
        when chaining multiple tool calls.
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
    # Tool execution: run Python code and return SHORT TEXT to Bedrock
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a toolUse and return a toolResult message.

        IMPORTANT:
        - We send results as SHORT TEXT, not JSON, to avoid LASSO jailbreak
          false positives and normalization issues.
        - Shape is aligned with AWS docs:

            tool_result = {
                "toolUseId": ...,
                "content": [ { "text": "..." } ]
            }
            tool_result_message = {
                "role": "user",
                "content": [ { "toolResult": tool_result } ]
            }
        """
        name = tool_use["name"]
        input_data = tool_use.get("input", {}) or {}
        tid = tool_use["toolUseId"]

        logger.info("Executing tool via BedrockToolAgent: %s", name)
        logger.debug("Tool input: %s", _pretty_json(input_data))

        try:
            # ---- Run the actual Python-side tool & build a SHORT TEXT summary ----
            if name == "list_edges":
                edges = self.vco_client.list_edges()
                parts: List[str] = []
                for e in edges[:10]:  # only summarize first 10 edges
                    parts.append(
                        f"{e.name} (logical_id={e.logical_id}, status={e.status})"
                    )
                summary_text = (
                    f"Found {len(edges)} edges. "
                    f"First {len(parts)}: " + "; ".join(parts)
                )

            elif name == "get_edge_health":
                lid = input_data.get("logical_id")
                minutes = int(input_data.get("minutes", 15))
                if not lid:
                    raise ValueError("Missing logical_id")

                h: VeloEdgeHealth = self.vco_client.get_edge_health(
                    logical_id=lid,
                    minutes=minutes,
                )
                summary_text = (
                    f"Edge {lid} health over last {minutes} minutes: "
                    f"cpu_avg={h.cpu_pct.average}%, "
                    f"mem_avg={h.memory_pct.average}%, "
                    f"flows_avg={h.flow_count.average}, "
                    f"drops_avg={h.handoff_queue_drops.average}"
                )

            elif name == "Test_Simple_Tool":
                summary_text = "Simple tool sanity test successful!"

            # ---------- NEW: WAN link stats ----------
            elif name == "get_wan_link_stats_for_edge":
                edge_name = input_data.get("edge_name")
                if not edge_name:
                    raise ValueError("Missing edge_name")
                minutes = int(input_data.get("minutes", 60))

                edge = self._get_edge_by_name_or_raise(edge_name)
                stats = self.vco_client.get_edge_link_stats(
                    logical_id=edge.logical_id,
                    minutes=minutes,
                )

                # stats may be a Pydantic model with `.data`, or a raw dict
                data = getattr(stats, "data", stats)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]

                num_records = len(data) if isinstance(data, list) else 0

                summary_text = (
                    f"Retrieved WAN link statistics for edge '{edge.name}' "
                    f"(logical_id={edge.logical_id}) over the last {minutes} minutes. "
                    f"Record count: {num_records}. "
                    "Use this to understand WAN interface status and utilization."
                )

            # ---------- NEW: WAN flow stats ----------
            elif name == "get_wan_flow_stats_for_edge":
                edge_name = input_data.get("edge_name")
                if not edge_name:
                    raise ValueError("Missing edge_name")
                minutes = int(input_data.get("minutes", 60))

                edge = self._get_edge_by_name_or_raise(edge_name)
                stats = self.vco_client.get_edge_flow_stats(
                    logical_id=edge.logical_id,
                    minutes=minutes,
                )

                data = getattr(stats, "data", stats)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]

                num_records = len(data) if isinstance(data, list) else 0

                summary_text = (
                    f"Retrieved WAN flow statistics for edge '{edge.name}' "
                    f"(logical_id={edge.logical_id}) over the last {minutes} minutes. "
                    f"Record count: {num_records}. "
                    "Use this to understand traffic mix, top talkers, and application usage."
                )

            else:
                raise ValueError(f"Unknown tool: {name}")

            logger.debug("Tool %s summary_text: %s", name, summary_text)

            tool_result = {
                "toolUseId": tid,
                "content": [
                    {
                        "text": summary_text,
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
            error_text = f"Error executing tool {name}: {e}"
            tool_result = {
                "toolUseId": tid,
                "content": [
                    {
                        "text": error_text,
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
