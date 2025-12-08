class BedrockToolAgent:
    """
    Native Bedrock tool-calling using Converse, shaped exactly like AWS docs.
    """

    def __init__(self, vco_client: VcoClient):
        self.vco_client = vco_client
        self.model_id = model_or_profile_id
        self.client = bedrock_client
        # NOTE: we won't pass this via system= (LASSO seems picky); keep simple for now.
        self.system_prompt = (
            "You are a VeloCloud Orchestrator SD-WAN assistant. "
            "Use tools to answer questions accurately."
        )
        logger.info("Initialized BedrockToolAgent model_id=%s", self.model_id)

    def _tool_specs(self) -> List[Dict[str, Any]]:
        """
        Tool definitions, matching AWS's `toolConfig` structure.
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
        ]

    def _normalize_result_to_dict(self, raw_result: Any) -> Dict[str, Any]:
        """
        Normalize any tool result into a dict so it can go into `json`.
        This matches Bedrock requirements: json must be an object, not a bare string.
        """
        if isinstance(raw_result, BaseModel):
            return raw_result.model_dump()

        if isinstance(raw_result, dict):
            return raw_result

        if isinstance(raw_result, list):
            items: List[Any] = []
            for item in raw_result:
                if isinstance(item, BaseModel):
                    items.append(item.model_dump())
                else:
                    items.append(item)
            return {"items": items}

        # For simple types (str, int, float, etc.)
        return {"result": raw_result}

    def chat(self, question: str) -> str:
        """
        Single-turn chat using Converse + tools.

        We intentionally keep the message history minimal:

        - original user question
        - the latest assistant toolUse message
        - the matching user toolResult message(s)

        This avoids confusing the LASSO normalizer with multiple
        historical toolUse/toolResult pairs in a single request.
        """
        # Start with just the user question
        original_user_message = {"role": "user", "content": [{"text": question}]}
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

            # Which tools (if any) does the model want to use?
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

            # For each toolUse we execute the tool and build a toolResult message
            tool_result_messages: List[Dict[str, Any]] = []
            for tool_use in tool_uses:
                tool_result_message = self._execute_tool(tool_use)
                tool_result_messages.append(tool_result_message)

            # 🔑 HERE IS THE IMPORTANT CHANGE:
            #
            # Instead of appending to the existing list (which accumulates
            # multiple toolUse/toolResult pairs), we rebuild `messages`
            # with just:
            #   - original user question
            #   - the latest assistant toolUse message
            #   - the toolResult message(s) for this iteration
            new_messages: List[Dict[str, Any]] = [original_user_message, output_message]
            new_messages.extend(tool_result_messages)
            messages = new_messages

            logger.debug(
                "Next-iteration messages count=%d", len(messages)
            )

        logger.error("Exceeded max tool-calling iterations without a final answer.")
        raise RuntimeError("Exceeded max tool-calling iterations without a final answer.")
        
def _execute_tool(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a toolUse and return a toolResult message.
    """
    name = tool_use["name"]
    input_data = tool_use.get("input", {}) or {}
    tid = tool_use["toolUseId"]

    logger.info("Executing tool via BedrockToolAgent: %s", name)
    logger.debug("Tool input: %s", _pretty_json(input_data))

    try:
        # ---- Run the actual Python tool & build a SHORT TEXT summary ----
        if name == "list_edges":
            edges = self.vco_client.list_edges()
            # Build a compact text summary (no huge JSON)
            parts = []
            for e in edges[:10]:  # cap at first 10 to stay small
                # Adjust attributes to match your VeloEdge model
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

            h = self.vco_client.get_edge_health(
                logical_id=lid,
                minutes=minutes,
            )
            # Adjust attribute names to match your VeloEdgeHealth model
            summary_text = (
                f"Edge {lid} health over last {minutes} minutes: "
                f"cpu_avg={h.cpu_pct.average}%, "
                f"mem_avg={h.memory_pct.average}%, "
                f"flows_avg={h.flow_count.average}, "
                f"drops_avg={h.handoff_queue_drops.average}"
            )

        elif name == "Test_Simple_Tool":
            summary_text = "Simple tool sanity test successful!"

        else:
            raise ValueError(f"Unknown tool: {name}")

        logger.debug("Tool %s summary_text: %s", name, summary_text)

        # NOTE: We use TEXT here, not JSON.
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

