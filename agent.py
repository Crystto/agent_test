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
        Single-turn chat using Converse + tools, matching the AWS example shape.
        """
        # As in AWS example, we start with just a user message, no system= param.
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"text": question}]}
        ]

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

            # Extract any requested tool uses
            tool_uses = [
                block["toolUse"] for block in content_blocks if "toolUse" in block
            ]
            logger.debug("Found %d toolUse blocks.", len(tool_uses))

            # No tools requested => final answer
            if not tool_uses:
                text = "\n".join(
                    block["text"] for block in content_blocks if "text" in block
                ).strip()
                logger.info("BedrockToolAgent final answer produced.")
                logger.debug("Final answer text: %s", text)
                return text

            # Append model's toolUse message (role: assistant)
            messages.append(output_message)

            # For each toolUse, execute tool and add toolResult message
            for tool_use in tool_uses:
                tool_result_message = self._execute_tool(tool_use)
                messages.append(tool_result_message)

        logger.error("Exceeded max tool-calling iterations without a final answer.")
        raise RuntimeError("Exceeded max tool-calling iterations without a final answer.")

    def _execute_tool(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single toolUse block and return a toolResult message.
        Shape is aligned with AWS docs:

        tool_result = {
            "toolUseId": ...,
            "content": [ { "json": {...} } ]  # or { "text": ... } in some cases
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
            # ---- Call the actual Python-side tool ----
            if name == "list_edges":
                raw_result = self.vco_client.list_edges()

            elif name == "get_edge_health":
                lid = input_data.get("logical_id")
                minutes = int(input_data.get("minutes", 15))
                if not lid:
                    raise ValueError("Missing logical_id")
                raw_result = self.vco_client.get_edge_health(lid, minutes)

            elif name == "Test_Simple_Tool":
                raw_result = "Simple tool sanity test successful!"

            else:
                raise ValueError(f"Unknown tool: {name}")

            payload = self._normalize_result_to_dict(raw_result)
            logger.debug("Tool %s normalized payload: %s", name, _pretty_json(payload))

            tool_result = {
                "toolUseId": tid,
                "content": [
                    {
                        "json": payload,
                    }
                ],
            }

            # NOTE: role is "user", matching AWS example.
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

