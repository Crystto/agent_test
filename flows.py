        # NEW: link stats
        {
            "toolSpec": {
                "name": "get_link_stats",
                "description": (
                    "Get WAN/link statistics for a specific SD-WAN edge over "
                    "a lookback window in minutes."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "logical_id": {
                                "type": "string",
                                "description": "The edge logicalId.",
                            },
                            "minutes": {
                                "type": "integer",
                                "description": "Lookback window in minutes.",
                                "default": 15,
                            },
                        },
                        "required": ["logical_id"],
                    }
                },
            }
        },
        # NEW: flow stats
        {
            "toolSpec": {
                "name": "get_flow_stats",
                "description": (
                    "Get flow/traffic statistics for a specific SD-WAN edge "
                    "over a lookback window in minutes."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "logical_id": {
                                "type": "string",
                                "description": "The edge logicalId.",
                            },
                            "minutes": {
                                "type": "integer",
                                "description": "Lookback window in minutes.",
                                "default": 15,
                            },
                        },
                        "required": ["logical_id"],
                    }
                },
            }
        },
    ]


def _execute_tool(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
    name = tool_use["name"]
    input_data = tool_use.get("input", {}) or {}
    tid = tool_use["toolUseId"]

    logger.info("Executing tool via BedrockToolAgent: %s", name)
    logger.debug("Tool input: %s", _pretty_json(input_data))

    try:
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

        # NEW: link stats
        elif name == "get_link_stats":
            lid = input_data.get("logical_id")
            minutes = int(input_data.get("minutes", 15))
            if not lid:
                raise ValueError("Missing logical_id")
            raw_result = self.vco_client.get_link_stats(
                logical_id=lid,
                minutes=minutes,
            )

        # NEW: flow stats
        elif name == "get_flow_stats":
            lid = input_data.get("logical_id")
            minutes = int(input_data.get("minutes", 15))
            if not lid:
                raise ValueError("Missing logical_id")
            raw_result = self.vco_client.get_flow_stats(
                logical_id=lid,
                minutes=minutes,
            )

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
