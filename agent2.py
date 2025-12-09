from __future__ import annotations

"""
Agent module for the VCO LLM project.

Two backends:
  - PydanticAI + BedrockConverseModel (existing path)
  - Native Bedrock Converse tool-calling (using AWS Bedrock Converse API)

Switch via: settings.use_pydantic_ai (bool)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import boto3
import urllib3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from backend.api.vco_client import VcoClient
from backend.core.config import get_settings
from backend.models.edge_health import VeloEdge, VeloEdgeHealth

# ---------------------------------------------------------------------------
# Logging / Warnings
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

settings = get_settings()

use_pydantic_ai = getattr(settings, "use_pydantic_ai", False)
bedrock_debug_logging = getattr(settings, "bedrock_debug_logging", False)


def _pretty_json(obj: Any, max_len: int = 8000) -> str:
    """Safely pretty-print JSON for logs with a max length."""
    try:
        text = json.dumps(obj, indent=2, sort_keys=True, default=str)
    except TypeError:
        text = repr(obj)
    if len(text) > max_len:
        return text[:max_len] + "... [truncated]"
    return text


# ---------------------------------------------------------------------------
# Shared Bedrock client via LASSO
# ---------------------------------------------------------------------------

def _create_bedrock_client_with_lasso():
    """
    Create a Bedrock Runtime client via LASSO with logging hooks.

    This client is shared by:
      - PydanticAI BedrockConverseModel
      - Native Bedrock Converse tool-calling agent
    """
    aws_region = getattr(settings, "aws_region", "us-east-1")
    aws_access_key_id = getattr(settings, "aws_access_key_id", None)
    aws_secret_access_key = getattr(
        settings,
        "aws_secret_access_key",
        getattr(settings, "aws_secert_access_key", None),
    )
    aws_session_token = getattr(settings, "aws_session_token", None)

    lasso_proxy_endpoint = getattr(settings, "lasso_proxy_endpoint", None)
    lasso_x_api_key = getattr(settings, "lasso_x_api_key", None)

    logger.info(
        "Configuring Bedrock via LASSO: region=%s endpoint=%s",
        aws_region,
        lasso_proxy_endpoint,
    )

    if not lasso_proxy_endpoint:
        raise RuntimeError("Missing required setting: lasso_proxy_endpoint")

    try:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=aws_region,
            endpoint_url=lasso_proxy_endpoint,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            verify=False,
            config=Config(signature_version="v4"),
        )
        logger.info("Created Bedrock Runtime client via LASSO.")
    except Exception:
        logger.exception("Failed to create Bedrock client.")
        raise

    # Inject LASSO header
    if lasso_x_api_key:
        def add_custom_headers(request, **kwargs):
            logger.debug("Injecting LASSO header.")
            request.headers.add_header("lasso-x-api-key", lasso_x_api_key)

        bedrock_client.meta.events.register_first(
            "before-sign.bedrock-runtime.*",
            add_custom_headers,
        )
        logger.info("Registered LASSO header hook.")
    else:
        logger.warning("No LASSO X-API key set; proxy may reject requests.")

    # Optional deep debug logging for Converse
    if bedrock_debug_logging:

        def log_headers(request, **kwargs):
            logger.debug("=== Converse HEADERS ===")
            for key, value in request.headers.items():
                k = key.lower()
                if k in ("authorization", "x-amz-security-token", "lasso-x-api-key"):
                    logger.debug("%s: [REDACTED]", key)
                else:
                    logger.debug("%s: %s", key, value)
            logger.debug("=== END HEADERS ===")

        def log_body(model, params, **kwargs):
            body = params.get("body")
            logger.debug("=== Converse REQUEST BODY ===")
            if not body:
                logger.debug("[Empty Body]")
            else:
                try:
                    pretty = json.dumps(json.loads(body), indent=2, sort_keys=True)
                    logger.debug(pretty)
                except Exception:
                    logger.debug(body)
            logger.debug("=== END BODY ===")

        def log_url(model, params, **kwargs):
            base = getattr(model, "endpoint_url", "<unknown>")
            path = params.get("url_path", "")
            logger.debug("Converse URL: %s%s", base, path)

        def log_resp(model, http_response, parsed, **kwargs):
            logger.debug(
                "=== Converse RESPONSE === status=%s requestId=%s",
                http_response.status_code,
                http_response.headers.get("x-amzn-requestid"),
            )
            logger.debug("=== END RESPONSE ===")

        bedrock_client.meta.events.register(
            "before-sign.bedrock-runtime.Converse",
            log_headers,
        )
        bedrock_client.meta.events.register(
            "before-call.bedrock-runtime.Converse",
            log_url,
        )
        bedrock_client.meta.events.register(
            "before-call.bedrock-runtime.Converse",
            log_body,
        )
        bedrock_client.meta.events.register(
            "after-call.bedrock-runtime.Converse",
            log_resp,
        )

    return bedrock_client


# Shared Bedrock client + PydanticAI model
bedrock_client = _create_bedrock_client_with_lasso()

model_or_profile_id = getattr(
    settings,
    "bedrock_inference_profile_id",
    "us.anthropic.claude-4-opus-20240229-v1:0",
)

provider = BedrockProvider(bedrock_client=bedrock_client)
bedrock_model = BedrockConverseModel(model_name=model_or_profile_id, provider=provider)


# ---------------------------------------------------------------------------
# Dependencies for tools
# ---------------------------------------------------------------------------

@dataclass
class VcoDeps:
    vco_client: VcoClient


def build_vco_client_from_settings() -> VcoClient:
    logger.debug("Creating VcoClient from settings.")
    client = VcoClient()
    logger.info(
        "Initialized VcoClient enterprise_id=%s base_url=%s verify_ssl=%s",
        client.enterprise_id,
        client.base_url,
        client.verify_ssl,
    )
    return client


def get_default_deps() -> VcoDeps:
    return VcoDeps(vco_client=build_vco_client_from_settings())


# ---------------------------------------------------------------------------
# PydanticAI Agent + tools (existing path)
# ---------------------------------------------------------------------------

agent = Agent(
    model=bedrock_model,
    deps_type=VcoDeps,
    system_prompt=(
        "You are a VeloCloud Orchestrator SD-WAN assistant.\n"
        "- Use tools when appropriate.\n"
        "- Never hallucinate metrics or devices.\n"
        "- Explain results clearly for network engineers.\n"
    ),
)


@agent.tool
def list_edges(ctx: RunContext[VcoDeps]) -> List[VeloEdge]:
    logger.info("[TOOL] list_edges()")
    edges = ctx.deps.vco_client.list_edges()
    logger.info("[RESULT] list_edges -> %d edges", len(edges))
    return edges


@agent.tool
def get_edge_health(
    ctx: RunContext[VcoDeps],
    logical_id: str,
    minutes: int = 15,
) -> VeloEdgeHealth:
    """
    Get detailed health for a specific edge.
    Note: VcoClient.get_edge_health expects minutes as a keyword-only arg.
    """
    logger.info("[TOOL] get_edge_health(logical_id=%s, minutes=%s)", logical_id, minutes)
    health = ctx.deps.vco_client.get_edge_health(
        logical_id=logical_id,
        minutes=minutes,
    )
    logger.info(
        "[RESULT] edge_health -> cpu=%s mem=%s flows=%s drops=%s",
        health.cpu_pct.average,
        health.memory_pct.average,
        health.flow_count.average,
        health.handoff_queue_drops.average,
    )
    return health


@agent.tool
def Test_Simple_Tool(ctx: RunContext[VcoDeps]) -> str:
    """
    Sanity-check tool: just returns a fixed string.
    """
    logger.info("[TOOL] Test_Simple_Tool invoked")
    result = "Simple tool sanity test successful!"
    logger.info("[RESULT] %s", result)
    return result


# ---------------------------------------------------------------------------
# Native Bedrock Converse Tool-Calling Agent (JSON toolResult version)
# ---------------------------------------------------------------------------

class BedrockToolAgent:
    """
    Native Bedrock tool-calling using Converse.

    Version B:
    - toolResult payloads are sent as JSON (`content: [{"json": ...}]`).
    - We normalize Pydantic models / lists into JSON-safe dicts via
      `_normalize_result_to_dict(mode="json")`.
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
        ]

    def _normalize_result_to_dict(self, raw_result: Any) -> Dict[str, Any]:
        """
        Normalize any tool result into a JSON-safe dict so it can go into `json`.

        - Pydantic models -> model_dump(mode="json")
        - Lists of models -> list of JSON-safe dicts
        - Dicts -> passed through
        - Other types -> wrapped in {"result": value}
        """
        if isinstance(raw_result, BaseModel):
            return raw_result.model_dump(mode="json")

        if isinstance(raw_result, dict):
            return raw_result

        if isinstance(raw_result, list):
            items: List[Any] = []
            for item in raw_result:
                if isinstance(item, BaseModel):
                    items.append(item.model_dump(mode="json"))
                else:
                    items.append(item)
            return {"items": items}

        # Simple types (str, int, float, etc.)
        return {"result": raw_result}

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


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def ask_vco_agent(question: str) -> str:
    """
    Unified helper: choose backend based on settings.use_pydantic_ai.

    - If use_pydantic_ai=True: use PydanticAI Agent + tools.
    - Else: use native Bedrock Converse tool-calling (BedrockToolAgent).
    """
    mode = "pydantic_ai" if use_pydantic_ai else "bedrock_native"
    logger.info("[AGENT INPUT] mode=%s question=%s", mode, question)

    deps = get_default_deps()

    try:
        if use_pydantic_ai:
            result = agent.run_sync(question, deps=deps)
            output = result.output
        else:
            raw_agent = BedrockToolAgent(deps.vco_client)
            output = raw_agent.chat(question)

        logger.info("[AGENT OUTPUT] mode=%s %s", mode, output)
        return output

    except ClientError as e:
        logger.exception("ClientError calling Bedrock.")
        return (
            "There was a client error while contacting the Bedrock backend. "
            "Check backend logs and LASSO configuration for details."
        )

    except BotoCoreError as e:
        logger.exception("BotoCoreError calling Bedrock.")
        return (
            "There was a connection/configuration error while contacting Bedrock. "
            "Check backend logs and AWS configuration for details."
        )

    except Exception:
        logger.exception("ask_vco_agent failed")
        return "An unexpected error occurred while processing your request. Check backend logs."


__all__ = [
    "agent",
    "ask_vco_agent",
    "VcoDeps",
    "get_default_deps",
    "BedrockToolAgent",
]
