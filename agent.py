from __future__ import annotations

"""
Agent module for the VCO LLM project.

Two backends:
  - PydanticAI + BedrockConverseModel (existing path)
  - Native Bedrock Converse tool-calling (fallback / improved reliability path)

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
    aws_region = getattr(settings, "aws_region", "us-east-1")
    aws_access_key_id = getattr(settings, "aws_access_key_id", None)
    aws_secret_access_key = getattr(
        settings, "aws_secret_access_key", getattr(settings, "aws_secert_access_key", None)
    )
    aws_session_token = getattr(settings, "aws_session_token", None)

    lasso_proxy_endpoint = getattr(settings, "lasso_proxy_endpoint", None)
    lasso_x_api_key = getattr(settings, "lasso_x_api_key", None)

    logger.info("Configuring Bedrock via LASSO: region=%s endpoint=%s",
                aws_region, lasso_proxy_endpoint)

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
            "before-sign.bedrock-runtime.*", add_custom_headers
        )
        logger.info("Registered LASSO header hook.")
    else:
        logger.warning("No LASSO X-API key set.")

    # Debug logging hooks
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
            logger.debug("=== Converse RESPONSE === status=%s requestId=%s",
                         http_response.status_code,
                         http_response.headers.get("x-amzn-requestid"))
            logger.debug("=== END RESPONSE ===")

        bedrock_client.meta.events.register(
            "before-sign.bedrock-runtime.Converse", log_headers
        )
        bedrock_client.meta.events.register(
            "before-call.bedrock-runtime.Converse", log_url
        )
        bedrock_client.meta.events.register(
            "before-call.bedrock-runtime.Converse", log_body
        )
        bedrock_client.meta.events.register(
            "after-call.bedrock-runtime.Converse", log_resp
        )

    return bedrock_client


# Create shared Bedrock client + PydanticAI model
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
    logger.info("Initialized VcoClient enterprise_id=%s base_url=%s verify_ssl=%s",
                client.enterprise_id, client.base_url, client.verify_ssl)
    return client


def get_default_deps() -> VcoDeps:
    return VcoDeps(vco_client=build_vco_client_from_settings())


# ---------------------------------------------------------------------------
# PydanticAI Agent + tools
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
def get_edge_health(ctx: RunContext[VcoDeps], logical_id: str, minutes: int = 15) -> VeloEdgeHealth:
    logger.info("[TOOL] get_edge_health(logical_id=%s, minutes=%s)", logical_id, minutes)
    health = ctx.deps.vco_client.get_edge_health(logical_id=logical_id, minutes=minutes)
    logger.info("[RESULT] edge_health -> cpu=%s mem=%s flows=%s drops=%s",
                health.cpu_pct.average,
                health.memory_pct.average,
                health.flow_count.average,
                health.handoff_queue_drops.average)
    return health


@agent.tool
def get_enterprise_health(ctx: RunContext[VcoDeps], minutes: int = 15) -> Dict[str, Any]:
    logger.info("[TOOL] get_enterprise_health(minutes=%s)", minutes)
    data = ctx.deps.vco_client.get_enterprise_health(minutes=minutes)
    logger.info("[RESULT] enterprise_health keys=%s", list(data.keys()))
    return data


# ---------------------------------------------------------------------------
# Native Bedrock Converse Tool-Calling Agent
# ---------------------------------------------------------------------------

class BedrockToolAgent:
    """Implements native Bedrock Converse tool-calling."""

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
        return [
            {
                "toolSpec": {
                    "name": "list_edges",
                    "description": "List all SD-WAN edge devices.",
                    "inputSchema": {"json": {"type": "object"}},
                }
            },
            {
                "toolSpec": {
                    "name": "get_edge_health",
                    "description": "Get health metrics for one edge.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "logical_id": {"type": "string"},
                                "minutes": {"type": "integer", "default": 15},
                            },
                            "required": ["logical_id"],
                        }
                    },
                }
            },
            {
                "toolSpec": {
                    "name": "get_enterprise_health",
                    "description": "Get enterprise-wide SD-WAN health.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"minutes": {"type": "integer", "default": 15}},
                        }
                    },
                }
            },
        ]

    def chat(self, question: str) -> str:
        messages = [
            {"role": "user", "content": [{"text": question}]}
        ]

        for step in range(5):
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": self.system_prompt}],
                messages=messages,
                toolConfig={
                    "tools": self._tool_specs(),
                    "toolChoice": {"auto": {}},
                },
            )

            output = response["output"]["message"]
            blocks = output.get("content", [])
            tool_uses = [b["toolUse"] for b in blocks if "toolUse" in b]

            if not tool_uses:
                # Final answer
                text = "\n".join(b["text"] for b in blocks if "text" in b).strip()
                return text

            # Append tool-use message
            messages.append(output)

            # Execute each tool
            for t in tool_uses:
                messages.append(self._execute_tool(t))

        raise RuntimeError("Exceeded max tool-calling iterations")
      
    def _result_to_content_blocks(self, result: Any) -> List[Dict[str, Any]]:
        """
        Convert a Python tool result into Converse content blocks for toolResult.

        - Strings -> text blocks
        - Pydantic models -> dict via .model_dump()
        - Lists of models -> list of dicts
        - Dict/List -> json block
        """
        # If it's a simple string, send as text
        if isinstance(result, str):
            return [{"text": result}]

        # If it's a Pydantic model, dump to dict
        if isinstance(result, BaseModel):
            result = result.model_dump()

        # If it's a list of Pydantic models, dump each
        if isinstance(result, list):
            normalized = []
            for item in result:
                if isinstance(item, BaseModel):
                    normalized.append(item.model_dump())
                else:
                    normalized.append(item)
            result = normalized

        # For dicts/lists (or other JSON-native types), send as json
        return [{"json": result}]

    def _execute_tool(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        name = tool_use["name"]
        input_data = tool_use.get("input", {}) or {}
        tid = tool_use["toolUseId"]

        try:
            if name == "list_edges":
                result = self.vco_client.list_edges()

            elif name == "get_edge_health":
                lid = input_data.get("logical_id")
                minutes = int(input_data.get("minutes", 15))
                if not lid:
                    raise ValueError("Missing logical_id")
                result = self.vco_client.get_edge_health(lid, minutes)

            elif name == "get_enterprise_health":
                minutes = int(input_data.get("minutes", 15))
                result = self.vco_client.get_enterprise_health(minutes)

            else:
                raise ValueError(f"Unknown tool: {name}")

            return {
                "role": "tool",
                "content": [{
                    "toolResult": {
                        "toolUseId": tid,
                        "content": [{"json": result}],
                    }
                }],
            }

        except Exception as e:
            logger.exception("Tool execution failed")
            return {
                "role": "tool",
                "content": [{
                    "toolResult": {
                        "toolUseId": tid,
                        "status": "error",
                        "content": [{"json": {"error": str(e)}}],
                    }
                }],
            }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def ask_vco_agent(question: str) -> str:
    """Unified helper: choose backend."""
    logger.info("[AGENT INPUT] mode=%s question=%s",
                "pydantic_ai" if use_pydantic_ai else "bedrock_native",
                question)

    deps = get_default_deps()

    try:
        if use_pydantic_ai:
            result = agent.run_sync(question, deps=deps)
            return result.output

        else:
            raw_agent = BedrockToolAgent(deps.vco_client)
            return raw_agent.chat(question)

    except Exception:
        logger.exception("ask_vco_agent failed")
        return "An error occurred. Check backend logs."



__all__ = [
    "agent",
    "ask_vco_agent",
    "VcoDeps",
    "get_default_deps",
    "BedrockToolAgent",
]
