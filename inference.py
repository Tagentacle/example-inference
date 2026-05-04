"""
Tagentacle Inference Node: Stateless LLM Gateway.

Provides Service "/inference/chat" that accepts OpenAI-compatible format
and proxies to any OpenAI-compatible LLM API (OpenAI, DeepSeek, OpenRouter, Ollama, etc).

The node is stateless — it holds no conversation history.
Context management is the Agent Node's responsibility.

Required configuration (from secrets.toml or environment):
    OPENAI_API_KEY  — API key for the LLM provider
    OPENAI_BASE_URL — endpoint, e.g. https://api.openai.com/v1
                      or https://openrouter.ai/api/v1

Both are REQUIRED with no default — the node refuses to start if either
is missing. This is deliberate: a default endpoint silently routes
traffic to the wrong provider when an operator forgets to configure it.
"""

import asyncio
import logging
import os
import sys

from openai import AsyncOpenAI
from tagentacle_py_core import Node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("INFERENCE_MODEL", "moonshotai/kimi-k2.5")


async def main():
    node = Node("inference_node")
    await node.connect()

    # --- Resolve credentials (no defaults; missing = fail-fast) ---
    api_key = node.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = (
        node.secrets.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    )

    missing = [
        k
        for k, v in (("OPENAI_API_KEY", api_key), ("OPENAI_BASE_URL", base_url))
        if not v
    ]
    if missing:
        logger.error(
            "Missing required secrets: %s. "
            "Set them in config/secrets.toml or as environment variables. "
            "No default endpoint is provided to prevent silent misrouting.",
            ", ".join(missing),
        )
        sys.exit(2)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logger.info(f"Inference Node ready. base_url={base_url}")

    @node.service("/inference/chat")
    async def handle_chat(payload: dict) -> dict:
        """
        Service handler for LLM chat completion.

        Request payload (OpenAI-compatible):
            {
                "model": "gpt-4o-mini",           # optional, defaults to DEFAULT_MODEL
                "messages": [...],                  # required: OpenAI messages array
                "tools": [...],                     # optional: OpenAI function tool schemas
                "temperature": 0.7,                 # optional
                "max_tokens": 4096                  # optional
            }

        Response payload:
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "...",
                            "tool_calls": [...]     # if tools were invoked
                        }
                    }
                ],
                "model": "gpt-4o-mini",
                "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
            }
        """
        messages = payload.get("messages")
        if not messages:
            return {"error": "missing 'messages' in payload"}

        model = payload.get("model", DEFAULT_MODEL)
        temperature = payload.get("temperature", 0.7)
        max_tokens = payload.get("max_tokens")

        # Build kwargs
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        tools = payload.get("tools")
        if tools:
            kwargs["tools"] = tools

        try:
            logger.info(
                f"Calling LLM: model={model}, messages_count={len(messages)}, "
                f"tools_count={len(tools) if tools else 0}"
            )
            completion = await client.chat.completions.create(**kwargs)

            # Convert pydantic model to plain dict for bus serialization
            result = completion.model_dump(exclude_none=True)
            logger.info(
                f"LLM response: finish_reason={result['choices'][0].get('finish_reason')}"
            )
            return result

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"error": str(e)}

    logger.info("Service '/inference/chat' registered. Spinning...")
    await node.spin()


if __name__ == "__main__":
    asyncio.run(main())
