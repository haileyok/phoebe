"""
Generalized Agent runtime.

This module replaces the original agent.py with a fully config-driven,
provider-agnostic, session-aware implementation. The Agent class knows
nothing about Phoebe, Sara, or any specific persona — all of that lives
in AgentConfig objects defined in agents/<name>/config.py.

Supported providers (all via the same two client classes):
  anthropic  → native Claude API with prompt caching
  openai     → OpenAI chat completions
  openapi    → any OpenAI-compatible endpoint:
               Kimi (Moonshot), Qwen (DashScope), Gemini, DeepSeek,
               Ollama, vLLM, Together, Fireworks, Groq, etc.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
import httpx
from anthropic.types import TextBlock, ToolUseBlock

from src.agent.config import AgentConfig, ModelConfig
from src.agent.session import InMemorySessionStore, Session, SessionStore
from src.agent.tools import ToolExecutor, NullToolExecutor

import json

logger = logging.getLogger(__name__)


# ── Response types (unchanged from original) ──────────────────────────────────

@dataclass
class AgentTextBlock:
    text: str


@dataclass
class AgentToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AgentResponse:
    content: list[AgentTextBlock | AgentToolUseBlock]
    stop_reason: Literal["end_turn", "tool_use"]
    reasoning_content: str | None = None


# ── Client implementations ────────────────────────────────────────────────────

class AnthropicClient:
    def __init__(self, api_key: str, model_name: str, max_tokens: int = 16_000) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model_name = model_name
        self._max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": self._max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": self._inject_cache_breakpoints(messages),
        }
        if tools:
            tools = [dict(t) for t in tools]
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = tools

        async with self._client.messages.stream(**kwargs) as stream:
            msg = await stream.get_final_message()

        content: list[AgentTextBlock | AgentToolUseBlock] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                content.append(AgentTextBlock(text=block.text))
            elif isinstance(block, ToolUseBlock):
                content.append(AgentToolUseBlock(id=block.id, name=block.name, input=block.input))  # type: ignore

        return AgentResponse(
            content=content,
            stop_reason=msg.stop_reason or "end_turn",  # type: ignore
        )

    @staticmethod
    def _inject_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages:
            return messages
        messages = list(messages)
        last_msg = dict(messages[-1])
        content = last_msg["content"]
        if isinstance(content, str):
            last_msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        elif isinstance(content, list) and content:
            content = [dict(b) for b in content]
            content[-1] = dict(content[-1])
            content[-1]["cache_control"] = {"type": "ephemeral"}
            last_msg["content"] = content
        messages[-1] = last_msg
        return messages


class OpenAICompatibleClient:
    """
    Handles any OpenAI-compatible /chat/completions endpoint.
    Covers: OpenAI, Kimi, Qwen, Gemini (via compat layer), DeepSeek,
            Ollama, vLLM, Together, Fireworks, Groq, and more.
    """

    def __init__(self, api_key: str, model_name: str, endpoint: str, max_tokens: int = 16_000) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._endpoint = endpoint.rstrip("/")
        self._max_tokens = max_tokens
        self._http = httpx.AsyncClient(timeout=300.0)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        oai_messages = self._convert_messages(messages, system)
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": oai_messages,
            "max_tokens": self._max_tokens,
        }
        if tools:
            payload["tools"] = self._convert_tools(tools)

        resp = await self._http.post(
            f"{self._endpoint}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        if not resp.is_success:
            logger.error("API error %d: %s", resp.status_code, resp.text[:1000])
            resp.raise_for_status()

        return self._parse_response(resp.json())

    def _convert_messages(self, messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    text_parts, tool_calls = [], []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                            })
                    oai_msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or ""}
                    if msg.get("reasoning_content"):
                        oai_msg["reasoning_content"] = msg["reasoning_content"]
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    result.append(oai_msg)
                elif role == "user":
                    if content and content[0].get("type") == "tool_result":
                        for block in content:
                            result.append({
                                "role": "tool",
                                "tool_call_id": block["tool_use_id"],
                                "content": block.get("content", ""),
                            })
                    else:
                        text = " ".join(b.get("text", str(b)) for b in content)
                        result.append({"role": "user", "content": text})
        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                **({"parameters": t["input_schema"]} if "input_schema" in t else {}),
            }}
            for t in tools
        ]

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")
        content: list[AgentTextBlock | AgentToolUseBlock] = []
        if message.get("content"):
            content.append(AgentTextBlock(text=message["content"]))
        for tc in message.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            content.append(AgentToolUseBlock(id=tc["id"], name=tc["function"]["name"], input=args))
        return AgentResponse(
            content=content,
            stop_reason="tool_use" if finish_reason == "tool_calls" else "end_turn",
            reasoning_content=message.get("reasoning_content"),
        )


def _build_client(model: ModelConfig, max_tokens: int) -> AnthropicClient | OpenAICompatibleClient:
    import os
    api_key = os.getenv(model.api_key_env, "")
    match model.provider:
        case "anthropic":
            return AnthropicClient(api_key=api_key, model_name=model.model_name, max_tokens=max_tokens)
        case "openai":
            return OpenAICompatibleClient(
                api_key=api_key, model_name=model.model_name,
                endpoint="https://api.openai.com/v1", max_tokens=max_tokens,
            )
        case "openapi":
            assert model.endpoint, "model_endpoint required for openapi provider"
            return OpenAICompatibleClient(
                api_key=api_key, model_name=model.model_name,
                endpoint=model.endpoint, max_tokens=max_tokens,
            )
        case _:
            raise ValueError(f"Unknown provider: {model.provider}")


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    Provider-agnostic, config-driven, session-aware agent runtime.

    Usage:
        config = load_agent_config("phoebe")   # or any agents/<name>/config.py
        store  = RedisSessionStore(url=REDIS_URL)
        agent  = Agent(config=config, session_store=store)

        response = await agent.chat(
            user_message="evaluate this submission",
            session_id="discord-user-123",
            mode="judge",
        )
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_executor: ToolExecutor | None = None,
        session_store: SessionStore | None = None,
        model_override: ModelConfig | None = None,
    ) -> None:
        self._config = config
        self._tool_executor = tool_executor or NullToolExecutor()
        self._session_store = session_store or InMemorySessionStore()
        model = model_override or config.default_model
        self._client = _build_client(model, config.max_tokens)

    def _get_tools(self) -> list[dict[str, Any]] | None:
        defs = self._tool_executor.get_tool_definitions()
        return defs if defs else None

    async def _handle_tool_call(self, tool_use: AgentToolUseBlock) -> dict[str, Any]:
        return await self._tool_executor.execute(tool_use.name, tool_use.input)

    async def chat(
        self,
        user_message: str,
        session_id: str = "default",
        mode: str | None = None,
    ) -> str:
        """
        Send a message and get a response.

        session_id  — identifies the conversation (per user, per channel, etc.)
        mode        — overrides the config default_mode for this call
        """
        session = await self._session_store.get_or_create(
            session_id=session_id,
            agent_name=self._config.name,
            mode=mode or self._config.default_mode,
        )
        system_prompt = self._config.get_system_prompt(mode)
        session.append({"role": "user", "content": user_message})

        text_response = await self._run_loop(session, system_prompt)

        await self._session_store.save(session)
        return text_response

    async def _run_loop(self, session: Session, system_prompt: str) -> str:
        while True:
            resp = await self._client.complete(
                messages=session.messages,
                system=system_prompt,
                tools=self._get_tools(),
            )

            assistant_content: list[dict[str, Any]] = []
            text_response = ""

            for block in resp.content:
                if isinstance(block, AgentTextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                    text_response += block.text
                elif isinstance(block, AgentToolUseBlock):
                    assistant_content.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if resp.reasoning_content:
                assistant_msg["reasoning_content"] = resp.reasoning_content
            session.append(assistant_msg)

            if resp.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if isinstance(block, AgentToolUseBlock):
                        logger.info("Tool call: %s | input keys: %s", block.name, list(block.input.keys()))
                        result = await self._handle_tool_call(block)
                        is_error = "error" in result
                        logger.info("Tool result (%s): %s", "error" if is_error else "ok", str(result)[:500])
                        content_str = str(result)
                        if len(content_str) > self._config.max_tool_result_length:
                            content_str = content_str[:self._config.max_tool_result_length] + "\n... (truncated)"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content_str,
                        })
                session.append({"role": "user", "content": tool_results})
            else:
                return text_response

    async def run(self) -> None:
        """
        Background loop for autonomous operation (scheduled campaigns, etc.).
        Override in subclasses or replace with a scheduler hook.
        """
        while True:
            logger.info("[%s] background tick", self._config.name)
            await asyncio.sleep(30)
