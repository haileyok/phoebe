from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
import httpx
from anthropic.types import TextBlock, ToolUseBlock

from src.agent.config import AgentConfig, ModelConfig
from src.agent.session import InMemorySessionStore, SessionStore
from src.agent.tools import ToolExecutor

logger = logging.getLogger(__name__)


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


class AgentClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        pass


class AnthropicClient(AgentClient):
    def __init__(
        self, api_key: str, model_name: str = "claude-sonnet-4-5-20250929"
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model_name = model_name

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        if system is None:
            raise ValueError("system prompt is required; pass it via Agent.chat() or provide it directly")
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": 16_000,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": self._inject_cache_breakpoints(messages),
        }

        if tools:
            tools = [dict(t) for t in tools]
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = tools

        async with self._client.messages.stream(**kwargs) as stream:  # type: ignore
            msg = await stream.get_final_message()

        content: list[AgentTextBlock | AgentToolUseBlock] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                content.append(AgentTextBlock(text=block.text))
            elif isinstance(block, ToolUseBlock):
                content.append(
                    AgentToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.input,  # type: ignore
                    )
                )

        return AgentResponse(
            content=content,
            stop_reason=msg.stop_reason or "end_turn",  # type: ignore
        )

    @staticmethod
    def _inject_cache_breakpoints(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Adds cache_control breakpoints to the conversation so that the prefix
        is cached across successive calls. Places a single breakpoint in the
        last message's content block, combined with the sys-prompt and tool
        defs breakpoints. Stays within the 4-breakpoint limit Anthropic requires.
        """
        if not messages:
            return messages

        messages = list(messages)
        last_msg = dict(messages[-1])
        content = last_msg["content"]

        if isinstance(content, str):
            last_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            content = [dict(b) for b in content]
            content[-1] = dict(content[-1])
            content[-1]["cache_control"] = {"type": "ephemeral"}
            last_msg["content"] = content

        messages[-1] = last_msg
        return messages


class OpenAICompatibleClient(AgentClient):
    """Client for OpenAI-compatible APIs (OpenAI, Moonshot, DeepSeek, Ollama, etc.)."""

    def __init__(self, api_key: str, model_name: str, endpoint: str) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._endpoint = endpoint.rstrip("/")
        self._http = httpx.AsyncClient(timeout=300.0)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        if system is None:
            raise ValueError("system prompt is required; pass it via Agent.chat() or provide it directly")
        oai_messages = self._convert_messages(messages, system)

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": oai_messages,
            "max_tokens": 16_000,
        }

        if tools:
            payload["tools"] = self._convert_tools(tools)

        resp = await self._http.post(
            f"{self._endpoint}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not resp.is_success:
            logger.error("API error %d: %s", resp.status_code, resp.text[:1000])
            resp.raise_for_status()
        data = resp.json()

        return self._parse_response(data)

    def _convert_messages(
        self, messages: list[dict[str, Any]], system: str
    ) -> list[dict[str, Any]]:
        """Convert Anthropic-format messages (tool_use, tool_result, reasoning_content) to OAI."""
        result: list[dict[str, Any]] = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": block["id"],
                                    "type": "function",
                                    "function": {
                                        "name": block["name"],
                                        "arguments": json.dumps(block["input"]),
                                    },
                                }
                            )
                    oai_msg: dict[str, Any] = {"role": "assistant"}
                    if msg.get("reasoning_content"):
                        oai_msg["reasoning_content"] = msg["reasoning_content"]
                    # some openai-compatible apis reject content: null on
                    # assistant messages with tool_calls, so omit it when empty
                    if text_parts:
                        oai_msg["content"] = "\n".join(text_parts)
                    else:
                        oai_msg["content"] = ""
                    if tool_calls:
                        oai_msg["tool_calls"] = tool_calls
                    result.append(oai_msg)
                elif role == "user":
                    if content and content[0].get("type") == "tool_result":
                        for block in content:
                            result.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block["tool_use_id"],
                                    "content": block.get("content", ""),
                                }
                            )
                    else:
                        text = " ".join(b.get("text", str(b)) for b in content)
                        result.append({"role": "user", "content": text})

        return result

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic tool defs to OAI function-calling format."""
        result = []
        for t in tools:
            func: dict[str, Any] = {
                "name": t["name"],
                "description": t.get("description", ""),
            }
            if "input_schema" in t:
                func["parameters"] = t["input_schema"]
            result.append({"type": "function", "function": func})
        return result

    def _parse_response(self, data: dict[str, Any]) -> AgentResponse:
        """Convert an OAI chat completion response to AgentResponse."""
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        content: list[AgentTextBlock | AgentToolUseBlock] = []

        if message.get("content"):
            content.append(AgentTextBlock(text=message["content"]))

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}
                content.append(
                    AgentToolUseBlock(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        input=args,
                    )
                )

        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        reasoning_content = message.get("reasoning_content")
        return AgentResponse(
            content=content,
            stop_reason=stop_reason,
            reasoning_content=reasoning_content,
        )


def _build_client(
    model_config: ModelConfig, model_override: ModelConfig | None = None
) -> AgentClient:
    """Factory: select the right AgentClient for a given ModelConfig."""
    mc = model_override or model_config
    api_key = os.environ.get(mc.api_key_env, "")
    if mc.provider == "anthropic":
        return AnthropicClient(api_key=api_key, model_name=mc.model_name)
    elif mc.provider == "openai":
        return OpenAICompatibleClient(
            api_key=api_key,
            model_name=mc.model_name,
            endpoint="https://api.openai.com/v1",
        )
    elif mc.provider == "kimi":
        # Kimi (Moonshot AI) — temperature must be in [0, 1]
        return OpenAICompatibleClient(
            api_key=api_key,
            model_name=mc.model_name,
            endpoint=mc.endpoint or "https://api.moonshot.ai/v1",
        )

    elif mc.provider == "glm":
        # GLM (Zhipu AI) — no parallel tool calls, no tool_choice=required
        return OpenAICompatibleClient(
            api_key=api_key,
            model_name=mc.model_name,
            endpoint=mc.endpoint or "https://open.bigmodel.cn/api/paas/v4",
        )

    elif mc.provider == "deepseek":
        # DeepSeek — reasoning_content already handled by _parse_response()
        return OpenAICompatibleClient(
            api_key=api_key,
            model_name=mc.model_name,
            endpoint=mc.endpoint or "https://api.deepseek.com",
        )

    elif mc.provider == "openapi":
        if not mc.endpoint:
            raise ValueError(
                f"ModelConfig.endpoint is required for provider 'openapi'"
            )
        return OpenAICompatibleClient(
            api_key=api_key,
            model_name=mc.model_name,
            endpoint=mc.endpoint,
        )
    raise ValueError(f"Unknown provider: {mc.provider}")


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        tool_executor: ToolExecutor | None = None,
        session_store: SessionStore | None = None,
        model_override: ModelConfig | None = None,
    ) -> None:
        self._config = config
        self._tool_executor = tool_executor
        self._session_store: SessionStore = session_store or InMemorySessionStore()
        self._client: AgentClient = _build_client(config.default_model, model_override)

    def _get_tools(self) -> list[dict[str, Any]] | None:
        if self._tool_executor is None:
            return None
        defs = self._tool_executor.get_tool_definitions()
        return defs or None

    async def _handle_tool_call(self, tool_use: AgentToolUseBlock) -> dict[str, Any]:
        if self._tool_executor:
            return await self._tool_executor.execute(tool_use.name, tool_use.input)
        return {"error": f"No tool executor. Called: {tool_use.name}"}

    async def chat(
        self,
        user_message: str,
        session_id: str = "default",
        mode: str | None = None,
    ) -> str:
        """Send a message and get a response, handling tool calls."""
        resolved_mode = mode or self._config.default_mode
        session = await self._session_store.get_or_create(
            session_id,
            agent_name=self._config.name,
            mode=resolved_mode,
        )
        system_prompt = self._config.get_system_prompt(resolved_mode)

        session.append({"role": "user", "content": user_message})
        await self._session_store.save(session)

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
                elif isinstance(block, AgentToolUseBlock):  # type: ignore
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if resp.reasoning_content:
                assistant_msg["reasoning_content"] = resp.reasoning_content
            session.append(assistant_msg)

            if resp.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if isinstance(block, AgentToolUseBlock):
                        logger.info("Tool call: %s\n%s", block.name, block.input)
                        result = await self._handle_tool_call(block)
                        is_error = "error" in result
                        logger.info(
                            "Tool result (%s): %s",
                            "error" if is_error else "ok",
                            str(result)[:500],
                        )
                        content_str = str(result)
                        if len(content_str) > self._config.max_tool_result_length:
                            content_str = (
                                content_str[: self._config.max_tool_result_length]
                                + "\n... (truncated)"
                            )
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": content_str,
                            }
                        )
                session.append({"role": "user", "content": tool_results})
            else:
                await self._session_store.save(session)
                return text_response

    async def run(self):
        while True:
            logger.info("running tasks...")
            await asyncio.sleep(30)
