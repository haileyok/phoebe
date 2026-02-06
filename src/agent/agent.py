import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
import httpx
from anthropic.types import TextBlock, ToolUseBlock

from src.agent.prompt import build_system_prompt
from src.tools.executor import ToolExecutor

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
        system_text = system or build_system_prompt()
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": 16_000,
            "system": [
                {
                    "type": "text",
                    "text": system_text,
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
            stop_reason=msg.stop_reason or "end_turn",  # type: ignore TODO: fix this
        )

    @staticmethod
    def _inject_cache_breakpoints(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        a helper that adds cache_control breakpoints to the conversation so that
        the conversation prefix is cached across successive calls. we place a single
        breakpoint in th last message's content block, combined with the sys-prompt
        and tool defs breakpoints. ensures that we stay in the 4-breakpoint limit
        that ant requires
        """
        if not messages:
            return messages

        # shallow-copy the list so we don't mutate the caller's conversation
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
    """client for openapi compatible apis like openai, moonshot, etc"""

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
        oai_messages = self._convert_messages(messages, system or build_system_prompt())

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
        """for anthropic chats, we'll convert the outputs into a similar format"""
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
        """convert anthropic tool defs to oai function calling format"""
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
        """convert an oai chat completion resp to agentresponse"""
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


MAX_TOOL_RESULT_LENGTH = 10_000


class Agent:
    def __init__(
        self,
        model_api: Literal["anthropic", "openai", "openapi"],
        model_name: str,
        model_api_key: str | None,
        model_endpoint: str | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        match model_api:
            case "anthropic":
                assert model_api_key
                self._client: AgentClient = AnthropicClient(
                    api_key=model_api_key, model_name=model_name
                )
            case "openai":
                assert model_api_key
                self._client = OpenAICompatibleClient(
                    api_key=model_api_key,
                    model_name=model_name,
                    endpoint="https://api.openai.com/v1",
                )
            case "openapi":
                assert model_api_key
                assert model_endpoint, "model_endpoint is required for openapi"
                self._client = OpenAICompatibleClient(
                    api_key=model_api_key,
                    model_name=model_name,
                    endpoint=model_endpoint,
                )

        self._tool_executor = tool_executor
        self._conversation: list[dict[str, Any]] = []

    def _get_tools(self) -> list[dict[str, Any]] | None:
        """get tool definitions for the agent"""

        if self._tool_executor is None:
            return None
        return [self._tool_executor.get_execute_code_tool_definition()]

    async def _handle_tool_call(self, tool_use: AgentToolUseBlock) -> dict[str, Any]:
        """handle a tool call from the model"""
        if tool_use.name == "execute_code" and self._tool_executor:
            code = tool_use.input.get("code", "")
            result = await self._tool_executor.execute_code(code)
            return result
        else:
            return {"error": f"Unknown tool: {tool_use.name}"}

    async def chat(self, user_message: str) -> str:
        """send a message and get a response, handling tool calls"""
        self._conversation.append({"role": "user", "content": user_message})

        while True:
            resp = await self._client.complete(
                messages=self._conversation,
                tools=self._get_tools(),
            )

            assistant_content: list[dict[str, Any]] = []
            text_response = ""

            for block in resp.content:
                if isinstance(block, AgentTextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                    text_response += block.text
                elif isinstance(block, AgentToolUseBlock):  # type: ignore TODO: for now this errors because there are no other types, but ignore for now
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
            self._conversation.append(assistant_msg)

            # find any tool calls that we need to handle
            if resp.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if isinstance(block, AgentToolUseBlock):
                        code = block.input.get("code", "")
                        logger.info("Tool call: %s\n%s", block.name, code)
                        result = await self._handle_tool_call(block)
                        is_error = "error" in result
                        summary = str(result)[:500]
                        logger.info(
                            "Tool result (%s): %s",
                            "error" if is_error else "ok",
                            summary,
                        )
                        content_str = str(result)
                        if len(content_str) > MAX_TOOL_RESULT_LENGTH:
                            content_str = (
                                content_str[:MAX_TOOL_RESULT_LENGTH]
                                + "\n... (truncated)"
                            )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": content_str,
                            }
                        )

                self._conversation.append({"role": "user", "content": tool_results})
            else:
                # once there are no more tool calls, we proceed to the text response
                return text_response

    async def run(self):
        while True:
            logger.info("running tasks...")
            await asyncio.sleep(30)
