from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Any, Literal

import anthropic
from anthropic.types import TextBlock, ToolUseBlock
from pydantic import BaseModel

from src.agent.prompt import build_system_prompt
from src.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AgentClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> anthropic.types.Message:
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
    ) -> anthropic.types.Message:
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
            "messages": messages,
        }

        if tools:
            tools = [dict(t) for t in tools]  # shallow copy
            tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = tools

        async with self._client.messages.stream(**kwargs) as stream:  # type: ignore
            return await stream.get_final_message()


MAX_TOOL_RESULT_LENGTH = 10_000


class Agent:
    def __init__(
        self,
        model_api: Literal["anthropic", "openai", "openapi"],
        model_name: str,
        model_api_key: str | None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        if model_api != "anthropic":
            # TODO: implement other APIs
            raise NotImplementedError()

        if model_api == "anthropic":
            assert model_api_key
            self._client = AnthropicClient(api_key=model_api_key, model_name=model_name)

        self._tool_executor = tool_executor
        self._conversation: list[dict[str, Any]] = []

    def _get_tools(self) -> list[dict[str, Any]] | None:
        """get tool definitions for the agent"""

        if self._tool_executor is None:
            return None
        return [self._tool_executor.get_execute_code_tool_definition()]

    async def _handle_tool_call(self, tool_use: ToolUseBlock) -> dict[str, Any]:
        """handle a tool call from the model"""
        if tool_use.name == "execute_code" and self._tool_executor:
            code = tool_use.input.get("code", "")  # type: ignore
            result = await self._tool_executor.execute_code(code)  # type: ignore
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
                if isinstance(block, TextBlock):
                    assistant_content.append({"type": "text", "text": block.text})
                    text_response += block.text
                elif isinstance(block, ToolUseBlock):
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            self._conversation.append(
                {"role": "assistant", "content": assistant_content}
            )

            # find any tool calls that we need to handle
            if resp.stop_reason == "tool_use":
                tool_results: list[dict[str, Any]] = []
                for block in resp.content:
                    if isinstance(block, ToolUseBlock):
                        code = (
                            block.input.get("code", "")
                            if isinstance(block.input, dict)  # type: ignore
                            else ""
                        )
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
                            content_str = content_str[:MAX_TOOL_RESULT_LENGTH] + "\n... (truncated)"

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": content_str,
                            }
                        )

                self._conversation.append({"role": "user", "content": tool_results})
            else:
                # once there are no mroe tool calls, we proceed to the text response
                return text_response

    async def run(self):
        while True:
            logger.info("running tasks...")
            await asyncio.sleep(30)
