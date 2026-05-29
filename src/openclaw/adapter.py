"""
OpenClaw Adapter
================

Integrates any Sara-framework agent into the OpenClaw multi-agent ecosystem.

OpenClaw provides:
  - Multi-channel messaging  (Discord, Telegram, Slack, WhatsApp, Signal)
  - Sub-agent spawning
  - MCP skill plugin system
  - Cron/event scheduler
  - Persistent memory (Redis-backed)

This module provides:
  - OpenClawMCPExecutor   bridges Sara tool calls → OpenClaw MCP dispatcher
  - OpenClawCodeExecutor  wraps openclaw sandbox runner as execute_code tool
  - OpenClawAgent         maps OpenClaw message events → Agent.chat()
  - AgentRegistry         factory that loads named agents by config
  - register_all()        entry point to wire agents into a running OpenClaw instance

Usage (in your OpenClaw bot entrypoint):
    from src.openclaw.adapter import register_all
    await register_all(openclaw_instance, redis_url="redis://localhost:6379")
"""

import logging
import os
from typing import Any, Callable, Awaitable

from src.agent.agent import Agent
from src.agent.config import AgentConfig, ModelConfig
from src.agent.session import InMemorySessionStore
from src.agent.tools import ToolExecutor, CompositeToolExecutor, make_tool

logger = logging.getLogger(__name__)


# ── OpenClaw MCP Tool Executor ─────────────────────────────────────────────────

class OpenClawMCPExecutor(ToolExecutor):
    """
    Bridges Sara tool calls to OpenClaw's MCP skill plugin dispatcher.

    Any tool Sara tries to call is forwarded to OpenClaw's registered MCP
    servers, giving Sara automatic access to all skills registered in the
    OpenClaw instance.

    Use this when deploying Sara inside an OpenClaw bot and you want Sara to
    call MCP skills (web search, calendar, database queries, etc.) transparently.

    Args:
        mcp_dispatcher:  async callable ``(tool_name, input_dict) -> result_dict``.
                         Pass ``openclaw.mcp.call`` or equivalent.
        local_tools:     optional Anthropic-format tool defs handled locally
                         (not forwarded to MCP).
        mcp_tool_names:  if provided, only these names are forwarded to MCP;
                         if None, all unmatched tool calls are forwarded.
    """

    def __init__(
        self,
        mcp_dispatcher: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        local_tools: list[dict[str, Any]] | None = None,
        mcp_tool_names: list[str] | None = None,
    ) -> None:
        self._dispatcher = mcp_dispatcher
        self._local_tools = local_tools or []
        self._mcp_tool_names: set[str] | None = set(mcp_tool_names) if mcp_tool_names else None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Returns local tool defs + any explicitly declared MCP tools.
        For dynamic MCP tool discovery, override this method to call
        ``openclaw.mcp.list_tools()`` at startup.
        """
        return self._local_tools

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self._mcp_tool_names is None or tool_name in self._mcp_tool_names:
            try:
                logger.info("Dispatching to OpenClaw MCP: %s", tool_name)
                return await self._dispatcher(tool_name, tool_input)
            except Exception as e:
                logger.error("MCP dispatch error for %s: %s", tool_name, e)
                return {"error": f"MCP dispatch failed for {tool_name}: {e}"}
        return {"error": f"No handler for tool: {tool_name}"}


# ── OpenClaw Code Executor ─────────────────────────────────────────────────────

class OpenClawCodeExecutor(ToolExecutor):
    """
    Wraps OpenClaw's sandboxed code runner as an ``execute_code`` tool.

    Use this as a drop-in replacement for the original Deno-based ToolExecutor
    when running inside an OpenClaw deployment that provides its own sandbox.

    Args:
        sandbox_runner: async callable ``(code: str) -> result_dict``.
                        Pass ``openclaw.sandbox.run`` or equivalent.
    """

    def __init__(self, sandbox_runner: Callable[[str], Awaitable[dict[str, Any]]]) -> None:
        self._runner = sandbox_runner

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [make_tool(
            name="execute_code",
            description="Execute code in a sandboxed environment and return the result.",
            properties={
                "code": {"type": "string", "description": "Code to execute"},
            },
            required=["code"],
        )]

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "execute_code":
            return await self._runner(tool_input.get("code", ""))
        return {"error": f"Unknown tool: {tool_name}"}


# ── OpenClaw Agent Wrapper ─────────────────────────────────────────────────────

class OpenClawAgent:
    """
    Wraps a Sara Agent for use inside an OpenClaw deployment.

    Handles:
    - Per-user/per-channel session isolation
    - Mode selection from channel config or explicit message metadata
    - Model override per channel (e.g. fast model on Discord, full on API)
    - Graceful error responses — on_message() never raises to OpenClaw's event loop

    Usage:
        agent = OpenClawAgent.from_config(
            config=PHOEBE_CONFIG,
            redis_url="redis://localhost:6379",
            mcp_dispatcher=openclaw.mcp.call,
            sandbox_runner=openclaw.sandbox.run,
        )

        @openclaw.on_message(channel="discord")
        async def handle(event):
            response = await agent.on_message(
                user_id=event.user_id,
                channel_id=event.channel_id,
                text=event.text,
            )
            await event.reply(response)
    """

    def __init__(
        self,
        agent: Agent,
        default_mode: str | None = None,
        channel_mode_map: dict[str, str] | None = None,
    ) -> None:
        self._agent = agent
        self._default_mode = default_mode
        self._channel_mode_map = channel_mode_map or {}

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        redis_url: str | None = None,
        mcp_dispatcher: Callable | None = None,
        sandbox_runner: Callable | None = None,
        model_override: ModelConfig | None = None,
        channel_mode_map: dict[str, str] | None = None,
    ) -> "OpenClawAgent":
        if redis_url:
            try:
                from src.agent.session import RedisSessionStore
            except ImportError as e:
                raise ImportError("pip install redis[asyncio] to use Redis session storage") from e
            store = RedisSessionStore(url=redis_url)
        else:
            logger.warning("No Redis URL provided; using in-memory sessions (not persistent)")
            store = InMemorySessionStore()

        executors: list[ToolExecutor] = []
        if sandbox_runner:
            executors.append(OpenClawCodeExecutor(sandbox_runner))
        if mcp_dispatcher:
            executors.append(OpenClawMCPExecutor(mcp_dispatcher))

        tool_executor: ToolExecutor | None = (
            CompositeToolExecutor(executors) if executors else None
        )

        agent = Agent(
            config=config,
            tool_executor=tool_executor,
            session_store=store,
            model_override=model_override,
        )

        return cls(
            agent=agent,
            default_mode=config.default_mode,
            channel_mode_map=channel_mode_map or {},
        )

    def _resolve_mode(self, channel_id: str | None, explicit_mode: str | None) -> str | None:
        if explicit_mode:
            return explicit_mode
        if channel_id and channel_id in self._channel_mode_map:
            return self._channel_mode_map[channel_id]
        return self._default_mode

    def _make_session_id(self, user_id: str, channel_id: str | None) -> str:
        """Session scope: per-user-per-channel. Override for shared channel sessions."""
        channel = channel_id or "direct"
        return f"{self._agent._config.name}:{channel}:{user_id}"

    async def on_message(
        self,
        user_id: str,
        text: str,
        channel_id: str | None = None,
        mode: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Main entry point for OpenClaw message events.
        Always returns a string — never raises.
        """
        sid = session_id or self._make_session_id(user_id, channel_id)
        resolved_mode = self._resolve_mode(channel_id, mode)

        try:
            return await self._agent.chat(
                user_message=text,
                session_id=sid,
                mode=resolved_mode,
            )
        except Exception as e:
            logger.error("Agent error for session %s: %s", sid, e, exc_info=True)
            return f"Sorry, I encountered an error: {type(e).__name__}. Please try again."

    async def reset_session(self, user_id: str, channel_id: str | None = None) -> None:
        """Clear conversation history for a user/channel."""
        sid = self._make_session_id(user_id, channel_id)
        await self._agent._session_store.delete(sid)


# ── Agent Registry ─────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Factory for named agents. Loads AgentConfig objects from agents/<name>/config.py.

    Use this when you want to manage multiple agents (Phoebe, Sara, etc.) in
    a single OpenClaw deployment without hardcoding agent construction everywhere.

    Usage:
        registry = AgentRegistry(redis_url="redis://localhost:6379")
        phoebe = registry.get("phoebe", mcp_dispatcher=openclaw.mcp.call)
        sara   = registry.get("sara")
    """

    _configs: dict[str, AgentConfig] = {}

    def __init__(
        self,
        redis_url: str | None = None,
        sandbox_runner: Callable | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._sandbox_runner = sandbox_runner
        self._instances: dict[str, OpenClawAgent] = {}
        self._load_builtin_configs()

    def _load_builtin_configs(self) -> None:
        try:
            from agents.phoebe.config import PHOEBE_CONFIG
            self.register_config(PHOEBE_CONFIG)
        except ImportError:
            pass
        try:
            from agents.sara_base.config import SARA_BASE_CONFIG
            self.register_config(SARA_BASE_CONFIG)
        except ImportError:
            pass

    @classmethod
    def register_config(cls, config: AgentConfig) -> None:
        """Register an AgentConfig under its lowercased name."""
        cls._configs[config.name.lower()] = config

    def get(
        self,
        name: str,
        mcp_dispatcher: Callable | None = None,
        model_override: ModelConfig | None = None,
        channel_mode_map: dict[str, str] | None = None,
    ) -> OpenClawAgent:
        key = name.lower()
        instance_key = f"{key}:{id(mcp_dispatcher)}"

        if instance_key not in self._instances:
            config = self._configs.get(key)
            if config is None:
                raise KeyError(
                    f"No agent config registered for '{name}'. "
                    f"Available: {list(self._configs.keys())}"
                )
            self._instances[instance_key] = OpenClawAgent.from_config(
                config=config,
                redis_url=self._redis_url,
                mcp_dispatcher=mcp_dispatcher,
                sandbox_runner=self._sandbox_runner,
                model_override=model_override,
                channel_mode_map=channel_mode_map,
            )

        return self._instances[instance_key]


# ── Top-level registration helper ─────────────────────────────────────────────

async def register_all(
    openclaw_instance: Any,
    redis_url: str | None = None,
    channel_maps: dict[str, dict[str, str]] | None = None,
) -> dict[str, "OpenClawAgent"]:
    """
    Wire all known Sara-framework agents into a running OpenClaw instance.

    Returns a dict of agent_name -> OpenClawAgent for further configuration.

    Args:
        openclaw_instance: running OpenClaw instance; ``sandbox_runner`` and
                           ``mcp_call`` attributes are used if present.
        redis_url:         Redis URL for session persistence. Falls back to
                           ``REDIS_URL`` env var, then InMemorySessionStore.
        channel_maps:      per-agent channel-to-mode mappings, keyed by the
                           agent's lowercase name.  Example::

                               channel_maps={
                                   "phoebe": {
                                       "1234567890": "judge",
                                       "0987654321": "redteam",
                                       "1122334455": "admin",
                                   }
                               }

    Example OpenClaw entrypoint::

        import openclaw
        from src.openclaw.adapter import register_all

        oc = openclaw.OpenClaw(token=os.getenv("DISCORD_TOKEN"))

        agents = await register_all(
            openclaw_instance=oc,
            redis_url=os.getenv("REDIS_URL"),
            channel_maps={
                "phoebe": {
                    "1234567890": "judge",
                    "0987654321": "redteam",
                },
            },
        )

        @oc.on_message
        async def handle(event):
            agent = agents.get("phoebe") or agents.get("sara")
            if agent:
                reply = await agent.on_message(
                    user_id=str(event.author.id),
                    channel_id=str(event.channel.id),
                    text=event.content,
                )
                await event.channel.send(reply)

        await oc.start()
    """
    registry = AgentRegistry(
        redis_url=redis_url or os.getenv("REDIS_URL"),
        sandbox_runner=getattr(openclaw_instance, "sandbox_runner", None),
    )

    mcp_dispatcher = getattr(openclaw_instance, "mcp_call", None)
    channel_maps = channel_maps or {}

    registered: dict[str, OpenClawAgent] = {}
    for name in list(AgentRegistry._configs.keys()):
        try:
            registered[name] = registry.get(
                name=name,
                mcp_dispatcher=mcp_dispatcher,
                channel_mode_map=channel_maps.get(name),
            )
            logger.info("Registered agent: %s", name)
        except Exception as e:
            logger.error("Failed to register agent %s: %s", name, e)

    return registered
