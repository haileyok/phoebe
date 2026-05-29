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
  - OpenClawAgent         wrapper that maps OpenClaw message events → Agent.chat()
  - OpenClawToolExecutor  bridges Sara tool calls → OpenClaw MCP dispatcher
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
from src.agent.session import RedisSessionStore, InMemorySessionStore
from src.agent.tools import ToolExecutor, CompositeToolExecutor, make_tool

logger = logging.getLogger(__name__)


# ── OpenClaw MCP Tool Executor ─────────────────────────────────────────────────

class OpenClawMCPExecutor(ToolExecutor):
    """
    Bridges Sara tool calls to OpenClaw's MCP skill plugin dispatcher.

    Any tool Sara tries to call that isn't handled locally is forwarded
    to OpenClaw's registered MCP servers. This means Sara automatically
    gets access to all skills registered in the OpenClaw instance.

    Args:
        mcp_dispatcher: async callable (tool_name, input_dict) -> result_dict
                        This is openclaw.mcp.call or equivalent.
        local_tools:    optional list of Anthropic-format tool defs that are
                        handled locally (not forwarded to MCP).
    """

    def __init__(
        self,
        mcp_dispatcher: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
        local_tools: list[dict[str, Any]] | None = None,
        mcp_tool_names: list[str] | None = None,
    ) -> None:
        self._dispatcher = mcp_dispatcher
        self._local_tools = local_tools or []
        # If provided, only these tool names are forwarded to MCP
        # If None, all unknown tools are forwarded
        self._mcp_tool_names = set(mcp_tool_names) if mcp_tool_names else None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """
        Returns local tool defs + any MCP tools explicitly declared.
        For dynamic MCP tool discovery, override this method to query
        openclaw.mcp.list_tools() at startup.
        """
        return self._local_tools

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        # Check if this is an MCP-dispatched tool
        if self._mcp_tool_names is None or tool_name in self._mcp_tool_names:
            try:
                logger.info("Dispatching to OpenClaw MCP: %s", tool_name)
                return await self._dispatcher(tool_name, tool_input)
            except Exception as e:
                logger.error("MCP dispatch error for %s: %s", tool_name, e)
                return {"error": f"MCP dispatch failed for {tool_name}: {e}"}
        return {"error": f"No handler for tool: {tool_name}"}


class OpenClawCodeExecutor(ToolExecutor):
    """
    execute_code tool backed by OpenClaw's sandboxed code runner.
    Drop-in replacement for the original ToolExecutor.execute_code.
    """

    def __init__(self, sandbox_runner: Callable[[str], Awaitable[dict[str, Any]]]) -> None:
        self._runner = sandbox_runner

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [make_tool(
            name="execute_code",
            description="Execute Python code in a sandboxed environment and return the result.",
            properties={
                "code": {"type": "string", "description": "Python code to execute"},
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
    - Mode selection from channel config or message metadata
    - Model override per channel (e.g. fast model on Discord, full model on API)
    - Graceful error responses (never raises to OpenClaw's event loop)

    Usage:
        agent = OpenClawAgent.from_config(
            config=PHOEBE_CONFIG,
            redis_url="redis://localhost:6379",
            mcp_dispatcher=openclaw.mcp.call,
            sandbox_runner=openclaw.sandbox.run,
        )

        # In your OpenClaw message handler:
        @openclaw.on_message(channel="discord")
        async def handle(event):
            response = await agent.on_message(
                user_id=event.user_id,
                channel_id=event.channel_id,
                text=event.text,
                mode=event.metadata.get("mode"),
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
        # Session store: Redis if available, else in-memory
        if redis_url:
            store = RedisSessionStore(url=redis_url)
        else:
            logger.warning("No Redis URL provided; using in-memory sessions (not persistent)")
            store = InMemorySessionStore()

        # Tool executor: compose code runner + MCP dispatcher
        executors = []
        if sandbox_runner:
            executors.append(OpenClawCodeExecutor(sandbox_runner))
        if mcp_dispatcher:
            executors.append(OpenClawMCPExecutor(mcp_dispatcher))

        tool_executor = CompositeToolExecutor(executors) if executors else None

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
        """
        Session scope: per-user-per-channel by default.
        Override for shared channel sessions (e.g. a broadcast bot).
        """
        if channel_id:
            return f"{self._agent._config.name}:{channel_id}:{user_id}"
        return f"{self._agent._config.name}:{user_id}"

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
        """Clear conversation history for a user."""
        sid = self._make_session_id(user_id, channel_id)
        await self._agent._session_store.delete(sid)


# ── Agent Registry ─────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Factory for named agents. Loads configs from agents/<name>/config.py.

    Usage:
        registry = AgentRegistry(redis_url="redis://localhost:6379")
        phoebe = registry.get("phoebe", mcp_dispatcher=openclaw.mcp.call)
        sara   = registry.get("sara_base")
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
    phoebe_channel_map: dict[str, str] | None = None,
) -> dict[str, OpenClawAgent]:
    """
    Wire all known Sara-framework agents into a running OpenClaw instance.

    Returns a dict of agent_name -> OpenClawAgent for further configuration.

    Example OpenClaw entrypoint:
        import openclaw
        from src.openclaw.adapter import register_all

        oc = openclaw.OpenClaw(token=os.getenv("DISCORD_TOKEN"))

        agents = await register_all(
            openclaw_instance=oc,
            redis_url=os.getenv("REDIS_URL"),
            phoebe_channel_map={
                "1234567890": "judge",    # #bounty-eval channel → judge mode
                "0987654321": "redteam",  # #red-team channel   → redteam mode
                "1122334455": "admin",    # #admin channel      → admin mode
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

    registered: dict[str, OpenClawAgent] = {}

    for name in list(AgentRegistry._configs.keys()):
        cmap = phoebe_channel_map if name == "phoebe" else None
        try:
            registered[name] = registry.get(
                name=name,
                mcp_dispatcher=mcp_dispatcher,
                channel_mode_map=cmap,
            )
            logger.info("Registered agent: %s", name)
        except Exception as e:
            logger.error("Failed to register agent %s: %s", name, e)

    return registered
