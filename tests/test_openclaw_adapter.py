"""
Tests for src/openclaw/adapter.py (Step 7).

All tests use InMemorySessionStore — no Redis, no external deps.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.agent.config import AgentConfig, ModelConfig
from src.agent.session import InMemorySessionStore
from src.agent.tools import NullToolExecutor
from src.agent.agent import Agent, AgentResponse, AgentTextBlock
from src.openclaw.adapter import (
    AgentRegistry,
    OpenClawAgent,
    OpenClawCodeExecutor,
    OpenClawMCPExecutor,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_agent_registry():
    """Isolate AgentRegistry._configs between tests — it's a class-level dict."""
    AgentRegistry._configs.clear()
    # Re-load the built-in configs so registry tests start from a known state
    AgentRegistry._configs.clear()
    try:
        from agents.phoebe.config import PHOEBE_CONFIG
        AgentRegistry.register_config(PHOEBE_CONFIG)
    except ImportError:
        pass
    try:
        from agents.sara_base.config import SARA_BASE_CONFIG
        AgentRegistry.register_config(SARA_BASE_CONFIG)
    except ImportError:
        pass
    yield
    AgentRegistry._configs.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config(name: str = "TestBot", modes: dict | None = None) -> AgentConfig:
    return AgentConfig(
        name=name,
        description="test",
        default_mode="default",
        modes=modes or {"default": "You are a test assistant."},
        default_model=ModelConfig.anthropic(),
    )


def _mock_agent(reply: str = "ok") -> Agent:
    cfg = _config()
    agent = Agent(config=cfg, session_store=InMemorySessionStore())
    client = AsyncMock()
    client.complete.return_value = AgentResponse(
        content=[AgentTextBlock(text=reply)],
        stop_reason="end_turn",
    )
    agent._client = client
    return agent


# ── OpenClawMCPExecutor ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_executor_dispatches_to_callable():
    async def dispatcher(tool_name: str, tool_input: dict) -> dict:
        return {"dispatched_to": tool_name, "input": tool_input}

    ex = OpenClawMCPExecutor(mcp_dispatcher=dispatcher)
    result = await ex.execute("my_skill", {"arg": "value"})
    assert result["dispatched_to"] == "my_skill"
    assert result["input"] == {"arg": "value"}


@pytest.mark.asyncio
async def test_mcp_executor_restricted_names_blocks_unlisted():
    async def dispatcher(tool_name: str, tool_input: dict) -> dict:
        return {"ok": True}

    ex = OpenClawMCPExecutor(mcp_dispatcher=dispatcher, mcp_tool_names=["allowed_tool"])
    result = await ex.execute("blocked_tool", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_mcp_executor_restricted_names_allows_listed():
    async def dispatcher(tool_name: str, tool_input: dict) -> dict:
        return {"ok": True}

    ex = OpenClawMCPExecutor(mcp_dispatcher=dispatcher, mcp_tool_names=["allowed_tool"])
    result = await ex.execute("allowed_tool", {})
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_mcp_executor_dispatch_error_returns_error_dict():
    async def bad_dispatcher(tool_name: str, tool_input: dict) -> dict:
        raise RuntimeError("network down")

    ex = OpenClawMCPExecutor(mcp_dispatcher=bad_dispatcher)
    result = await ex.execute("some_tool", {})
    assert "error" in result
    assert "network down" in result["error"]


def test_mcp_executor_local_tools_exposed():
    from src.agent.tools import make_tool
    local = [make_tool("local_tool", "a local tool", {"x": {"type": "string"}})]
    ex = OpenClawMCPExecutor(mcp_dispatcher=AsyncMock(), local_tools=local)
    defs = ex.get_tool_definitions()
    assert any(d["name"] == "local_tool" for d in defs)


# ── OpenClawCodeExecutor ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_code_executor_calls_sandbox_runner():
    async def runner(code: str) -> dict:
        return {"output": f"ran: {code}"}

    ex = OpenClawCodeExecutor(sandbox_runner=runner)
    result = await ex.execute("execute_code", {"code": "print('hello')"})
    assert result["output"] == "ran: print('hello')"


@pytest.mark.asyncio
async def test_code_executor_unknown_tool_returns_error():
    ex = OpenClawCodeExecutor(sandbox_runner=AsyncMock())
    result = await ex.execute("not_execute_code", {})
    assert "error" in result


def test_code_executor_exposes_execute_code_tool_def():
    ex = OpenClawCodeExecutor(sandbox_runner=AsyncMock())
    defs = ex.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "execute_code"


# ── OpenClawAgent ─────────────────────────────────────────────────────────────

def test_make_session_id_with_channel():
    agent = _mock_agent()
    oc_agent = OpenClawAgent(agent=agent)
    sid = oc_agent._make_session_id("user123", "chan456")
    assert sid == "TestBot:chan456:user123"


def test_make_session_id_without_channel():
    agent = _mock_agent()
    oc_agent = OpenClawAgent(agent=agent)
    sid = oc_agent._make_session_id("user123", None)
    assert sid == "TestBot:direct:user123"


def test_resolve_mode_explicit_wins():
    agent = _mock_agent()
    oc_agent = OpenClawAgent(
        agent=agent,
        default_mode="default",
        channel_mode_map={"chan1": "admin"},
    )
    assert oc_agent._resolve_mode("chan1", "redteam") == "redteam"


def test_resolve_mode_channel_map():
    agent = _mock_agent()
    oc_agent = OpenClawAgent(
        agent=agent,
        default_mode="default",
        channel_mode_map={"chan1": "admin"},
    )
    assert oc_agent._resolve_mode("chan1", None) == "admin"


def test_resolve_mode_falls_back_to_default():
    agent = _mock_agent()
    oc_agent = OpenClawAgent(agent=agent, default_mode="default")
    assert oc_agent._resolve_mode("unknown_chan", None) == "default"


@pytest.mark.asyncio
async def test_on_message_returns_string():
    oc_agent = OpenClawAgent(agent=_mock_agent("hello from agent"))
    result = await oc_agent.on_message(user_id="u1", text="hi")
    assert result == "hello from agent"


@pytest.mark.asyncio
async def test_on_message_never_raises():
    """on_message() must catch all exceptions and return a string."""
    cfg = _config()
    agent = Agent(config=cfg, session_store=InMemorySessionStore())
    broken_client = AsyncMock()
    broken_client.complete.side_effect = RuntimeError("boom")
    agent._client = broken_client

    oc_agent = OpenClawAgent(agent=agent)
    result = await oc_agent.on_message(user_id="u1", text="hi")
    assert isinstance(result, str)
    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_on_message_session_isolation_by_channel():
    """Different channels get different sessions."""
    cfg = _config()
    store = InMemorySessionStore()
    agent = Agent(config=cfg, session_store=store)
    client = AsyncMock()
    client.complete.return_value = AgentResponse(
        content=[AgentTextBlock(text="reply")],
        stop_reason="end_turn",
    )
    agent._client = client

    oc_agent = OpenClawAgent(agent=agent)
    await oc_agent.on_message(user_id="u1", text="msg", channel_id="chan_A")
    await oc_agent.on_message(user_id="u1", text="msg", channel_id="chan_B")

    sess_a = await store.get_or_create("TestBot:chan_A:u1")
    sess_b = await store.get_or_create("TestBot:chan_B:u1")
    assert len(sess_a.messages) == 2  # user + assistant
    assert len(sess_b.messages) == 2


@pytest.mark.asyncio
async def test_reset_session_clears_history():
    cfg = _config()
    store = InMemorySessionStore()
    agent = Agent(config=cfg, session_store=store)
    client = AsyncMock()
    client.complete.return_value = AgentResponse(
        content=[AgentTextBlock(text="reply")],
        stop_reason="end_turn",
    )
    agent._client = client

    oc_agent = OpenClawAgent(agent=agent)
    await oc_agent.on_message(user_id="u1", text="hello", channel_id="c1")

    sess_before = await store.get_or_create("TestBot:c1:u1")
    assert len(sess_before.messages) == 2

    await oc_agent.reset_session(user_id="u1", channel_id="c1")

    sess_after = await store.get_or_create("TestBot:c1:u1")
    assert len(sess_after.messages) == 0


@pytest.mark.asyncio
async def test_on_message_explicit_session_id_overrides():
    oc_agent = OpenClawAgent(agent=_mock_agent("resp"))
    result = await oc_agent.on_message(
        user_id="u1", text="hi", channel_id="c1", session_id="custom-sid"
    )
    assert result == "resp"
    sess = await oc_agent._agent._session_store.get_or_create("custom-sid")
    assert len(sess.messages) == 2


# ── AgentRegistry ─────────────────────────────────────────────────────────────

def test_registry_loads_phoebe_and_sara():
    # fixture reloads builtins — just assert they're present
    assert "phoebe" in AgentRegistry._configs
    assert "sara" in AgentRegistry._configs


def test_registry_get_returns_openclaw_agent():
    registry = AgentRegistry()
    agent = registry.get("phoebe")
    assert isinstance(agent, OpenClawAgent)


def test_registry_get_unknown_raises_key_error():
    registry = AgentRegistry()
    with pytest.raises(KeyError, match="No agent config"):
        registry.get("does_not_exist")


def test_registry_get_same_name_returns_same_instance():
    registry = AgentRegistry()
    a1 = registry.get("phoebe")
    a2 = registry.get("phoebe")
    assert a1 is a2


def test_registry_custom_config_registration():
    registry = AgentRegistry()
    custom = _config(name="Custom")
    AgentRegistry.register_config(custom)
    assert "custom" in AgentRegistry._configs
    agent = registry.get("custom")
    assert isinstance(agent, OpenClawAgent)


@pytest.mark.asyncio
async def test_register_all_returns_known_agents():
    from src.openclaw.adapter import register_all

    fake_openclaw = MagicMock()
    fake_openclaw.sandbox_runner = None
    fake_openclaw.mcp_call = None

    registered = await register_all(fake_openclaw, redis_url=None)
    assert "phoebe" in registered
    assert "sara" in registered
    assert all(isinstance(v, OpenClawAgent) for v in registered.values())


@pytest.mark.asyncio
async def test_register_all_channel_maps_applied():
    """channel_maps routes each agent to its own per-channel mode map."""
    from src.openclaw.adapter import register_all

    fake_openclaw = MagicMock()
    fake_openclaw.sandbox_runner = None
    fake_openclaw.mcp_call = None

    registered = await register_all(
        fake_openclaw,
        redis_url=None,
        channel_maps={"phoebe": {"chan-x": "admin"}},
    )
    phoebe_agent = registered["phoebe"]
    assert phoebe_agent._channel_mode_map.get("chan-x") == "admin"
    # sara gets no channel map
    sara_agent = registered["sara"]
    assert sara_agent._channel_mode_map == {}


# ── OpenClawAgent.from_config() direct tests ─────────────────────────────────

def test_from_config_no_redis_uses_in_memory_store():
    """from_config() without redis_url must use InMemorySessionStore."""
    cfg = _config()
    oc_agent = OpenClawAgent.from_config(config=cfg, redis_url=None)
    assert isinstance(oc_agent._agent._session_store, InMemorySessionStore)


def test_from_config_no_executors_sets_tool_executor_none():
    """from_config() with no sandbox_runner or mcp_dispatcher → tool_executor is None."""
    cfg = _config()
    oc_agent = OpenClawAgent.from_config(config=cfg, redis_url=None)
    assert oc_agent._agent._tool_executor is None


def test_from_config_with_sandbox_runner_sets_code_executor():
    """from_config() with sandbox_runner wraps it in CodeToolExecutor via Composite."""
    from src.openclaw.adapter import OpenClawCodeExecutor
    from src.agent.tools import CompositeToolExecutor

    cfg = _config()
    oc_agent = OpenClawAgent.from_config(
        config=cfg, redis_url=None, sandbox_runner=AsyncMock()
    )
    assert isinstance(oc_agent._agent._tool_executor, CompositeToolExecutor)
    defs = oc_agent._agent._tool_executor.get_tool_definitions()
    assert any(d["name"] == "execute_code" for d in defs)


def test_from_config_default_mode_from_config():
    cfg = _config()
    oc_agent = OpenClawAgent.from_config(config=cfg)
    assert oc_agent._default_mode == cfg.default_mode
