"""
Framework unit tests for steps 1–5 of the Sara generalization refactor.

All tests use InMemorySessionStore and NullToolExecutor — no external deps required.
"""

import asyncio
import tempfile
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from src.agent.config import AgentConfig, ModelConfig
from src.agent.session import FileSessionStore, InMemorySessionStore, Session
from src.agent.tools import (
    CompositeToolExecutor,
    NullToolExecutor,
    make_tool,
)
from src.agent.agent import Agent, AgentResponse, AgentTextBlock, AgentToolUseBlock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config(name: str = "TestBot", modes: dict | None = None) -> AgentConfig:
    return AgentConfig(
        name=name,
        description="test agent",
        default_mode="default",
        modes=modes or {"default": "You are a test assistant."},
        default_model=ModelConfig.anthropic(),
    )


def _mock_client(text: str = "ok", stop_reason: str = "end_turn") -> AsyncMock:
    client = AsyncMock()
    client.complete.return_value = AgentResponse(
        content=[AgentTextBlock(text=text)],
        stop_reason=stop_reason,
    )
    return client


# ── AgentConfig tests ─────────────────────────────────────────────────────────

def test_agent_config_get_system_prompt():
    cfg = _config(modes={"a": "prompt-a", "b": "prompt-b"})
    assert cfg.get_system_prompt("a") == "prompt-a"
    assert cfg.get_system_prompt("b") == "prompt-b"


def test_agent_config_default_mode():
    cfg = _config(modes={"default": "default-prompt"})
    assert cfg.get_system_prompt() == "default-prompt"


def test_agent_config_invalid_mode_raises():
    cfg = _config()
    with pytest.raises(ValueError, match="no mode"):
        cfg.get_system_prompt("nonexistent")


def test_agent_config_available_modes():
    cfg = _config(modes={"x": "px", "y": "py"})
    assert cfg.available_modes() == ["x", "y"]


# ── Session tests ─────────────────────────────────────────────────────────────

def test_session_serialization_roundtrip():
    s = Session(
        session_id="sess-42",
        agent_name="Phoebe",
        mode="judge",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"foo": "bar"},
    )
    s2 = Session.from_dict(s.to_dict())
    assert s2.session_id == s.session_id
    assert s2.agent_name == s.agent_name
    assert s2.mode == s.mode
    assert s2.messages == s.messages
    assert s2.metadata == s.metadata


def test_session_append_updates_timestamp():
    import time
    s = Session(session_id="s", agent_name="a", mode="m")
    before = s.updated_at
    time.sleep(0.01)
    s.append({"role": "user", "content": "hi"})
    assert s.updated_at > before
    assert len(s.messages) == 1


def test_session_clear():
    s = Session(session_id="s", agent_name="a", mode="m")
    s.append({"role": "user", "content": "hi"})
    s.clear()
    assert s.messages == []


# ── InMemorySessionStore tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_in_memory_store_creates_new_session():
    store = InMemorySessionStore()
    s = await store.get_or_create("s1", agent_name="Bot", mode="x")
    assert s.session_id == "s1"
    assert s.agent_name == "Bot"
    assert s.messages == []


@pytest.mark.asyncio
async def test_in_memory_store_returns_existing_session():
    store = InMemorySessionStore()
    s = await store.get_or_create("s1", agent_name="Bot", mode="x")
    s.append({"role": "user", "content": "hello"})
    await store.save(s)

    s2 = await store.get_or_create("s1")
    assert len(s2.messages) == 1
    assert s2.messages[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_in_memory_store_isolation():
    store = InMemorySessionStore()
    s1 = await store.get_or_create("s1")
    s1.append({"role": "user", "content": "in s1"})
    await store.save(s1)

    s2 = await store.get_or_create("s2")
    assert len(s2.messages) == 0


@pytest.mark.asyncio
async def test_in_memory_store_delete():
    store = InMemorySessionStore()
    s = await store.get_or_create("s1")
    s.append({"role": "user", "content": "hi"})
    await store.save(s)
    await store.delete("s1")

    s2 = await store.get_or_create("s1")
    assert len(s2.messages) == 0


@pytest.mark.asyncio
async def test_in_memory_store_list_sessions():
    store = InMemorySessionStore()
    await store.get_or_create("s1", agent_name="BotA")
    await store.get_or_create("s2", agent_name="BotB")
    await store.get_or_create("s3", agent_name="BotA")

    all_sessions = await store.list_sessions()
    assert set(all_sessions) == {"s1", "s2", "s3"}

    bot_a = await store.list_sessions(agent_name="BotA")
    assert set(bot_a) == {"s1", "s3"}


# ── FileSessionStore tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_file_store_persistence():
    with tempfile.TemporaryDirectory() as d:
        store = FileSessionStore(base_dir=d)
        s = await store.get_or_create("s1", agent_name="Bot", mode="m")
        s.append({"role": "user", "content": "persisted"})
        await store.save(s)

        store2 = FileSessionStore(base_dir=d)
        s2 = await store2.get_or_create("s1")
        assert len(s2.messages) == 1
        assert s2.messages[0]["content"] == "persisted"


@pytest.mark.asyncio
async def test_file_store_delete():
    with tempfile.TemporaryDirectory() as d:
        store = FileSessionStore(base_dir=d)
        s = await store.get_or_create("s1")
        s.append({"role": "user", "content": "hi"})
        await store.save(s)
        await store.delete("s1")

        s2 = await store.get_or_create("s1")
        assert len(s2.messages) == 0


# ── NullToolExecutor tests ────────────────────────────────────────────────────

def test_null_executor_returns_no_tools():
    ex = NullToolExecutor()
    assert ex.get_tool_definitions() == []


@pytest.mark.asyncio
async def test_null_executor_execute_returns_error():
    ex = NullToolExecutor()
    result = await ex.execute("any_tool", {})
    assert "error" in result


# ── CompositeToolExecutor tests ───────────────────────────────────────────────

class _EchoExecutor(NullToolExecutor):
    def __init__(self, tool_name: str, response: str):
        self._tool_name = tool_name
        self._response = response

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [make_tool(self._tool_name, "echo tool", {"x": {"type": "string"}})]

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        return {"result": self._response}


@pytest.mark.asyncio
async def test_composite_routes_to_correct_executor():
    a = _EchoExecutor("tool_a", "from-a")
    b = _EchoExecutor("tool_b", "from-b")
    comp = CompositeToolExecutor([a, b])

    assert await comp.execute("tool_a", {}) == {"result": "from-a"}
    assert await comp.execute("tool_b", {}) == {"result": "from-b"}


@pytest.mark.asyncio
async def test_composite_unknown_tool_returns_error():
    comp = CompositeToolExecutor([_EchoExecutor("tool_a", "x")])
    result = await comp.execute("unknown", {})
    assert "error" in result


def test_composite_deduplicates_tool_definitions():
    a = _EchoExecutor("shared", "from-a")
    b = _EchoExecutor("shared", "from-b")
    comp = CompositeToolExecutor([a, b])
    defs = comp.get_tool_definitions()
    names = [d["name"] for d in defs]
    assert names.count("shared") == 1


def test_composite_first_match_routing():
    """First-registered executor wins for duplicate tool names."""
    a = _EchoExecutor("shared", "first")
    b = _EchoExecutor("shared", "second")
    comp = CompositeToolExecutor([a, b])

    async def _run():
        return await comp.execute("shared", {})

    result = asyncio.get_event_loop().run_until_complete(_run())
    assert result == {"result": "first"}


# ── Agent.chat() tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_chat_simple_response():
    agent = Agent(config=_config(), session_store=InMemorySessionStore())
    agent._client = _mock_client("Hello, world!")

    response = await agent.chat("hi")
    assert response == "Hello, world!"


@pytest.mark.asyncio
async def test_agent_chat_backward_compat_positional_only():
    """Old callers pass only user_message — must still work."""
    agent = Agent(config=_config(), session_store=InMemorySessionStore())
    agent._client = _mock_client("ok")

    result = await agent.chat("test")
    assert result == "ok"


@pytest.mark.asyncio
async def test_agent_chat_accumulates_messages():
    agent = Agent(config=_config(), session_store=InMemorySessionStore())
    agent._client = _mock_client("reply")

    await agent.chat("first", session_id="s1")
    await agent.chat("second", session_id="s1")

    session = await agent._session_store.get_or_create("s1")
    roles = [m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_agent_chat_session_isolation():
    agent = Agent(config=_config(), session_store=InMemorySessionStore())
    agent._client = _mock_client("reply")

    await agent.chat("msg", session_id="user-A")
    await agent.chat("msg", session_id="user-B")

    sa = await agent._session_store.get_or_create("user-A")
    sb = await agent._session_store.get_or_create("user-B")
    assert len(sa.messages) == 2
    assert len(sb.messages) == 2


@pytest.mark.asyncio
async def test_agent_chat_mode_selects_correct_prompt():
    cfg = _config(modes={"default": "default-prompt", "alt": "alt-prompt"})
    agent = Agent(config=cfg, session_store=InMemorySessionStore())
    captured_system: list[str] = []

    async def mock_complete(messages, system=None, tools=None):
        captured_system.append(system)
        return AgentResponse(content=[AgentTextBlock(text="ok")], stop_reason="end_turn")

    client = AsyncMock()
    client.complete.side_effect = mock_complete
    agent._client = client

    await agent.chat("hi", mode="alt")
    assert captured_system[0] == "alt-prompt"


@pytest.mark.asyncio
async def test_agent_chat_persists_user_message_before_api_call():
    """F3 regression: user message must be saved before the API call."""
    store = InMemorySessionStore()
    agent = Agent(config=_config(), session_store=store)

    call_order: list[str] = []

    async def mock_complete(messages, system=None, tools=None):
        # By the time this runs, the user message must already be saved
        s = await store.get_or_create("sess")
        call_order.append(f"api:{len(s.messages)}")
        return AgentResponse(content=[AgentTextBlock(text="ok")], stop_reason="end_turn")

    client = AsyncMock()
    client.complete.side_effect = mock_complete
    agent._client = client

    await agent.chat("hello", session_id="sess")
    assert call_order == ["api:1"]  # user message saved before API call


@pytest.mark.asyncio
async def test_agent_chat_tool_call_loop():
    """Agent handles one tool_use turn then resolves to end_turn."""
    cfg = _config()
    store = InMemorySessionStore()

    tool_ex = _EchoExecutor("my_tool", "tool-output")
    agent = Agent(config=cfg, tool_executor=tool_ex, session_store=store)

    call_count = 0

    async def mock_complete(messages, system=None, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(
                content=[AgentToolUseBlock(id="t1", name="my_tool", input={"x": "1"})],
                stop_reason="tool_use",
            )
        return AgentResponse(
            content=[AgentTextBlock(text="done")],
            stop_reason="end_turn",
        )

    client = AsyncMock()
    client.complete.side_effect = mock_complete
    agent._client = client

    result = await agent.chat("go", session_id="s")
    assert result == "done"
    assert call_count == 2


# ── Backward-compat shim ──────────────────────────────────────────────────────

def test_build_system_prompt_shim_judge():
    from src.agent.prompt import build_system_prompt
    p = build_system_prompt("judge")
    assert "Phoebe" in p


def test_build_system_prompt_shim_admin():
    from src.agent.prompt import build_system_prompt
    p = build_system_prompt("admin")
    assert "admin" in p.lower()


def test_build_system_prompt_shim_redteam():
    from src.agent.prompt import build_system_prompt
    p = build_system_prompt("redteam")
    assert "red team" in p.lower()


# ── Phoebe config ─────────────────────────────────────────────────────────────

def test_phoebe_config_modes():
    from agents.phoebe.config import PHOEBE_CONFIG
    assert PHOEBE_CONFIG.name == "Phoebe"
    assert set(PHOEBE_CONFIG.available_modes()) == {"judge", "admin", "redteam"}


def test_phoebe_config_default_mode():
    from agents.phoebe.config import PHOEBE_CONFIG
    assert PHOEBE_CONFIG.default_mode == "judge"


# ── Sara base config ──────────────────────────────────────────────────────────

def test_sara_base_config_modes():
    from agents.sara_base.config import SARA_BASE_CONFIG
    assert SARA_BASE_CONFIG.name == "Sara"
    assert set(SARA_BASE_CONFIG.available_modes()) == {"assistant", "analyst", "researcher"}
