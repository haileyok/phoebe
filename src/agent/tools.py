"""
Pluggable tool executor interface.

Each agent or deployment context provides its own ToolExecutor implementation.
The Agent class only calls the abstract interface — it never imports
deployment-specific tool code directly.

Built-in executors:
  - NullToolExecutor      → no tools (text-only agents)
  - CodeToolExecutor      → execute_code only (original Phoebe behaviour)
  - CompositeToolExecutor → combine multiple executors

Third-party executors live in their own modules:
  - src/openclaw/adapter.py  → OpenClaw MCP bridge
  - agents/phoebe/tools.py   → Sandbox Arena tools
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


# ── Abstract base ─────────────────────────────────────────────────────────────

class ToolExecutor(ABC):
    """
    Abstract interface every tool backend must implement.

    get_tool_definitions() returns Anthropic-format tool dicts.
    The framework converts these to OAI format automatically when
    using an OpenAI-compatible client.
    """

    @abstractmethod
    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Anthropic-format tool definitions for this executor."""
        ...

    @abstractmethod
    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool by name with the given input dict.
        Always returns a dict. On error, return {"error": "<message>"}.
        """
        ...

    # Legacy compat shim — original code called get_execute_code_tool_definition()
    def get_execute_code_tool_definition(self) -> dict[str, Any]:
        defs = self.get_tool_definitions()
        for d in defs:
            if d.get("name") == "execute_code":
                return d
        raise RuntimeError("No execute_code tool registered in this executor")


# ── Null executor ─────────────────────────────────────────────────────────────

class NullToolExecutor(ToolExecutor):
    """Use when the agent needs no tools (pure language tasks)."""

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return []

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        return {"error": f"No tools available. Called: {tool_name}"}


# ── Code executor (original behaviour) ───────────────────────────────────────

class CodeToolExecutor(ToolExecutor):
    """
    Wraps the original ToolExecutor.execute_code behaviour.
    Import your existing ToolExecutor and pass it in.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [self._inner.get_execute_code_tool_definition()]

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "execute_code":
            code = tool_input.get("code", "")
            return await self._inner.execute_code(code)
        return {"error": f"Unknown tool: {tool_name}"}


# ── Composite executor ────────────────────────────────────────────────────────

class CompositeToolExecutor(ToolExecutor):
    """
    Fan-out executor: combine multiple ToolExecutors into one.

    Tool name resolution is first-match; add executors in priority order.

    Example:
        executor = CompositeToolExecutor([
            CodeToolExecutor(inner),
            OpenClawMCPExecutor(mcp_client),
            SearchToolExecutor(search_client),
        ])
    """

    def __init__(self, executors: list[ToolExecutor]) -> None:
        self._executors = executors
        self._registry: dict[str, ToolExecutor] = {}
        for ex in executors:
            for defn in ex.get_tool_definitions():
                name = defn["name"]
                if name not in self._registry:
                    self._registry[name] = ex

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result = []
        for ex in self._executors:
            for d in ex.get_tool_definitions():
                if d["name"] not in seen:
                    result.append(d)
                    seen.add(d["name"])
        return result

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        executor = self._registry.get(tool_name)
        if executor is None:
            return {"error": f"No executor registered for tool: {tool_name}"}
        return await executor.execute(tool_name, tool_input)


# ── Tool definition helpers ───────────────────────────────────────────────────

def make_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Helper to build an Anthropic-format tool definition."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }
