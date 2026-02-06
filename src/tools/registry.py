from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.clickhouse.clickhouse import Clickhouse
from src.osprey.osprey import Osprey
from src.ozone.ozone import Ozone


class ToolParameter(BaseModel):
    """tool parameter definition"""

    name: str
    type: Literal["string", "number", "boolean", "object", "array"]
    description: str
    required: bool = True
    default: Any = None


class Tool(BaseModel):
    """tool definition that can be executed in deno sandbox"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: list[ToolParameter]
    handler: Callable[..., Awaitable[Any]]  # async function


class ToolContext:
    """a context that has access to various backend services that are available to deno sandboxed tools"""

    def __init__(
        self,
        clickhouse: Clickhouse | None = None,
        osprey: Osprey | None = None,
        ozone: Ozone | None = None,
    ) -> None:
        self._clickhouse = clickhouse
        self._osprey = osprey
        self._ozone = ozone

    @property
    def clickhouse(self) -> Clickhouse:
        if self._clickhouse is None:
            raise RuntimeError("Clickhouse client not configured")
        return self._clickhouse

    @property
    def osprey(self) -> Osprey:
        if self._osprey is None:
            raise RuntimeError("Osprey client not configured")
        return self._osprey

    @property
    def ozone(self) -> Ozone:
        if self._ozone is None:
            raise RuntimeError("Ozone client not configured")
        return self._ozone


class ToolRegistry:
    """a registry of all the available tools"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter] | None = None,
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """the main tool decorator for tools that you create"""

        def decorator(
            func: Callable[..., Awaitable[Any]],
        ) -> Callable[..., Awaitable[Any]]:
            tool = Tool(
                name=name,
                description=description,
                parameters=parameters or [],
                handler=func,
            )
            self.register(tool)
            return func

        return decorator

    async def execute(self, ctx: ToolContext, name: str, params: dict[str, Any]) -> Any:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        if len(params) == 1:
            param_names = {p.name for p in tool.parameters}
            val = next(iter(params.values()))
            if isinstance(val, dict) and set(val.keys()) <= param_names:
                params = val

        return await tool.handler(ctx, **params)

    def generate_tool_documentation(self) -> str:
        """generated tool documentation is passed to the llm so it knows how to run a given command inside its execute_tool call"""

        lines = ["# Available Tools\n"]
        lines.append(
            "These tools are available to call from TypeScript code in execute_code:\n"
        )

        by_namespace: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            namespace = tool.name.split(".")[0]
            by_namespace.setdefault(namespace, []).append(tool)

        for namespace, tools in sorted(by_namespace.items()):
            lines.append(f"## {namespace}\n")
            for tool in sorted(tools, key=lambda t: t.name):
                lines.append(f"### {tool.name}")
                lines.append(f"{tool.description}\n")
                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param in tool.parameters:
                        req = "" if param.required else " (optional)"
                        default = (
                            f", default: {param.default}"
                            if param.default is not None
                            else ""
                        )
                        lines.append(
                            f"- `{param.name}` ({param.type}{req}{default}): {param.description}"
                        )
                    lines.append("")

        return "\n".join(lines)

    def generate_typescript_types(self) -> str:
        lines = [
            "// Auto-generated - do not edit",
            'import { callTool } from "./runtime.ts";',
            "",
        ]

        by_namespace: dict[str, list[Tool]] = {}
        for tool in self._tools.values():
            namespace = tool.name.split(".")[0]
            by_namespace.setdefault(namespace, []).append(tool)

        for namespace, tools in sorted(by_namespace.items()):
            lines.append(f"export const {namespace} = {{")
            for i, tool in enumerate(sorted(tools, key=lambda t: t.name)):
                method_name = tool.name.split(".", 1)[1]

                required_params = [p for p in tool.parameters if p.required]
                optional_params = [p for p in tool.parameters if not p.required]
                ordered_params = required_params + optional_params

                params: list[str] = []
                for param in ordered_params:
                    ts_type = self._python_type_to_ts(param.type)
                    if param.required:
                        params.append(f"{param.name}: {ts_type}")
                    else:
                        params.append(f"{param.name}?: {ts_type}")

                param_str = ", ".join(params)

                param_names = [p.name for p in tool.parameters]
                params_obj = (
                    "{ " + ", ".join(param_names) + " }" if param_names else "{}"
                )

                lines.append(f"  /** {tool.description} */")
                lines.append(
                    f'  {method_name}: ({param_str}): Promise<unknown> => callTool("{tool.name}", {params_obj}),'
                )
                if i < len(tools) - 1:
                    lines.append("")

            lines.append("};")
            lines.append("")

        return "\n".join(lines)

    def _python_type_to_ts(self, py_type: str) -> str:
        mapping = {
            "string": "string",
            "number": "number",
            "boolean": "boolean",
            "object": "Record<string, unknown>",
            "array": "unknown[]",
        }
        return mapping.get(py_type, "unknown")

    def _default_to_ts(self, value: Any) -> str:
        if value is None:
            return "undefined"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)


TOOL_REGISTRY = ToolRegistry()
