import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

DENO_DIR = Path(__file__).parent / "deno"


class ToolExecutor:
    """executor that runs Typescript code in a deno subprocess"""

    def __init__(self, registry: ToolRegistry, ctx: ToolContext) -> None:
        self._registry = registry
        self._ctx = ctx
        self._database_schema: str | None = None
        self._osprey_features: str | None = None
        self._osprey_labels: str | None = None
        self._osprey_udfs: str | None = None
        self._osprey_rule_files: str | None = None
        self._tool_definition: dict[str, Any] | None = None

    async def initialize(self) -> None:
        # prefetch data for inclusion in the tool description, so that the agent doesn't waste tool calls on discovery
        try:
            schema = await self._registry.execute(self._ctx, "clickhouse.getSchema", {})
            lines = [f"  {col['name']} ({col['type']})" for col in schema]
            self._database_schema = "\n".join(lines)
            logger.info("Prefetched database schema (%d columns)", len(schema))
        except Exception:
            logger.warning("Failed to prefetch database schema", exc_info=True)

        try:
            config = await self._registry.execute(self._ctx, "osprey.getConfig", {})
            feature_lines = [f"  {name}: {ftype}" for name, ftype in config["features"].items()]
            self._osprey_features = "\n".join(feature_lines)

            label_lines = [
                f"  {l['name']}: {l['description']} (valid for: {l['valid_for']})"
                for l in config["labels"]
            ]
            self._osprey_labels = "\n".join(label_lines)

            logger.info(
                "Prefetched osprey config (%d features, %d labels)",
                len(config["features"]),
                len(config["labels"]),
            )
        except Exception:
            logger.warning("Failed to prefetch osprey config", exc_info=True)

        try:
            udfs = await self._registry.execute(self._ctx, "osprey.getUdfs", {})
            udf_lines: list[str] = []
            for cat in udfs["categories"]:
                udf_lines.append(f"## {cat['name']}")
                for udf in cat["udfs"]:
                    udf_lines.append(f"  {udf['signature']}")
                    if udf.get("doc"):
                        first_line = udf["doc"].strip().split("\n")[0]
                        udf_lines.append(f"    {first_line}")
            self._osprey_udfs = "\n".join(udf_lines)
            logger.info("Prefetched osprey UDFs")
        except Exception:
            logger.warning("Failed to prefetch osprey UDFs", exc_info=True)

        try:
            result = await self._registry.execute(self._ctx, "osprey.listRuleFiles", {})
            self._osprey_rule_files = "\n".join(f"  {f}" for f in result["files"])
            logger.info("Prefetched osprey rule files (%d files)", len(result["files"]))
        except Exception:
            logger.warning("Failed to prefetch osprey rule files", exc_info=True)

    async def execute_code(self, code: str) -> dict[str, Any]:
        """
        execute Typescript code in a deno subprocess.

        code has access to tools defined in the registry via the generated typescript
        stubs. calls are bridged to pythin via stdin/out
        """

        self._write_generated_tools()

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", delete=False, dir=DENO_DIR
        ) as f:
            # start by adding all the imports that we need...
            full_code = f"""
import {{ output, debug }} from "./runtime.ts";
import * as tools from "./tools.ts";
export {{ tools }};

{code}
"""
            f.write(full_code)
            temp_path = f.name

        try:
            return await self._run_deno(temp_path)
        finally:
            os.unlink(temp_path)

    def _write_generated_tools(self) -> None:
        """generate tool stubs and write them to the deno directory"""

        tools_ts = self._registry.generate_typescript_types()
        tools_path = DENO_DIR / "tools.ts"
        tools_path.write_text(tools_ts)

    async def _run_deno(self, script_path: str) -> dict[str, Any]:
        """run the input script in a deno subprocess"""

        # spawn a subprocess that executes deno. the deno sandbox has no permissions for network, execing, etc
        # all it can do is read the input directory and read/write from stdin/out
        process = await asyncio.create_subprocess_exec(
            "deno",
            "run",
            "--allow-read=" + str(DENO_DIR),  # allow just reading of the deno directory
            script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        outputs: list[Any] = []
        debug_messages: list[str] = []
        error: str | None = None

        try:
            while True:
                # start reading the response from deno, with a 30 second timeout
                line = await asyncio.wait_for(process.stdout.readline(), timeout=30.0)

                # if there are no more lines we're finished...
                if not line:
                    break

                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    message = json.loads(line_str)
                except json.JSONDecodeError:
                    debug_messages.append(line_str)
                    continue

                # whenever we encounter a tool call, we then need to execute that tool and give
                # it the response
                if "__tool_call__" in message:
                    tool_name = message["tool"]
                    params = message["params"]
                    logger.info(f"Tool call: {tool_name} with params: {params}")

                    try:
                        result = await self._registry.execute(
                            self._ctx, tool_name, params
                        )
                        response = json.dumps({"__tool_result__": result}, default=str)
                    except Exception as e:
                        logger.exception(f"Tool error: {tool_name}")
                        response = json.dumps({"__tool_error__": str(e)})

                    process.stdin.write((response + "\n").encode())
                    await process.stdin.drain()

                elif "__output__" in message:
                    outputs.append(message["__output__"])

                elif "__debug__" in message:
                    debug_messages.append(message["__debug__"])

                else:
                    debug_messages.append(line_str)

        # make sure that we kill deno subprocess if the execution times out
        except asyncio.TimeoutError:
            process.kill()
            error = "execution timed out after 30 seconds"
        # also kill it for any other exceptiosn we encounter
        except Exception as e:
            process.kill()
            error = str(e)

        await process.wait()

        stderr_content = await process.stderr.read()
        if stderr_content:
            stderr_str = stderr_content.decode().strip()
            if stderr_str:
                if error:
                    error += f"\n\nStderr:\n{stderr_str}"
                else:
                    error = stderr_str

        success = process.returncode == 0 and error is None

        result: dict[str, Any] = {
            "success": success,
            "debug": debug_messages,
        }

        if outputs:
            result["output"] = outputs[-1] if len(outputs) == 1 else outputs

        if error:
            result["error"] = error

        return result

    def get_execute_code_tool_definition(self) -> dict[str, Any]:
        """get the anthropic tool definition for execute_code, including all the docs for available backend tools"""

        if self._tool_definition is not None:
            return self._tool_definition

        tool_docs = self._registry.generate_tool_documentation()

        schema_section = ""
        if self._database_schema:
            schema_section = f"""

# Database Schema

The `default.osprey_execution_results` table has these columns:
{self._database_schema}

Use these exact column names when writing SQL queries. Do NOT guess column names.
"""

        osprey_section = ""
        if self._osprey_features:
            osprey_section += f"""

# Available Osprey Features

These features are available in rule conditions (pre-loaded — no need to call getConfig):
{self._osprey_features}
"""
        if self._osprey_labels:
            osprey_section += f"""
# Available Labels

Use these label names with AtprotoLabel effects:
{self._osprey_labels}
"""
        if self._osprey_udfs:
            osprey_section += f"""
# Available UDFs and Effects

These are the available functions for use in rules (pre-loaded — no need to call getUdfs):
{self._osprey_udfs}
"""
        if self._osprey_rule_files:
            osprey_section += f"""
# Existing Rule Files

These .sml files already exist in the ruleset (pre-loaded — no need to call listRuleFiles unless you need a fresh list after mutations):
{self._osprey_rule_files}
"""

        description = f"""Execute Typescript code in a sandboxed Deno runtime.

The code has access to backend tools via the `tools` namespace. Use `output()` to return results.

Example:
```typescript
const result = await tools.clickhouse.query("SELECT count() FROM events");
output(result);
```

{tool_docs}{schema_section}{osprey_section}"""

        self._tool_definition = {
            "name": "execute_code",
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Typescript code to execute. Has access to `tools` namespace and `output()` function.",
                    }
                },
                "required": ["code"],
            },
        }
        return self._tool_definition
