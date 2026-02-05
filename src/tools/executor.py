import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

# Path to the Deno runtime files
DENO_DIR = Path(__file__).parent / "deno"


class ToolExecutor:
    """An executor that runs Typescript code in a deno subprocess"""

    def __init__(self, registry: ToolRegistry, ctx: ToolContext) -> None:
        self._registry = registry
        self._ctx = ctx

    async def execute_code(self, code: str) -> dict[str, Any]:
        """
        execute Typescript code in a deno subprocess.

        code has access to tools defined in the registry via the generated typescript
        stubs. calls are bridged to pythin via stdin/out

        Returns:
            A dict with keys:
            - "success": bool
            - "output": The final output (if any)
            - "debug": List of debug messages
            - "error": Error message (if failed)
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
                        response = json.dumps({"__tool_result__": result})
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

        tool_docs = self._registry.generate_tool_documentation()

        description = f"""Execute Typescript code in a sandboxed Deno runtime.

The code has access to backend tools via the `tools` namespace. Use `output()` to return results.

Example:
```typescript
const result = await tools.clickhouse.query("SELECT count() FROM events");
output(result);
```

{tool_docs}"""

        return {
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
