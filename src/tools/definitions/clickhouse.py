from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="clickhouse.query",
    description="Execute a SQL query against ClickHouse and return the results. All queries must include a LIMIT, and all queries must be executed on defualt.osprey_execution_results",
    parameters=[
        ToolParameter(
            name="sql",
            type="string",
            description="SQL query to execute",
        ),
    ],
)
async def clickhouse_query(ctx: ToolContext, sql: str) -> dict[str, Any]:
    resp = await ctx.clickhouse.query(sql)

    return {
        "columns": resp.column_names,  # type: ignore
        "rows": resp.result_rows,  # type: ignore
    }


@TOOL_REGISTRY.tool(
    name="clickhouse.getSchema",
    description="Get database schema information including tables and their columns. Schema is for the table default.osprey_execution_results",
    parameters=[],
)
async def clickhouse_get_schema(
    ctx: ToolContext, database: str | None = None
) -> list[dict[str, Any]]:
    resp = await ctx.clickhouse.get_schema()

    return resp
