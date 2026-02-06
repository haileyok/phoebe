from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="content.similarity",
    description="Find similar posts in the network using ClickHouse's ngramDistance function. Useful for detecting coordinated spam, copypasta, or templated abuse content. Returns posts ordered by similarity score.",
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="The input text to find similar posts for",
        ),
        ToolParameter(
            name="threshold",
            type="number",
            description="Similarity threshold (0.0 = identical, 1.0 = completely different). Lower values return more similar results.",
            required=False,
            default=0.4,
        ),
        ToolParameter(
            name="limit",
            type="number",
            description="Maximum number of results to return (max 100)",
            required=False,
            default=20,
        ),
    ],
)
async def content_similarity(
    ctx: ToolContext,
    text: str,
    threshold: float = 0.4,
    limit: int = 20,
) -> dict[str, Any]:
    limit = min(max(1, int(limit)), 100)
    threshold = max(0.0, min(1.0, float(threshold)))

    escaped_text = text.replace("'", "\\'")

    sql = f"""
        SELECT
            UserId AS user_id,
            UserHandle AS handle,
            PostText AS post_text,
            ngramDistance(PostText, '{escaped_text}') AS distance,
            __timestamp AS timestamp
        FROM default.osprey_execution_results
        WHERE PostText != ''
          AND ngramDistance(PostText, '{escaped_text}') < {threshold}
        ORDER BY distance ASC
        LIMIT {limit}
    """

    resp = await ctx.clickhouse.query(sql)

    rows = []
    for row in resp.result_rows:  # type: ignore
        rows.append(  # type: ignore
            {
                "user_id": row[0],
                "handle": row[1],
                "post_text": row[2],
                "similarity_score": round(1.0 - row[3], 4),  # type: ignore
                "distance": round(row[3], 4),  # type: ignore
                "timestamp": str(row[4]),  # type: ignore
            }
        )

    return {
        "success": True,
        "input_text": text[:200],
        "threshold": threshold,
        "result_count": len(rows),  # type: ignore
        "results": rows,
    }
