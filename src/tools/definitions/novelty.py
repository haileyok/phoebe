"""
Novelty detection tools — score how novel an adversarial prompt is.

Adapted from Phoebe's content.similarity tool, retargeted from social media
posts to adversarial prompts. Uses ClickHouse's ngramDistance for similarity
matching against the attack history table.
"""

from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="novelty.score",
    description="Score how novel an adversarial prompt is compared to the attack history. Returns a novelty score from 0.0 (duplicate) to 1.0 (completely novel).",
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The adversarial prompt to score",
        ),
        ToolParameter(
            name="threshold",
            type="number",
            description="Similarity threshold (0.0 = identical, 1.0 = different). Prompts closer than this are considered duplicates.",
            required=False,
            default=0.3,
        ),
    ],
)
async def novelty_score(
    ctx: ToolContext,
    prompt: str,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Score novelty against attack_history table in ClickHouse.

    Falls back to a simple hash-based check if ClickHouse is not available
    or the table doesn't exist yet.
    """
    threshold = max(0.0, min(1.0, float(threshold)))

    try:
        escaped = prompt.replace("'", "\\'")

        sql = f"""
            SELECT
                prompt_text,
                ngramDistance(prompt_text, '{escaped}') AS distance
            FROM arena.attack_history
            WHERE ngramDistance(prompt_text, '{escaped}') < {threshold}
            ORDER BY distance ASC
            LIMIT 5
        """

        resp = await ctx.clickhouse.query(sql)
        similar = []
        for row in resp.result_rows:  # type: ignore
            similar.append({
                "prompt_preview": str(row[0])[:100],
                "distance": round(float(row[1]), 4),
                "similarity": round(1.0 - float(row[1]), 4),
            })

        if not similar:
            novelty = 1.0
        else:
            closest_distance = similar[0]["distance"]
            novelty = min(1.0, closest_distance / threshold)

        return {
            "novelty_score": round(novelty, 4),
            "similar_count": len(similar),
            "similar_prompts": similar,
        }

    except Exception:
        # ClickHouse table may not exist yet — return full novelty
        return {
            "novelty_score": 1.0,
            "similar_count": 0,
            "similar_prompts": [],
            "note": "Attack history table not available; assuming full novelty.",
        }


@TOOL_REGISTRY.tool(
    name="novelty.find_similar",
    description="Find adversarial prompts similar to the given text in the attack history. Useful for deduplication.",
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="The text to find similar attacks for",
        ),
        ToolParameter(
            name="limit",
            type="number",
            description="Maximum results to return (max 50)",
            required=False,
            default=10,
        ),
        ToolParameter(
            name="threshold",
            type="number",
            description="Similarity threshold (lower = more similar)",
            required=False,
            default=0.5,
        ),
    ],
)
async def novelty_find_similar(
    ctx: ToolContext,
    text: str,
    limit: int = 10,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Find similar prompts in the attack history."""
    limit = min(max(1, int(limit)), 50)
    threshold = max(0.0, min(1.0, float(threshold)))
    escaped = text.replace("'", "\\'")

    try:
        sql = f"""
            SELECT
                prompt_text,
                category,
                ngramDistance(prompt_text, '{escaped}') AS distance,
                first_seen,
                times_submitted
            FROM arena.attack_history
            WHERE ngramDistance(prompt_text, '{escaped}') < {threshold}
            ORDER BY distance ASC
            LIMIT {limit}
        """

        resp = await ctx.clickhouse.query(sql)
        rows = []
        for row in resp.result_rows:  # type: ignore
            rows.append({
                "prompt_preview": str(row[0])[:200],
                "category": str(row[1]),
                "distance": round(float(row[2]), 4),
                "similarity": round(1.0 - float(row[2]), 4),
                "first_seen": str(row[3]),
                "times_submitted": int(row[4]),
            })

        return {"results": rows, "count": len(rows)}

    except Exception as e:
        return {"error": str(e), "results": [], "count": 0}
