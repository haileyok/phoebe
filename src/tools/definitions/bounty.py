"""
Bounty management tools — query and manage arena bounties.

Uses the ClickHouse-backed ArenaStore for persistent state.
Available in the Deno sandbox so Phoebe's agent can inspect bounties
during evaluation, and for interactive chat mode.
"""

from typing import Any

from src.arena.taxonomy import ALL_CATEGORIES, CATEGORY_DESCRIPTIONS, SafetyCategory
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="bounty.list",
    description="List all active bounties in the arena.",
    parameters=[],
)
async def bounty_list(ctx: ToolContext) -> dict[str, Any]:
    """List active bounties from ClickHouse."""
    store = ctx.arena.store
    active = await store.list_active_bounties()
    bounties = [
        {
            "bounty_id": b.bounty_id,
            "target_model": b.target_model_name,
            "pool_usdc": b.pool_usdc,
            "remaining_usdc": b.remaining_usdc,
            "categories": b.categories,
            "max_per_finding": b.max_payout_per_finding,
        }
        for b in active
    ]
    return {"bounties": bounties, "count": len(bounties)}


@TOOL_REGISTRY.tool(
    name="bounty.get",
    description="Get details of a specific bounty.",
    parameters=[
        ToolParameter(
            name="bounty_id",
            type="string",
            description="The bounty ID to look up",
        ),
    ],
)
async def bounty_get(ctx: ToolContext, bounty_id: str) -> dict[str, Any]:
    """Get bounty details from ClickHouse."""
    store = ctx.arena.store
    bounty = await store.get_bounty(bounty_id)
    if not bounty:
        return {"error": f"Bounty {bounty_id} not found"}
    return bounty.model_dump()


@TOOL_REGISTRY.tool(
    name="bounty.taxonomy",
    description="Get the full safety taxonomy with category descriptions and current attack coverage.",
    parameters=[],
)
async def bounty_taxonomy(ctx: ToolContext) -> dict[str, Any]:
    """Return taxonomy categories and coverage from ClickHouse."""
    store = ctx.arena.store
    coverage = await store.get_category_coverage()

    categories = []
    for cat in SafetyCategory:
        categories.append({
            "id": cat.value,
            "description": CATEGORY_DESCRIPTIONS.get(cat, ""),
            "attacks_found": coverage.get(cat.value, 0),
        })

    return {"categories": categories, "total_categories": len(ALL_CATEGORIES)}
