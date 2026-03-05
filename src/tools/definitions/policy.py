"""
Policy tools — query the GA Guard content moderation policy taxonomy.

Exposes the policy categories, block/allow rules, and compliance anchors
so that the agent, safety classifier, and red team mode can reference
the structured policy during evaluation and attack generation.
"""

from typing import Any

from src.osprey.policy import (
    ALL_POLICY_CATEGORIES,
    GA_GUARD_POLICY,
    PolicyCategory,
    classify_category,
    format_policy_for_classifier,
    get_policy,
)
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="policy.list",
    description="List all GA Guard policy categories with their goals and compliance anchors.",
    parameters=[],
)
async def policy_list(ctx: ToolContext) -> dict[str, Any]:
    """List all policy categories."""
    categories = []
    for spec in get_policy():
        categories.append({
            "id": spec.id.value,
            "name": spec.name,
            "goal": spec.goal,
            "block_rules_count": len(spec.block_rules),
            "allow_rules_count": len(spec.allow_rules),
            "compliance_anchors": spec.compliance_anchors,
        })
    return {"categories": categories, "total": len(categories)}


@TOOL_REGISTRY.tool(
    name="policy.get",
    description="Get the full block/allow rules for a specific GA Guard policy category. Use this to understand exactly what should be blocked vs. allowed for a given category.",
    parameters=[
        ToolParameter(
            name="category",
            type="string",
            description=f"Policy category ID. One of: {', '.join(ALL_POLICY_CATEGORIES)}. Also accepts aliases like 'pii', 'hate_speech', 'violence', 'jailbreak', etc.",
        ),
    ],
)
async def policy_get(ctx: ToolContext, category: str) -> dict[str, Any]:
    """Get detailed policy for a category."""
    cat = classify_category(category)
    if cat is None:
        return {
            "error": f"Unknown category: {category}",
            "available": ALL_POLICY_CATEGORIES,
        }

    spec = GA_GUARD_POLICY[cat]
    return {
        "id": spec.id.value,
        "name": spec.name,
        "goal": spec.goal,
        "block_rules": [{"description": r.description} for r in spec.block_rules],
        "allow_rules": [{"description": r.description} for r in spec.allow_rules],
        "compliance_anchors": spec.compliance_anchors,
    }


@TOOL_REGISTRY.tool(
    name="policy.classify",
    description="Map a free-form category string to its GA Guard policy category. Useful for normalizing category names from bounties/submissions to the canonical policy taxonomy.",
    parameters=[
        ToolParameter(
            name="category",
            type="string",
            description="A category string to normalize (e.g., 'violent_crime', 'cbrn', 'hate_speech', 'jailbreak').",
        ),
    ],
)
async def policy_classify(ctx: ToolContext, category: str) -> dict[str, Any]:
    """Map a category string to a GA Guard policy category."""
    cat = classify_category(category)
    if cat is None:
        return {
            "input": category,
            "matched": False,
            "available": ALL_POLICY_CATEGORIES,
        }

    spec = GA_GUARD_POLICY[cat]
    return {
        "input": category,
        "matched": True,
        "policy_category": cat.value,
        "policy_name": spec.name,
        "goal": spec.goal,
    }


@TOOL_REGISTRY.tool(
    name="policy.prompt",
    description="Get the full GA Guard policy formatted as a classifier prompt. Useful for passing to safety.classify or for understanding the complete policy in context.",
    parameters=[
        ToolParameter(
            name="categories",
            type="string",
            description="Comma-separated category IDs to include (or 'all' for the full policy).",
            required=False,
        ),
    ],
)
async def policy_prompt(
    ctx: ToolContext, categories: str | None = None
) -> dict[str, Any]:
    """Get the policy formatted as a classifier prompt."""
    cat_list = None
    if categories and categories.strip().lower() != "all":
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]

    prompt_text = format_policy_for_classifier(cat_list)
    return {"prompt": prompt_text}
