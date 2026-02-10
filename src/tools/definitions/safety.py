"""
Safety classification tools — evaluate whether target model outputs are unsafe.

Available in the Deno sandbox so Phoebe's agent can classify outputs during
the scoring pipeline.
"""

from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="safety.classify",
    description="Classify whether a target model's output is unsafe for a given safety category. Returns {unsafe: bool, severity: 1-5, explanation: str}.",
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The adversarial prompt that was sent to the target model",
        ),
        ToolParameter(
            name="model_output",
            type="string",
            description="The target model's response",
        ),
        ToolParameter(
            name="category",
            type="string",
            description="The safety category being tested (e.g. 'cbrn', 'misinformation')",
        ),
    ],
)
async def safety_classify(
    ctx: ToolContext,
    prompt: str,
    model_output: str,
    category: str,
) -> dict[str, Any]:
    """Classify a single prompt/output pair."""
    return await ctx.safety_classifier.classify(
        prompt=prompt,
        model_output=model_output,
        category=category,
    )


@TOOL_REGISTRY.tool(
    name="safety.classify_batch",
    description="Classify multiple prompt/output pairs. Each item should have 'prompt', 'model_output', and 'category' fields.",
    parameters=[
        ToolParameter(
            name="items",
            type="array",
            description="Array of {prompt, model_output, category} objects",
        ),
    ],
)
async def safety_classify_batch(
    ctx: ToolContext,
    items: list[dict[str, str]],
) -> dict[str, Any]:
    """Classify multiple prompt/output pairs."""
    if len(items) > 20:
        return {"error": "Maximum 20 items per batch"}

    results = await ctx.safety_classifier.classify_batch(items)
    return {"results": results, "count": len(results)}
