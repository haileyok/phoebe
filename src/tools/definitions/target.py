"""
Target model tools — query the model being red-teamed.

These tools are available in the Deno sandbox so Phoebe's LLM agent can
interact with the target model during evaluation.
"""

from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="target.generate",
    description="Send a prompt to the target model and return its response. Payment is handled automatically via x402 if the target model requires it.",
    parameters=[
        ToolParameter(
            name="prompt",
            type="string",
            description="The prompt to send to the target model",
        ),
        ToolParameter(
            name="max_tokens",
            type="number",
            description="Maximum tokens in the response",
            required=False,
            default=1000,
        ),
    ],
)
async def target_generate(
    ctx: ToolContext,
    prompt: str,
    max_tokens: int = 1000,
) -> dict[str, Any]:
    """Query the target model with a single prompt."""
    arena = ctx.arena
    bounty = arena.active_bounty

    if bounty is None:
        return {"error": "No active bounty set. Use arena.set_bounty first."}

    try:
        resp = await ctx.x402_client.post(
            bounty.target_model_endpoint,
            json={
                "model": bounty.target_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
        )

        if not resp.is_success:
            return {"error": f"Target model HTTP {resp.status_code}", "body": resp.text[:500]}

        data = resp.json()

        # extract text from Anthropic or OpenAI response formats
        text = ""
        if "content" in data and isinstance(data["content"], list):
            text = data["content"][0].get("text", "")
        elif "choices" in data:
            text = data["choices"][0]["message"]["content"]

        return {
            "output": text,
            "payment_cost": resp.payment_cost,
        }

    except Exception as e:
        return {"error": str(e)}


@TOOL_REGISTRY.tool(
    name="target.batch_generate",
    description="Send multiple prompts to the target model. Returns results for each prompt.",
    parameters=[
        ToolParameter(
            name="prompts",
            type="array",
            description="Array of prompt strings to send",
        ),
        ToolParameter(
            name="max_tokens",
            type="number",
            description="Maximum tokens per response",
            required=False,
            default=500,
        ),
    ],
)
async def target_batch_generate(
    ctx: ToolContext,
    prompts: list[str],
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Query the target model with multiple prompts."""
    import asyncio

    if len(prompts) > 20:
        return {"error": "Maximum 20 prompts per batch"}

    async def query_one(prompt: str) -> dict[str, Any]:
        return await target_generate(ctx, prompt=prompt, max_tokens=max_tokens)

    results = await asyncio.gather(*(query_one(p) for p in prompts))
    return {"results": list(results), "count": len(results)}
