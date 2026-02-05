from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


@TOOL_REGISTRY.tool(
    name="ozone.applyLabel",
    description="Apply a moderation label to a subject (account or record)",
    parameters=[
        ToolParameter(
            name="subject",
            type="string",
            description="The subject to label (DID for account, AT-URI for content)",
        ),
        ToolParameter(
            name="label",
            type="string",
            description="The label to apply",
        ),
    ],
)
async def ozone_apply_label(
    ctx: ToolContext, subject: str, label: str
) -> dict[str, Any]:
    raise NotImplementedError("Ozone client not yet implemented")


@TOOL_REGISTRY.tool(
    name="ozone.removeLabel",
    description="Remove a moderation label from a subject (account or record)",
    parameters=[
        ToolParameter(
            name="subject",
            type="string",
            description="The subject to unlabel (DID for account, AT-URI for content)",
        ),
        ToolParameter(
            name="label",
            type="string",
            description="The label to remove",
        ),
    ],
)
async def ozone_remove_label(
    ctx: ToolContext, subject: str, label: str
) -> dict[str, Any]:
    raise NotImplementedError("Ozone client not yet implemented")
