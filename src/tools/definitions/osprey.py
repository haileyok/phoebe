from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext


@TOOL_REGISTRY.tool(
    name="osprey.getConfig",
    description="Get Osprey configuration including available features, labels, and rules",
    parameters=[],
)
async def osprey_get_config(ctx: ToolContext) -> dict[str, Any]:
    """Get Osprey configuration."""
    config = await ctx.osprey.get_config()
    return {
        "features": config.get_available_features(),
        "labels": [
            {
                "name": name,
                "description": info.description,
                "connotation": info.connotation,
                "valid_for": info.valid_for,
            }
            for name, info in config.label_info_mapping.items()
        ],
        "rules": config.get_existing_rules(),
        "actions": config.known_action_names,
    }


@TOOL_REGISTRY.tool(
    name="osprey.getUdfs",
    description="Get available UDFs (user-defined functions) for rule writing",
    parameters=[],
)
async def osprey_get_udfs(ctx: ToolContext) -> dict[str, Any]:
    """Get available UDFs for rule writing."""
    catalog = await ctx.osprey.get_udfs()
    return {
        "categories": [
            {
                "name": cat.name,
                "udfs": [
                    {
                        "name": udf.name,
                        "signature": udf.signature(),
                        "doc": udf.doc,
                        "arguments": [
                            {
                                "name": arg.name,
                                "type": arg.type,
                                "default": arg.default,
                                "doc": arg.doc,
                            }
                            for arg in udf.argument_specs
                        ],
                    }
                    for udf in cat.udfs
                ],
            }
            for cat in catalog.udf_categories
        ]
    }
