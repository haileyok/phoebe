from typing import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


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


@TOOL_REGISTRY.tool(
    name="osprey.listRuleFiles",
    description="List existing .sml rule files in the ruleset. Use this before saving a rule to check for naming collisions.",
    parameters=[
        ToolParameter(
            name="directory",
            type="string",
            description="Subdirectory to search within (relative to ruleset root), e.g. 'rules/my_rules'. Defaults to 'rules/'.",
            required=False,
        ),
    ],
)
async def osprey_list_rule_files(
    ctx: ToolContext, directory: str | None = None
) -> dict[str, Any]:
    """list .sml files in the ruleset"""
    files = ctx.osprey.list_rule_files(directory)
    return {"files": files}


@TOOL_REGISTRY.tool(
    name="osprey.readRuleFile",
    description="Read the contents of an existing .sml rule file in the ruleset.",
    parameters=[
        ToolParameter(
            name="file_path",
            type="string",
            description="Path to the .sml file relative to the ruleset root, e.g. 'rules/my_rules/example.sml'.",
            required=True,
        ),
    ],
)
async def osprey_read_rule_file(ctx: ToolContext, file_path: str) -> dict[str, Any]:
    """read an .sml rule file"""
    content = ctx.osprey.read_rule_file(file_path)
    return {"file_path": file_path, "content": content}


@TOOL_REGISTRY.tool(
    name="osprey.saveRule",
    description="Save an .sml rule file to the ruleset. Creates parent directories if needed. New files (except index.sml) are auto-registered in the parent directory's index.sml. Call osprey.listRuleFiles first to check for existing files.",
    parameters=[
        ToolParameter(
            name="file_path",
            type="string",
            description="Path for the .sml file relative to the ruleset root, e.g. 'rules/my_rules/new_rule.sml'.",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The full SML content to write to the file.",
            required=True,
        ),
        ToolParameter(
            name="require_if",
            type="string",
            description="Optional condition for the Require() entry in index.sml, e.g. a feature flag. Only used when creating new files.",
            required=False,
        ),
    ],
)
async def osprey_save_rule(
    ctx: ToolContext,
    file_path: str,
    content: str,
    require_if: str | None = None,
) -> dict[str, Any]:
    """save an .sml rule file to the ruleset"""
    return ctx.osprey.save_rule(file_path, content, require_if)


@TOOL_REGISTRY.tool(
    name="osprey.validateRules",
    description="Validate the Osprey ruleset using the linter",
    parameters=[],
)
async def osprey_validate_rules(ctx: ToolContext) -> dict[str, Any]:
    """validates the osprey ruleset using the linter"""
    success, error = await ctx.osprey.validate_rules()

    return {
        "success": success,
        "error": error if not success else None,
    }
