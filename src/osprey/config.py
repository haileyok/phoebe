from pydantic import BaseModel


class CurrentUser(BaseModel):
    email: str


class DefaultSummaryFeature(BaseModel):
    actions: list[str]
    features: list[str]


class FeatureLocation(BaseModel):
    name: str
    source_line: int
    source_path: str
    source_snippet: str


class LabelInfo(BaseModel):
    connotation: str
    description: str
    valid_for: list[str]


class OspreyConfig(BaseModel):
    current_user: CurrentUser
    default_summary_features: list[DefaultSummaryFeature] = []
    external_links: dict[str, str] = {}
    feature_name_to_entity_type_mapping: dict[str, str] = {}
    feature_name_to_value_type_mapping: dict[str, str] = {}
    known_action_names: list[str] = []
    known_feature_locations: list[FeatureLocation] = []
    label_info_mapping: dict[str, LabelInfo] = {}
    rule_info_mapping: dict[str, str] = {}

    def get_available_features(self) -> dict[str, str]:
        return self.feature_name_to_value_type_mapping

    def get_available_labels(self) -> list[str]:
        return list(self.label_info_mapping.keys())

    def get_existing_rules(self) -> dict[str, str]:
        return self.rule_info_mapping

    def get_feature_examples(self, feature_name: str) -> FeatureLocation | None:
        for loc in self.known_feature_locations:
            if loc.name == feature_name:
                return loc
        return None

    def format_features_for_llm(self) -> str:
        lines = ["# Available Features\n"]
        lines.append("These features are already extracted and available in rules:\n")

        by_type: dict[str, list[str]] = {}
        for name, typ in sorted(self.feature_name_to_value_type_mapping.items()):
            by_type.setdefault(typ, []).append(name)

        for typ, names in sorted(by_type.items()):
            lines.append(f"\n## Type: `{typ}`")
            for name in sorted(names):
                lines.append(f"- {name}")

        return "\n".join(lines)

    def format_labels_for_llm(self) -> str:
        lines = ["# Available Labels\n"]
        lines.append(
            "You can add new labels if you feel none of the current ones fit, but be sure to update the config."
        )

        for name, info in sorted(self.label_info_mapping.items()):
            valid_for = ", ".join(info.valid_for)
            lines.append(f"- `{name}`: {info.description} (valid for: {valid_for})")

        return "\n".join(lines)

    def format_existing_rules_for_llm(self) -> str:
        lines = ["# Existing Rules\n"]
        lines.append("These rules already exist and can be referenced:\n")

        for name, desc in sorted(self.rule_info_mapping.items()):
            lines.append(f"- `{name}`: {desc}")

        return "\n".join(lines)

    def format_feature_examples_for_llm(self, feature_names: list[str]) -> str:
        lines = ["# Feature Definition Examples\n"]

        for name in feature_names:
            loc = self.get_feature_examples(name)
            if loc:
                lines.append(f"## {name}")
                lines.append(f"Source: `{loc.source_path}:{loc.source_line}`")
                lines.append(f"```python\n{loc.source_snippet}\n```\n")

        return "\n".join(lines)
