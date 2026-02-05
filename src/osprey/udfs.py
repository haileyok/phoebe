from pydantic import BaseModel


class UdfArgumentSpec(BaseModel):
    name: str
    type: str
    default: str | None = None
    doc: str | None = None


class Udf(BaseModel):
    name: str
    return_type: str
    argument_specs: list[UdfArgumentSpec] = []
    doc: str | None = None
    category: str | None = None

    def signature(self) -> str:
        args = ", ".join(
            f"{arg.name}: {arg.type}" + (f" = {arg.default}" if arg.default else "")
            for arg in self.argument_specs
        )
        return f"{self.name}({args}) -> {self.return_type}"


class UdfCategory(BaseModel):
    name: str | None = None
    udfs: list[Udf] = []


class UdfCatalog(BaseModel):
    udf_categories: list[UdfCategory] = []

    def all_udfs(self) -> list[Udf]:
        return [udf for cat in self.udf_categories for udf in cat.udfs]

    def find_udf(self, name: str) -> Udf | None:
        name_lower = name.lower()
        for udf in self.all_udfs():
            if udf.name.lower() == name_lower:
                return udf
        return None

    def udfs_by_category(self, category: str) -> list[Udf]:
        for cat in self.udf_categories:
            if cat.name and cat.name.lower() == category.lower():
                return cat.udfs
        return []

    def format_for_llm(self) -> str:
        lines = ["# Available UDFs\n"]

        for cat in self.udf_categories:
            cat_name = cat.name or "Other"
            lines.append(f"## {cat_name}\n")

            for udf in cat.udfs:
                lines.append(f"### {udf.name}")
                lines.append("```")
                lines.append(udf.signature())
                lines.append("```")
                if udf.doc:
                    lines.append(udf.doc)

                if udf.argument_specs:
                    lines.append("\n**Parameters:**")
                    for arg in udf.argument_specs:
                        arg_doc = arg.doc or arg.type
                        default = f" (default: {arg.default})" if arg.default else ""
                        lines.append(f"- `{arg.name}`: {arg_doc}{default}")

                lines.append("")

        return "\n".join(lines)
