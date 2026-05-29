"""
Skill Configuration Studio.

Converts natural-language org descriptions into structured SkillFiles
using Sara's judge mode — Sara configures Sara.
"""

from __future__ import annotations

import json
import logging

from src.agent.agent import Agent
from src.agent.config import AgentConfig, ModelConfig
from src.agent.session import InMemorySessionStore
from src.agent.tools import NullToolExecutor
from src.sarabox.models import SkillFile
from src.sarabox.taxonomy import get_taxonomy_for_org_type

logger = logging.getLogger(__name__)

SKILL_BUILDER_PROMPT = """\
You are Sara's skill configuration assistant. Your job is to parse an
organisation's natural-language description into a structured safety skill file.

Given the org description, you must:
1. Identify the org type (dao, defi, nft, bridge, custom)
2. Select the most relevant attack categories from the taxonomy
3. Customise each category's description and examples for this specific org
4. Set severity levels based on the org's risk profile
5. Write a system_prompt_extension: 2-3 sentences of domain context that
   Sara should use when classifying prompts for this org

Respond ONLY with valid JSON matching the SkillFile schema.
Do not include any explanation or markdown fencing.
"""


def _make_skill_builder_config() -> AgentConfig:
    """Build an AgentConfig for the skill builder assistant."""
    return AgentConfig(
        name="SkillBuilder",
        description="Sara's skill configuration assistant",
        default_mode="build",
        modes={"build": SKILL_BUILDER_PROMPT},
        default_model=ModelConfig.anthropic(),
        tool_tags=[],
    )


class SkillBuilder:

    def __init__(self, model: ModelConfig | None = None):
        self._agent = Agent(
            config=_make_skill_builder_config(),
            tool_executor=NullToolExecutor(),
            session_store=InMemorySessionStore(),
            model_override=model or ModelConfig.anthropic(),
        )

    async def build_from_description(
        self,
        org_id: str,
        description: str,
        org_type: str = "custom",
    ) -> SkillFile:
        """
        Convert a natural-language org description into a SkillFile.

        This calls Sara's judge mode once with the description and the
        relevant taxonomy as context. Returns a structured SkillFile.
        """
        taxonomy = get_taxonomy_for_org_type(org_type)
        prompt = (
            f"Organisation description: {description}\n"
            f"Organisation type: {org_type}\n"
            f"Available attack categories: {json.dumps(taxonomy, indent=2)}\n\n"
            f"Generate a SkillFile JSON for this organisation."
        )
        response = await self._agent.chat(
            prompt,
            session_id=f"skill-build-{org_id}",
            mode="build",
        )
        # Strip any accidental markdown fencing
        clean = (
            response.strip()
            .lstrip("```json")
            .lstrip("```")
            .rstrip("```")
            .strip()
        )
        data = json.loads(clean)
        data["org_id"] = org_id
        data["org_type"] = org_type
        return SkillFile(**data)

    async def preview_classification(
        self,
        skill_file: SkillFile,
        sample_prompt: str,
    ) -> dict:
        """
        Show how Sara would classify a sample prompt using this skill file.
        Used for real-time preview in the configuration UI.
        """
        from src.sarabox.classifier import SaraBoxClassifier
        classifier = SaraBoxClassifier(skill_file=skill_file)
        result = await classifier.classify(sample_prompt)
        return result.model_dump()
