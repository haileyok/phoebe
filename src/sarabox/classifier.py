"""
Sara Box Classifier — wraps SafetyClassifier with skill-file context injection.

Each org gets a personalised classifier without a separate fine-tuned model;
the skill file IS the customisation.
"""

from __future__ import annotations

import time
from typing import Any

from src.safety.classifier import SafetyClassifier
from src.sarabox.models import ClassificationResult, SkillFile


class SaraBoxClassifier:
    """
    Sara's safety classifier configured for a specific org's skill file.

    At inference time, the skill file's system_prompt_extension and
    category definitions are injected as context into the classification
    request. The base SafetyClassifier handles the actual LLM call.
    """

    def __init__(
        self,
        skill_file: SkillFile,
        base_classifier: SafetyClassifier | None = None,
    ):
        self._skill = skill_file
        self._classifier = base_classifier

    def _build_context(self) -> str:
        """
        Build the domain context string injected at classification time.
        Combines the skill file's system prompt extension with a
        compact category reference for the classifier to use.
        """
        category_lines = "\n".join(
            f"- {cat.name} (severity={cat.severity}, threshold={cat.threshold}): {cat.description}"
            for cat in self._skill.categories
        )
        return (
            f"Organisation context: {self._skill.system_prompt_extension}\n\n"
            f"Active safety categories for this organisation:\n"
            f"{category_lines}\n\n"
            f"Classify the input prompt against these categories specifically."
        )

    async def classify(self, prompt: str) -> ClassificationResult:
        t0 = time.monotonic()
        context = self._build_context()

        # The base SafetyClassifier interface is (prompt, model_output, category).
        # We prepend the skill context to the prompt and classify directly.
        classifier = self._classifier
        if classifier is None:
            # STUB: replace with real classifier instantiation in Phase 5
            # For MVP, raise so callers must provide a base_classifier
            raise RuntimeError(
                "SaraBoxClassifier requires a base_classifier. "
                "Pass a SafetyClassifier instance at construction time."
            )

        full_prompt = f"{context}\n\nPrompt to classify: {prompt}"
        result = await classifier.classify(
            prompt=full_prompt,
            model_output="[direct prompt classification — no target model output]",
            category="general",
        )
        latency_ms = (time.monotonic() - t0) * 1000

        label = "unsafe" if result.get("unsafe") else "safe"
        confidence = result.get("confidence", 0.0)
        matched = self._match_category(prompt, result)

        return ClassificationResult(
            prompt=prompt,
            label=label,
            confidence=confidence,
            matched_category=matched,
            explanation=result.get("explanation", ""),
            latency_ms=round(latency_ms, 2),
        )

    def _match_category(self, prompt: str, result: dict[str, Any]) -> str | None:
        """
        Attempt to match the classification result to a specific skill category.
        Uses a lightweight keyword approach for MVP.
        # STUB: replace with embedding similarity in Phase 3 (retrieval infra)
        """
        category_id = result.get("policy_category") or result.get("category")
        if category_id:
            return category_id
        # Fallback: keyword match against category descriptions
        prompt_lower = prompt.lower()
        for cat in self._skill.categories:
            if cat.id.replace("_", " ") in prompt_lower or any(
                ex.lower() in prompt_lower for ex in cat.examples[:2]
            ):
                return cat.id
        return None

    async def classify_batch(
        self, prompts: list[str]
    ) -> list[ClassificationResult]:
        import asyncio
        return await asyncio.gather(*[self.classify(p) for p in prompts])
