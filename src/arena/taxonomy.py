"""
Safety taxonomy for the red teaming arena.

Defines the categories of unsafe behavior that red teamers can test for,
severity levels, and coverage tracking.

Categories are aligned with the GA Guard policy taxonomy (7 top-level
categories) while preserving the finer-grained legacy categories for
backward compatibility with existing ClickHouse data.
"""

from enum import Enum

from src.osprey.policy import GA_GUARD_POLICY, PolicyCategory, classify_category


class SafetyCategory(str, Enum):
    """
    Safety taxonomy categories. The top-level categories map 1:1 to the
    GA Guard policy taxonomy. Legacy fine-grained categories are retained
    for backward compatibility.
    """

    # -- GA Guard top-level categories --
    PII_IP = "pii_ip"
    ILLICIT_ACTIVITIES = "illicit_activities"
    HATE = "hate"
    SEXUAL_CONTENT = "sexual_content"
    PROMPT_SECURITY = "prompt_security"
    VIOLENCE_SELF_HARM = "violence_self_harm"
    MISINFORMATION = "misinformation"

    # -- Legacy fine-grained categories (map to GA Guard parents) --
    VIOLENT_CRIME = "violent_crime"
    NON_VIOLENT_CRIME = "non_violent_crime"
    SEX_RELATED = "sex_related"
    CHILD_SAFETY = "child_safety"
    INDISCRIMINATE_WEAPONS = "indiscriminate_weapons"
    CBRN = "cbrn"
    SUICIDE_SELF_HARM = "suicide_self_harm"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    PRIVACY_VIOLATION = "privacy_violation"
    MANIPULATION = "manipulation"
    DEFAMATION = "defamation"
    ELECTION_INTERFERENCE = "election_interference"
    UNAUTHORIZED_PRACTICE = "unauthorized_practice"
    COPYRIGHT_VIOLATION = "copyright_violation"


class Severity(int, Enum):
    """Attack severity levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


# Map legacy fine-grained categories to GA Guard top-level categories
LEGACY_TO_GA_GUARD: dict[SafetyCategory, PolicyCategory] = {
    SafetyCategory.VIOLENT_CRIME: PolicyCategory.VIOLENCE_SELF_HARM,
    SafetyCategory.NON_VIOLENT_CRIME: PolicyCategory.ILLICIT_ACTIVITIES,
    SafetyCategory.SEX_RELATED: PolicyCategory.SEXUAL_CONTENT,
    SafetyCategory.CHILD_SAFETY: PolicyCategory.SEXUAL_CONTENT,
    SafetyCategory.INDISCRIMINATE_WEAPONS: PolicyCategory.ILLICIT_ACTIVITIES,
    SafetyCategory.CBRN: PolicyCategory.ILLICIT_ACTIVITIES,
    SafetyCategory.SUICIDE_SELF_HARM: PolicyCategory.VIOLENCE_SELF_HARM,
    SafetyCategory.HATE_SPEECH: PolicyCategory.HATE,
    SafetyCategory.HARASSMENT: PolicyCategory.HATE,
    SafetyCategory.PRIVACY_VIOLATION: PolicyCategory.PII_IP,
    SafetyCategory.MANIPULATION: PolicyCategory.MISINFORMATION,
    SafetyCategory.DEFAMATION: PolicyCategory.MISINFORMATION,
    SafetyCategory.ELECTION_INTERFERENCE: PolicyCategory.MISINFORMATION,
    SafetyCategory.UNAUTHORIZED_PRACTICE: PolicyCategory.ILLICIT_ACTIVITIES,
    SafetyCategory.COPYRIGHT_VIOLATION: PolicyCategory.PII_IP,
}

# GA Guard top-level categories (the canonical set)
GA_GUARD_CATEGORIES: list[str] = [c.value for c in PolicyCategory]


def resolve_policy_category(category: str) -> PolicyCategory | None:
    """Resolve any category string (legacy or GA Guard) to a PolicyCategory."""
    result = classify_category(category)
    if result:
        return result

    try:
        legacy = SafetyCategory(category)
        return LEGACY_TO_GA_GUARD.get(legacy)
    except ValueError:
        return None


# Human-readable descriptions for UI / leaderboard
CATEGORY_DESCRIPTIONS: dict[SafetyCategory, str] = {
    # GA Guard top-level
    SafetyCategory.PII_IP: "Block prompts containing or seeking identifiable personal data, secrets, or IP",
    SafetyCategory.ILLICIT_ACTIVITIES: "Block operationalization of crime, weapons, or illegal substances",
    SafetyCategory.HATE: "Prevent hate, harassment, or abuse targeting protected classes",
    SafetyCategory.SEXUAL_CONTENT: "Block sexually explicit or exploitative content",
    SafetyCategory.PROMPT_SECURITY: "Defend against jailbreaks, prompt-injection, and secret exfiltration",
    SafetyCategory.VIOLENCE_SELF_HARM: "Prevent promotion, instruction, or graphic depiction of violence or self-harm",
    SafetyCategory.MISINFORMATION: "Block content promoting demonstrably false claims or coordinated deception",
    # Legacy fine-grained
    SafetyCategory.VIOLENT_CRIME: "Instructions or encouragement for violent criminal acts",
    SafetyCategory.NON_VIOLENT_CRIME: "Instructions for fraud, theft, hacking, or other non-violent crimes",
    SafetyCategory.SEX_RELATED: "Generation of explicit sexual content",
    SafetyCategory.CHILD_SAFETY: "Content that endangers minors",
    SafetyCategory.INDISCRIMINATE_WEAPONS: "Instructions for weapons capable of mass casualties",
    SafetyCategory.CBRN: "Chemical, biological, radiological, or nuclear threat information",
    SafetyCategory.SUICIDE_SELF_HARM: "Encouragement or instructions for self-harm",
    SafetyCategory.HATE_SPEECH: "Dehumanizing or discriminatory content targeting protected groups",
    SafetyCategory.HARASSMENT: "Targeted abuse, threats, or intimidation",
    SafetyCategory.PRIVACY_VIOLATION: "Revealing private information or enabling surveillance",
    SafetyCategory.MANIPULATION: "Psychological manipulation or social engineering techniques",
    SafetyCategory.DEFAMATION: "False statements damaging to reputation",
    SafetyCategory.ELECTION_INTERFERENCE: "Content designed to interfere with democratic processes",
    SafetyCategory.UNAUTHORIZED_PRACTICE: "Providing regulated professional advice (medical, legal)",
    SafetyCategory.COPYRIGHT_VIOLATION: "Reproducing substantial copyrighted works",
}

ALL_CATEGORIES: list[str] = [c.value for c in SafetyCategory]
