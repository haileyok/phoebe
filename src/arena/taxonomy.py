"""
Safety taxonomy for the red teaming arena.

Defines the categories of unsafe behavior that red teamers can test for,
severity levels, and coverage tracking.
"""

from enum import Enum


class SafetyCategory(str, Enum):
    """
    Safety taxonomy categories. Based on ML Commons AI Safety Benchmark
    and common red teaming frameworks.
    """

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
    MISINFORMATION = "misinformation"
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


# Human-readable descriptions for UI / leaderboard
CATEGORY_DESCRIPTIONS: dict[SafetyCategory, str] = {
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
    SafetyCategory.MISINFORMATION: "Generation of convincing false information",
    SafetyCategory.MANIPULATION: "Psychological manipulation or social engineering techniques",
    SafetyCategory.DEFAMATION: "False statements damaging to reputation",
    SafetyCategory.ELECTION_INTERFERENCE: "Content designed to interfere with democratic processes",
    SafetyCategory.UNAUTHORIZED_PRACTICE: "Providing regulated professional advice (medical, legal)",
    SafetyCategory.COPYRIGHT_VIOLATION: "Reproducing substantial copyrighted works",
}

ALL_CATEGORIES: list[str] = [c.value for c in SafetyCategory]
