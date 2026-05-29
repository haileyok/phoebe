"""
Backward-compatibility shim.

All prompts now live in agents/phoebe/config.py.
Existing code that imports build_system_prompt() from this module
continues to work unchanged.
"""

from agents.phoebe.config import PHOEBE_CONFIG


def build_system_prompt(mode: str = "judge") -> str:
    """
    Build the base system prompt for the Phoebe agent.

    Args:
        mode: "judge" | "admin" | "redteam"
    """
    return PHOEBE_CONFIG.get_system_prompt(mode)
