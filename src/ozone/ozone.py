from typing import Any


class Ozone:
    """Client for Ozone moderation service."""

    def __init__(self) -> None:
        # TODO: Implement Ozone client
        pass

    async def apply_label(self, subject: str, label: str) -> dict[str, Any]:
        """apply a moderation label to a subject"""

        raise NotImplementedError("Ozone client not yet implemented")

    async def remove_label(self, subject: str, label: str) -> dict[str, Any]:
        """remove a moderation label from a subject"""
        raise NotImplementedError("Ozone client not yet implemented")
