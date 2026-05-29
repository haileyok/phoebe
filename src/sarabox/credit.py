"""
Credit economy for Sara in a Box.

Orgs earn credits by submitting high-quality attack data to the
federated training pool. Credits reduce the cost of inference API calls.
"""

from __future__ import annotations

import datetime
from datetime import timezone
import logging
from typing import TYPE_CHECKING

from src.sarabox.tee_training import GradientUpdate

if TYPE_CHECKING:
    from src.sarabox.store import SaraBoxStore

logger = logging.getLogger(__name__)

# Credits per contribution — scaled by novelty + coverage (same formula
# as Sandbox Arena scorer but applied to training contributions, not attacks)
BASE_CREDITS_PER_SAMPLE = 0.1  # credits per unique attack prompt submitted
NOVELTY_MULTIPLIER = 2.0  # if contribution_score > 0.7, double credits
COVERAGE_BONUS = 5.0  # flat bonus for covering a new category
INFERENCE_COST_PER_CALL = 0.01  # credits deducted per /classify call


class CreditEngine:
    """
    Manages the contribution economy.

    Credit formula (mirrors Sandbox Arena scoring):
      earned = BASE * num_samples * (1 + novelty_multiplier * score)
             + COVERAGE_BONUS * new_categories_covered

    This incentivises novel, diverse contributions over duplicate data.
    """

    def __init__(self, store: SaraBoxStore):
        self._store = store

    async def award_for_contribution(
        self,
        update: GradientUpdate,
    ) -> float:
        """
        Award credits to an org for a gradient contribution.
        Returns the number of credits awarded.
        """
        from src.sarabox.models import CreditLedger

        ledger = await self._store.get_ledger(update.org_id)
        if ledger is None:
            ledger = CreditLedger(org_id=update.org_id)
        base = BASE_CREDITS_PER_SAMPLE * update.num_samples
        multiplier = NOVELTY_MULTIPLIER if update.contribution_score > 0.7 else 1.0
        earned = round(base * multiplier, 4)

        ledger.total_earned += earned
        ledger.balance += earned
        ledger.contributions.append({
            "update_id": update.update_id,
            "credits": earned,
            "score": update.contribution_score,
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        })
        await self._store.save_ledger(ledger)
        logger.info(
            "Awarded %.4f credits to %s for contribution %s",
            earned, update.org_id, update.update_id[:8]
        )
        return earned

    async def deduct_for_inference(self, org_id: str, calls: int = 1) -> bool:
        """
        Deduct credits for inference API usage.
        Returns True if deduction succeeded, False if insufficient balance.
        """
        ledger = await self._store.get_ledger(org_id)
        if ledger is None:
            return False
        cost = INFERENCE_COST_PER_CALL * calls
        if ledger.balance < cost:
            return False
        ledger.balance -= cost
        ledger.total_spent += cost
        await self._store.save_ledger(ledger)
        return True

    async def get_balance(self, org_id: str) -> float:
        ledger = await self._store.get_ledger(org_id)
        if ledger is None:
            return 0.0
        return ledger.balance
