"""
ClickHouse-backed persistence for Sara in a Box.

Follows the exact pattern of src/arena/store.py — async client,
same table creation pattern, same query style.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.clickhouse.clickhouse import Clickhouse
from src.sarabox.models import (
    AttackCategory,
    AttackSubmission,
    CreditLedger,
    OrgConfig,
    SkillFile,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL for sarabox tables
# ---------------------------------------------------------------------------

SARABOX_DDL = [
    "CREATE DATABASE IF NOT EXISTS sarabox",
    """
    CREATE TABLE IF NOT EXISTS sarabox.skill_files (
        skill_id         String,
        org_id           String,
        org_type         String,
        display_name     String,
        skill_json       String,
        created_at       Float64,
        version          String,
        is_private       UInt8
    ) ENGINE = ReplacingMergeTree()
      ORDER BY skill_id
    """,
    """
    CREATE TABLE IF NOT EXISTS sarabox.org_configs (
        org_id           String,
        org_type         String,
        description      String,
        opt_in           UInt8,
        credit_balance   Float64,
        skill_file_id    String,
        created_at       Float64
    ) ENGINE = ReplacingMergeTree()
      ORDER BY org_id
    """,
    """
    CREATE TABLE IF NOT EXISTS sarabox.submissions (
        submission_id    String,
        org_id           String,
        skill_id         String,
        commitment_hash  String,
        num_prompts      UInt32,
        labels_json      String,
        submitted_at     Float64
    ) ENGINE = ReplacingMergeTree()
      ORDER BY submission_id
    """,
    """
    CREATE TABLE IF NOT EXISTS sarabox.credit_ledgers (
        org_id           String,
        total_earned     Float64,
        total_spent      Float64,
        balance          Float64,
        contributions_json String,
        updated_at       Float64
    ) ENGINE = ReplacingMergeTree()
      ORDER BY org_id
    """,
    """
    CREATE TABLE IF NOT EXISTS sarabox.gradient_log (
        update_id        String,
        org_id           String,
        delta_hash       String,
        num_samples      UInt32,
        contribution_score Float64,
        attestation_quote String,
        created_at       Float64
    ) ENGINE = MergeTree()
      ORDER BY (created_at, update_id)
    """,
]


class SaraBoxStore:
    """Persistent store backed by ClickHouse for Sara in a Box."""

    def __init__(self, clickhouse: Clickhouse) -> None:
        self._ch = clickhouse

    async def initialize(self) -> None:
        """Create sarabox database and tables if they don't exist."""
        for ddl in SARABOX_DDL:
            try:
                await self._ch.query(ddl.strip())
            except Exception:
                logger.warning("DDL may already exist: %s", ddl[:80], exc_info=True)
        logger.info("SaraBox ClickHouse tables initialized")

    # ------------------------------------------------------------------
    # Skill files
    # ------------------------------------------------------------------

    async def save_skill_file(self, skill: SkillFile) -> None:
        cats = json.dumps([c.model_dump() for c in skill.categories])
        sql = f"""
            INSERT INTO sarabox.skill_files VALUES (
                '{_esc(skill.skill_id)}',
                '{_esc(skill.org_id)}',
                '{_esc(skill.org_type)}',
                '{_esc(skill.display_name)}',
                '{_esc(json.dumps(skill.model_dump()))}',
                {skill.created_at.timestamp()},
                '{_esc(skill.version)}',
                {1 if skill.is_private else 0}
            )
        """
        await self._ch.query(sql)

    async def get_skill_file(self, skill_id: str) -> SkillFile | None:
        sql = f"""
            SELECT skill_json FROM sarabox.skill_files FINAL
            WHERE skill_id = '{_esc(skill_id)}'
            LIMIT 1
        """
        resp = await self._ch.query(sql)
        rows = resp.result_rows  # type: ignore
        if not rows:
            return None
        data = json.loads(str(rows[0][0]))
        return SkillFile(**data)

    # ------------------------------------------------------------------
    # Org configs
    # ------------------------------------------------------------------

    async def save_org_config(self, org: OrgConfig) -> None:
        sql = f"""
            INSERT INTO sarabox.org_configs VALUES (
                '{_esc(org.org_id)}',
                '{_esc(org.org_type)}',
                '{_esc(org.natural_language_description)}',
                {1 if org.federated_training_opt_in else 0},
                {org.credit_balance},
                '{_esc(org.skill_file_id or "")}',
                {time.time()}
            )
        """
        await self._ch.query(sql)

    async def get_org_config(self, org_id: str) -> OrgConfig | None:
        sql = f"""
            SELECT * FROM sarabox.org_configs FINAL
            WHERE org_id = '{_esc(org_id)}'
            LIMIT 1
        """
        resp = await self._ch.query(sql)
        rows = resp.result_rows  # type: ignore
        if not rows:
            return None
        r = rows[0]
        return OrgConfig(
            org_id=str(r[0]),
            org_type=str(r[1]),
            natural_language_description=str(r[2]),
            federated_training_opt_in=bool(r[3]),
            credit_balance=float(r[4]),
            skill_file_id=str(r[5]) if r[5] else None,
        )

    # ------------------------------------------------------------------
    # Submissions
    # ------------------------------------------------------------------

    async def save_submission(self, sub: AttackSubmission) -> None:
        sql = f"""
            INSERT INTO sarabox.submissions VALUES (
                '{_esc(sub.submission_id)}',
                '{_esc(sub.org_id)}',
                '{_esc(sub.skill_id)}',
                '{_esc(sub.commitment_hash)}',
                {len(sub.prompts)},
                '{_esc(json.dumps(sub.labels))}',
                {sub.submitted_at.timestamp()}
            )
        """
        await self._ch.query(sql)

    # ------------------------------------------------------------------
    # Credit ledgers
    # ------------------------------------------------------------------

    async def get_ledger(self, org_id: str) -> CreditLedger | None:
        sql = f"""
            SELECT * FROM sarabox.credit_ledgers FINAL
            WHERE org_id = '{_esc(org_id)}'
            LIMIT 1
        """
        resp = await self._ch.query(sql)
        rows = resp.result_rows  # type: ignore
        if not rows:
            return None
        r = rows[0]
        return CreditLedger(
            org_id=str(r[0]),
            total_earned=float(r[1]),
            total_spent=float(r[2]),
            balance=float(r[3]),
            contributions=json.loads(str(r[4])) if r[4] else [],
        )

    async def save_ledger(self, ledger: CreditLedger) -> None:
        sql = f"""
            INSERT INTO sarabox.credit_ledgers VALUES (
                '{_esc(ledger.org_id)}',
                {ledger.total_earned},
                {ledger.total_spent},
                {ledger.balance},
                '{_esc(json.dumps(ledger.contributions))}',
                {time.time()}
            )
        """
        await self._ch.query(sql)

    # ------------------------------------------------------------------
    # Gradient log
    # ------------------------------------------------------------------

    async def save_gradient_log(self, update_id: str, org_id: str, delta_hash: str, num_samples: int, contribution_score: float, attestation_quote: str) -> None:
        sql = f"""
            INSERT INTO sarabox.gradient_log VALUES (
                '{_esc(update_id)}',
                '{_esc(org_id)}',
                '{_esc(delta_hash)}',
                {num_samples},
                {contribution_score},
                '{_esc(attestation_quote)}',
                {time.time()}
            )
        """
        await self._ch.query(sql)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    """Escape for ClickHouse SQL string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")
