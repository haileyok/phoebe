"""
ClickHouse-backed persistent store for the Sandbox Arena.

Replaces the in-memory dicts from the prototype with durable storage.
All arena state — bounties, submissions, evaluations, attack history,
and the leaderboard — is persisted to ClickHouse tables.
"""

import json
import logging
import time
from typing import Any

from src.arena.models import (
    AttackPrompt,
    Bounty,
    BountyStatus,
    EvaluationResult,
    PromptEvaluation,
    Submission,
    SubmissionStatus,
)
from src.clickhouse.clickhouse import Clickhouse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL for arena tables. Run once on startup via ReplacingMergeTree.
# ---------------------------------------------------------------------------

ARENA_DDL = [
    "CREATE DATABASE IF NOT EXISTS arena",
    """
    CREATE TABLE IF NOT EXISTS arena.bounties (
        bounty_id        String,
        funder_wallet    String,
        target_model_endpoint String,
        target_model_name String,
        categories       Array(String),
        pool_usdc        Float64,
        remaining_usdc   Float64,
        max_payout_per_finding Float64,
        created_at       Float64,
        expires_at       Nullable(Float64),
        status           String
    ) ENGINE = ReplacingMergeTree()
      ORDER BY bounty_id
    """,
    """
    CREATE TABLE IF NOT EXISTS arena.submissions (
        submission_id    String,
        bounty_id        String,
        teamer_wallet    String,
        prompts_json     String,
        submitted_at     Float64,
        status           String
    ) ENGINE = ReplacingMergeTree()
      ORDER BY submission_id
    """,
    """
    CREATE TABLE IF NOT EXISTS arena.evaluations (
        submission_id    String,
        bounty_id        String,
        evaluations_json String,
        total_score      Float64,
        payout_usdc      Float64,
        category_coverage String,
        duplicate_penalty Float64,
        evaluated_at     Float64,
        tx_hash          String DEFAULT ''
    ) ENGINE = ReplacingMergeTree()
      ORDER BY submission_id
    """,
    """
    CREATE TABLE IF NOT EXISTS arena.attack_history (
        prompt_hash      String,
        prompt_text      String,
        category         String,
        first_seen       Float64,
        times_submitted  UInt32,
        avg_severity     Float32 DEFAULT 0,
        last_submission_id String DEFAULT ''
    ) ENGINE = ReplacingMergeTree(times_submitted)
      ORDER BY prompt_hash
    """,
    """
    CREATE TABLE IF NOT EXISTS arena.leaderboard (
        wallet           String,
        total_score      Float64,
        total_payout     Float64,
        submissions_count UInt32,
        successful_attacks UInt32,
        last_submission_at Float64
    ) ENGINE = ReplacingMergeTree(submissions_count)
      ORDER BY wallet
    """,
    """
    CREATE TABLE IF NOT EXISTS arena.payment_log (
        tx_hash          String,
        timestamp        Float64,
        from_wallet      String,
        to_wallet        String,
        amount_usdc      Float64,
        payment_type     String,
        submission_id    String DEFAULT '',
        bounty_id        String DEFAULT ''
    ) ENGINE = MergeTree()
      ORDER BY (timestamp, tx_hash)
    """,
]


class ArenaStore:
    """
    Persistent store backed by ClickHouse.

    Uses ReplacingMergeTree so upserts (INSERT with same ORDER BY key)
    replace the previous row on merge. FINAL keyword in SELECTs ensures
    we always read the latest version.
    """

    def __init__(self, clickhouse: Clickhouse) -> None:
        self._ch = clickhouse

    async def initialize(self) -> None:
        """Create arena database and tables if they don't exist."""
        for ddl in ARENA_DDL:
            try:
                await self._ch.query(ddl.strip())
            except Exception:
                logger.warning("DDL may already exist: %s", ddl[:80], exc_info=True)
        logger.info("Arena ClickHouse tables initialized")

    # ------------------------------------------------------------------
    # Bounties
    # ------------------------------------------------------------------

    async def save_bounty(self, bounty: Bounty) -> None:
        cats = "','".join(_esc(c) for c in bounty.categories)
        expires = str(bounty.expires_at) if bounty.expires_at else "NULL"
        sql = f"""
            INSERT INTO arena.bounties VALUES (
                '{_esc(bounty.bounty_id)}',
                '{_esc(bounty.funder_wallet)}',
                '{_esc(bounty.target_model_endpoint)}',
                '{_esc(bounty.target_model_name)}',
                ['{cats}'],
                {bounty.pool_usdc},
                {bounty.remaining_usdc},
                {bounty.max_payout_per_finding},
                {bounty.created_at},
                {expires},
                '{bounty.status.value}'
            )
        """
        await self._ch.query(sql)

    async def get_bounty(self, bounty_id: str) -> Bounty | None:
        sql = f"""
            SELECT * FROM arena.bounties FINAL
            WHERE bounty_id = '{_esc(bounty_id)}'
            LIMIT 1
        """
        resp = await self._ch.query(sql)
        rows = resp.result_rows  # type: ignore
        if not rows:
            return None
        return _row_to_bounty(rows[0])

    async def list_active_bounties(self) -> list[Bounty]:
        sql = """
            SELECT * FROM arena.bounties FINAL
            WHERE status = 'active' AND remaining_usdc > 0
            ORDER BY created_at DESC
        """
        resp = await self._ch.query(sql)
        return [_row_to_bounty(r) for r in resp.result_rows]  # type: ignore

    # ------------------------------------------------------------------
    # Submissions
    # ------------------------------------------------------------------

    async def save_submission(self, sub: Submission) -> None:
        prompts_json = json.dumps([p.model_dump() for p in sub.prompts])
        sql = f"""
            INSERT INTO arena.submissions VALUES (
                '{_esc(sub.submission_id)}',
                '{_esc(sub.bounty_id)}',
                '{_esc(sub.teamer_wallet)}',
                '{_esc(prompts_json)}',
                {sub.submitted_at},
                '{sub.status.value}'
            )
        """
        await self._ch.query(sql)

    async def get_submission(self, submission_id: str) -> Submission | None:
        sql = f"""
            SELECT * FROM arena.submissions FINAL
            WHERE submission_id = '{_esc(submission_id)}'
            LIMIT 1
        """
        resp = await self._ch.query(sql)
        rows = resp.result_rows  # type: ignore
        if not rows:
            return None
        return _row_to_submission(rows[0])

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    async def save_evaluation(self, ev: EvaluationResult, tx_hash: str = "") -> None:
        evals_json = json.dumps([e.model_dump() for e in ev.prompt_evaluations])
        cov_json = json.dumps(ev.category_coverage)
        sql = f"""
            INSERT INTO arena.evaluations VALUES (
                '{_esc(ev.submission_id)}',
                '{_esc(ev.bounty_id)}',
                '{_esc(evals_json)}',
                {ev.total_score},
                {ev.payout_usdc},
                '{_esc(cov_json)}',
                {ev.duplicate_penalty},
                {ev.evaluated_at},
                '{_esc(tx_hash)}'
            )
        """
        await self._ch.query(sql)

    async def get_evaluation(self, submission_id: str) -> EvaluationResult | None:
        sql = f"""
            SELECT * FROM arena.evaluations FINAL
            WHERE submission_id = '{_esc(submission_id)}'
            LIMIT 1
        """
        resp = await self._ch.query(sql)
        rows = resp.result_rows  # type: ignore
        if not rows:
            return None
        return _row_to_evaluation(rows[0])

    # ------------------------------------------------------------------
    # Attack history (novelty scoring via ngramDistance)
    # ------------------------------------------------------------------

    async def record_attack(
        self,
        prompt_hash: str,
        prompt_text: str,
        category: str,
        severity: float,
        submission_id: str,
    ) -> None:
        sql = f"""
            INSERT INTO arena.attack_history VALUES (
                '{_esc(prompt_hash)}',
                '{_esc(prompt_text[:5000])}',
                '{_esc(category)}',
                {time.time()},
                1,
                {severity},
                '{_esc(submission_id)}'
            )
        """
        await self._ch.query(sql)

    async def compute_novelty(self, prompt_text: str, threshold: float = 0.3) -> dict[str, Any]:
        """
        Compute novelty via ClickHouse ngramDistance against the full
        attack history. Returns 1.0 for novel, 0.0 for exact duplicate.
        """
        escaped = _esc(prompt_text)
        sql = f"""
            SELECT
                prompt_text,
                category,
                ngramDistance(prompt_text, '{escaped}') AS distance,
                times_submitted
            FROM arena.attack_history FINAL
            WHERE ngramDistance(prompt_text, '{escaped}') < {threshold}
            ORDER BY distance ASC
            LIMIT 10
        """
        try:
            resp = await self._ch.query(sql)
            similar = []
            for row in resp.result_rows:  # type: ignore
                similar.append({
                    "prompt_preview": str(row[0])[:100],
                    "category": str(row[1]),
                    "distance": round(float(row[2]), 4),
                    "times_submitted": int(row[3]),
                })

            if not similar:
                return {"novelty_score": 1.0, "similar_count": 0, "similar_prompts": []}

            closest = similar[0]["distance"]
            novelty = min(1.0, closest / threshold)
            return {
                "novelty_score": round(novelty, 4),
                "similar_count": len(similar),
                "similar_prompts": similar,
            }
        except Exception:
            logger.warning("Novelty query failed; assuming full novelty", exc_info=True)
            return {"novelty_score": 1.0, "similar_count": 0, "similar_prompts": []}

    # ------------------------------------------------------------------
    # Leaderboard
    # ------------------------------------------------------------------

    async def update_leaderboard(
        self,
        wallet: str,
        score_delta: float,
        payout_delta: float,
        successful_attacks: int,
    ) -> None:
        current = await self._get_leaderboard_entry(wallet)
        sql = f"""
            INSERT INTO arena.leaderboard VALUES (
                '{_esc(wallet)}',
                {current['total_score'] + score_delta},
                {current['total_payout'] + payout_delta},
                {current['submissions_count'] + 1},
                {current['successful_attacks'] + successful_attacks},
                {time.time()}
            )
        """
        await self._ch.query(sql)

    async def _get_leaderboard_entry(self, wallet: str) -> dict[str, Any]:
        sql = f"""
            SELECT total_score, total_payout, submissions_count, successful_attacks
            FROM arena.leaderboard FINAL
            WHERE wallet = '{_esc(wallet)}'
            LIMIT 1
        """
        try:
            resp = await self._ch.query(sql)
            rows = resp.result_rows  # type: ignore
            if rows:
                return {
                    "total_score": float(rows[0][0]),
                    "total_payout": float(rows[0][1]),
                    "submissions_count": int(rows[0][2]),
                    "successful_attacks": int(rows[0][3]),
                }
        except Exception:
            pass
        return {"total_score": 0, "total_payout": 0, "submissions_count": 0, "successful_attacks": 0}

    async def get_leaderboard(self, limit: int = 100) -> list[dict[str, Any]]:
        sql = f"""
            SELECT wallet, total_score, total_payout, submissions_count,
                   successful_attacks
            FROM arena.leaderboard FINAL
            ORDER BY total_score DESC
            LIMIT {limit}
        """
        resp = await self._ch.query(sql)
        entries = []
        for i, row in enumerate(resp.result_rows):  # type: ignore
            entries.append({
                "rank": i + 1,
                "wallet": str(row[0]),
                "total_score": round(float(row[1]), 4),
                "total_payout": round(float(row[2]), 4),
                "submissions_count": int(row[3]),
                "successful_attacks": int(row[4]),
            })
        return entries

    # ------------------------------------------------------------------
    # Payment log
    # ------------------------------------------------------------------

    async def log_payment(
        self,
        tx_hash: str,
        from_wallet: str,
        to_wallet: str,
        amount_usdc: float,
        payment_type: str,
        submission_id: str = "",
        bounty_id: str = "",
    ) -> None:
        sql = f"""
            INSERT INTO arena.payment_log VALUES (
                '{_esc(tx_hash)}',
                {time.time()},
                '{_esc(from_wallet)}',
                '{_esc(to_wallet)}',
                {amount_usdc},
                '{_esc(payment_type)}',
                '{_esc(submission_id)}',
                '{_esc(bounty_id)}'
            )
        """
        await self._ch.query(sql)

    # ------------------------------------------------------------------
    # Coverage stats (aggregated from evaluations)
    # ------------------------------------------------------------------

    async def get_category_coverage(self) -> dict[str, int]:
        sql = """
            SELECT category_coverage FROM arena.evaluations FINAL
        """
        try:
            resp = await self._ch.query(sql)
            merged: dict[str, int] = {}
            for row in resp.result_rows:  # type: ignore
                cov = json.loads(str(row[0]))
                for cat, count in cov.items():
                    merged[cat] = merged.get(cat, 0) + int(count)
            return merged
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    """Escape for ClickHouse SQL string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _row_to_bounty(row: tuple) -> Bounty:
    return Bounty(
        bounty_id=str(row[0]),
        funder_wallet=str(row[1]),
        target_model_endpoint=str(row[2]),
        target_model_name=str(row[3]),
        categories=list(row[4]) if row[4] else [],
        pool_usdc=float(row[5]),
        remaining_usdc=float(row[6]),
        max_payout_per_finding=float(row[7]),
        created_at=float(row[8]),
        expires_at=float(row[9]) if row[9] is not None else None,
        status=BountyStatus(str(row[10])),
    )


def _row_to_submission(row: tuple) -> Submission:
    prompts_data = json.loads(str(row[3]))
    prompts = [AttackPrompt(**p) for p in prompts_data]
    return Submission(
        submission_id=str(row[0]),
        bounty_id=str(row[1]),
        teamer_wallet=str(row[2]),
        prompts=prompts,
        submitted_at=float(row[4]),
        status=SubmissionStatus(str(row[5])),
    )


def _row_to_evaluation(row: tuple) -> EvaluationResult:
    evals_data = json.loads(str(row[2]))
    prompt_evals = [PromptEvaluation(**e) for e in evals_data]
    category_cov = json.loads(str(row[5]))
    return EvaluationResult(
        submission_id=str(row[0]),
        bounty_id=str(row[1]),
        prompt_evaluations=prompt_evals,
        total_score=float(row[3]),
        payout_usdc=float(row[4]),
        category_coverage=category_cov,
        duplicate_penalty=float(row[6]),
        evaluated_at=float(row[7]),
    )
