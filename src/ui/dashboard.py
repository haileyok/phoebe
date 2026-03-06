"""
T&S Analyst Dashboard — Starlette routes for the web UI.

Serves the single-page dashboard and provides API endpoints for:
- POST /api/tns/classify  — classify a prompt against GA Guard policy
- GET  /api/tns/breaches  — list recent security breaches with filters
- GET  /api/tns/stats     — dashboard summary statistics
- GET  /tns               — serve the dashboard HTML
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from src.osprey.policy import (
    GA_GUARD_POLICY,
    PolicyCategory,
    classify_category,
    format_policy_for_classifier,
)

logger = logging.getLogger(__name__)

DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()


class TNSDashboard:
    """
    Trust & Safety analyst dashboard.

    Provides a web UI for:
    - Checking prompts against the GA Guard policy
    - Viewing recent security breach logs
    - Filtering breaches by time, category, and severity
    """

    def __init__(
        self,
        safety_classifier: Any,
        store: Any,
    ) -> None:
        self._classifier = safety_classifier
        self._store = store

    def routes(self) -> list[Route]:
        return [
            Route("/tns", self._serve_dashboard, methods=["GET"]),
            Route("/api/tns/classify", self._classify_prompt, methods=["POST"]),
            Route("/api/tns/breaches", self._list_breaches, methods=["GET"]),
            Route("/api/tns/stats", self._get_stats, methods=["GET"]),
        ]

    async def _serve_dashboard(self, request: Request) -> Response:
        return HTMLResponse(DASHBOARD_HTML)

    async def _classify_prompt(self, request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        prompt = body.get("prompt", "").strip()
        if not prompt:
            return JSONResponse({"error": "prompt is required"}, status_code=400)

        if len(prompt) > 20_000:
            return JSONResponse({"error": "prompt too long (max 20000 chars)"}, status_code=400)

        category = body.get("category")

        # If a specific category is requested, classify against that one
        if category:
            resolved = classify_category(category)
            if not resolved:
                return JSONResponse({"error": f"unknown category: {category}"}, status_code=400)
            result = await self._classifier.classify(
                prompt=prompt,
                model_output=prompt,  # for input-only classification, prompt is the content
                category=resolved.value,
            )
        else:
            # Classify against all categories, return the most severe violation
            results = []
            for cat in PolicyCategory:
                r = await self._classifier.classify(
                    prompt=prompt,
                    model_output=prompt,
                    category=cat.value,
                )
                if r.get("unsafe"):
                    results.append(r)

            if results:
                result = max(results, key=lambda r: r.get("severity", 0))
            else:
                # No violations found — return safe with the first category checked
                result = {
                    "unsafe": False,
                    "severity": 0,
                    "policy_category": "none",
                    "matched_block_rule": "",
                    "explanation": "Prompt does not match any block rules across all GA Guard categories.",
                }

        # Log the classification to ClickHouse for the breach log
        if result.get("unsafe"):
            await self._log_breach(prompt, result)

        return JSONResponse(result)

    async def _list_breaches(self, request: Request) -> Response:
        from_ts = request.query_params.get("from")
        to_ts = request.query_params.get("to")
        category = request.query_params.get("category")
        limit = min(int(request.query_params.get("limit", "500")), 2000)

        conditions = ["1=1"]

        if from_ts:
            try:
                from_epoch = _iso_to_epoch(from_ts)
                conditions.append(f"timestamp >= {from_epoch}")
            except Exception:
                pass

        if to_ts:
            try:
                to_epoch = _iso_to_epoch(to_ts)
                conditions.append(f"timestamp <= {to_epoch}")
            except Exception:
                pass

        if category:
            conditions.append(f"policy_category = '{_esc(category)}'")

        where = " AND ".join(conditions)

        sql = f"""
            SELECT timestamp, policy_category, severity, prompt_text,
                   unsafe, matched_block_rule, explanation
            FROM arena.tns_breach_log
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """

        try:
            resp = await self._store._ch.query(sql)
            breaches = []
            for row in resp.result_rows:  # type: ignore
                breaches.append({
                    "timestamp": float(row[0]),
                    "policy_category": str(row[1]),
                    "severity": int(row[2]),
                    "prompt_text": str(row[3]),
                    "unsafe": bool(row[4]),
                    "matched_block_rule": str(row[5]),
                    "explanation": str(row[6]),
                })
            return JSONResponse({"breaches": breaches, "count": len(breaches)})
        except Exception as e:
            logger.warning("Breach log query failed: %s", e)
            return JSONResponse({"breaches": [], "count": 0})

    async def _get_stats(self, request: Request) -> Response:
        now = time.time()
        day_ago = now - 86400

        stats: dict[str, Any] = {
            "total_breaches": 0,
            "last_24h": 0,
            "by_category": {},
        }

        try:
            # Total count
            resp = await self._store._ch.query(
                "SELECT count() FROM arena.tns_breach_log"
            )
            stats["total_breaches"] = int(resp.result_rows[0][0])  # type: ignore

            # Last 24h
            resp = await self._store._ch.query(
                f"SELECT count() FROM arena.tns_breach_log WHERE timestamp >= {day_ago}"
            )
            stats["last_24h"] = int(resp.result_rows[0][0])  # type: ignore

            # By category
            resp = await self._store._ch.query(
                "SELECT policy_category, count() as cnt FROM arena.tns_breach_log "
                "GROUP BY policy_category ORDER BY cnt DESC"
            )
            for row in resp.result_rows:  # type: ignore
                stats["by_category"][str(row[0])] = int(row[1])

        except Exception as e:
            logger.warning("Stats query failed: %s", e)

        return JSONResponse(stats)

    async def _log_breach(self, prompt: str, result: dict[str, Any]) -> None:
        """Log a classified breach to the tns_breach_log table."""
        try:
            sql = f"""
                INSERT INTO arena.tns_breach_log VALUES (
                    {time.time()},
                    '{_esc(result.get("policy_category", ""))}',
                    {int(result.get("severity", 0))},
                    '{_esc(prompt[:5000])}',
                    {1 if result.get("unsafe") else 0},
                    '{_esc(result.get("matched_block_rule", ""))}',
                    '{_esc(result.get("explanation", ""))}'
                )
            """
            await self._store._ch.query(sql)
        except Exception as e:
            logger.warning("Failed to log breach: %s", e)


# DDL for the T&S breach log table
TNS_DDL = [
    """
    CREATE TABLE IF NOT EXISTS arena.tns_breach_log (
        timestamp        Float64,
        policy_category  String,
        severity         UInt8,
        prompt_text      String,
        unsafe           UInt8,
        matched_block_rule String DEFAULT '',
        explanation      String DEFAULT ''
    ) ENGINE = MergeTree()
      ORDER BY (timestamp, policy_category)
    """,
]


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _iso_to_epoch(iso_str: str) -> float:
    """Convert ISO 8601 string to epoch timestamp."""
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp()
