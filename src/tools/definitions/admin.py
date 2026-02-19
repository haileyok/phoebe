"""
Admin tools — arena operator tools for managing bounties, reviewing
submissions, approving payouts, and monitoring arena health.

Only available to the admin agent (not the judge or chat agents).
"""

import time
from typing import Any

from src.arena.models import Bounty, BountyStatus, SubmissionStatus
from src.arena.taxonomy import ALL_CATEGORIES, CATEGORY_DESCRIPTIONS, SafetyCategory
from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


# ---------------------------------------------------------------------------
# Bounty management
# ---------------------------------------------------------------------------


@TOOL_REGISTRY.tool(
    name="admin.create_bounty",
    description="Create a new red teaming bounty with a USDC pool.",
    parameters=[
        ToolParameter(name="target_model_endpoint", type="string", description="API endpoint of the target model to red team"),
        ToolParameter(name="target_model_name", type="string", description="Human-readable name of the target model"),
        ToolParameter(name="pool_usdc", type="number", description="Total USDC pool for this bounty"),
        ToolParameter(name="max_payout_per_finding", type="number", description="Max USDC payout per individual finding (default 10.0)"),
        ToolParameter(name="categories", type="string", description="Comma-separated safety categories to target (or 'all')"),
        ToolParameter(name="funder_wallet", type="string", description="Wallet address of the bounty funder"),
        ToolParameter(name="expires_in_days", type="number", description="Number of days until bounty expires (optional)"),
    ],
)
async def admin_create_bounty(
    ctx: ToolContext,
    target_model_endpoint: str,
    target_model_name: str,
    pool_usdc: float,
    funder_wallet: str,
    max_payout_per_finding: float = 10.0,
    categories: str = "all",
    expires_in_days: float | None = None,
) -> dict[str, Any]:
    """Create a new bounty and persist to ClickHouse."""
    store = ctx.arena.store

    if categories.strip().lower() == "all":
        cat_list = [c.value for c in SafetyCategory]
    else:
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]

    expires_at = None
    if expires_in_days is not None:
        expires_at = time.time() + (expires_in_days * 86400)

    bounty = Bounty(
        funder_wallet=funder_wallet,
        target_model_endpoint=target_model_endpoint,
        target_model_name=target_model_name,
        categories=cat_list,
        pool_usdc=pool_usdc,
        remaining_usdc=pool_usdc,
        max_payout_per_finding=max_payout_per_finding,
        expires_at=expires_at,
        status=BountyStatus.ACTIVE,
    )

    await store.save_bounty(bounty)
    return {"status": "created", "bounty": bounty.model_dump()}


@TOOL_REGISTRY.tool(
    name="admin.pause_bounty",
    description="Pause an active bounty (stops accepting new submissions).",
    parameters=[
        ToolParameter(name="bounty_id", type="string", description="The bounty ID to pause"),
    ],
)
async def admin_pause_bounty(ctx: ToolContext, bounty_id: str) -> dict[str, Any]:
    """Set a bounty's status to cancelled (paused)."""
    store = ctx.arena.store
    bounty = await store.get_bounty(bounty_id)
    if not bounty:
        return {"error": f"Bounty {bounty_id} not found"}

    bounty.status = BountyStatus.CANCELLED
    await store.save_bounty(bounty)
    return {"status": "paused", "bounty_id": bounty_id}


@TOOL_REGISTRY.tool(
    name="admin.resume_bounty",
    description="Resume a paused/cancelled bounty back to active.",
    parameters=[
        ToolParameter(name="bounty_id", type="string", description="The bounty ID to resume"),
    ],
)
async def admin_resume_bounty(ctx: ToolContext, bounty_id: str) -> dict[str, Any]:
    """Re-activate a paused bounty."""
    store = ctx.arena.store
    bounty = await store.get_bounty(bounty_id)
    if not bounty:
        return {"error": f"Bounty {bounty_id} not found"}

    if bounty.remaining_usdc <= 0:
        return {"error": "Cannot resume — bounty pool is exhausted"}

    bounty.status = BountyStatus.ACTIVE
    await store.save_bounty(bounty)
    return {"status": "resumed", "bounty_id": bounty_id}


@TOOL_REGISTRY.tool(
    name="admin.expire_bounty",
    description="Manually expire a bounty.",
    parameters=[
        ToolParameter(name="bounty_id", type="string", description="The bounty ID to expire"),
    ],
)
async def admin_expire_bounty(ctx: ToolContext, bounty_id: str) -> dict[str, Any]:
    """Mark a bounty as expired."""
    store = ctx.arena.store
    bounty = await store.get_bounty(bounty_id)
    if not bounty:
        return {"error": f"Bounty {bounty_id} not found"}

    bounty.status = BountyStatus.EXPIRED
    await store.save_bounty(bounty)
    return {"status": "expired", "bounty_id": bounty_id}


@TOOL_REGISTRY.tool(
    name="admin.fund_bounty",
    description="Add additional USDC to an existing bounty's pool.",
    parameters=[
        ToolParameter(name="bounty_id", type="string", description="The bounty ID to fund"),
        ToolParameter(name="amount_usdc", type="number", description="Amount of USDC to add"),
    ],
)
async def admin_fund_bounty(ctx: ToolContext, bounty_id: str, amount_usdc: float) -> dict[str, Any]:
    """Top up a bounty's remaining pool."""
    store = ctx.arena.store
    bounty = await store.get_bounty(bounty_id)
    if not bounty:
        return {"error": f"Bounty {bounty_id} not found"}

    bounty.pool_usdc += amount_usdc
    bounty.remaining_usdc += amount_usdc
    if bounty.status == BountyStatus.EXHAUSTED:
        bounty.status = BountyStatus.ACTIVE
    await store.save_bounty(bounty)
    return {
        "status": "funded",
        "bounty_id": bounty_id,
        "added_usdc": amount_usdc,
        "new_pool": bounty.pool_usdc,
        "new_remaining": bounty.remaining_usdc,
    }


# ---------------------------------------------------------------------------
# Submission review
# ---------------------------------------------------------------------------


@TOOL_REGISTRY.tool(
    name="admin.list_submissions",
    description="List submissions for a bounty, optionally filtered by status.",
    parameters=[
        ToolParameter(name="bounty_id", type="string", description="The bounty ID (optional — omit for all bounties)"),
        ToolParameter(name="status", type="string", description="Filter by status: queued, evaluating, scored, paid, rejected (optional)"),
        ToolParameter(name="limit", type="number", description="Max results (default 50)"),
    ],
)
async def admin_list_submissions(
    ctx: ToolContext,
    bounty_id: str = "",
    status: str = "",
    limit: float = 50,
) -> dict[str, Any]:
    """Query submissions from ClickHouse."""
    store = ctx.arena.store
    where_clauses = []
    if bounty_id:
        where_clauses.append(f"bounty_id = '{bounty_id}'")
    if status:
        where_clauses.append(f"status = '{status}'")

    where = " AND ".join(where_clauses) if where_clauses else "1=1"
    sql = f"""
        SELECT submission_id, bounty_id, teamer_wallet, submitted_at, status
        FROM arena.submissions FINAL
        WHERE {where}
        ORDER BY submitted_at DESC
        LIMIT {int(limit)}
    """
    resp = await store._ch.query(sql)
    subs = []
    for row in resp.result_rows:  # type: ignore
        subs.append({
            "submission_id": str(row[0]),
            "bounty_id": str(row[1]),
            "teamer_wallet": str(row[2]),
            "submitted_at": float(row[3]),
            "status": str(row[4]),
        })
    return {"submissions": subs, "count": len(subs)}


@TOOL_REGISTRY.tool(
    name="admin.get_submission",
    description="Get full details of a specific submission including its prompts.",
    parameters=[
        ToolParameter(name="submission_id", type="string", description="The submission ID"),
    ],
)
async def admin_get_submission(ctx: ToolContext, submission_id: str) -> dict[str, Any]:
    """Get a submission and its evaluation."""
    store = ctx.arena.store
    sub = await store.get_submission(submission_id)
    if not sub:
        return {"error": f"Submission {submission_id} not found"}

    result: dict[str, Any] = sub.model_dump()

    ev = await store.get_evaluation(submission_id)
    if ev:
        result["evaluation"] = ev.summary()

    return result


@TOOL_REGISTRY.tool(
    name="admin.reject_submission",
    description="Reject a submission (e.g. for policy violation).",
    parameters=[
        ToolParameter(name="submission_id", type="string", description="The submission ID to reject"),
        ToolParameter(name="reason", type="string", description="Reason for rejection"),
    ],
)
async def admin_reject_submission(ctx: ToolContext, submission_id: str, reason: str = "") -> dict[str, Any]:
    """Mark a submission as rejected."""
    store = ctx.arena.store
    sub = await store.get_submission(submission_id)
    if not sub:
        return {"error": f"Submission {submission_id} not found"}

    sub.status = SubmissionStatus.REJECTED
    await store.save_submission(sub)
    return {"status": "rejected", "submission_id": submission_id, "reason": reason}


# ---------------------------------------------------------------------------
# Leaderboard & analytics
# ---------------------------------------------------------------------------


@TOOL_REGISTRY.tool(
    name="admin.leaderboard",
    description="Get the red teamer leaderboard with scores and payouts.",
    parameters=[
        ToolParameter(name="limit", type="number", description="Number of top entries (default 25)"),
    ],
)
async def admin_leaderboard(ctx: ToolContext, limit: float = 25) -> dict[str, Any]:
    """Fetch leaderboard from ClickHouse."""
    store = ctx.arena.store
    entries = await store.get_leaderboard(limit=int(limit))
    return {"leaderboard": entries, "count": len(entries)}


@TOOL_REGISTRY.tool(
    name="admin.payment_log",
    description="View recent payment activity (submission fees and bounty payouts).",
    parameters=[
        ToolParameter(name="limit", type="number", description="Number of recent payments (default 50)"),
        ToolParameter(name="payment_type", type="string", description="Filter by type: submission_fee, bounty_payout (optional)"),
    ],
)
async def admin_payment_log(ctx: ToolContext, limit: float = 50, payment_type: str = "") -> dict[str, Any]:
    """Query the payment log."""
    store = ctx.arena.store
    where = f"payment_type = '{payment_type}'" if payment_type else "1=1"
    sql = f"""
        SELECT tx_hash, timestamp, from_wallet, to_wallet, amount_usdc,
               payment_type, submission_id, bounty_id
        FROM arena.payment_log
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT {int(limit)}
    """
    resp = await store._ch.query(sql)
    payments = []
    for row in resp.result_rows:  # type: ignore
        payments.append({
            "tx_hash": str(row[0]),
            "timestamp": float(row[1]),
            "from_wallet": str(row[2]),
            "to_wallet": str(row[3]),
            "amount_usdc": float(row[4]),
            "payment_type": str(row[5]),
            "submission_id": str(row[6]),
            "bounty_id": str(row[7]),
        })
    return {"payments": payments, "count": len(payments)}


@TOOL_REGISTRY.tool(
    name="admin.arena_stats",
    description="Get overall arena health metrics: active bounties, total submissions, payouts, coverage.",
    parameters=[],
)
async def admin_arena_stats(ctx: ToolContext) -> dict[str, Any]:
    """Aggregate arena health dashboard."""
    store = ctx.arena.store

    active_bounties = await store.list_active_bounties()
    coverage = await store.get_category_coverage()
    leaderboard = await store.get_leaderboard(limit=1)

    # total pool remaining
    total_pool = sum(b.remaining_usdc for b in active_bounties)

    # count stats via raw queries
    try:
        sub_resp = await store._ch.query("SELECT count() FROM arena.submissions FINAL")
        total_submissions = int(sub_resp.result_rows[0][0])  # type: ignore
    except Exception:
        total_submissions = 0

    try:
        eval_resp = await store._ch.query(
            "SELECT count(), sum(payout_usdc), sum(total_score) FROM arena.evaluations FINAL"
        )
        row = eval_resp.result_rows[0]  # type: ignore
        total_evaluations = int(row[0])
        total_payouts = float(row[1])
        total_score = float(row[2])
    except Exception:
        total_evaluations = 0
        total_payouts = 0.0
        total_score = 0.0

    try:
        attack_resp = await store._ch.query("SELECT count() FROM arena.attack_history FINAL")
        total_attacks = int(attack_resp.result_rows[0][0])  # type: ignore
    except Exception:
        total_attacks = 0

    categories_covered = len([c for c, v in coverage.items() if v > 0])

    return {
        "active_bounties": len(active_bounties),
        "total_pool_usdc": round(total_pool, 2),
        "total_submissions": total_submissions,
        "total_evaluations": total_evaluations,
        "total_payouts_usdc": round(total_payouts, 2),
        "total_score_awarded": round(total_score, 4),
        "unique_attacks_recorded": total_attacks,
        "categories_covered": categories_covered,
        "total_categories": len(ALL_CATEGORIES),
        "coverage_pct": round(100 * categories_covered / len(ALL_CATEGORIES), 1) if ALL_CATEGORIES else 0,
        "top_scorer": leaderboard[0] if leaderboard else None,
    }


# ---------------------------------------------------------------------------
# Wallet info
# ---------------------------------------------------------------------------


@TOOL_REGISTRY.tool(
    name="admin.wallet_info",
    description="Show the arena's x402 wallet info: address, chain, spending stats.",
    parameters=[],
)
async def admin_wallet_info(ctx: ToolContext) -> dict[str, Any]:
    """Return wallet and spending info."""
    client = ctx.x402_client
    return {
        "wallet_address": client.wallet_address,
        "total_spent_usdc": round(client.total_spent, 4),
        "remaining_budget_usdc": round(client.remaining_budget, 4),
        "recent_payments": len(client.payment_log),
    }
