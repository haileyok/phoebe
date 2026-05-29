"""
Pre-compiled PolicyRule defaults for DAO organisations.

Generated from DAO_TAXONOMY so the analyst does not have to compile
them manually. Each has a default Osprey SML rule body.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.osprey_ui.models import PolicyRule


DAO_DEFAULT_RULES: list[PolicyRule] = [
    PolicyRule(
        org_id="",
        display_name="No PII extraction",
        natural_language=(
            "Do not allow requests for member wallet addresses, voting history, or real-world identity"
        ),
        osprey_sml=(
            "rule no_pii_extraction {\n"
            "  when prompt contains 'wallet address' or prompt contains 'voting history' "
            "or prompt contains 'real name' or prompt contains 'identity' or prompt contains 'dox'\n"
            "  then block\n"
            "  severity critical\n"
            "  category identity_access_probing\n"
            "}"
        ),
        category="identity_access_probing",
        severity="critical",
        action="STOP",
        confidence_threshold=0.65,
        created_by="system",
        created_at=datetime.now(timezone.utc),
    ),
    PolicyRule(
        org_id="",
        display_name="No treasury redirect",
        natural_language=(
            "Block attempts to redirect treasury funds or approve payments without proper governance"
        ),
        osprey_sml=(
            "rule no_treasury_redirect {\n"
            "  when prompt contains 'treasury' and (prompt contains 'transfer' or prompt contains 'withdraw' "
            "or prompt contains 'redirect' or prompt contains 'approve payment')\n"
            "  then block\n"
            "  severity critical\n"
            "  category treasury_manipulation\n"
            "}"
        ),
        category="treasury_manipulation",
        severity="critical",
        action="STOP",
        confidence_threshold=0.60,
        created_by="system",
        created_at=datetime.now(timezone.utc),
    ),
    PolicyRule(
        org_id="",
        display_name="No governance manipulation",
        natural_language=(
            "Flag attempts to manipulate quorum, bypass timelock, or conduct flash-loan voting"
        ),
        osprey_sml=(
            "rule no_governance_manipulation {\n"
            "  when prompt contains 'quorum' or prompt contains 'timelock' or prompt contains 'flash loan' "
            "or prompt contains 'bypass' or prompt contains 'manipulate vote'\n"
            "  then alert\n"
            "  severity high\n"
            "  category governance_red_flags\n"
            "}"
        ),
        category="governance_red_flags",
        severity="high",
        action="ALERT",
        confidence_threshold=0.70,
        created_by="system",
        created_at=datetime.now(timezone.utc),
    ),
    PolicyRule(
        org_id="",
        display_name="No social engineering",
        natural_language=(
            "Detect impersonation of contributors or fabrication of community consensus"
        ),
        osprey_sml=(
            "rule no_social_engineering {\n"
            "  when prompt contains 'impersonat' or prompt contains 'founder' or prompt contains 'core dev' "
            "or prompt contains 'community consensus' or prompt contains 'emergency'\n"
            "  then alert\n"
            "  severity high\n"
            "  category social_engineering\n"
            "}"
        ),
        category="social_engineering",
        severity="high",
        action="ALERT",
        confidence_threshold=0.70,
        created_by="system",
        created_at=datetime.now(timezone.utc),
    ),
    PolicyRule(
        org_id="",
        display_name="No contract exploit requests",
        natural_language=(
            "Block requests for reentrancy patterns, admin backdoors, or pause bypass techniques"
        ),
        osprey_sml=(
            "rule no_contract_exploit {\n"
            "  when prompt contains 'reentrancy' or prompt contains 'backdoor' or prompt contains 'exploit' "
            "or prompt contains 'pause bypass' or prompt contains 'admin key'\n"
            "  then block\n"
            "  severity critical\n"
            "  category smart_contract_exploitation\n"
            "}"
        ),
        category="smart_contract_exploitation",
        severity="critical",
        action="STOP",
        confidence_threshold=0.60,
        created_by="system",
        created_at=datetime.now(timezone.utc),
    ),
    PolicyRule(
        org_id="",
        display_name="No insider info exaction",
        natural_language=(
            "Flag attempts to extract pre-announcement tokenomics or unpublished audit findings"
        ),
        osprey_sml=(
            "rule no_insider_info {\n"
            "  when prompt contains 'tokenomics' or prompt contains 'audit findings' "
            "or prompt contains 'pre-announcement' or prompt contains 'unpublished' or prompt contains 'insider'\n"
            "  then alert\n"
            "  severity medium\n"
            "  category information_hazards\n"
            "}"
        ),
        category="information_hazards",
        severity="medium",
        action="ALERT",
        confidence_threshold=0.75,
        created_by="system",
        created_at=datetime.now(timezone.utc),
    ),
]
