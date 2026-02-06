import asyncio
from datetime import datetime, timezone
from typing import Any

import whois

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter


def _normalize_date(value: Any) -> str | None:
    """normalize python-whois date values, which can be a single datetime, a list, or None."""
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value) if value else None


@TOOL_REGISTRY.tool(
    name="whois.lookup",
    description="Look up WHOIS registration data for a domain. Returns registrar, creation/expiration dates, name servers, registrant info, and domain age in days. Domain age is a key T&S signal — newly registered domains are heavily used for spam and phishing.",
    parameters=[
        ToolParameter(
            name="domain",
            type="string",
            description="The domain name to look up (e.g. example.com)",
        ),
    ],
)
async def whois_lookup(ctx: ToolContext, domain: str) -> dict[str, Any]:
    try:
        w = await asyncio.to_thread(whois.whois, domain)
    except Exception as e:
        return {"success": False, "domain": domain, "error": str(e)}

    creation_date = _normalize_date(w.creation_date)
    expiration_date = _normalize_date(w.expiration_date)
    updated_date = _normalize_date(w.updated_date)

    # compute domain age
    domain_age_days: int | None = None
    if creation_date:
        try:
            raw = w.creation_date
            if isinstance(raw, list):
                raw = raw[0]
            if isinstance(raw, datetime):
                delta = datetime.now(timezone.utc) - raw.replace(tzinfo=timezone.utc)
                domain_age_days = delta.days
        except Exception:
            pass

    name_servers = w.name_servers
    if isinstance(name_servers, set):
        name_servers = sorted(name_servers)

    return {
        "success": True,
        "domain": domain,
        "registrar": w.registrar,
        "creation_date": creation_date,
        "expiration_date": expiration_date,
        "updated_date": updated_date,
        "domain_age_days": domain_age_days,
        "name_servers": name_servers,
        "dnssec": w.dnssec if hasattr(w, "dnssec") else None,
        "registrant": {
            "name": w.name if hasattr(w, "name") else None,
            "org": w.org if hasattr(w, "org") else None,
            "country": w.country if hasattr(w, "country") else None,
            "state": w.state if hasattr(w, "state") else None,
            "city": w.city if hasattr(w, "city") else None,
            "emails": w.emails if hasattr(w, "emails") else None,
        },
    }
