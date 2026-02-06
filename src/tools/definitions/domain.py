import asyncio
import re
from typing import Any

import httpx
from dns import asyncresolver

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

_DOMAIN_REGEX = re.compile(r"^https?://")


async def _check_http(domain: str) -> tuple[str | int, str | None]:
    """check the http status and see if the domain redirects elsewhere"""
    # give it a shot with https first
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            response = await client.head(f"https://{domain}")
            redirects_to = response.headers.get("Location")
            return response.status_code, redirects_to
    except Exception:
        pass

    # otherwise try http
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            response = await client.head(f"http://{domain}")
            redirects_to = response.headers.get("Location")
            return response.status_code, redirects_to
    except Exception:
        return "unreachable", None


async def _query_dns(
    resolver: asyncresolver.Resolver, domain: str, record_type: str
) -> list[str] | str | None:
    """query domains for a given domain and record type, with an input resolver"""
    try:
        answers = await resolver.resolve(domain, record_type)

        if record_type == "SOA":
            # soa returns a single answer
            return str(answers[0]) if answers else None  # type: ignore
        elif record_type == "MX":
            # mx have priority
            return [f"{answer.preference} {answer.exchange}" for answer in answers]
        elif record_type == "TXT":
            # txt have quotes
            return [
                " ".join(
                    str(s, "utf-8") if isinstance(s, bytes) else str(s)
                    for s in answer.strings
                )
                for answer in answers
            ]
        else:
            return [str(answer) for answer in answers]
    except (resolver.NoAnswer, resolver.NXDOMAIN, resolver.NoNameservers):  # type: ignore
        return [] if record_type != "SOA" else None
    except Exception:
        return [] if record_type != "SOA" else None


@TOOL_REGISTRY.tool(
    name="domain.checkDomain",
    description="Lookup A, AAAA, NS, MX, TXT, CNAME, and SOA for a given input domain",
    parameters=[
        ToolParameter(
            name="domain",
            type="string",
            description="The domain name (not a URL) to check",
        ),
    ],
)
async def check_domain(ctx: ToolContext, domain: str):
    # defensive incase the model decides to stick a url in instead of a domain
    re.sub(_DOMAIN_REGEX, "", domain).split("/")[0]

    try:
        resolver = asyncresolver.Resolver()

        dns_tasks: dict[str, Any] = {
            "a": _query_dns(resolver, domain, "A"),
            "aaaa": _query_dns(resolver, domain, "AAAA"),
            "ns": _query_dns(resolver, domain, "NS"),
            "mx": _query_dns(resolver, domain, "MX"),
            "txt": _query_dns(resolver, domain, "TXT"),
            "cname": _query_dns(resolver, domain, "CNAME"),
            "soa": _query_dns(resolver, domain, "SOA"),
        }

        # run all of the lookups in parallel
        dns_results = await asyncio.gather(*dns_tasks.values(), return_exceptions=True)
        dns_data = dict(zip(dns_tasks.keys(), dns_results))

        a_records = (  # type: ignore
            dns_data.get("a", [])
            if not isinstance(dns_data.get("a"), Exception)
            else []
        )
        aaaa_records = (  # type: ignore
            dns_data.get("aaaa", [])
            if not isinstance(dns_data.get("aaaa"), Exception)
            else []
        )
        cname_records = (  # type: ignore
            dns_data.get("cname", [])
            if not isinstance(dns_data.get("cname"), Exception)
            else []
        )

        http_status, redirects_to = await _check_http(domain)

        result: dict[str, Any] = {
            "success": True,
            "domain": domain,
            "resolves": len(a_records) > 0  # type: ignore
            or len(aaaa_records) > 0  # type: ignore
            or len(cname_records) > 0,  # type: ignore
            "dns": {
                "a": a_records,
                "aaaa": aaaa_records,
                "cname": cname_records,
                "ns": dns_data.get("ns", [])
                if not isinstance(dns_data.get("ns"), Exception)
                else [],
                "mx": dns_data.get("mx", [])
                if not isinstance(dns_data.get("mx"), Exception)
                else [],
                "txt": dns_data.get("txt", [])
                if not isinstance(dns_data.get("txt"), Exception)
                else [],
                "soa": dns_data.get("soa")
                if not isinstance(dns_data.get("soa"), Exception)
                else None,
            },
            "http_status": http_status,
            "redirects_to": redirects_to,
        }

        return result

    except Exception as e:
        result = {"success": False, "domain": domain, "error": str(e)}
        return result
