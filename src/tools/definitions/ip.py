import re
from typing import Any

import httpx

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

_IP_REGEX = re.compile(
    r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$"
    r"|^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"
    r"|^::$"
    r"|^([0-9a-fA-F]{1,4}:){1,7}:$"
    r"|^::[0-9a-fA-F]{1,4}(:[0-9a-fA-F]{1,4}){0,5}$"
)


@TOOL_REGISTRY.tool(
    name="ip.lookup",
    description="GeoIP and ASN lookup for an IP address. Returns geographic location (country, region, city, coordinates, timezone), network information (ISP, org, ASN), and flags for mobile, proxy, and hosting IPs.",
    parameters=[
        ToolParameter(
            name="ip",
            type="string",
            description="The IP address to look up (IPv4 or IPv6)",
        ),
    ],
)
async def ip_lookup(ctx: ToolContext, ip: str) -> dict[str, Any]:
    ip = ip.strip()
    if not _IP_REGEX.match(ip):
        return {"success": False, "ip": ip, "error": "Invalid IP address format"}

    try:
        # ip-api.com free tier requires HTTP, not HTTPS
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"http://ip-api.com/json/{ip}",
                params={
                    "fields": "status,message,country,countryCode,region,regionName,city,zip,lat,lon,timezone,isp,org,as,asname,mobile,proxy,hosting,query"
                },
            )
            data = response.json()

        if data.get("status") == "fail":
            return {
                "success": False,
                "ip": ip,
                "error": data.get("message", "Lookup failed"),
            }

        return {
            "success": True,
            "ip": data.get("query", ip),
            "geo": {
                "country": data.get("country"),
                "country_code": data.get("countryCode"),
                "region": data.get("regionName"),
                "region_code": data.get("region"),
                "city": data.get("city"),
                "zip": data.get("zip"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "timezone": data.get("timezone"),
            },
            "network": {
                "isp": data.get("isp"),
                "org": data.get("org"),
                "asn": data.get("as"),
                "asn_name": data.get("asname"),
            },
            "flags": {
                "is_mobile": data.get("mobile", False),
                "is_proxy": data.get("proxy", False),
                "is_hosting": data.get("hosting", False),
            },
        }

    except Exception as e:
        return {"success": False, "ip": ip, "error": str(e)}
