from urllib.parse import urljoin, urlparse

import httpx
from whois import Any

from src.tools.registry import TOOL_REGISTRY, ToolContext, ToolParameter

_KNOWN_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "j.mp",
    "rb.gy",
    "shorturl.at",
    "tiny.cc",
    "bl.ink",
    "short.io",
    "cutt.ly",
    "rebrand.ly",
}


@TOOL_REGISTRY.tool(
    name="url.expand",
    description="Follow a URL through its redirect chain (up to 10 hops), recording each hop's URL and HTTP status code. Flags known URL shorteners. Useful for investigating obfuscated or shortened links in spam/phishing content.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="The URL to expand and follow through redirects",
        ),
    ],
)
async def url_expand(ctx: ToolContext, url: str) -> dict[str, Any]:
    hops: list[dict[str, Any]] = []
    current_url = url
    max_hops = 10
    visited: set[str] = set()

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
        for i in range(max_hops):
            if current_url in visited:
                break
            visited.add(current_url)

            try:
                # try HEAD first to avoid downloading large bodies
                try:
                    response = await client.head(current_url)
                except httpx.HTTPError:
                    response = await client.get(
                        current_url,
                        headers={"Range": "bytes=0-0"},
                    )

                hop = {
                    "hop": i + 1,
                    "url": current_url,
                    "status_code": response.status_code,
                }

                parsed = urlparse(current_url)
                if parsed.hostname and parsed.hostname.lower() in _KNOWN_SHORTENERS:
                    hop["is_shortener"] = True

                hops.append(hop)

                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        break
                    # handle relative redirect URLs
                    current_url = urljoin(current_url, location)
                else:
                    break

            except Exception as e:
                hops.append(
                    {
                        "hop": i + 1,
                        "url": current_url,
                        "error": str(e),
                    }
                )
                break

    final_url = hops[-1]["url"] if hops else url
    parsed_input = urlparse(url)
    is_shortener = (
        parsed_input.hostname is not None
        and parsed_input.hostname.lower() in _KNOWN_SHORTENERS
    )

    return {
        "success": True,
        "input_url": url,
        "final_url": final_url,
        "is_shortener": is_shortener,
        "total_hops": len(hops),
        "hops": hops,
    }
