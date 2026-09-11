"""
person_info.py
----------------
Looks up a short public bio/summary for an identified person, so the app
can show "who is this" context after a successful match - not just a
name, but achievements/background a person would otherwise Google
separately.

Uses Wikipedia's public REST API (no API key, no signup, generous rate
limits) rather than a paid web-search API, since the whole point is to
work out of the box on a fresh deploy with zero extra configuration. This
is a reasonable fit for the kind of well-known identities (celebrities,
public figures) this project's dataset uses - it will return nothing
useful for a private individual who doesn't have a Wikipedia page, which
is a real limitation worth knowing about (see README).
"""

import requests

_SEARCH_API = "https://en.wikipedia.org/w/api.php"
_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def fetch_person_summary(name: str, timeout: float = 6.0):
    """
    Returns a dict {title, extract, url, thumbnail} for the best-matching
    Wikipedia page for `name`, or None if nothing was found or the request
    failed (network error, timeout, disambiguation page, etc.) - callers
    should treat None as "no info available" and degrade gracefully rather
    than erroring, since this is a nice-to-have, not core functionality.
    """
    try:
        search_resp = requests.get(
            _SEARCH_API,
            params={"action": "query", "list": "search", "srsearch": name,
                    "format": "json", "srlimit": 1},
            timeout=timeout,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])
        if not results:
            return None
        title = results[0]["title"]

        summary_resp = requests.get(
            _SUMMARY_API.format(title=requests.utils.quote(title)),
            timeout=timeout,
        )
        summary_resp.raise_for_status()
        data = summary_resp.json()

        if data.get("type") == "disambiguation" or not data.get("extract"):
            return None

        thumbnail = data.get("thumbnail", {})
        return {
            "title": data.get("title", name),
            "extract": data["extract"],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
            "thumbnail": thumbnail.get("source") if thumbnail else None,
        }
    except Exception:
        return None
