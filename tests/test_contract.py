"""Contract: every call HttpWebApp makes matches a route of the web app's callback server (internal-server.ts) that the
news agent's token may call (NEWS_AGENT_ROUTES there), and the two lists are the same.

The route tables are parsed from the TypeScript source, so a renamed path, a changed method, or a route the web app
does not let the news agent call fails here instead of in production. Needs a moks-cms checkout (tests/moks_cms.py).
"""

from __future__ import annotations

import re

import httpx
import pytest

from app.webapp import CALLBACK_ROUTES, HttpWebApp
from tests.moks_cms import www

SERVER = "src/lib/seo-agent/internal-server.ts"
UUID = "11111111-1111-4111-8111-111111111111"
ROUTE = r'\["(GET|POST)", /(\^[^,]+\$)/'


def _parse(src: str) -> list[tuple[str, re.Pattern[str]]]:
    return [(method, re.compile(pattern.replace("\\/", "/"))) for method, pattern in re.findall(ROUTE, src)]


def routes() -> list[tuple[str, re.Pattern[str]]]:
    out = _parse(www(SERVER).read_text())
    assert len(out) >= 10, "could not parse the route table"
    return out


def news_routes() -> list[tuple[str, re.Pattern[str]]]:
    src = www(SERVER).read_text()
    start = src.index("export const NEWS_AGENT_ROUTES")
    block = src[start : src.index("];", start)]
    out = _parse(block)
    assert out, "could not parse NEWS_AGENT_ROUTES"
    return out


async def calls() -> list[tuple[str, str, str | None]]:
    seen: list[tuple[str, str, str | None]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, req.headers.get("authorization")))
        return httpx.Response(200, json={"queries": [], "skills": [], "crews": []})

    web = HttpWebApp("http://web:9465", "c" * 40, transport=httpx.MockTransport(handler))
    await web.gsc_queries(topic="istio")
    await web.news_fetch({"url": "https://e.org/f", "purpose": "feed"})
    await web.news_save_draft({})
    await web.news_store_media({})
    await web.news_generate_image({})
    await web.news_update_draft_image(UUID, {})
    await web.news_site_signals()
    await web.news_notify({})
    await web.news_collect_performance()
    await web.agent_skills()
    await web.crew_configs()
    await web.run_history_settings()
    await web.close()
    called = {m for m in dir(HttpWebApp) if not m.startswith("_") and m != "close"}
    assert len(seen) == len(called), "a client method is not exercised by this contract test"
    return seen


async def test_every_client_call_hits_a_route_the_news_token_may_call() -> None:
    table, allowed = routes(), news_routes()
    seen = await calls()
    assert len(seen) == len(CALLBACK_ROUTES)
    for method, path, auth in seen:
        assert path.startswith("/internal/seo-agent/"), path
        rel = path.removeprefix("/internal/seo-agent")
        assert any(m == method and p.match(rel) for m, p in table), f"{method} {rel} has no route in internal-server.ts"
        assert any(m == method and p.match(rel) for m, p in allowed), f"{method} {rel} is not in NEWS_AGENT_ROUTES: the web app refuses it for this token"
        assert auth == f"Bearer {'c' * 40}"
    assert len(allowed) == len(CALLBACK_ROUTES), "NEWS_AGENT_ROUTES allows a route this client never calls"


def test_the_news_token_cannot_reach_drafts_or_publishing() -> None:
    allowed = news_routes()
    for method, rel in (("POST", f"/drafts/{UUID}/publish"), ("POST", f"/drafts/{UUID}/save"), ("POST", f"/drafts/{UUID}/status"), ("POST", "/reports/generate"), ("GET", "/articles/published")):
        assert not any(m == method and p.match(rel) for m, p in allowed), f"{method} {rel} must not be open to the news agent"


async def test_web_app_errors_carry_status_and_transience() -> None:
    from app.webapp import WebAppError

    web = HttpWebApp("http://web:9465", "t", transport=httpx.MockTransport(lambda r: httpx.Response(403, json={"error": "This token may not call this route", "code": "forbidden"})))
    with pytest.raises(WebAppError) as e:
        await web.news_save_draft({})
    assert e.value.status == 403 and e.value.code == "forbidden" and not e.value.transient
    web2 = HttpWebApp("http://web:9465", "t", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(WebAppError) as e2:
        await web2.news_notify({})
    assert e2.value.transient
