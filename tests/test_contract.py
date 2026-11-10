"""Contract: every call HttpWebApp makes matches a route of the web app's callback server (internal-server.ts).

The route table is parsed from the TypeScript source, so a renamed path or changed method on either side fails here
instead of in production.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from app.webapp import HttpWebApp

SERVER = Path(__file__).resolve().parents[2] / "www" / "src" / "lib" / "seo-agent" / "internal-server.ts"
UUID = "11111111-1111-4111-8111-111111111111"


def routes() -> list[tuple[str, re.Pattern[str]]]:
    src = SERVER.read_text()
    out = []
    for method, pattern in re.findall(r'\["(GET|POST)", /(\^[^,]+\$)/,', src):
        out.append((method, re.compile(pattern.replace("\\/", "/"))))
    assert len(out) >= 10, "could not parse the route table"
    return out


async def test_every_client_call_hits_a_server_route() -> None:
    seen: list[tuple[str, str, str | None]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path, req.headers.get("authorization")))
        return httpx.Response(200, json={"articles": [], "queries": [], "items": [], "articleIds": [], "topics": []})

    web = HttpWebApp("http://web:9465", "c" * 40, transport=httpx.MockTransport(handler))
    await web.article_context(UUID, {})
    await web.validate(UUID, {})
    await web.save_draft(UUID, {})
    await web.set_draft_status(UUID, "generation_failed")
    await web.publish(UUID, UUID, [])
    await web.verify(UUID, 1, False)
    await web.submit(UUID)
    await web.read_article("/blog/x")
    await web.published_articles()
    await web.gsc_queries(topic="istio")
    await web.fetch_docs("https://kubernetes.io/")
    await web.weekly_candidates(5)
    await web.plan_links(UUID, [])
    await web.calendar_brief(UUID)
    # v4
    await web.run_job("gsc_sync", "2026-11-10", "schedule")
    await web.draft_summary(UUID)
    await web.reject_draft(UUID, "r", "admin")
    await web.add_review_note(UUID, "n", "admin")
    await web.review_queue({"sort": "priority"})
    await web.draft_diff(UUID)
    await web.open_draft_articles(10)
    for name in ("overview", "traffic-change", "keywords", "page-report", "audit-issues", "llm-usage"):
        await web.data(name, {})
    await web.topics()
    await web.report_generate("weekly")
    await web.report_narrative(UUID, "text", {"ok": True})
    await web.report_notify(UUID)
    await web.report_pdf(UUID, b"%PDF-1.4")
    await web.cluster_ask_gaps()
    # news desk
    await web.news_fetch({"url": "https://e.org/f", "purpose": "feed"})
    await web.news_save_draft({})
    await web.news_store_media({})
    await web.news_generate_image({})
    await web.news_update_draft_image(UUID, {})
    await web.news_site_signals()
    await web.news_notify({})
    await web.news_collect_performance()
    await web.close()
    called = {m for m in dir(HttpWebApp) if not m.startswith("_") and m != "close"}
    assert len(seen) >= len(called), "a client method is not exercised by this contract test"
    table = routes()
    for method, path, auth in seen:
        assert path.startswith("/internal/seo-agent/"), path
        rel = path.removeprefix("/internal/seo-agent")
        assert any(m == method and p.match(rel) for m, p in table), f"{method} {rel} has no route in internal-server.ts"
        assert auth == f"Bearer {'c' * 40}"


async def test_web_app_errors_carry_status_and_transience() -> None:
    from app.webapp import WebAppError

    web = HttpWebApp("http://web:9465", "t", transport=httpx.MockTransport(lambda r: httpx.Response(403, json={"error": "No valid admin approval", "code": "approval_required"})))
    with pytest.raises(WebAppError) as e:
        await web.publish(UUID, "x", [])
    assert e.value.status == 403 and e.value.code == "approval_required" and not e.value.transient
    web2 = HttpWebApp("http://web:9465", "t", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(WebAppError) as e2:
        await web2.submit(UUID)
    assert e2.value.transient
