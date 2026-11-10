"""Fixtures for the news desk tests: feeds (labelled test data, not real articles), a fake web app with the news
callbacks, a seeded in-memory store, and a fake crew whose output passes (or deliberately fails) the quality gates."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from app.main import memory_container
from app.news.schemas import Claim, FactCheckReport, FactSheet, ImageBrief, NewsDraft, NewsSourceRef
from tests.conftest import FakeWebApp, settings

JOB = "44444444-4444-4444-8444-444444444444"
SRC_A = "55555555-5555-4555-8555-555555555551"
SRC_B = "55555555-5555-4555-8555-555555555552"
SRC_BAD = "55555555-5555-4555-8555-555555555553"
NOW = datetime(2026, 11, 10, 6, 0, tzinfo=UTC)

# Test fixture text (written for these tests; it describes a fictional release in plain words).
K8S_TEXT = (
    "The Kubernetes project released version 1.35 today. In-place pod resize has graduated to stable, so operators can change "
    "the CPU and memory of a running container without recreating the pod. The release contains 42 enhancements in total. "
    "The kubelet now reports resize status through the pod status field. Cluster administrators should check that their "
    "container runtime supports the resize API before relying on it in production."
)
K8S_TEXT_B = (
    "Kubernetes 1.35 is out. The biggest change for platform teams is that in-place pod resize is now stable after several "
    "releases in beta. The project counts 42 enhancements in this cycle, including the stable resize feature and changes to "
    "how the kubelet reports resize progress."
)


def rss(items: list[tuple[str, str, str]]) -> str:
    body = "".join(f"<item><title>{t}</title><link>{u}</link><description>{d}</description><pubDate>Tue, 10 Nov 2026 04:00:00 GMT</pubDate></item>" for t, u, d in items)
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>Fixture feed</title>{body}</channel></rss>'


def atom(items: list[tuple[str, str, str]]) -> str:
    body = "".join(f'<entry><title>{t}</title><link rel="alternate" href="{u}"/><updated>2026-11-10T04:30:00Z</updated><content type="html">{d}</content></entry>' for t, u, d in items)
    return f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Releases</title>{body}</feed>'


FEEDS = {
    "https://feeds.example.org/k8s.xml": rss([("Kubernetes 1.35 released with stable in-place pod resize", "https://blog.example.org/k8s-135?utm_source=rss", K8S_TEXT)]),
    "https://releases.example.net/k8s.atom": atom([("Kubernetes v1.35: in-place pod resize goes stable", "https://releases.example.net/kubernetes/v1.35", K8S_TEXT_B)]),
}


@dataclass
class NewsWeb(FakeWebApp):
    feeds: dict[str, str] = field(default_factory=lambda: dict(FEEDS))
    failing: set[str] = field(default_factory=lambda: {"https://broken.example.com/feed.xml"})
    saved: list[dict[str, Any]] = field(default_factory=list)
    media: list[dict[str, Any]] = field(default_factory=list)
    news_notes: list[dict[str, Any]] = field(default_factory=list)
    gemini: list[bytes] = field(default_factory=list)
    save_errors: list[str] = field(default_factory=list)
    image_updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def news_fetch(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("news_fetch", body["url"]))
        u = body["url"]
        if u in self.failing:
            return {"status": 503, "error": "upstream 503"}
        if u in self.feeds:
            return {"status": 200, "finalUrl": u, "headers": {"etag": '"v1"'}, "body": self.feeds[u]}
        return {"status": 404, "error": "not found"}

    async def news_save_draft(self, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("news_save_draft", body["articleId"]))
        if self.save_errors:
            return {"errors": list(self.save_errors)}
        self.saved.append(body)
        n = len(self.saved)
        return {"draftId": f"00000000-0000-4000-8000-{n:012d}", "postId": f"00000000-0000-4000-9000-{n:012d}", "errors": [], "flags": []}

    async def news_store_media(self, body: dict[str, Any]) -> dict[str, Any]:
        self.media.append(body)
        mid = f"00000000-0000-4000-a000-{len(self.media):012d}"
        return {"mediaId": mid, "src": f"/media/{mid}/cover.webp", "variants": {k: f"/media/{mid}-{k}/cover-{k}.webp" for k in body["files"]}}

    async def news_generate_image(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.gemini:
            return {"error": "Image generation is not configured."}
        return {"data": base64.b64encode(self.gemini.pop(0)).decode(), "mimeType": "image/png", "model": "gemini-test"}

    async def news_update_draft_image(self, draft_id: str, image: dict[str, Any]) -> dict[str, Any]:
        self.image_updates.append((draft_id, image))
        return {"ok": True}

    async def news_site_signals(self) -> dict[str, Any]:
        return {"topArticles": [{"path": "/blog/ingress-tls", "clicks28": 210}], "queries": [{"query": "kubernetes pod resize", "impressions": 300}]}

    async def news_notify(self, body: dict[str, Any]) -> dict[str, Any]:
        self.news_notes.append(body)
        return {"sent": False, "reason": "no recipients"}

    async def news_collect_performance(self) -> dict[str, Any]:
        return {"rows": 0}


def png(w: int = 1600, h: int = 900, color: tuple[int, int, int] = (20, 60, 140)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


GOOD_BODY = """Version 1.35 of Kubernetes is out, and for anyone running stateful or bursty workloads the release brings a welcome change: in-place pod resize is now stable. You can raise or lower the CPU and memory of a running container without recreating its pod. The project counts 42 enhancements in this cycle.

## Why it matters

Until now, giving a busy service more memory meant a restart, which drops connections and empties warm caches. Teams worked around that by over-provisioning, which wastes money on every node. A stable resize path lets you right-size live workloads during an incident or a traffic spike, and it makes vertical autoscaling far less disruptive. It also changes how you should think about requests and limits: they become settings you can tune while the service runs, not decisions you lock in at deploy time.

## What to do

Check that your container runtime supports the resize API before you depend on it. Then try a resize on a non-critical deployment in a staging cluster and watch the pod status field, where the kubelet now reports progress. Update your runbooks so on-call engineers know they can resize instead of restarting. Finally, review any autoscaler settings that assumed a restart, since they may now act faster than your alerts expect."""


def good_draft(sources: list[str] | None = None) -> NewsDraft:
    urls = sources or ["https://blog.example.org/k8s-135", "https://releases.example.net/kubernetes/v1.35"]
    return NewsDraft(
        title="Kubernetes 1.35 makes in-place pod resize stable",
        dek="Operators can now change CPU and memory on running pods without a restart.",
        meta_description="Kubernetes 1.35 makes in-place pod resize stable: what changed, why it matters for platform teams, and what to check before you rely on it.",
        slug_suggestion="kubernetes-1-35-in-place-pod-resize-stable",
        body_mdx=GOOD_BODY,
        key_takeaways=["In-place pod resize is stable in Kubernetes 1.35.", "Check runtime support before relying on it."],
        why_it_matters="Resizing without restarts reduces over-provisioning and disruption.",
        headings=["Why it matters", "What to do"],
        sources=[NewsSourceRef(title="Kubernetes 1.35 released", publisher="Example Blog", url=u, accessed_at="2026-11-10") for u in urls],
        fact_check_report=FactCheckReport(checked=4),
        categories=["kubernetes"],
        tags=["kubernetes", "autoscaling"],
        schema_jsonld={"@context": "https://schema.org", "@type": "NewsArticle", "headline": "Kubernetes 1.35 makes in-place pod resize stable", "description": "In-place pod resize is stable."},
        image_brief=ImageBrief(subject="containers expanding inside a cluster grid", composition="wide, centred", avoid=["people", "logos"]),
        reading_time_minutes=2,
        confidence=0.8,
    )


def facts() -> FactSheet:
    return FactSheet(claims=[Claim(id="f1", claim="Kubernetes 1.35 released; in-place pod resize stable", source_url="https://blog.example.org/k8s-135", snippet="released version 1.35", kind="version"), Claim(id="f2", claim="42 enhancements", source_url="https://blog.example.org/k8s-135", snippet="42 enhancements", kind="number")])


@dataclass
class FakeNewsCrew:
    """Returns queued drafts (or exceptions). Records each run's kwargs."""

    drafts: list[Any] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "completed"

    async def run(self, brief: dict[str, Any], **kw: Any) -> Any:
        from app.news.crew import NewsCrewResult

        self.runs.append({"brief": brief, **kw})
        item = self.drafts.pop(0) if self.drafts else good_draft()
        if isinstance(item, Exception):
            raise item
        return NewsCrewResult(draft=item, facts=facts(), status="incomplete" if item.incomplete else "completed", tokens=9000, cost_usd=0.05, tool_calls=[("read_news_item", "x")])


def seed(store: Any, *, job_cfg: dict[str, Any] | None = None, image_provider: str = "template", enabled: bool = True) -> None:
    store.settings = {"max_drafts_per_day": 10}
    store.jobs[JOB] = {"id": JOB, "name": "Cloud native daily", "enabled": enabled, "cron": "0 7 * * *", "timezone": "Asia/Kolkata", "mode": "single_stories", "writing_profile_key": "news-default", "image_profile_key": "cover-template", "review_priority": 50, "version": 3, "config": {"length_min_words": 150, "length_max_words": 900, "min_score": 0.2, "max_articles_per_run": 3, "image": {"enabled": True}, **(job_cfg or {})}}
    store.writing.append({"id": "w1", "key": "news-default", "version": 1, "status": "active", "name": "Default", "required_sections": ["Why it matters"], "banned_phrases": ["game-changer", "in today's fast-paced world"], "system_style": "Plain, direct.", "structure_template": "", "tone": "neutral", "audience": "engineers"})
    store.imaging.append({"id": "i1", "key": "cover-template", "version": 1, "status": "active", "name": "Template", "provider": image_provider, "template_fallback": True, "variants": 2, "style_prompt": "flat vector", "negative_prompt": ""})
    for sid, name, url, typ, cred in ((SRC_A, "Example blog", "https://feeds.example.org/k8s.xml", "rss", 0.9), (SRC_B, "Example releases", "https://releases.example.net/k8s.atom", "github_releases", 0.95), (SRC_BAD, "Broken feed", "https://broken.example.com/feed.xml", "rss", 0.5)):
        store.srcs[sid] = {"id": sid, "name": name, "url": url, "type": typ, "credibility": cred, "enabled": True, "fetch_interval_minutes": 60, "last_fetch_at": None, "fetch_count": 0, "error_count": 0, "consecutive_errors": 0, "avg_items": 0, "items_total": 0, "created_at": NOW - timedelta(days=3)}
    store.topic_rows += [{"slug": "kubernetes", "label": "Kubernetes", "category_slug": "kubernetes", "keywords": ["kubernetes", "kubelet", "pod"], "tags": ["kubernetes"], "enabled": True}, {"slug": "cybersecurity", "label": "Security", "category_slug": "cybersecurity", "keywords": ["cve", "vulnerability"], "tags": ["security"], "enabled": True}]


def news_container(llm: Any, web: NewsWeb, crew: FakeNewsCrew | None = None, *, enabled: bool = True, **kw: Any) -> Any:
    c = memory_container(settings(**kw), llm=llm, web=web)
    store = c.deps.extra["news_store"]
    store.clock = lambda: NOW
    c.deps.extra["clock"] = lambda: NOW
    c.deps.extra["env"] = {"NEWS_DESK_ENABLED": "true" if enabled else "false"}
    c.deps.extra["news_crew"] = (lambda _llm: crew) if crew else None
    return c
