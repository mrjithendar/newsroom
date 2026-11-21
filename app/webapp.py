"""Client for the web app's internal callback API (moks-cms/src/lib/seo-agent/internal-server.ts, src/lib/news/callbacks.ts).

Forked from moks-seo-agent/app/webapp.py@bb9e553 on 2026-11-21 and cut down to what the news desk calls. The callback
API listens on a separate, cluster-internal port of the web pods (never routed by the ingress). The news agent uses
its own token, NEWS_AGENT_CALLBACK_TOKEN, which the web app accepts only on the routes below: the news desk's own
(`/news/*`), Search Console queries for the crew's tool, and the skills, crew configurations and run-history settings
every crew run reads. With it, the web app refuses everything else (drafts, publishing, reports…) with 403.

Fetching (SSRF-safe, robots.txt), CMS drafts, media storage, Gemini images and notifications stay in the web app.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from . import tracing

# The routes the web app accepts with NEWS_AGENT_CALLBACK_TOKEN (moks-cms internal-server.ts NEWS_AGENT_ROUTES).
CALLBACK_ROUTES = (
    "POST /news/fetch",
    "POST /news/drafts",
    "POST /news/drafts/{id}/image",
    "POST /news/media",
    "POST /news/images/generate",
    "GET /news/site-signals",
    "POST /news/notify",
    "POST /news/performance",
    "GET /gsc/queries",
    "GET /skills",
    "GET /crew-configs",
    "GET /run-history/settings",
)


class WebAppError(Exception):
    def __init__(self, status: int, message: str, code: str = "error"):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message

    @property
    def transient(self) -> bool:
        return self.status in (0, 408, 425, 429, 500, 502, 503, 504)


class WebApp(Protocol):
    async def gsc_queries(self, *, topic: str | None = None, article_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]: ...
    # News desk (moks-cms/src/lib/news/callbacks.ts)
    async def news_fetch(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def news_save_draft(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def news_store_media(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def news_generate_image(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def news_update_draft_image(self, draft_id: str, image: dict[str, Any]) -> dict[str, Any]: ...
    async def news_site_signals(self) -> dict[str, Any]: ...
    async def news_notify(self, body: dict[str, Any]) -> dict[str, Any]: ...
    async def news_collect_performance(self) -> dict[str, Any]: ...
    async def agent_skills(self) -> list[dict[str, Any]]: ...
    async def crew_configs(self) -> list[dict[str, Any]]: ...
    async def run_history_settings(self) -> dict[str, Any]: ...


class HttpWebApp:
    def __init__(self, base_url: str, token: str, timeout_s: float = 60, transport: httpx.AsyncBaseTransport | None = None):
        # The web app serves its callback API under /internal/seo-agent for both agents; the token says which one calls.
        self.client = httpx.AsyncClient(
            base_url=f"{base_url.rstrip('/')}/internal/seo-agent",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "news-agent/1"},
            timeout=timeout_s,
            transport=transport,
            follow_redirects=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            # traceparent: the web app's handling of this call joins the flow run's trace.
            r = await self.client.request(method, path, headers=tracing.inject(dict(kw.pop("headers", None) or {})), **kw)
        except httpx.HTTPError as e:
            raise WebAppError(0, f"{type(e).__name__}", "unreachable") from e
        if r.status_code >= 400:
            try:
                j = r.json()
            except ValueError:
                j = {}
            raise WebAppError(r.status_code, str(j.get("error", r.reason_phrase))[:300], str(j.get("code", "error")))
        return r.json() if r.content else {}

    async def gsc_queries(self, *, topic: str | None = None, article_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        params = {k: v for k, v in {"topic": topic, "articleId": article_id, "limit": str(limit)}.items() if v}
        return (await self._req("GET", "/gsc/queries", params=params))["queries"]

    # --- news desk ------------------------------------------------------------------------------------------------

    async def news_fetch(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/news/fetch", json=body, timeout=45)

    async def news_save_draft(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/news/drafts", json=body, timeout=120)

    async def news_store_media(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/news/media", json=body, timeout=120)

    async def news_generate_image(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/news/images/generate", json=body, timeout=300)

    async def news_update_draft_image(self, draft_id: str, image: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", f"/news/drafts/{draft_id}/image", json={"image": image})

    async def news_site_signals(self) -> dict[str, Any]:
        return await self._req("GET", "/news/site-signals")

    async def news_notify(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", "/news/notify", json=body)

    async def news_collect_performance(self) -> dict[str, Any]:
        return await self._req("POST", "/news/performance", json={}, timeout=120)

    # --- read before every crew run -------------------------------------------------------------------------------

    async def agent_skills(self) -> list[dict[str, Any]]:
        """The enabled agent skills (D-96, app/crews/skills.py), read before every crew run."""
        return list((await self._req("GET", "/skills")).get("skills") or [])

    async def crew_configs(self) -> list[dict[str, Any]]:
        """The active crew configuration versions (D-98, app/crews/config.py): [{crew, version, config}], before every crew run."""
        return list((await self._req("GET", "/crew-configs")).get("crews") or [])

    async def run_history_settings(self) -> dict[str, Any]:
        """Run history settings (app/run_history.py): {"transcripts": bool}, whether model calls' prompts are kept."""
        return dict(await self._req("GET", "/run-history/settings"))
