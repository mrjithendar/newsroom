"""Client for the web app's internal callback API (services/www/src/lib/seo-agent/internal-server.ts).

Everything that already exists in v2 is called, not reimplemented: article context, the change-set validators, draft
storage, the link planner, publish/verify/submit, the SSRF-safe fetcher and the weekly rules/scoring. v4 adds the
existing sync/audit job runner, reports, ask-gap clustering, the review queue/diff and the dashboard read models (for
the admin MCP tools), all as thin wrappers over the same TypeScript functions the admin UI uses. The callback API
listens on a separate, cluster-internal port of the web pods (never routed by the ingress) and requires
SEO_AGENT_CALLBACK_TOKEN. Publishing additionally requires a human approval recorded by the web app (see publish()).
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx


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
    async def article_context(self, article_id: str, body: dict[str, Any]) -> dict[str, Any]: ...
    async def validate(self, article_id: str, body: dict[str, Any]) -> dict[str, Any]: ...
    async def save_draft(self, draft_id: str, body: dict[str, Any]) -> dict[str, Any]: ...
    async def set_draft_status(self, draft_id: str, status: str, reason: str | None = None) -> dict[str, Any]: ...
    async def publish(self, draft_id: str, approval_id: str, accepted_change_ids: list[str], edits: dict[str, str] | None = None) -> dict[str, Any]: ...
    async def verify(self, draft_id: str, attempt: int, final: bool) -> dict[str, Any]: ...
    async def submit(self, draft_id: str) -> dict[str, Any]: ...
    async def read_article(self, ref: str) -> dict[str, Any]: ...
    async def published_articles(self) -> list[dict[str, Any]]: ...
    async def gsc_queries(self, *, topic: str | None = None, article_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]: ...
    async def fetch_docs(self, url: str) -> dict[str, Any]: ...
    async def weekly_candidates(self, limit: int) -> dict[str, Any]: ...
    async def plan_links(self, article_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]: ...
    async def calendar_brief(self, calendar_id: str) -> dict[str, Any]: ...
    async def run_job(self, job_type: str, slot: str | None, trigger: str) -> dict[str, Any]: ...
    async def draft_summary(self, draft_id: str) -> dict[str, Any]: ...
    async def reject_draft(self, draft_id: str, reason: str | None, actor: str) -> dict[str, Any]: ...
    async def add_review_note(self, draft_id: str, note: str, actor: str) -> dict[str, Any]: ...
    async def review_queue(self, params: dict[str, Any]) -> dict[str, Any]: ...
    async def draft_diff(self, draft_id: str) -> dict[str, Any]: ...
    async def open_draft_articles(self, limit: int) -> list[str]: ...
    async def data(self, name: str, params: dict[str, Any]) -> dict[str, Any]: ...
    async def topics(self) -> list[dict[str, Any]]: ...
    async def report_generate(self, period: str) -> dict[str, Any]: ...
    async def report_narrative(self, report_id: str, narrative: str, check: dict[str, Any]) -> dict[str, Any]: ...
    async def report_notify(self, report_id: str) -> dict[str, Any]: ...
    async def report_pdf(self, report_id: str, pdf: bytes) -> dict[str, Any]: ...
    async def cluster_ask_gaps(self) -> dict[str, Any]: ...


class HttpWebApp:
    def __init__(self, base_url: str, token: str, timeout_s: float = 60, transport: httpx.AsyncBaseTransport | None = None):
        self.client = httpx.AsyncClient(
            base_url=f"{base_url.rstrip('/')}/internal/seo-agent",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "seo-agent/4"},
            timeout=timeout_s,
            transport=transport,
            follow_redirects=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = await self.client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise WebAppError(0, f"{type(e).__name__}", "unreachable") from e
        if r.status_code >= 400:
            try:
                j = r.json()
            except ValueError:
                j = {}
            raise WebAppError(r.status_code, str(j.get("error", r.reason_phrase))[:300], str(j.get("code", "error")))
        return r.json() if r.content else {}

    async def article_context(self, article_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", f"/articles/{article_id}/context", json=body)

    async def validate(self, article_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", f"/articles/{article_id}/validate", json=body)

    async def save_draft(self, draft_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", f"/drafts/{draft_id}/save", json=body)

    async def set_draft_status(self, draft_id: str, status: str, reason: str | None = None) -> dict[str, Any]:
        return await self._req("POST", f"/drafts/{draft_id}/status", json={"status": status, "reason": reason})

    async def publish(self, draft_id: str, approval_id: str, accepted_change_ids: list[str], edits: dict[str, str] | None = None) -> dict[str, Any]:
        return await self._req("POST", f"/drafts/{draft_id}/publish", json={"approvalId": approval_id, "acceptedChangeIds": accepted_change_ids, "edits": edits or {}})

    async def verify(self, draft_id: str, attempt: int, final: bool) -> dict[str, Any]:
        return await self._req("POST", f"/drafts/{draft_id}/verify", json={"attempt": attempt, "final": final})

    async def submit(self, draft_id: str) -> dict[str, Any]:
        return await self._req("POST", f"/drafts/{draft_id}/submit", json={})

    async def read_article(self, ref: str) -> dict[str, Any]:
        return await self._req("GET", "/articles/read", params={"ref": ref})

    async def published_articles(self) -> list[dict[str, Any]]:
        return (await self._req("GET", "/articles/published"))["articles"]

    async def gsc_queries(self, *, topic: str | None = None, article_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        params = {k: v for k, v in {"topic": topic, "articleId": article_id, "limit": str(limit)}.items() if v}
        return (await self._req("GET", "/gsc/queries", params=params))["queries"]

    async def fetch_docs(self, url: str) -> dict[str, Any]:
        return await self._req("POST", "/fetch-docs", json={"url": url})

    async def weekly_candidates(self, limit: int) -> dict[str, Any]:
        return await self._req("POST", "/weekly/candidates", json={"limit": limit})

    async def plan_links(self, article_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._req("POST", f"/articles/{article_id}/links", json={"candidates": candidates})

    async def calendar_brief(self, calendar_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/calendar/{calendar_id}/brief")

    # --- v4 ---------------------------------------------------------------------------------------------------

    async def run_job(self, job_type: str, slot: str | None, trigger: str) -> dict[str, Any]:
        return await self._req("POST", "/jobs/run", json={"type": job_type, "slot": slot, "trigger": trigger}, timeout=900)

    async def draft_summary(self, draft_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/drafts/{draft_id}")

    async def reject_draft(self, draft_id: str, reason: str | None, actor: str) -> dict[str, Any]:
        return await self._req("POST", f"/drafts/{draft_id}/reject", json={"reason": reason, "actor": actor})

    async def add_review_note(self, draft_id: str, note: str, actor: str) -> dict[str, Any]:
        return await self._req("POST", f"/drafts/{draft_id}/notes", json={"note": note, "actor": actor})

    async def review_queue(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._req("GET", "/review/queue", params={k: str(v) for k, v in params.items() if v is not None})

    async def draft_diff(self, draft_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/drafts/{draft_id}/diff")

    async def open_draft_articles(self, limit: int) -> list[str]:
        return (await self._req("GET", "/drafts/open-articles", params={"limit": str(limit)}))["articleIds"]

    async def data(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._req("GET", f"/data/{name}", params={k: str(v) for k, v in params.items() if v is not None})

    async def topics(self) -> list[dict[str, Any]]:
        return (await self._req("GET", "/articles/topics"))["topics"]

    async def report_generate(self, period: str) -> dict[str, Any]:
        return await self._req("POST", "/reports/generate", json={"period": period})

    async def report_narrative(self, report_id: str, narrative: str, check: dict[str, Any]) -> dict[str, Any]:
        return await self._req("POST", f"/reports/{report_id}/narrative", json={"narrative": narrative, "check": check})

    async def report_notify(self, report_id: str) -> dict[str, Any]:
        return await self._req("POST", f"/reports/{report_id}/notify", json={})

    async def report_pdf(self, report_id: str, pdf: bytes) -> dict[str, Any]:
        """The rendered PDF → the web app, which stores it in R2 with its existing storage client (D-61)."""
        import base64

        return await self._req("POST", f"/reports/{report_id}/pdf", json={"pdf": base64.b64encode(pdf).decode()}, timeout=120)

    async def cluster_ask_gaps(self) -> dict[str, Any]:
        return await self._req("POST", "/ask-gaps/cluster", json={})
