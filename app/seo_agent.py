"""What the news agent asks the seo-agent for (moks-seo-agent/app/main.py), with its own token SEO_AGENT_NEWS_TOKEN.

The seo-agent accepts that token on three routes only and answers 403 on every other one:

    POST /search               {q, k} → {hits: [...]}: the blog's own index, for site match and the crew's
                               vector_search_site tool. Read-only; no model call.
    POST /llm-budget/reserve   {max_usd, purpose} → {id}, or 402 with the reason: the one LLM budget
    POST /llm-budget/settle    {id}: let go of a hold once this service has written its usage row

**The one LLM budget** stays one budget (Admin → AI & SEO → Settings → LLM budget). This service is an outside caller,
like the web app's own LLM calls (moks-cms/src/lib/seo-intel/llm-gate.ts), with the same rules:

- no seo-agent set up here (no SEO_AGENT_URL or no SEO_AGENT_NEWS_TOKEN): the call is allowed. There is no budget
  anywhere but the provider's own spend limit (the Anthropic Console workspace).
- the seo-agent is older than the one budget (404/405): the call is allowed, with a warning in the log.
- the seo-agent is set up but does not answer (or answers 5xx): the call is refused. A limit that cannot be checked
  is not skipped.

The usage row of every call goes to `seo_llm_usage` (purpose `news.*`) as before, so the budget and the spend views
count it. The hold covers the call's upper bound while it is in flight and is settled right after the row is written.

**Site search** is optional: the seo-agent down, slow or not set up → no hits (site match skipped, the crew's search
tool says it is unavailable). It never fails a run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from . import tracing
from .llm import BudgetExceeded, BudgetGuard, Reservation, SpendStore

log = logging.getLogger("news_agent.seo_agent")

UNCHECKED = "The LLM budget could not be checked: the seo-agent did not answer. No call was made."
SEARCH_TIMEOUT_S = 8.0
BUDGET_TIMEOUT_S = 8.0
SEARCH_COOLDOWN_S = 60.0  # after a failed search, skip the seo-agent for this long (one story's batch stays quick)


@dataclass
class Hit:
    """One chunk of a site article (the fields of moks-seo-agent/app/index.py Hit that the news desk reads)."""

    article_id: str
    title: str
    path: str
    heading_path: str
    anchor: str | None
    content: str
    score: float  # cosine similarity of the chunk
    category: str = ""

    @property
    def url(self) -> str:
        return f"{self.path}#{self.anchor}" if self.anchor else self.path


class SeoAgentClient:
    def __init__(self, base_url: str, token: str, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._warned: set[str] = set()
        self.client = httpx.AsyncClient(
            base_url=self.base_url or "http://seo-agent.invalid",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "news-agent/1"},
            transport=transport,
            follow_redirects=False,
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    async def close(self) -> None:
        await self.client.aclose()

    def _warn_once(self, key: str, msg: str, **extra: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(msg, extra={"event": f"seo_agent_{key}", **extra})

    async def _post(self, path: str, body: dict[str, Any], limit_s: float) -> httpx.Response:
        return await self.client.post(path, json=body, timeout=limit_s, headers=tracing.inject({}))

    # --- the one LLM budget ---------------------------------------------------------------------------------------

    async def reserve(self, max_usd: float, purpose: str) -> str | None:
        """Holds a call's upper bound against the one budget. Returns the hold's id, or None when there is no budget
        to hold against (not set up here, or an older seo-agent). BudgetExceeded when the call must not be made."""
        if not self.configured:
            self._warn_once("unconfigured", "no seo-agent set up (SEO_AGENT_URL / SEO_AGENT_NEWS_TOKEN): LLM calls are not checked against the one LLM budget")
            return None
        try:
            r = await self._post("/llm-budget/reserve", {"max_usd": round(max(0.0, min(1000.0, max_usd)), 6), "purpose": purpose}, BUDGET_TIMEOUT_S)
        except httpx.HTTPError as e:
            raise BudgetExceeded("llm_unchecked", UNCHECKED) from e
        if r.status_code == 200:
            hold = (r.json() or {}).get("id")
            return hold if isinstance(hold, str) else None
        if r.status_code in (404, 405):
            self._warn_once("budget_missing", "the seo-agent has no /llm-budget/reserve (older release): LLM calls are not checked against the one LLM budget")
            return None
        if r.status_code == 402:
            try:
                j = r.json()
            except ValueError:
                j = {}
            raise BudgetExceeded(str(j.get("scope") or "llm"), str(j.get("error") or "The LLM budget is reached."))
        if r.status_code in (401, 403):
            raise BudgetExceeded("llm_unchecked", f"The LLM budget could not be checked: the seo-agent refused SEO_AGENT_NEWS_TOKEN ({r.status_code}). No call was made.")
        raise BudgetExceeded("llm_unchecked", UNCHECKED)

    async def settle(self, hold: str) -> None:
        """Best effort: a hold that is never settled lapses on the seo-agent after 15 minutes."""
        try:
            await self._post("/llm-budget/settle", {"id": hold}, BUDGET_TIMEOUT_S)
        except httpx.HTTPError:
            log.warning("llm budget hold not settled (it lapses on its own)", extra={"event": "llm_budget_settle_failed"})

    # --- site search ----------------------------------------------------------------------------------------------

    async def search(self, q: str, k: int = 6) -> list[Hit]:
        if not self.configured:
            return []
        r = await self._post("/search", {"q": q[:500], "k": max(1, min(10, k))}, SEARCH_TIMEOUT_S)
        if r.status_code != 200:
            raise httpx.HTTPStatusError(f"search answered {r.status_code}", request=r.request, response=r)
        out = []
        for h in (r.json() or {}).get("hits") or []:
            try:
                out.append(Hit(article_id=str(h.get("article_id") or ""), title=str(h["title"]), path=str(h["path"]), heading_path=str(h.get("heading") or h.get("heading_path") or ""), anchor=h.get("anchor"), content=str(h.get("content") or ""), score=float(h["score"]), category=str(h.get("category") or "")))
            except (KeyError, TypeError, ValueError):
                continue
        return out


class RemoteRetriever:
    """The site search for the news desk (`deps.retriever`): the seo-agent's index over POST /search."""

    def __init__(self, client: SeoAgentClient, monotonic: Any = time.monotonic):
        self.client = client
        self.monotonic = monotonic
        self._down_until = 0.0

    @property
    def available(self) -> bool:
        return self.client.configured and self.monotonic() >= self._down_until

    async def search(self, query: str, k: int = 6, **_: Any) -> list[Hit]:
        if not self.available or not query.strip():
            return []
        try:
            return await self.client.search(query, k)
        except (httpx.HTTPError, ValueError) as e:
            self._down_until = self.monotonic() + SEARCH_COOLDOWN_S
            log.warning("site search failed; skipped for a minute", extra={"event": "site_search_failed", "error": type(e).__name__})
            return []


class RemoteBudgetGuard(BudgetGuard):
    """BudgetGuard whose caps live in the seo-agent: every call's upper bound is held there before the call and let go
    after its usage row is written here. No limit of its own; no Ask-the-Blog cap (there is no Ask here)."""

    def __init__(self, store: SpendStore, client: SeoAgentClient, clock: Any = None):
        super().__init__(store, None, 0.0, clock=clock)
        self.client = client
        self._settling: set[asyncio.Task[None]] = set()

    async def _admit_usd(self, est: float, purpose: str, hold: bool) -> Reservation:
        hold_id = await self.client.reserve(est, purpose)
        if not hold:
            if hold_id:
                await self.client.settle(hold_id)
            return Reservation(None, 0.0, released=True)
        return Reservation(None, est, hold_id=hold_id)

    def release(self, r: Reservation) -> None:
        if r.released:
            return
        r.released = True
        if r.hold_id:
            t = asyncio.get_running_loop().create_task(self.client.settle(r.hold_id))
            self._settling.add(t)
            t.add_done_callback(self._settling.discard)

    async def exhausted(self) -> bool:
        """Whether the one budget has nothing left: a zero-dollar hold the seo-agent refuses."""
        try:
            hold = await self.client.reserve(0.0, "news.check")
        except BudgetExceeded as e:
            return e.scope != "llm_unchecked"  # unreachable: say so at the first real call, not here
        if hold:
            await self.client.settle(hold)
        return False

    async def status(self) -> dict[str, Any]:
        """The budget itself is the seo-agent's (GET /llm-budget there). Only this service's own spend is known here."""
        now: datetime = self.clock()
        from .llm_budgets import period_start

        return {
            "daily_budget_usd": None,
            "monthly_budget_usd": None,
            "day_spend_usd": await self.store.llm_spend(period_start("day", now)),
            "month_spend_usd": await self.store.llm_spend(period_start("month", now)),
            "ask_day_spend_usd": 0.0,
        }
