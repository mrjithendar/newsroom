"""Claude access for every flow step and crew: Anthropic SDK client, budget guard, usage logging.

- One admin budget for v2 and v4: usage rows go to the web app's `seo_llm_usage` table (same columns, same prices), so
  SEO_LLM_MONTHLY_BUDGET_USD caps the legacy path and the agents together. Ask-the-Blog is separate: its rows (purpose
  "ask…") never count against the admin budget, and it has its own daily cap (ASK_BLOG_DAILY_BUDGET_USD), which also
  bounds its month (≤ 31 × daily).
- Before every call an upper-bound estimate (characters / 3.5 in, full max_tokens out) is checked; a call that could
  cross the cap is never sent (BudgetExceeded → the flow run moves to paused_budget).
- Stable system text (rules, style guide) is sent as cached system blocks (cache_control ephemeral).
- Tests inject FakeLlm: deterministic structured answers per purpose, with fake token counts.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from . import metrics, tracing
from .llm_budgets import Limit, LimitsSource, period_end, period_start

T = TypeVar("T", bound=BaseModel)

# USD per million tokens; identical to moks-cms/src/lib/seo-intel/claude.ts (first-party list prices, 2026-10).
PRICES: dict[str, dict[str, float]] = {
    "claude-sonnet-5-5": {"input": 2, "output": 10, "cache_read": 0.2, "cache_write": 2.5},
    "claude-haiku-5-5": {"input": 0.1, "output": 0.5, "cache_read": 0.01, "cache_write": 0.125},
    "claude-opus-5-5": {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5},
    "claude-fable-5-1": {"input": 10, "output": 50, "cache_read": 0.25, "cache_write": 12.5},
}
FALLBACK = PRICES["claude-fable-5-1"]


def price_of(model: str) -> dict[str, float]:
    raw = os.environ.get("SEO_LLM_PRICES_JSON")
    if raw:
        try:
            o = json.loads(raw).get(model)
            if isinstance(o, dict) and o.get("input", -1) >= 0 and o.get("output", -1) >= 0:
                return {
                    "input": float(o["input"]),
                    "output": float(o["output"]),
                    "cache_read": float(o.get("cacheRead", o["input"] * 0.1)),
                    "cache_write": float(o.get("cacheWrite", o["input"] * 1.25)),
                }
        except (ValueError, AttributeError, TypeError):
            pass
    if model in PRICES:
        return PRICES[model]
    # The API may answer with a dated id ("claude-sonnet-5-5-20261001"): price it as its family, not the fallback.
    family = max((k for k in PRICES if model.startswith(f"{k}-")), key=len, default=None)
    return PRICES[family] if family else FALLBACK


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    request_id: str | None = None
    latency_ms: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    def cost_usd(self, mode: str = "sync") -> float:
        p = price_of(self.model)
        raw = (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_read_tokens * p["cache_read"]
            + self.cache_write_tokens * p["cache_write"]
        ) / 1_000_000
        return raw / 2 if mode == "batch" else raw


def estimate_cost_usd(model: str, input_chars: int, max_tokens: int) -> float:
    return Usage(model, input_tokens=-(-input_chars * 2 // 7), output_tokens=max_tokens).cost_usd()


class BudgetExceeded(Exception):
    def __init__(self, scope: str, detail: str):
        super().__init__(detail)
        self.scope = scope
        self.detail = detail


class TokenCapExceeded(Exception):
    """A per-run token cap (CREW_MAX_TOKENS_PER_RUN) was reached."""


class InvalidOutput(Exception):
    def __init__(self, detail: str, usage: Usage | None = None):
        super().__init__(detail)
        self.usage = usage


class Truncated(InvalidOutput):
    """stop_reason == "max_tokens": the answer was cut off. A subclass of InvalidOutput, so every existing handler
    treats a truncated answer like an unusable one instead of shipping half a text."""


# --- Spend accounting (shared table seo_llm_usage) ------------------------------------------------------------------


class SpendStore(Protocol):
    async def month_spend(self, now: datetime) -> float: ...
    async def day_spend(self, now: datetime, purpose_prefix: str) -> float: ...
    async def llm_spend(self, since: datetime, *, model: str | None = None, provider: str | None = None) -> float: ...
    async def llm_spend_by_model(self, month_start: datetime, day_start: datetime) -> list[dict[str, Any]]: ...
    async def record(self, row: dict[str, Any]) -> None: ...


class PgSpendStore:
    def __init__(self, pool: Any):
        self.pool = pool

    async def month_spend(self, now: datetime) -> float:
        start = datetime(now.year, now.month, 1, tzinfo=UTC)
        async with self.pool.connection() as c:
            cur = await c.execute(
                "select coalesce(sum(cost_usd), 0)::float8 as s from seo_llm_usage where at >= %s and purpose not like 'ask%%' and purpose not like 'news%%'", (start,)
            )
            spent = float((await cur.fetchone())["s"])
            cur = await c.execute(
                "select coalesce(sum(estimated_cost_usd), 0)::float8 as s from seo_llm_batches where status = 'in_progress'"
            )
            return spent + float((await cur.fetchone())["s"])

    async def day_spend(self, now: datetime, purpose_prefix: str) -> float:
        start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        async with self.pool.connection() as c:
            cur = await c.execute(
                "select coalesce(sum(cost_usd), 0)::float8 as s from seo_llm_usage where at >= %s and purpose like %s",
                (start, f"{purpose_prefix}%"),
            )
            return float((await cur.fetchone())["s"])

    async def llm_spend(self, since: datetime, *, model: str | None = None, provider: str | None = None) -> float:
        """Every purpose since `since`, for one model or one provider (app/llm_budgets.py), plus legacy batches still in
        flight at their estimate (they are Anthropic batches)."""
        async with self.pool.connection() as c:
            cur = await c.execute(
                "select coalesce(sum(cost_usd), 0)::float8 as s from seo_llm_usage where at >= %s"
                " and (%s::text is null or model = %s) and (%s::text is null or provider = %s)",
                (since, model, model, provider, provider),
            )
            spent = float((await cur.fetchone())["s"])
            cur = await c.execute(
                "select coalesce(sum(estimated_cost_usd), 0)::float8 as s from seo_llm_batches where status = 'in_progress'"
                " and (%s::text is null or model = %s) and (%s::text is null or %s = 'anthropic')",
                (model, model, provider, provider),
            )
            return spent + float((await cur.fetchone())["s"])

    async def llm_spend_by_model(self, month_start: datetime, day_start: datetime) -> list[dict[str, Any]]:
        """This month's spend per provider and model, today's alongside, the purposes' first word, and legacy batches
        in flight (counted today, as Anthropic spend). For GET /llm-budgets."""
        async with self.pool.connection() as c:
            cur = await c.execute(
                "select provider, model, coalesce(sum(cost_usd) filter (where at >= %s), 0)::float8 as today,"
                " coalesce(sum(cost_usd), 0)::float8 as month, count(*)::int as calls,"
                " array_agg(distinct split_part(purpose, '.', 1)) as purposes"
                " from seo_llm_usage where at >= %s group by provider, model",
                (day_start, month_start),
            )
            rows = [dict(r) for r in await cur.fetchall()]
            cur = await c.execute(
                "select model, coalesce(sum(estimated_cost_usd), 0)::float8 as s from seo_llm_batches where status = 'in_progress' group by model"
            )
            batches = list(await cur.fetchall())
        by_key = {(r["provider"], r["model"]): r for r in rows}
        for b in batches:
            r = by_key.setdefault(("anthropic", b["model"]), {"provider": "anthropic", "model": b["model"], "today": 0.0, "month": 0.0, "calls": 0, "purposes": []})
            r["today"] += float(b["s"])
            r["month"] += float(b["s"])
            r["purposes"] = [*(r["purposes"] or []), "batch"]
        return list(by_key.values())

    _attribution: bool | None = None  # seo_llm_usage has run_id/agent (web app migration 0022); checked once

    async def _has_attribution(self, c: Any) -> bool:
        """The web app owns the table and may be deployed after this service: write run_id/agent only once its
        migration has added them, so an older schema never fails a model call."""
        if self._attribution is None:
            cur = await c.execute(
                "select count(*)::int as n from information_schema.columns where table_name = 'seo_llm_usage' and column_name in ('run_id', 'agent')"
            )
            self._attribution = int((await cur.fetchone())["n"]) == 2
        return self._attribution

    async def record(self, row: dict[str, Any]) -> None:
        draft_id = row.get("draft_id")
        async with self.pool.connection() as c:
            extra_cols, extra_vals = ("", "")
            if await self._has_attribution(c):
                extra_cols, extra_vals = (", run_id, agent", ", %(run_id)s, %(agent)s")
            await c.execute(
                f"""insert into seo_llm_usage (provider, model, purpose, prompt_version, mode, request_id, input_tokens,
                     output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, latency_ms, outcome, error_code, draft_id{extra_cols})
                   values ('anthropic', %(model)s, %(purpose)s, %(prompt_version)s, 'sync', %(request_id)s, %(input_tokens)s,
                     %(output_tokens)s, %(cache_read_tokens)s, %(cache_write_tokens)s, %(cost_usd)s, %(latency_ms)s,
                     %(outcome)s, %(error_code)s,
                     (select id from seo_drafts where id::text = %(draft_id)s){extra_vals})""",  # noqa: S608 - constant column lists
                {**row, "draft_id": draft_id, "run_id": row.get("run_id"), "agent": row.get("agent")},
            )


@dataclass
class MemorySpendStore:
    rows: list[dict[str, Any]] = field(default_factory=list)
    preexisting_usd: float = 0.0

    async def month_spend(self, now: datetime) -> float:
        return self.preexisting_usd + sum(r["cost_usd"] for r in self.rows if not r["purpose"].startswith(("ask", "news")))

    async def day_spend(self, now: datetime, purpose_prefix: str) -> float:
        return sum(r["cost_usd"] for r in self.rows if r["purpose"].startswith(purpose_prefix))

    async def llm_spend(self, since: datetime, *, model: str | None = None, provider: str | None = None) -> float:
        """Rows without an `at` count as now. Every row here is an Anthropic call."""
        rows = [r for r in self.rows if r.get("at", since) >= since and (model is None or r["model"] == model) and provider in (None, "anthropic")]
        return sum(r["cost_usd"] for r in rows)

    async def llm_spend_by_model(self, month_start: datetime, day_start: datetime) -> list[dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in self.rows:
            at = r.get("at", day_start)
            if at < month_start:
                continue
            m = out.setdefault(r["model"], {"provider": "anthropic", "model": r["model"], "today": 0.0, "month": 0.0, "calls": 0, "purposes": []})
            m["month"] += r["cost_usd"]
            m["today"] += r["cost_usd"] if at >= day_start else 0.0
            m["calls"] += 1
            m["purposes"] = sorted({*m["purposes"], r["purpose"].split(".")[0]})
        return list(out.values())

    async def record(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


@dataclass
class Reservation:
    """An in-flight call's upper-bound estimate, counted as spend until the call is booked (or fails)."""

    scope: str | None  # "monthly" | "ask_daily" | None (news: its own caps)
    usd: float
    released: bool = False
    model: str | None = None  # held against the model's and the provider's limits too (app/llm_budgets.py)


PROVIDER = "anthropic"  # every call this service makes is a Claude call (seo_llm_usage.provider)



class BudgetGuard:
    """Hard caps checked before every call.

    Concurrency: `reserve()` checks the cap and holds the call's upper-bound estimate as pending spend (per scope,
    under an asyncio.Lock) until the call has been booked, so N concurrent calls cannot each read the same spend and
    all pass. The reservation is in-process only: replicas do not see each other's pending calls (the service runs
    one replica; a cross-replica reservation would need a row lock or an advisory lock on the shared table).
    """

    def __init__(
        self,
        store: SpendStore,
        monthly_usd: float,
        ask_daily_usd: float,
        clock: Callable[[], datetime] | None = None,
        limits: LimitsSource | None = None,
    ):
        self.store = store
        self.monthly_usd = monthly_usd
        self.ask_daily_usd = ask_daily_usd
        self.clock = clock or (lambda: datetime.now(UTC))
        self.limits = limits
        self._lock = asyncio.Lock()
        self.pending: dict[str, float] = {"monthly": 0.0, "ask_daily": 0.0}
        self.pending_models: dict[str, float] = {}  # in-flight estimates per model, for the LLM budgets

    @staticmethod
    def _scope(purpose: str) -> str | None:
        if purpose.startswith("news"):
            return None  # the news desk has its own per-job and global caps (app/news/budget.py), checked before every call
        return "ask_daily" if purpose.startswith("ask") else "monthly"

    async def _llm_limits(self, model: str) -> tuple[Limit | None, Limit | None]:
        if self.limits is None:
            return None, None
        lims = await self.limits.limits()
        return lims.model(model), lims.provider(PROVIDER)

    async def _check_llm_limits(self, model: str, est: float, now: datetime, own: Limit | None, total: Limit | None) -> None:
        """The model's and the provider's own limits (app/llm_budgets.py), across every purpose:
        spend this UTC day or month, plus the calls in flight here, plus this call's upper bound. Under the lock."""
        checks = ((own, f"The {{unit}} budget of {model}", {"model": model}, self.pending_models.get(model, 0.0)),
                  (total, f"The {{unit}} total for all {PROVIDER} models", {"provider": PROVIDER}, sum(self.pending_models.values())))
        for lim, label, by, pending in checks:
            if lim is None:
                continue
            for unit, max_usd in (("day", lim.daily_usd), ("month", lim.monthly_usd)):
                if max_usd is None:
                    continue
                used = await self.store.llm_spend(period_start(unit, now), **by) + pending
                if used + est > max_usd or (used > 0 and used >= max_usd):
                    word = "daily" if unit == "day" else "monthly"
                    scope = f"llm_{'model' if 'model' in by else 'provider'}_{word}"
                    metrics.BUDGET_BLOCKED.labels(scope=scope).inc()
                    more = f", and this call could cost up to ${est:.2f}" if used < max_usd else ""
                    reset = period_end(unit, now).strftime("%Y-%m-%d %H:%M UTC")
                    raise BudgetExceeded(scope, f"{label.format(unit=word)} is reached: ${used:.2f} of ${max_usd:.2f} used {'today' if unit == 'day' else 'this month'}{more}."
                                                f" It resets {reset}; change it in Admin → Automation & AI → LLM budgets.")

    async def _admit(self, model: str, input_chars: int, max_tokens: int, purpose: str, hold: bool) -> Reservation:
        est = estimate_cost_usd(model, input_chars, max_tokens)
        scope = self._scope(purpose)
        own, total = await self._llm_limits(model)
        if scope is None and own is None and total is None:
            return Reservation(None, 0.0, released=True)
        async with self._lock:
            now = self.clock()
            if own is not None or total is not None:
                await self._check_llm_limits(model, est, now, own, total)
            if scope == "monthly":
                month = await self.store.month_spend(now)
                if month + self.pending["monthly"] + est > self.monthly_usd:
                    metrics.BUDGET_BLOCKED.labels(scope="monthly").inc()
                    raise BudgetExceeded("monthly", f"monthly LLM budget ${self.monthly_usd:.2f} would be exceeded (spent ${month:.2f})")
            elif scope == "ask_daily":
                day = await self.store.day_spend(now, "ask")
                if day + self.pending["ask_daily"] + est > self.ask_daily_usd:
                    metrics.BUDGET_BLOCKED.labels(scope="ask_daily").inc()
                    raise BudgetExceeded("ask_daily", f"daily Ask-the-Blog budget ${self.ask_daily_usd:.2f} reached")
            if not hold:
                return Reservation(scope, 0.0, released=True)
            if scope is not None:
                self.pending[scope] += est
            self.pending_models[model] = self.pending_models.get(model, 0.0) + est
            return Reservation(scope, est, model=model)

    async def ensure(self, model: str, input_chars: int, max_tokens: int, purpose: str) -> None:
        """Check only (no reservation): would this call fit under its cap, counting the calls in flight?"""
        await self._admit(model, input_chars, max_tokens, purpose, hold=False)

    def release(self, r: Reservation) -> None:
        if r.released:
            return
        r.released = True
        if r.scope is not None:
            self.pending[r.scope] = max(0.0, self.pending[r.scope] - r.usd)
        if r.model is not None:
            left = self.pending_models.get(r.model, 0.0) - r.usd
            if left > 1e-12:
                self.pending_models[r.model] = left
            else:
                self.pending_models.pop(r.model, None)

    @asynccontextmanager
    async def reserve(self, model: str, input_chars: int, max_tokens: int, purpose: str) -> AsyncIterator[Reservation]:
        """`async with guard.reserve(...)`: BudgetExceeded before the call, the estimate held while it runs, released
        on exit (normal, exception or a closed generator). Book the real usage with record() inside the block: for a
        moment the call counts twice, which only errs on the safe side."""
        r = await self._admit(model, input_chars, max_tokens, purpose, hold=True)
        try:
            yield r
        finally:
            self.release(r)

    async def status(self) -> dict[str, float]:
        now = self.clock()
        month = await self.store.month_spend(now)
        return {"monthly_budget_usd": self.monthly_usd, "month_spend_usd": month, "ask_day_spend_usd": await self.store.day_spend(now, "ask")}

    async def record(
        self,
        usage: Usage,
        purpose: str,
        prompt_version: str,
        outcome: str = "ok",
        error_code: str | None = None,
        draft_id: str | None = None,
        *,
        run_id: str | None = None,
        agent: str | None = None,
        reservation: Reservation | None = None,
    ) -> float:
        cost = usage.cost_usd()
        try:
            await self.store.record(
                {
                    "model": usage.model,
                    "purpose": purpose,
                    "prompt_version": prompt_version,
                    "request_id": usage.request_id,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cache_read_tokens": usage.cache_read_tokens,
                    "cache_write_tokens": usage.cache_write_tokens,
                    "cost_usd": cost,
                    "latency_ms": usage.latency_ms,
                    "outcome": outcome,
                    "error_code": error_code,
                    "draft_id": draft_id,
                    "run_id": run_id,
                    "agent": agent,
                }
            )
        finally:
            if reservation is not None:
                self.release(reservation)
        metrics.LLM_TOKENS.labels(model=usage.model, purpose=metrics.purpose_label(purpose)).inc(usage.total_tokens)
        metrics.LLM_COST.labels(purpose=metrics.purpose_label(purpose)).inc(cost)
        metrics.LLM_CALLS.labels(purpose=metrics.purpose_label(purpose), outcome=metrics.llm_outcome_label(outcome)).inc()
        return cost


# --- Model clients --------------------------------------------------------------------------------------------------


@dataclass
class CallSpec:
    model: str
    purpose: str
    prompt_version: str
    cached_system: list[str]
    user: str
    max_tokens: int = 4096
    draft_id: str | None = None
    effort: str | None = None  # output_config.effort; None = the model's default
    run_id: str | None = None  # the flow run this call belongs to (seo_flow_runs.run_id), for usage attribution
    agent: str | None = None  # the crew agent key (e.g. "writer") that made the call
    # News desk: images for a vision check (media type, bytes), sent before the text. Alt text is written by a model
    # looking at the final image, never from the generation prompt (docs/news/IMAGES.md).
    images: list[tuple[str, bytes]] = field(default_factory=list)

    @property
    def input_chars(self) -> int:
        # An image costs roughly (w × h) / 750 tokens; budget estimates count a 1200×630 image as ~1,000 tokens.
        return len(self.user) + sum(len(s) for s in self.cached_system) + 3500 * len(self.images)


UsageCallback = Callable[[Usage], None]


class Llm(Protocol):
    async def structured(self, spec: CallSpec, schema: type[T]) -> tuple[T, Usage]: ...
    async def text(self, spec: CallSpec) -> tuple[str, Usage]: ...
    def stream(self, spec: CallSpec) -> AsyncIterator[str]: ...


class UsageReportingLlm(Llm, Protocol):
    """AnthropicLlm and FakeLlm: `stream(spec, on_usage=cb)` calls `cb` once with the booked usage when the stream
    ends (also on a refusal, a truncation or a client disconnect), so a caller can add the streamed step to its own
    cost (Ask the Blog). Wrappers that only implement Llm (NewsLlm) do not take it."""

    def stream(self, spec: CallSpec, *, on_usage: UsageCallback | None = None) -> AsyncIterator[str]: ...


def _system_blocks(cached: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": t} for t in cached if t]
    if blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}  # one breakpoint after the stable prefix
    return blocks


def _answered_model(msg: Any, requested: str) -> str:
    """The model that actually answered: a server-side fallback may route the request to another model, and that is
    the one billed."""
    m = getattr(msg, "model", None)
    return m if isinstance(m, str) and m else requested


def _usage_from(msg: Any, model: str, started: float) -> Usage:
    """Anthropic usage: input_tokens is already the uncached part; cache reads/writes are reported separately."""
    u = getattr(msg, "usage", None)
    return Usage(
        model=_answered_model(msg, model),
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        request_id=getattr(msg, "_request_id", None) or getattr(msg, "id", None),
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def estimated_usage(spec: CallSpec, started: float, output_chars: int | None = None, snapshot: Any = None) -> Usage:
    """Upper-bound usage when the API's own numbers are not available (a stream closed early, a reply the SDK could
    not parse). Input: the snapshot's input tokens if the stream got that far, else the guard's chars / 3.5. Output:
    the larger of the snapshot's count and streamed chars / 3.5 (more than the real count for English text), or the
    full max_tokens when nothing is known. Never more than max_tokens."""
    u = getattr(snapshot, "usage", None)
    snap_in = int(getattr(u, "input_tokens", 0) or 0)
    snap_out = int(getattr(u, "output_tokens", 0) or 0)
    est_in = math.ceil(spec.input_chars * 2 / 7)
    out = spec.max_tokens if output_chars is None else min(spec.max_tokens, max(snap_out, math.ceil(output_chars * 2 / 7)))
    sid = getattr(snapshot, "id", None)
    return Usage(
        model=_answered_model(snapshot, spec.model),
        input_tokens=snap_in or est_in,
        output_tokens=out,
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        request_id=sid if isinstance(sid, str) else None,
        latency_ms=int((time.monotonic() - started) * 1000),
    )


# Server-side refusal fallback (routes a declined request to a model the API picks by refusal category). Only the
# models that accept the "default" form get it; Haiku has no server-side fallback.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1")


def _text_of(msg: Any) -> str:
    return "".join(getattr(b, "text", "") for b in (getattr(msg, "content", None) or []) if getattr(b, "type", "") == "text")


class Refused(Exception):
    """stop_reason == "refusal" (after any server-side fallback)."""


def _attribution(spec: CallSpec) -> dict[str, str | None]:
    # run_id / agent are optional CallSpec fields (flows and crews set them); older specs simply lack them.
    return {"run_id": getattr(spec, "run_id", None), "agent": getattr(spec, "agent", None)}


class AnthropicLlm:
    """The official Anthropic SDK (AsyncAnthropic) behind the budget guard. No LangChain (D-54).

    - Structured output: `messages.parse(output_format=<Pydantic model>)` (output_config.format under the hood), then
      the caller validates again server-side (the web app's validators).
    - Stable system text is sent as cached system blocks; usage rows record cache reads/writes.
    - The SDK retries connection errors, 408/409/429/5xx twice; flows add nothing on top except their one
      validation retry.
    - Every call is booked exactly once (seo_llm_usage row + Prometheus + one `llm.call` span), whatever its ending:
      ok | invalid | truncated (stop_reason max_tokens → Truncated) | refused | cancelled (stream closed early).
      Usage is priced on the model that answered (`msg.model`), not the one requested.
    """

    def __init__(self, api_key: str, guard: BudgetGuard, timeout_s: float = 120, client: Any = None):
        self.guard = guard
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key or None, timeout=timeout_s, max_retries=2)
        self.client = client

    def _kw(self, spec: CallSpec) -> dict[str, Any]:
        content: Any = spec.user
        if spec.images:
            import base64

            content = [*({"type": "image", "source": {"type": "base64", "media_type": mt, "data": base64.b64encode(b).decode()}} for mt, b in spec.images), {"type": "text", "text": spec.user}]
        kw: dict[str, Any] = {"model": spec.model, "max_tokens": spec.max_tokens, "system": _system_blocks(spec.cached_system), "messages": [{"role": "user", "content": content}]}
        if spec.model in FALLBACK_MODELS:
            kw.update(betas=[FALLBACK_BETA], fallbacks="default")
        if spec.effort:
            kw["output_config"] = {"effort": spec.effort}
        return kw

    def _api(self, spec: CallSpec) -> Any:
        return self.client.beta.messages if spec.model in FALLBACK_MODELS else self.client.messages

    async def _book(self, spec: CallSpec, usage: Usage, r: Reservation, started_ns: int, outcome: str = "ok", error_code: str | None = None) -> float:
        attrs = _attribution(spec)
        cost = await self.guard.record(usage, spec.purpose, spec.prompt_version, outcome, error_code, spec.draft_id, reservation=r, run_id=attrs["run_id"], agent=attrs["agent"])
        tracing.emit_span(
            "llm.call",
            start_ns=started_ns,
            attrs={
                "llm.model": usage.model,
                "llm.requested_model": spec.model,
                "llm.purpose": spec.purpose,
                "llm.prompt_version": spec.prompt_version,
                "llm.input_tokens": usage.input_tokens,
                "llm.output_tokens": usage.output_tokens,
                "llm.cache_read_tokens": usage.cache_read_tokens,
                "llm.cache_write_tokens": usage.cache_write_tokens,
                "llm.cost_usd": round(cost, 6),
                "llm.outcome": outcome,
                **({"llm.error_code": error_code} if error_code else {}),
                **{f"llm.{k}": v for k, v in attrs.items() if v},
            },
            error=None if outcome == "ok" else outcome,
        )
        return cost

    async def _check(self, msg: Any, spec: CallSpec, usage: Usage, r: Reservation, started_ns: int) -> None:
        stop = getattr(msg, "stop_reason", None)
        if stop == "refusal":
            await self._book(spec, usage, r, started_ns, "refused", "refusal")
            raise Refused("the model declined this request")
        if stop == "max_tokens":
            await self._book(spec, usage, r, started_ns, "truncated", "max_tokens")
            raise Truncated(f"output cut off at max_tokens={spec.max_tokens}", usage)

    async def structured(self, spec: CallSpec, schema: type[T]) -> tuple[T, Usage]:
        async with self.guard.reserve(spec.model, spec.input_chars, spec.max_tokens, spec.purpose) as r:
            started, started_ns = time.monotonic(), time.time_ns()
            try:
                msg = await self._api(spec).parse(**self._kw(spec), output_format=schema)
            except Exception as e:
                if type(e).__name__ == "ValidationError":  # the SDK could not parse the model's JSON into the schema
                    # The SDK raises before handing back the message, so its usage is not obtainable here: book the
                    # upper bound (estimated input, full max_tokens out) rather than nothing. JSON that ends early
                    # ("EOF while parsing") is a max_tokens stop, where the full max_tokens is also the true output.
                    usage = estimated_usage(spec, started)
                    cut = "EOF while parsing" in str(e)
                    await self._book(spec, usage, r, started_ns, "invalid", "max_tokens" if cut else "schema_estimated")
                    raise (Truncated if cut else InvalidOutput)(str(e)[:500], usage) from e
                raise
            usage = _usage_from(msg, spec.model, started)
            stop = getattr(msg, "stop_reason", None)
            if stop == "refusal":
                await self._check(msg, spec, usage, r, started_ns)
            parsed = getattr(msg, "parsed_output", None)
            if parsed is None:
                # A max_tokens stop leaves no parsed output: outcome "invalid" like a schema miss, error_code tells.
                await self._book(spec, usage, r, started_ns, "invalid", "max_tokens" if stop == "max_tokens" else "schema")
                raise (Truncated if stop == "max_tokens" else InvalidOutput)(f"no structured output (stop_reason={stop})", usage)
            await self._book(spec, usage, r, started_ns)
            return parsed, usage

    async def text(self, spec: CallSpec) -> tuple[str, Usage]:
        async with self.guard.reserve(spec.model, spec.input_chars, spec.max_tokens, spec.purpose) as r:
            started, started_ns = time.monotonic(), time.time_ns()
            msg = await self._api(spec).create(**self._kw(spec))
            usage = _usage_from(msg, spec.model, started)
            await self._check(msg, spec, usage, r, started_ns)
            await self._book(spec, usage, r, started_ns)
            return _text_of(msg), usage

    async def stream(self, spec: CallSpec, *, on_usage: UsageCallback | None = None) -> AsyncIterator[str]:
        """Yields text deltas. Booked exactly once in every ending: the final message's usage when the stream
        completes; when the consumer stops early (client disconnect → GeneratorExit / cancellation) or the transport
        fails mid-stream, an upper-bound estimate from the snapshot and the characters streamed so far."""
        async with self.guard.reserve(spec.model, spec.input_chars, spec.max_tokens, spec.purpose) as r:
            started, started_ns = time.monotonic(), time.time_ns()
            streamed_chars = 0
            snapshot: Any = None
            booked = False
            try:
                async with self._api(spec).stream(**self._kw(spec)) as s:
                    async for text in s.text_stream:
                        snapshot = getattr(s, "current_message_snapshot", None)
                        if text:
                            streamed_chars += len(text)
                            yield text
                    msg = await s.get_final_message()
                usage = _usage_from(msg, spec.model, started)
                booked = True
                if on_usage:
                    on_usage(usage)
                await self._check(msg, spec, usage, r, started_ns)
                await self._book(spec, usage, r, started_ns)
            finally:
                if not booked:
                    # Closed early (GeneratorExit / CancelledError) or failed after the request went out: the tokens
                    # generated so far are billed. Shielded so a second cancellation cannot skip the booking.
                    est = estimated_usage(spec, started, streamed_chars, snapshot)
                    if on_usage:
                        on_usage(est)
                    await asyncio.shield(self._book(spec, est, r, started_ns, "cancelled", "stream_closed"))


class FakeLlm:
    """Deterministic model for tests: answers are queued per purpose (BaseModel instances, dicts, str or exceptions).

    It still goes through the BudgetGuard (reserve + record), so budget tests exercise the real accounting.
    """

    def __init__(self, guard: BudgetGuard, answers: dict[str, list[Any]] | None = None, tokens: tuple[int, int] = (1200, 300)):
        self.guard = guard
        self.answers: dict[str, deque[Any]] = {k: deque(v) for k, v in (answers or {}).items()}
        self.tokens = tokens
        self.calls: list[CallSpec] = []

    def queue(self, purpose: str, *items: Any) -> None:
        self.answers.setdefault(purpose, deque()).extend(items)

    def _next(self, spec: CallSpec) -> Any:
        q = self.answers.get(spec.purpose)
        if not q:
            raise AssertionError(f"FakeLlm: no answer queued for purpose {spec.purpose!r}")
        item = q.popleft()
        if isinstance(item, Exception):
            raise item
        return item(spec) if callable(item) and not isinstance(item, type) else item

    async def structured(self, spec: CallSpec, schema: type[T]) -> tuple[T, Usage]:
        async with self.guard.reserve(spec.model, spec.input_chars, spec.max_tokens, spec.purpose) as r:
            self.calls.append(spec)
            usage = Usage(spec.model, input_tokens=self.tokens[0], output_tokens=self.tokens[1], request_id=f"fake_{len(self.calls)}")
            item = self._next(spec)
            try:
                value = item if isinstance(item, schema) else schema.model_validate(item)
            except Exception as e:
                await self.guard.record(usage, spec.purpose, spec.prompt_version, "invalid", "schema", spec.draft_id, reservation=r, run_id=getattr(spec, "run_id", None), agent=getattr(spec, "agent", None))
                raise InvalidOutput(str(e), usage) from e
            await self.guard.record(usage, spec.purpose, spec.prompt_version, draft_id=spec.draft_id, reservation=r, run_id=getattr(spec, "run_id", None), agent=getattr(spec, "agent", None))
            return value, usage

    async def text(self, spec: CallSpec) -> tuple[str, Usage]:
        async with self.guard.reserve(spec.model, spec.input_chars, spec.max_tokens, spec.purpose) as r:
            self.calls.append(spec)
            usage = Usage(spec.model, input_tokens=self.tokens[0], output_tokens=self.tokens[1])
            out = str(self._next(spec))
            await self.guard.record(usage, spec.purpose, spec.prompt_version, reservation=r, run_id=getattr(spec, "run_id", None), agent=getattr(spec, "agent", None))
            return out, usage

    async def stream(self, spec: CallSpec, *, on_usage: UsageCallback | None = None) -> AsyncIterator[str]:
        text, usage = await self.text(spec)
        if on_usage:
            on_usage(usage)
        for word in text.split(" "):
            yield word + " "
