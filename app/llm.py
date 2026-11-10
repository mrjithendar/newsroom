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

import json
import os
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from . import metrics

T = TypeVar("T", bound=BaseModel)

# USD per million tokens; identical to services/www/src/lib/seo-intel/claude.ts (first-party list prices, 2026-10).
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
    return PRICES.get(model, FALLBACK)


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


# --- Spend accounting (shared table seo_llm_usage) ------------------------------------------------------------------


class SpendStore(Protocol):
    async def month_spend(self, now: datetime) -> float: ...
    async def day_spend(self, now: datetime, purpose_prefix: str) -> float: ...
    async def record(self, row: dict[str, Any]) -> None: ...


class PgSpendStore:
    def __init__(self, pool: Any):
        self.pool = pool

    async def month_spend(self, now: datetime) -> float:
        start = datetime(now.year, now.month, 1, tzinfo=UTC)
        async with self.pool.connection() as c:
            cur = await c.execute(
                "select coalesce(sum(cost_usd), 0)::float8 as s from seo_llm_usage where at >= %s and purpose not like 'ask%%'", (start,)
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

    async def record(self, row: dict[str, Any]) -> None:
        draft_id = row.get("draft_id")
        async with self.pool.connection() as c:
            await c.execute(
                """insert into seo_llm_usage (provider, model, purpose, prompt_version, mode, request_id, input_tokens,
                     output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, latency_ms, outcome, error_code, draft_id)
                   values ('anthropic', %(model)s, %(purpose)s, %(prompt_version)s, 'sync', %(request_id)s, %(input_tokens)s,
                     %(output_tokens)s, %(cache_read_tokens)s, %(cache_write_tokens)s, %(cost_usd)s, %(latency_ms)s,
                     %(outcome)s, %(error_code)s,
                     (select id from seo_drafts where id::text = %(draft_id)s))""",
                {**row, "draft_id": draft_id},
            )


@dataclass
class MemorySpendStore:
    rows: list[dict[str, Any]] = field(default_factory=list)
    preexisting_usd: float = 0.0

    async def month_spend(self, now: datetime) -> float:
        return self.preexisting_usd + sum(r["cost_usd"] for r in self.rows if not r["purpose"].startswith("ask"))

    async def day_spend(self, now: datetime, purpose_prefix: str) -> float:
        return sum(r["cost_usd"] for r in self.rows if r["purpose"].startswith(purpose_prefix))

    async def record(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


class BudgetGuard:
    def __init__(self, store: SpendStore, monthly_usd: float, ask_daily_usd: float, clock: Callable[[], datetime] | None = None):
        self.store = store
        self.monthly_usd = monthly_usd
        self.ask_daily_usd = ask_daily_usd
        self.clock = clock or (lambda: datetime.now(UTC))

    async def ensure(self, model: str, input_chars: int, max_tokens: int, purpose: str) -> None:
        now = self.clock()
        est = estimate_cost_usd(model, input_chars, max_tokens)
        if not purpose.startswith("ask"):
            month = await self.store.month_spend(now)
            if month + est > self.monthly_usd:
                metrics.BUDGET_BLOCKED.labels(scope="monthly").inc()
                raise BudgetExceeded("monthly", f"monthly LLM budget ${self.monthly_usd:.2f} would be exceeded (spent ${month:.2f})")
        else:
            day = await self.store.day_spend(now, "ask")
            if day + est > self.ask_daily_usd:
                metrics.BUDGET_BLOCKED.labels(scope="ask_daily").inc()
                raise BudgetExceeded("ask_daily", f"daily Ask-the-Blog budget ${self.ask_daily_usd:.2f} reached")

    async def status(self) -> dict[str, float]:
        now = self.clock()
        month = await self.store.month_spend(now)
        return {"monthly_budget_usd": self.monthly_usd, "month_spend_usd": month, "ask_day_spend_usd": await self.store.day_spend(now, "ask")}

    async def record(self, usage: Usage, purpose: str, prompt_version: str, outcome: str = "ok", error_code: str | None = None, draft_id: str | None = None) -> float:
        cost = usage.cost_usd()
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
            }
        )
        metrics.LLM_TOKENS.labels(model=usage.model, purpose=metrics.purpose_label(purpose)).inc(usage.total_tokens)
        metrics.LLM_COST.labels(purpose=metrics.purpose_label(purpose)).inc(cost)
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

    @property
    def input_chars(self) -> int:
        return len(self.user) + sum(len(s) for s in self.cached_system)


class Llm(Protocol):
    async def structured(self, spec: CallSpec, schema: type[T]) -> tuple[T, Usage]: ...
    async def text(self, spec: CallSpec) -> tuple[str, Usage]: ...
    def stream(self, spec: CallSpec) -> AsyncIterator[str]: ...


def _system_blocks(cached: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": t} for t in cached if t]
    if blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}  # one breakpoint after the stable prefix
    return blocks


def _usage_from(msg: Any, model: str, started: float) -> Usage:
    """Anthropic usage: input_tokens is already the uncached part; cache reads/writes are reported separately."""
    u = getattr(msg, "usage", None)
    return Usage(
        model=model,
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
        request_id=getattr(msg, "_request_id", None) or getattr(msg, "id", None),
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


class AnthropicLlm:
    """The official Anthropic SDK (AsyncAnthropic) behind the budget guard. No LangChain (D-54).

    - Structured output: `messages.parse(output_format=<Pydantic model>)` (output_config.format under the hood), then
      the caller validates again server-side (the web app's validators).
    - Stable system text is sent as cached system blocks; usage rows record cache reads/writes.
    - The SDK retries connection errors, 408/409/429/5xx twice; flows add nothing on top except their one
      validation retry.
    """

    def __init__(self, api_key: str, guard: BudgetGuard, timeout_s: float = 120, client: Any = None):
        self.guard = guard
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=api_key or None, timeout=timeout_s, max_retries=2)
        self.client = client

    def _kw(self, spec: CallSpec) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": spec.model, "max_tokens": spec.max_tokens, "system": _system_blocks(spec.cached_system), "messages": [{"role": "user", "content": spec.user}]}
        if spec.model in FALLBACK_MODELS:
            kw.update(betas=[FALLBACK_BETA], fallbacks="default")
        if spec.effort:
            kw["output_config"] = {"effort": spec.effort}
        return kw

    def _api(self, spec: CallSpec) -> Any:
        return self.client.beta.messages if spec.model in FALLBACK_MODELS else self.client.messages

    async def _check(self, msg: Any, spec: CallSpec, usage: Usage) -> None:
        if getattr(msg, "stop_reason", None) == "refusal":
            await self.guard.record(usage, spec.purpose, spec.prompt_version, "refused", "refusal", spec.draft_id)
            raise Refused("the model declined this request")

    async def structured(self, spec: CallSpec, schema: type[T]) -> tuple[T, Usage]:
        await self.guard.ensure(spec.model, spec.input_chars, spec.max_tokens, spec.purpose)
        started = time.monotonic()
        try:
            msg = await self._api(spec).parse(**self._kw(spec), output_format=schema)
        except Exception as e:
            if type(e).__name__ == "ValidationError":  # the SDK could not parse the model's JSON into the schema
                raise InvalidOutput(str(e)[:500]) from e
            raise
        usage = _usage_from(msg, spec.model, started)
        await self._check(msg, spec, usage)
        parsed = getattr(msg, "parsed_output", None)
        if parsed is None:
            await self.guard.record(usage, spec.purpose, spec.prompt_version, "invalid", "schema", spec.draft_id)
            raise InvalidOutput(f"no structured output (stop_reason={getattr(msg, 'stop_reason', None)})", usage)
        await self.guard.record(usage, spec.purpose, spec.prompt_version, draft_id=spec.draft_id)
        return parsed, usage

    async def text(self, spec: CallSpec) -> tuple[str, Usage]:
        await self.guard.ensure(spec.model, spec.input_chars, spec.max_tokens, spec.purpose)
        started = time.monotonic()
        msg = await self._api(spec).create(**self._kw(spec))
        usage = _usage_from(msg, spec.model, started)
        await self._check(msg, spec, usage)
        await self.guard.record(usage, spec.purpose, spec.prompt_version, draft_id=spec.draft_id)
        return _text_of(msg), usage

    async def stream(self, spec: CallSpec) -> AsyncIterator[str]:
        await self.guard.ensure(spec.model, spec.input_chars, spec.max_tokens, spec.purpose)
        started = time.monotonic()
        async with self._api(spec).stream(**self._kw(spec)) as s:
            async for text in s.text_stream:
                if text:
                    yield text
            msg = await s.get_final_message()
        usage = _usage_from(msg, spec.model, started)
        await self._check(msg, spec, usage)
        await self.guard.record(usage, spec.purpose, spec.prompt_version)


class InvalidOutput(Exception):
    def __init__(self, detail: str, usage: Usage | None = None):
        super().__init__(detail)
        self.usage = usage


class FakeLlm:
    """Deterministic model for tests: answers are queued per purpose (BaseModel instances, dicts, str or exceptions).

    It still goes through the BudgetGuard (ensure + record), so budget tests exercise the real accounting.
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
        await self.guard.ensure(spec.model, spec.input_chars, spec.max_tokens, spec.purpose)
        self.calls.append(spec)
        usage = Usage(spec.model, input_tokens=self.tokens[0], output_tokens=self.tokens[1], request_id=f"fake_{len(self.calls)}")
        item = self._next(spec)
        try:
            value = item if isinstance(item, schema) else schema.model_validate(item)
        except Exception as e:
            await self.guard.record(usage, spec.purpose, spec.prompt_version, "invalid", "schema", spec.draft_id)
            raise InvalidOutput(str(e), usage) from e
        await self.guard.record(usage, spec.purpose, spec.prompt_version, draft_id=spec.draft_id)
        return value, usage

    async def text(self, spec: CallSpec) -> tuple[str, Usage]:
        await self.guard.ensure(spec.model, spec.input_chars, spec.max_tokens, spec.purpose)
        self.calls.append(spec)
        usage = Usage(spec.model, input_tokens=self.tokens[0], output_tokens=self.tokens[1])
        out = str(self._next(spec))
        await self.guard.record(usage, spec.purpose, spec.prompt_version)
        return out, usage

    async def stream(self, spec: CallSpec) -> AsyncIterator[str]:
        text, _ = await self.text(spec)
        for word in text.split(" "):
            yield word + " "
