"""LLM budgets, owned by the seo-agent: an optional daily and monthly limit for each model, and for each provider (all
of its models together, e.g. the credit on the Anthropic account). Every Claude call of the SEO platform is made
here, so this is where the limits live and are enforced.

- Stored in the agent's own table `seo_agent_llm_budgets` (created at start-up, owned by the agent's role; no web app
  migration), one row per model or provider. A missing row or a NULL column is no limit; nothing is preset.
- Set through the internal API: `GET /llm-budgets` (limits, spend today and this month, forecast) and
  `PUT /llm-budgets` (the whole set, validated here). The web app's Admin → Automation & AI → LLM budgets card is a thin
  client of these two calls and keeps nothing itself.
- Checked by BudgetGuard (app/llm.py) before every call, for every purpose: drafts, flows, Ask the Blog and the news
  desk. Spend is every row of seo_llm_usage (the web app's own few Claude calls included) plus legacy batches in
  flight, by UTC day and month, the periods Anthropic's spend limits reset on. The monthly admin cap and the feature
  caps still apply on top.

Reading: cached for a short time. A database error before the first successful read means no limits from here (the
other caps still apply and the provider's spend limit is the hard stop); after one, an outage keeps the last known
limits, so it never lifts a limit that was set. A save takes effect at once in this process.

The table is created by `ensure_schema()`. When it is first created, limits saved by the web app's earlier LLM budgets
card (the `setting` row "llm_budgets", moks-cms #157, since moved here) are copied in once.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger("seo-agent.llm-budgets")

TABLE = "seo_agent_llm_budgets"
LEGACY_SETTING = "llm_budgets"  # the web app's earlier copy, imported once
MAX_USD = 100_000.0
MAX_MODELS, MAX_PROVIDERS = 100, 20
MODEL_ID = re.compile(r"^[A-Za-z0-9._:\-/@]{1,120}$")
PROVIDER_ID = re.compile(r"^[a-z0-9_-]{2,40}$")

DDL = f"""create table if not exists {TABLE} (
    kind text not null check (kind in ('model', 'provider')),
    name text not null,
    daily_usd double precision check (daily_usd >= 0),
    monthly_usd double precision check (monthly_usd >= 0),
    updated_by text,
    updated_at timestamptz not null default now(),
    primary key (kind, name))"""


@dataclass(frozen=True)
class Limit:
    daily_usd: float | None = None
    monthly_usd: float | None = None

    @property
    def empty(self) -> bool:
        return self.daily_usd is None and self.monthly_usd is None

    def out(self) -> dict[str, float | None]:
        return {"daily_usd": self.daily_usd, "monthly_usd": self.monthly_usd}


@dataclass(frozen=True)
class LlmLimits:
    models: dict[str, Limit] = field(default_factory=dict)
    providers: dict[str, Limit] = field(default_factory=dict)

    def model(self, name: str) -> Limit | None:
        return self.models.get(name)

    def provider(self, name: str) -> Limit | None:
        return self.providers.get(name)

    def out(self) -> dict[str, dict[str, dict[str, float | None]]]:
        return {"models": {k: v.out() for k, v in sorted(self.models.items())}, "providers": {k: v.out() for k, v in sorted(self.providers.items())}}


NO_LIMITS = LlmLimits()


# --- Validation (PUT /llm-budgets) -----------------------------------------------------------------------------------


class LimitIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    daily_usd: float | None = Field(default=None, ge=0, le=MAX_USD, allow_inf_nan=False)
    monthly_usd: float | None = Field(default=None, ge=0, le=MAX_USD, allow_inf_nan=False)


class BudgetsIn(BaseModel):
    """The whole set; what is not in it has no limit afterwards."""

    model_config = ConfigDict(extra="forbid")
    models: dict[str, LimitIn] = Field(default_factory=dict)
    providers: dict[str, LimitIn] = Field(default_factory=dict)
    actor: str | None = Field(default=None, max_length=200)

    @field_validator("models")
    @classmethod
    def _models(cls, v: dict[str, LimitIn]) -> dict[str, LimitIn]:
        bad = [k for k in v if not MODEL_ID.fullmatch(k)]
        if bad:
            raise ValueError(f"not a model id: {', '.join(bad[:5])}")
        if len(v) > MAX_MODELS:
            raise ValueError(f"at most {MAX_MODELS} models")
        return v

    @field_validator("providers")
    @classmethod
    def _providers(cls, v: dict[str, LimitIn]) -> dict[str, LimitIn]:
        bad = [k for k in v if not PROVIDER_ID.fullmatch(k)]
        if bad:
            raise ValueError(f"not a provider name: {', '.join(bad[:5])}")
        if len(v) > MAX_PROVIDERS:
            raise ValueError(f"at most {MAX_PROVIDERS} providers")
        return v

    def limits(self) -> LlmLimits:
        def keep(d: dict[str, LimitIn]) -> dict[str, Limit]:
            return {k: lim for k, v in d.items() if not (lim := Limit(v.daily_usd, v.monthly_usd)).empty}

        return LlmLimits(keep(self.models), keep(self.providers))


def _usd(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > MAX_USD:
        return None
    return float(v)


def parse_legacy(value: Any) -> LlmLimits:
    """The web app's earlier document ({models|providers: {id: {dailyUsd, monthlyUsd}}}), tolerantly."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return NO_LIMITS
    if not isinstance(value, dict):
        return NO_LIMITS

    def entries(d: Any, key: re.Pattern[str]) -> dict[str, Limit]:
        out: dict[str, Limit] = {}
        for name, e in (d or {}).items() if isinstance(d, dict) else ():
            if isinstance(name, str) and key.fullmatch(name) and isinstance(e, dict):
                lim = Limit(_usd(e.get("dailyUsd")), _usd(e.get("monthlyUsd")))
                if not lim.empty:
                    out[name] = lim
        return out

    return LlmLimits(entries(value.get("models"), MODEL_ID), entries(value.get("providers"), PROVIDER_ID))


# --- Stores ----------------------------------------------------------------------------------------------------------


class LimitsSource(Protocol):
    async def limits(self) -> LlmLimits: ...


class BudgetStore(LimitsSource, Protocol):
    async def ensure_schema(self) -> None: ...
    async def save(self, limits: LlmLimits, actor: str | None) -> LlmLimits: ...


@dataclass
class MemoryBudgets:
    """Tests and the in-memory mode without a database."""

    value: LlmLimits = NO_LIMITS

    async def ensure_schema(self) -> None:
        return None

    async def limits(self) -> LlmLimits:
        return self.value

    async def save(self, limits: LlmLimits, actor: str | None) -> LlmLimits:
        self.value = limits
        return limits


def _rows_to_limits(rows: list[dict[str, Any]]) -> LlmLimits:
    models: dict[str, Limit] = {}
    providers: dict[str, Limit] = {}
    for r in rows:
        lim = Limit(_usd(r["daily_usd"]), _usd(r["monthly_usd"]))
        if lim.empty:
            continue
        (models if r["kind"] == "model" else providers)[r["name"]] = lim
    return LlmLimits(models, providers)


class PgBudgets:
    def __init__(self, pool: Any, ttl_s: float = 30.0):
        self.pool, self.ttl = pool, ttl_s
        self._at = 0.0
        self._value: LlmLimits | None = None

    async def ensure_schema(self) -> None:
        async with self.pool.connection() as c:
            cur = await c.execute("select to_regclass(%s) is not null as present", (TABLE,))
            existed = bool((await cur.fetchone())["present"])
            await c.execute(DDL)
            if existed:
                return
            cur = await c.execute("select value from setting where key = %s", (LEGACY_SETTING,))
            row = await cur.fetchone()
            legacy = parse_legacy(row["value"]) if row else NO_LIMITS
        if legacy.models or legacy.providers:
            await self.save(legacy, "imported from the web app's LLM budgets")
            log.info("imported LLM budgets from the web app", extra={"event": "llm_budgets_imported"})

    async def limits(self) -> LlmLimits:
        now = time.monotonic()
        if self._value is not None and now - self._at < self.ttl:
            return self._value
        try:
            async with self.pool.connection() as c:
                cur = await c.execute(f"select kind, name, daily_usd, monthly_usd from {TABLE}")  # noqa: S608 - constant name
                self._value = _rows_to_limits(list(await cur.fetchall()))
        except Exception:
            log.warning("could not read the LLM budgets; keeping the last known ones", exc_info=True)
            if self._value is None:
                return NO_LIMITS  # never read successfully: no limits from here, and try again on the next call
        self._at = now
        return self._value

    async def save(self, limits: LlmLimits, actor: str | None) -> LlmLimits:
        """Replaces the whole set in one transaction."""
        rows = [("model", k, v.daily_usd, v.monthly_usd) for k, v in limits.models.items()]
        rows += [("provider", k, v.daily_usd, v.monthly_usd) for k, v in limits.providers.items()]
        async with self.pool.connection() as c, c.transaction():
            await c.execute(f"delete from {TABLE}")  # noqa: S608 - constant name
            for kind, name, daily, monthly in rows:
                await c.execute(
                    f"insert into {TABLE} (kind, name, daily_usd, monthly_usd, updated_by) values (%s, %s, %s, %s, %s)",  # noqa: S608
                    (kind, name, daily, monthly, actor),
                )
        self._value, self._at = limits, time.monotonic()
        return limits


# --- Periods and the spend overview (GET /llm-budgets) -----------------------------------------------------------------


def period_start(unit: str, now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day, tzinfo=UTC) if unit == "day" else datetime(now.year, now.month, 1, tzinfo=UTC)


def period_end(unit: str, now: datetime) -> datetime:
    if unit == "day":
        return period_start("day", now) + timedelta(days=1)
    return datetime(now.year + (now.month == 12), now.month % 12 + 1, 1, tzinfo=UTC)


class SpendBreakdown(Protocol):
    async def llm_spend_by_model(self, month_start: datetime, day_start: datetime) -> list[dict[str, Any]]: ...


async def overview(limits: LlmLimits, store: SpendBreakdown, now: datetime) -> dict[str, Any]:
    """The limits with this month's spend by model and provider (today's alongside) and a month-end forecast at the
    month's average daily rate so far (at least one day, so an early burst does not read as a month of it)."""
    month, day = period_start("month", now), period_start("day", now)
    rows = await store.llm_spend_by_model(month, day)
    models = [
        {"model": r["model"], "provider": r["provider"], "today_usd": round(float(r["today"]), 6), "month_usd": round(float(r["month"]), 6),
         "calls": int(r["calls"]), "purposes": sorted({p for p in (r.get("purposes") or []) if p})}
        for r in rows
    ]
    elapsed = max((now - month).total_seconds() / 86400, 1.0)
    days = (period_end("month", now) - month).days
    providers: dict[str, dict[str, Any]] = {}
    for m in models:
        p = providers.setdefault(m["provider"], {"provider": m["provider"], "today_usd": 0.0, "month_usd": 0.0})
        p["today_usd"] += m["today_usd"]
        p["month_usd"] += m["month_usd"]
    for p in providers.values():
        p["forecast_usd"] = round(p["month_usd"] / elapsed * days, 6)
    return {
        "budgets": limits.out(),
        "month_start": month.date().isoformat(),
        "day_resets_at": period_end("day", now).isoformat(),
        "month_resets_at": period_end("month", now).isoformat(),
        "models": sorted(models, key=lambda m: (-m["month_usd"], m["model"])),
        "providers": sorted(providers.values(), key=lambda p: (-p["month_usd"], p["provider"])),
    }
