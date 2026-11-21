"""The LLM budget, owned by the seo-agent: ONE budget for everything, with an optional daily limit and an optional
monthly limit (USD). It counts every provider, every model and every purpose together: drafts, crews, the news desk,
Ask the Blog and the web app's own calls.

- Stored in the agent's own table `seo_agent_llm_budget` (one row, created at start-up, owned by the agent's role; no
  web app migration). An empty limit is no limit; nothing is preset.
- Set through the internal API: `GET /llm-budget?scope=day|month` (spent, limit, remaining, forecast, and spend by
  provider, model and purpose for that period; the breakdowns are for reading only, they limit nothing) and
  `PUT /llm-budget` `{daily_usd, monthly_usd, actor}`.
- Checked by BudgetGuard (app/llm.py) before every call of every purpose. Spend is every row of seo_llm_usage plus
  legacy batches in flight, by UTC day and month. Callers outside this process (the web app) hold their upper-bound
  estimate against the same budget with `POST /llm-budget/reserve` and let go of it with `POST /llm-budget/settle`.
- Still apart: Ask the Blog's own daily limit (a guard against readers, not the admin, running up spend) and, until
  the news desk moves over, the news desk's own caps (app/news/budget.py).

Before 2026-11-21 there was a limit per model and per provider instead (table `seo_agent_llm_budgets`). Those rows
are not enforced any more. They are left in place and shown read-only (`legacy_limits`), so the owner can see what
was set and decide the one budget; nothing here adds them up into a total.

Reading: cached for a short time. A database error before the first successful read means no limit from here (the
provider's own spend limit is then the hard stop); after one, an outage keeps the last known budget, so it never
lifts a limit that was set. A save takes effect at once in this process.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("seo-agent.llm-budget")

TABLE = "seo_agent_llm_budget"
LEGACY_TABLE = "seo_agent_llm_budgets"  # per-model and per-provider limits: read-only, no longer enforced
MAX_USD = 100_000.0
Unit = Literal["day", "month"]

DDL = f"""create table if not exists {TABLE} (
    id boolean primary key default true check (id),
    daily_usd double precision check (daily_usd >= 0),
    monthly_usd double precision check (monthly_usd >= 0),
    updated_by text,
    updated_at timestamptz not null default now())"""


@dataclass(frozen=True)
class Budget:
    daily_usd: float | None = None
    monthly_usd: float | None = None

    @property
    def empty(self) -> bool:
        return self.daily_usd is None and self.monthly_usd is None

    def limit(self, unit: Unit) -> float | None:
        return self.daily_usd if unit == "day" else self.monthly_usd

    def out(self) -> dict[str, float | None]:
        return {"daily_usd": self.daily_usd, "monthly_usd": self.monthly_usd}


NO_BUDGET = Budget()


class BudgetIn(BaseModel):
    """PUT /llm-budget. Both limits are sent; null is no limit."""

    model_config = ConfigDict(extra="forbid")
    daily_usd: float | None = Field(default=None, ge=0, le=MAX_USD, allow_inf_nan=False)
    monthly_usd: float | None = Field(default=None, ge=0, le=MAX_USD, allow_inf_nan=False)
    actor: str | None = Field(default=None, max_length=200)

    def budget(self) -> Budget:
        return Budget(self.daily_usd, self.monthly_usd)


def _usd(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > MAX_USD:
        return None
    return float(v)


# --- Stores ----------------------------------------------------------------------------------------------------------


class BudgetSource(Protocol):
    async def budget(self) -> Budget: ...


class BudgetStore(BudgetSource, Protocol):
    async def ensure_schema(self) -> None: ...
    async def save(self, budget: Budget, actor: str | None) -> Budget: ...
    async def legacy(self) -> dict[str, dict[str, dict[str, float | None]]]: ...


def _no_legacy() -> dict[str, dict[str, dict[str, float | None]]]:
    return {"models": {}, "providers": {}}


@dataclass
class MemoryBudget:
    """Tests and the in-memory mode without a database."""

    value: Budget = NO_BUDGET
    legacy_value: dict[str, dict[str, dict[str, float | None]]] = field(default_factory=_no_legacy)

    async def ensure_schema(self) -> None:
        return None

    async def budget(self) -> Budget:
        return self.value

    async def save(self, budget: Budget, actor: str | None) -> Budget:
        self.value = budget
        return budget

    async def legacy(self) -> dict[str, dict[str, dict[str, float | None]]]:
        return self.legacy_value


class PgBudget:
    def __init__(self, pool: Any, ttl_s: float = 30.0):
        self.pool, self.ttl = pool, ttl_s
        self._at = 0.0
        self._value: Budget | None = None

    async def ensure_schema(self) -> None:
        async with self.pool.connection() as c:
            await c.execute(DDL)

    async def budget(self) -> Budget:
        now = time.monotonic()
        if self._value is not None and now - self._at < self.ttl:
            return self._value
        try:
            async with self.pool.connection() as c:
                cur = await c.execute(f"select daily_usd, monthly_usd from {TABLE}")  # noqa: S608 - constant name
                row = await cur.fetchone()
                self._value = Budget(_usd(row["daily_usd"]), _usd(row["monthly_usd"])) if row else NO_BUDGET
        except Exception:
            log.warning("could not read the LLM budget; keeping the last known one", exc_info=True)
            if self._value is None:
                return NO_BUDGET  # never read successfully: no limit from here, and try again on the next call
        self._at = now
        return self._value

    async def save(self, budget: Budget, actor: str | None) -> Budget:
        async with self.pool.connection() as c:
            await c.execute(
                f"insert into {TABLE} (id, daily_usd, monthly_usd, updated_by) values (true, %s, %s, %s) "  # noqa: S608 - constant name
                "on conflict (id) do update set daily_usd = excluded.daily_usd, monthly_usd = excluded.monthly_usd,"
                " updated_by = excluded.updated_by, updated_at = now()",
                (budget.daily_usd, budget.monthly_usd, actor),
            )
        self._value, self._at = budget, time.monotonic()
        return budget

    async def legacy(self) -> dict[str, dict[str, dict[str, float | None]]]:
        """The per-model and per-provider limits from before the one budget, as they were saved. Read-only: nothing
        enforces them, and nothing here turns them into a total."""
        out = _no_legacy()
        try:
            async with self.pool.connection() as c:
                cur = await c.execute("select to_regclass(%s) is not null as present", (LEGACY_TABLE,))
                if not (await cur.fetchone())["present"]:
                    return out
                cur = await c.execute(f"select kind, name, daily_usd, monthly_usd from {LEGACY_TABLE} order by kind, name")  # noqa: S608 - constant name
                for r in await cur.fetchall():
                    daily, monthly = _usd(r["daily_usd"]), _usd(r["monthly_usd"])
                    if daily is None and monthly is None:
                        continue
                    out["models" if r["kind"] == "model" else "providers"][r["name"]] = {"daily_usd": daily, "monthly_usd": monthly}
        except Exception:
            log.warning("could not read the earlier per-model limits", exc_info=True)
        return out


# --- Periods and the overview (GET /llm-budget) ------------------------------------------------------------------------


def period_start(unit: str, now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day, tzinfo=UTC) if unit == "day" else datetime(now.year, now.month, 1, tzinfo=UTC)


def period_end(unit: str, now: datetime) -> datetime:
    if unit == "day":
        return period_start("day", now) + timedelta(days=1)
    return datetime(now.year + (now.month == 12), now.month % 12 + 1, 1, tzinfo=UTC)


class SpendBreakdown(Protocol):
    async def llm_spend(self, since: datetime, *, model: str | None = None, provider: str | None = None) -> float: ...
    async def llm_spend_breakdown(self, since: datetime) -> list[dict[str, Any]]: ...


def _grouped(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in rows:
        k = tuple(r[key] for key in keys)
        g = out.setdefault(k, {**{key: r[key] for key in keys}, "usd": 0.0, "calls": 0})
        g["usd"] += float(r["usd"])
        g["calls"] += int(r["calls"])
    for g in out.values():
        g["usd"] = round(g["usd"], 6)
    return sorted(out.values(), key=lambda g: (-g["usd"], *(str(g[key]) for key in keys)))


async def overview(budget: Budget, store: SpendBreakdown, now: datetime, scope: Unit = "month", pending_usd: float = 0.0) -> dict[str, Any]:
    """The budget and the chosen period: spent, limit, remaining, the calls in flight, a month-end forecast (month
    only, at the month's average daily rate so far: at least one day, so an early burst does not read as a month of
    it), and the period's spend by provider, by model and by purpose. One line for the other period too."""
    other: Unit = "day" if scope == "month" else "month"
    start = period_start(scope, now)
    rows = await store.llm_spend_breakdown(start)
    spent = round(sum(float(r["usd"]) for r in rows), 6)
    limit = budget.limit(scope)
    out: dict[str, Any] = {
        "budget": budget.out(),
        "scope": scope,
        "period_start": start.isoformat(),
        "resets_at": period_end(scope, now).isoformat(),
        "spent_usd": spent,
        "pending_usd": round(pending_usd, 6),
        "limit_usd": limit,
        "remaining_usd": None if limit is None else round(max(0.0, limit - spent - pending_usd), 6),
        "forecast_usd": None,
        "by_provider": _grouped(rows, ("provider",)),
        "by_model": _grouped(rows, ("provider", "model")),
        "by_purpose": _grouped(rows, ("purpose",)),
        "other": {
            "scope": other,
            "spent_usd": round(await store.llm_spend(period_start(other, now)), 6),
            "limit_usd": budget.limit(other),
            "resets_at": period_end(other, now).isoformat(),
        },
    }
    if scope == "month":
        elapsed = max((now - start).total_seconds() / 86400, 1.0)
        out["forecast_usd"] = round(spent / elapsed * (period_end("month", now) - start).days, 6)
    return out


class SpendByModel(Protocol):
    async def llm_spend_by_model(self, month_start: datetime, day_start: datetime) -> list[dict[str, Any]]: ...


REPLACED = "Limits per model and per provider were replaced by one LLM budget (PUT /llm-budget). Update the web app to set it."


async def legacy_overview(store: SpendByModel, now: datetime) -> dict[str, Any]:
    """GET /llm-budgets, for a web app released before the one budget: this month's spend by model and provider in
    the shape its card reads, with no limits, because none of that kind is enforced any more."""
    month, day = period_start("month", now), period_start("day", now)
    models = [
        {"model": r["model"], "provider": r["provider"], "today_usd": round(float(r["today"]), 6), "month_usd": round(float(r["month"]), 6),
         "calls": int(r["calls"]), "purposes": sorted({p for p in (r.get("purposes") or []) if p})}
        for r in await store.llm_spend_by_model(month, day)
    ]  # fmt: skip
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
        "budgets": {"models": {}, "providers": {}},
        "replaced_by": "/llm-budget",
        "month_start": month.date().isoformat(),
        "day_resets_at": period_end("day", now).isoformat(),
        "month_resets_at": period_end("month", now).isoformat(),
        "models": sorted(models, key=lambda m: (-m["month_usd"], m["model"])),
        "providers": sorted(providers.values(), key=lambda p: (-p["month_usd"], p["provider"])),
    }
