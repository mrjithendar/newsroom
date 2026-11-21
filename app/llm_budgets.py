"""The pieces of the one LLM budget that BudgetGuard (app/llm.py) needs: its shape and its periods.

Forked from moks-seo-agent/app/llm_budgets.py@bb9e553 on 2026-11-21. The budget itself is the seo-agent's: it keeps the
admin's daily and monthly limits (table seo_agent_llm_budget, Admin → AI & SEO → Settings → LLM budget) and checks
every caller against them. This service holds each call's upper bound there (app/seo_agent.py RemoteBudgetGuard) and
stores no limit of its own. `MemoryBudget` is for tests and the in-memory mode, where BudgetGuard checks locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

Unit = Literal["day", "month"]


@dataclass(frozen=True)
class Budget:
    daily_usd: float | None = None
    monthly_usd: float | None = None

    @property
    def empty(self) -> bool:
        return self.daily_usd is None and self.monthly_usd is None

    def limit(self, unit: Unit) -> float | None:
        return self.daily_usd if unit == "day" else self.monthly_usd


NO_BUDGET = Budget()


class BudgetSource(Protocol):
    async def budget(self) -> Budget: ...


@dataclass
class MemoryBudget:
    """Tests and the in-memory mode without a database."""

    value: Budget = NO_BUDGET

    async def budget(self) -> Budget:
        return self.value

    async def save(self, budget: Budget, actor: str | None) -> Budget:
        self.value = budget
        return budget


def period_start(unit: str, now: datetime) -> datetime:
    return datetime(now.year, now.month, now.day, tzinfo=UTC) if unit == "day" else datetime(now.year, now.month, 1, tzinfo=UTC)


def period_end(unit: str, now: datetime) -> datetime:
    if unit == "day":
        return period_start("day", now) + timedelta(days=1)
    return datetime(now.year + (now.month == 12), now.month % 12 + 1, 1, tzinfo=UTC)
