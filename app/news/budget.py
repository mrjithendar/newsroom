"""News budgets (docs/news/COSTS.md): per job and global, daily and monthly, separately for LLM and image spend.

Checked *before* every LLM call (upper-bound estimate: input chars / 3.5 + full max_tokens) and before every image
generation; when a call would cross a cap, nothing is sent, `BudgetExceeded` is raised and the run ends as
`paused_budget` with the cap named in its error. Days and months are calendar periods in the job's time zone (global
caps: UTC). Spend is read from news_runs (cost_usd / image_cost_usd), which every call updates as it completes.
LLM calls reserve their estimate first (`NewsBudget.reserve`: booked on the run's row under a lock, then verified),
so concurrent calls cannot all pass the same check and jointly overspend; the reservation is released when the call
ends and the actual cost is booked.

News LLM usage is also recorded in seo_llm_usage (purpose "news.*") for the shared usage page, but it is excluded from
the SEO admin monthly cap (app/llm.py BudgetGuard, web budgetState), so the news desk can never pause SEO drafting.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from ..llm import BudgetExceeded, CallSpec, Llm, Usage, estimate_cost_usd
from .config import Job, NewsSettings
from .store import NewsStore

T = TypeVar("T", bound=BaseModel)


def period_starts(now: datetime, tz: str) -> tuple[datetime, datetime]:
    try:
        z = ZoneInfo(tz)
    except Exception:
        z = ZoneInfo("UTC")
    local = now.astimezone(z)
    day = local.replace(hour=0, minute=0, second=0, microsecond=0)
    month = day.replace(day=1)
    return day.astimezone(UTC), month.astimezone(UTC)


@dataclass
class _LoopState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: float = 0.0


_STATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopState] = weakref.WeakKeyDictionary()


def _loop_state() -> _LoopState:
    """One reservation lock per event loop (an asyncio.Lock is bound to the loop that first waits on it)."""
    loop = asyncio.get_running_loop()
    st = _STATES.get(loop)
    if st is None:
        st = _STATES[loop] = _LoopState()
    return st


@dataclass
class NewsBudget:
    store: NewsStore
    settings: NewsSettings
    job: Job | None
    run_id: str
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    dry_run: bool = False

    async def status(self) -> dict[str, Any]:
        now = self.clock()
        gd, gm = period_starts(now, "UTC")
        g_day, g_month = await self.store.spend(None, gd), await self.store.spend(None, gm)
        out: dict[str, Any] = {
            "global": {"llm_day": g_day["llm"], "llm_month": g_month["llm"], "image_day": g_day["image"], "image_month": g_month["image"], "caps": {"llm_day": self.settings.llm_daily_usd, "llm_month": self.settings.llm_monthly_usd, "image_day": self.settings.image_daily_usd, "image_month": self.settings.image_monthly_usd}},
        }
        if self.job:
            jd, jm = period_starts(now, self.job.timezone)
            d, m = await self.store.spend(self.job.id, jd), await self.store.spend(self.job.id, jm)
            b = self.job.config.budget
            out["job"] = {"llm_day": d["llm"], "llm_month": m["llm"], "image_day": d["image"], "image_month": m["image"], "caps": {"llm_day": b.llm_daily_usd, "llm_month": b.llm_monthly_usd, "image_day": b.image_daily_usd, "image_month": b.image_monthly_usd}}
        return out

    async def check(self, kind: str, estimate_usd: float, *, booked: bool = False) -> None:
        """kind: llm | image. Raises BudgetExceeded naming the first cap the estimate would cross. booked: the estimate
        is already included in the recorded spend (a reservation)."""
        st = await self.status()
        checks = [("global", "day"), ("global", "month")] + ([("job", "day"), ("job", "month")] if self.job else [])
        for scope, period in checks:
            spent = st[scope][f"{kind}_{period}"] - (estimate_usd if booked else 0.0)
            cap = st[scope]["caps"][f"{kind}_{period}"]
            if spent + estimate_usd > cap:
                raise BudgetExceeded(f"news_{scope}_{kind}_{period}", f"{'this job' if scope == 'job' else 'the news desk'}'s {'daily' if period == 'day' else 'monthly'} {kind.upper() if kind == 'llm' else 'image'} budget ${cap:.2f} would be exceeded (spent ${spent:.2f}, this call ≈ ${estimate_usd:.3f})")

    async def reserve(self, kind: str, estimate_usd: float) -> float:
        """Check-and-reserve, atomically: under a per-event-loop lock, the estimate is booked on this run's news_runs row
        (so every process sees it in `spend`), then the caps are verified with the reservation included. Over a cap the
        reservation is released and BudgetExceeded raised, so concurrent calls can never jointly overspend; across
        processes the worst case is a spurious refusal, never an overrun. Returns the reserved amount (to `settle`).
        Without a run row to book on (run_id empty) it falls back to an in-process pending total."""
        if kind != "llm":
            await self.check(kind, estimate_usd)
            return 0.0
        state = _loop_state()
        async with state.lock:
            if not self.run_id:
                await self.check(kind, estimate_usd + state.pending)
                state.pending += estimate_usd
                return estimate_usd
            await self.store.add_run_cost(self.run_id, usd=estimate_usd)
            try:
                await self.check(kind, estimate_usd, booked=True)
            except BudgetExceeded:
                await self.store.add_run_cost(self.run_id, usd=-estimate_usd)
                raise
            return estimate_usd

    async def settle(self, reserved: float) -> None:
        """Release a reservation (the caller books the actual cost separately)."""
        if reserved <= 0:
            return
        if not self.run_id:
            state = _loop_state()
            state.pending = max(0.0, state.pending - reserved)
            return
        await self.store.add_run_cost(self.run_id, usd=-reserved)

    async def ratios(self) -> dict[str, float]:
        st = await self.status()
        out = {}
        for scope in ("global", "job"):
            if scope in st:
                for k in ("llm_day", "llm_month", "image_day", "image_month"):
                    cap = st[scope]["caps"][k]
                    out[f"{scope}_{k}"] = round(st[scope][k] / cap, 4) if cap else 0.0
        return out


class NewsLlm:
    """The service's Llm with the news caps in front and the cost booked to the news run (and its parent)."""

    def __init__(self, inner: Llm, budget: NewsBudget, run_ids: list[str], *, enabled: bool = True):
        self.inner = inner
        self.budget = budget
        self.run_ids = run_ids
        self.enabled = enabled
        self.guard = getattr(inner, "guard", None)
        self.tokens = 0
        self.cost_usd = 0.0

    async def _pre(self, spec: CallSpec) -> float:
        """Reserves the upper-bound estimate; returns it for `_settle` (always called, also when the call fails)."""
        if not self.enabled:
            raise BudgetExceeded("news_llm_off", "LLM calls are switched off for this run (dry run without LLM)")
        if not spec.purpose.startswith("news"):
            spec.purpose = f"news.{spec.purpose}"
        return await self.budget.reserve("llm", estimate_cost_usd(spec.model, spec.input_chars, spec.max_tokens))

    async def _post(self, usage: Usage) -> None:
        c = usage.cost_usd()
        self.tokens += usage.total_tokens
        self.cost_usd += c
        for rid in self.run_ids:
            await self.budget.store.add_run_cost(rid, usd=c, tokens=usage.total_tokens)

    async def structured(self, spec: CallSpec, schema: type[T]) -> tuple[T, Usage]:
        held = await self._pre(spec)
        try:
            out, usage = await self.inner.structured(spec, schema)
            await self._post(usage)  # the actual cost is booked before the reservation is released
        finally:
            await self.budget.settle(held)
        return out, usage

    async def text(self, spec: CallSpec) -> tuple[str, Usage]:
        held = await self._pre(spec)
        try:
            out, usage = await self.inner.text(spec)
            await self._post(usage)  # the actual cost is booked before the reservation is released
        finally:
            await self.budget.settle(held)
        return out, usage

    async def stream(self, spec: CallSpec) -> AsyncIterator[str]:
        held = await self._pre(spec)
        try:
            async for t in self.inner.stream(spec):
                yield t
        finally:
            await self.budget.settle(held)
