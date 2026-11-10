"""Shared Flow plumbing: dependencies, the base class every flow extends, and step instrumentation.

Every flow is a CrewAI `Flow[State]` with `@start` / `@listen` / `@router` methods. `SeoFlow` adds:
- `deps` (settings, LLM, web app client, retriever, run store, the runner for child runs, crew factories);
- `step(name)`: duration histogram, OTel span and the run's `current_step` while the step runs (CrewAI's persistence
  then records the finished step and the state, app/runs.py);
- `@persist(INSTANCE_ONLY)` (never CrewAI's default SQLite), no auto-created CrewAI memory, quiet console.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from crewai.flow import Flow
from pydantic import BaseModel, PrivateAttr

from .. import metrics
from ..config import Settings
from ..llm import Llm
from ..tracing import span
from ..webapp import WebApp, WebAppError

S = TypeVar("S", bound=BaseModel)


@dataclass
class Deps:
    settings: Settings
    llm: Llm
    web: WebApp
    retriever: Any
    store: Any = None  # RunStore
    runner: Any = None  # FlowRunner (child runs)
    optimization_crew: Callable[[], Any] | None = None
    editorial_crew: Callable[[], Any] | None = None
    report_crew: Callable[[], Any] | None = None
    reindex: Callable[[list[str] | None, bool], Awaitable[dict[str, int]]] | None = None
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    verify_delays: tuple[float, ...] = (40.0, 30.0, 60.0)
    extra: dict[str, Any] = field(default_factory=dict)


def transient(e: BaseException) -> bool:
    if isinstance(e, WebAppError):
        return e.transient
    return type(e).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError", "ServiceUnavailableError", "ConnectError", "ReadTimeout", "RemoteProtocolError"}


async def with_retry(fn: Callable[[], Awaitable[Any]], sleep: Callable[[float], Awaitable[None]], attempts: int = 3, base: float = 2.0) -> Any:
    """Transient errors only (connection, timeout, 429, 5xx): exponential backoff. Anything else raises at once."""
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:
            if i == attempts - 1 or not transient(e):
                raise
            await sleep(min(30.0, base * (2**i)))
    raise AssertionError("unreachable")


class SeoFlow(Flow[S]):
    """Base class. Subclasses set FLOW (the registry name) and define their steps."""

    FLOW: ClassVar[str] = "flow"
    _skip_auto_memory: ClassVar[bool] = True  # CrewAI would otherwise create a Memory (LanceDB) per flow
    _deps: Any = PrivateAttr(default=None)

    @classmethod
    def create(cls, deps: Deps, persistence: Any) -> SeoFlow[Any]:
        f = cls(persistence=persistence, suppress_flow_events=True, tracing=False)
        f._deps = deps
        return f

    @property
    def deps(self) -> Deps:
        return self._deps

    @asynccontextmanager
    async def step(self, name: str) -> AsyncIterator[None]:
        started = time.monotonic()
        store = self.deps.store
        run_id = getattr(self.state, "id", None)
        if store is not None and run_id:
            try:
                await store.update(run_id, current_step=name)
            except Exception:  # noqa: S110 - the run record must never break a flow
                pass
        with span(f"flow.{self.FLOW}.{name}", {"flow": self.FLOW, "step": name}):
            try:
                yield
            finally:
                metrics.STEP_DURATION.labels(flow=metrics.flow_label(self.FLOW), step=name).observe(time.monotonic() - started)

    async def retry(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        return await with_retry(fn, self.deps.sleep)
