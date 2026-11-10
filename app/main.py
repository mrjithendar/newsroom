"""seo-agent: FastAPI service (cluster-internal only; no Ingress/VirtualService route points here).

Every endpoint except /healthz, /readyz and /metrics requires `Authorization: Bearer $SEO_AGENT_TOKEN` (the web app's
admin API holds it). /ask is public *through the web app*, which rate-limits per IP before proxying.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest

from . import metrics
from .ask import AskService, MemoryAskLog, PgAskLog
from .config import Settings, load_settings
from .crews.editorial import EditorialCrew
from .graphs import ask_blog, internal_linking, new_article, optimize_article, weekly
from .graphs.common import Deps
from .index import MemoryRetriever, PgVectorIndex
from .llm import AnthropicLlm, BudgetGuard, MemorySpendStore, PgSpendStore
from .runs import MemoryRunStore, PgRunStore, RunError, RunManager
from .schemas import AskRequest, ReindexRequest, ResumeRun, StartRun
from .tracing import setup_tracing
from .webapp import HttpWebApp

log = logging.getLogger("seo_agent")
VERSION = os.environ.get("GIT_SHA", "dev")


@dataclass
class Container:
    settings: Settings
    deps: Deps
    runs: RunManager
    ask: AskService
    index: Any
    guard: BudgetGuard
    pool: Any = None
    checkpointer: Any = None
    ready: bool = False
    reasons: list[str] = field(default_factory=list)
    reindex_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_reindex: float = 0.0
    bg: list[asyncio.Task[Any]] = field(default_factory=list)

    async def reindex(self, article_ids: list[str] | None = None, full: bool = False) -> dict[str, int]:
        async with self.reindex_lock:
            if not getattr(self.index, "available", False):
                return {"articles": 0, "embedded": 0, "unchanged": 0, "deleted": 0}
            articles = await self.deps.web.published_articles()
            return await self.index.reindex(articles, full=full, only=set(article_ids) if article_ids else None)


def build_graphs(deps: Deps, checkpointer: Any) -> dict[str, Any]:
    return {
        "optimize_article": optimize_article.build(deps, checkpointer),
        "weekly_recommendations": weekly.build(deps, checkpointer),
        "internal_linking": internal_linking.build(deps, checkpointer),
        "new_article": new_article.build(deps, checkpointer),
    }


def wire(deps: Deps, runs: RunManager) -> None:
    """Child runs (weekly fan-out) go through the same run manager; wait returns at interrupt or end."""

    async def start_child(graph: str, payload: dict[str, Any], thread_id: str) -> dict[str, Any]:
        return await runs.start(graph, payload, thread_id)

    async def wait_child(thread_id: str) -> dict[str, Any]:
        await runs.wait(thread_id)
        r = await runs.store.get(thread_id)
        return dict(r or {})

    deps.start_child, deps.wait_child = start_child, wait_child


async def build_container(settings: Settings) -> Container:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    from .embeddings import make_embeddings

    # PgBouncer (session mode) in front of Postgres: no server-side prepared statements, autocommit for the saver.
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=6, open=False, kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None})
    await pool.open(wait=True, timeout=30)
    checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
    await checkpointer.setup()  # idempotent; its own checkpoint_migrations table
    guard = BudgetGuard(PgSpendStore(pool), settings.monthly_budget_usd, settings.ask_daily_budget_usd)
    llm = AnthropicLlm(settings.anthropic_api_key, guard)
    web = HttpWebApp(settings.callback_url, settings.callback_token)
    index = PgVectorIndex(pool, make_embeddings(settings))
    reasons: list[str] = []
    try:
        await index.ensure_schema()
        if not index.available:
            reasons.append(index.reason)
    except Exception as e:
        reasons.append(f"vector index: {type(e).__name__}")
    run_store = PgRunStore(pool)
    deps = Deps(settings=settings, llm=llm, web=web, retriever=index, runs=run_store)
    deps.crew_factory = lambda: EditorialCrew(settings, llm, web, index)
    runs = RunManager(build_graphs(deps, checkpointer), run_store)
    wire(deps, runs)
    ask = AskService(ask_blog.build(deps), index, PgAskLog(pool), settings.ask_enabled, settings.ask_cache_ttl_s)
    c = Container(settings=settings, deps=deps, runs=runs, ask=ask, index=index, guard=guard, pool=pool, checkpointer=checkpointer, reasons=reasons)

    async def reindex_article(article_id: str) -> Any:
        return await c.reindex([article_id])

    deps.reindex_article = reindex_article
    return c


def memory_container(settings: Settings, *, llm: Any, web: Any, retriever: Any = None, checkpointer: Any = None, crew_factory: Any = None, verify_delays: tuple[float, ...] = (0, 0, 0)) -> Container:
    """Everything in memory (tests): InMemorySaver unless a checkpointer is given, fake LLM/web app."""
    from langgraph.checkpoint.memory import InMemorySaver

    retriever = retriever or MemoryRetriever()
    run_store = MemoryRunStore()

    async def no_sleep(_s: float) -> None:
        return None

    deps = Deps(settings=settings, llm=llm, web=web, retriever=retriever, runs=run_store, sleep=no_sleep, verify_delays=verify_delays, crew_factory=crew_factory)
    runs = RunManager(build_graphs(deps, checkpointer or InMemorySaver()), run_store)
    wire(deps, runs)
    guard = getattr(llm, "guard", BudgetGuard(MemorySpendStore(), settings.monthly_budget_usd, settings.ask_daily_budget_usd))
    ask = AskService(ask_blog.build(deps), retriever, MemoryAskLog(), settings.ask_enabled, settings.ask_cache_ttl_s)
    return Container(settings=settings, deps=deps, runs=runs, ask=ask, index=retriever, guard=guard, ready=True)


async def _gauges(c: Container) -> None:
    while True:
        try:
            metrics.INTERRUPTS_PENDING.set(await c.runs.store.count_status("interrupted"))
            st = await c.guard.status()
            if c.settings.monthly_budget_usd > 0:
                metrics.BUDGET_RATIO.set(st["month_spend_usd"] / c.settings.monthly_budget_usd)
            if c.settings.ask_daily_budget_usd > 0:
                metrics.ASK_BUDGET_RATIO.set(st["ask_day_spend_usd"] / c.settings.ask_daily_budget_usd)
            if isinstance(c.ask.log, PgAskLog):
                await c.ask.log.prune(c.settings.ask_retention_days)
        except Exception:  # noqa: S110 - gauges are best effort
            pass
        await asyncio.sleep(60)


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container
        if c is None:
            settings = load_settings()
            if settings.production and (len(settings.agent_token) < 32 or len(settings.callback_token) < 32):
                raise RuntimeError("SEO_AGENT_TOKEN and SEO_AGENT_CALLBACK_TOKEN must be set (≥ 32 characters)")
            setup_tracing(settings.otel_endpoint, settings.service_name, VERSION)
            c = await build_container(settings)
            try:
                await c.runs.recover()
            except Exception as e:  # e.g. the web app's migration 0015 not applied yet: start, report on /readyz
                c.reasons.append(f"run recovery failed: {type(e).__name__} (is migration 0015 applied?)")
                log.warning("run recovery failed", extra={"event": "recover_failed"})
            c.ready = True
            c.bg.append(asyncio.create_task(_gauges(c)))
        app.state.c = c
        yield
        for t in c.bg:
            t.cancel()
        if c.pool is not None:
            await c.pool.close()

    app = FastAPI(title="seo-agent", version=VERSION, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def ctr(request: Request) -> Container:
        return request.app.state.c

    def auth(request: Request) -> None:
        c: Container = request.app.state.c
        token = c.settings.agent_token
        got = request.headers.get("authorization", "")
        if not token or not hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "unauthorized")

    @app.exception_handler(RunError)
    async def run_error(_r: Request, e: RunError) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=e.status)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": VERSION}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        c = ctr(request)
        if not c.ready:
            return JSONResponse({"status": "starting"}, status_code=503)
        try:
            if c.pool is not None:
                async with c.pool.connection() as conn:
                    await conn.execute("select 1")
        except Exception:
            return JSONResponse({"status": "db_unavailable"}, status_code=503)
        return JSONResponse({"status": "ready", "vector": getattr(c.index, "available", False), "notes": c.reasons})

    @app.get("/metrics")
    async def metrics_ep() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(metrics.REGISTRY).decode(), media_type="text/plain; version=0.0.4")

    @app.post("/runs", dependencies=[Depends(auth)])
    async def start_run(body: StartRun, request: Request) -> JSONResponse:
        c = ctr(request)
        tid = body.thread_id or body.input.get("draft_id")
        if not tid:
            raise HTTPException(400, "thread_id (or input.draft_id) is required")
        r = await c.runs.start(body.graph, body.input, str(tid))
        return JSONResponse(r, status_code=200 if r.get("existing") else 202)

    @app.get("/runs/{thread_id}", dependencies=[Depends(auth)])
    async def get_run(thread_id: str, request: Request) -> dict[str, Any]:
        return await ctr(request).runs.get(thread_id)

    @app.post("/runs/{thread_id}/resume", dependencies=[Depends(auth)])
    async def resume_run(thread_id: str, body: ResumeRun, request: Request) -> JSONResponse:
        if body.decision == "approve" and not body.approval_id:
            raise HTTPException(400, "approve needs the approval_id the web app recorded")
        return JSONResponse(await ctr(request).runs.resume(thread_id, body.model_dump()), status_code=202)

    @app.post("/runs/{thread_id}/cancel", dependencies=[Depends(auth)])
    async def cancel_run(thread_id: str, request: Request) -> dict[str, Any]:
        return await ctr(request).runs.cancel(thread_id)

    @app.post("/embeddings/reindex", dependencies=[Depends(auth)])
    async def reindex(body: ReindexRequest, request: Request) -> JSONResponse:
        c = ctr(request)
        import time

        if body.full and time.monotonic() - c.last_reindex < 300:
            raise HTTPException(429, "a full reindex ran less than 5 minutes ago")
        if c.reindex_lock.locked():
            raise HTTPException(409, "a reindex is already running")
        c.last_reindex = time.monotonic()
        return JSONResponse(await c.reindex(body.article_ids, body.full))

    @app.post("/ask", dependencies=[Depends(auth)])
    async def ask(body: AskRequest, request: Request) -> Any:
        c = ctr(request)
        if body.stream:
            return StreamingResponse(c.ask.stream(body.question, body.ip_hash), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        return await c.ask.ask(body.question, body.ip_hash)

    @app.get("/budget", dependencies=[Depends(auth)])
    async def budget(request: Request) -> dict[str, float]:
        return await ctr(request).guard.status()

    return app


def run() -> None:
    import uvicorn

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), proxy_headers=False, access_log=False)  # noqa: S104 - in-cluster ClusterIP


if __name__ == "__main__":
    run()
