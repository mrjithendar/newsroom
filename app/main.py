"""seo-agent: CrewAI Flows behind a small internal API, plus the MCP edge (app/mcp) on a second port.

Internal API (port 8080, ClusterIP; NetworkPolicy admits only the web pods, the trigger CronJobs and Prometheus).
Every endpoint except /healthz, /readyz and /metrics needs `Authorization: Bearer $SEO_AGENT_TOKEN`:

    POST /flows/{flow_name}/kickoff     {inputs, triggered_by: cron|manual, actor?} → {run_id}
    GET  /flows/runs                    ?flow=&status=&draft_id=&limit=
    GET  /flows/runs/{run_id}           status, current step, outputs, errors, cost
    POST /flows/runs/{run_id}/cancel
    POST /review/{draft_id}/decision    {approve|reject|regenerate, approval_id, accepted_change_ids, edits, feedback}
    POST /ask                           public RAG, proxied by the web app (rate-limited there)
    GET  /budget
    GET  /healthz /readyz /metrics

publish_flow cannot be started through /flows/{name}/kickoff (403): only /review/{id}/decision with an approval.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest
from pydantic import BaseModel, ConfigDict, Field

from . import metrics
from .ask import AskService, MemoryAskLog, PgAskLog
from .config import Settings, load_settings
from .crews.editorial import EditorialCrew
from .crews.optimization import OptimizationCrew
from .crews.report import ReportCrew
from .embeddings import HashEmbeddings
from .flows import SPECS, check_kickoff
from .flows.base import Deps
from .index import MemoryRetriever, PgVectorIndex
from .llm import AnthropicLlm, BudgetGuard, MemorySpendStore, PgSpendStore
from .news.store import MemoryNewsStore, PgNewsStore
from .runs import FlowRunner, MemoryFlowPersistence, MemoryLocks, MemoryRunStore, PgAdvisoryLocks, PgFlowPersistence, PgRunStore, RunError, view
from .schemas import AskRequest
from .services import review
from .services.ask_pipeline import AskPipeline
from .services.content import ContentError
from .tracing import setup_tracing
from .webapp import HttpWebApp

log = logging.getLogger("seo_agent")
VERSION = os.environ.get("GIT_SHA", "dev")


class Kickoff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: dict[str, Any] = Field(default_factory=dict)
    triggered_by: Literal["cron", "manual"] = "manual"
    actor: str | None = Field(default=None, max_length=200)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject", "regenerate"]
    approval_id: str | None = Field(default=None, max_length=80)
    accepted_change_ids: list[str] = Field(default_factory=list, max_length=200)
    edits: dict[str, str] = Field(default_factory=dict)
    feedback: str | None = Field(default=None, max_length=2000)
    actor: str | None = Field(default=None, max_length=200)


@dataclass
class Container:
    settings: Settings
    deps: Deps
    runner: FlowRunner
    store: Any
    ask: AskService
    index: Any
    guard: BudgetGuard
    pool: Any = None
    sync_pool: Any = None
    ready: bool = False
    reasons: list[str] = field(default_factory=list)
    reindex_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    bg: list[asyncio.Task[Any]] = field(default_factory=list)

    async def reindex(self, article_ids: list[str] | None = None, full: bool = False) -> dict[str, int]:
        async with self.reindex_lock:
            if not getattr(self.index, "available", False) or not hasattr(self.index, "reindex"):
                return {"articles": 0, "embedded": 0, "unchanged": 0, "deleted": 0}
            articles = await self.deps.web.published_articles()
            return await self.index.reindex(articles, full=full, only=set(article_ids) if article_ids else None)


def _crews(deps: Deps) -> None:
    s, llm, web, ret = deps.settings, deps.llm, deps.web, deps.retriever
    deps.optimization_crew = deps.optimization_crew or (lambda: OptimizationCrew(s, llm, web, ret))
    deps.editorial_crew = deps.editorial_crew or (lambda: EditorialCrew(s, llm, web, ret))
    deps.report_crew = deps.report_crew or (lambda: ReportCrew(s, llm, web, ret))


def _news(deps: Deps, store: Any, embeddings: Any) -> None:
    """News desk dependencies (app/flows/news.py). Tests may pre-set any of these in deps.extra."""

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embeddings.embed(texts)

    deps.extra.setdefault("news_store", store)
    deps.extra.setdefault("news_embed", embed)


async def build_container(settings: Settings) -> Container:
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool, ConnectionPool

    from .embeddings import make_embeddings

    kw = {"autocommit": True, "row_factory": dict_row, "prepare_threshold": None}  # PgBouncer (session mode) in front
    pool = AsyncConnectionPool(settings.database_url, min_size=1, max_size=6, open=False, kwargs=kw)
    await pool.open(wait=True, timeout=30)
    sync_pool = ConnectionPool(settings.database_url, min_size=1, max_size=2, open=True, kwargs=kw)  # CrewAI persistence
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
    store = PgRunStore(pool)
    deps = Deps(settings=settings, llm=llm, web=web, retriever=index, store=store)
    _crews(deps)
    _news(deps, PgNewsStore(pool), index.emb)
    runner = FlowRunner(SPECS, store, PgAdvisoryLocks(settings.database_url), lambda: PgFlowPersistence().bind(sync_pool), deps, timeout_s=settings.flow_timeout_s, policy=check_kickoff)
    deps.runner = runner
    ask = AskService(AskPipeline(deps), index, PgAskLog(pool), settings.ask_enabled, settings.ask_cache_ttl_s)
    c = Container(settings=settings, deps=deps, runner=runner, store=store, ask=ask, index=index, guard=guard, pool=pool, sync_pool=sync_pool, reasons=reasons)
    deps.reindex = c.reindex
    return c


def memory_container(settings: Settings, *, llm: Any, web: Any, retriever: Any = None, verify_delays: tuple[float, ...] = (0, 0, 0), **crews: Any) -> Container:
    """Everything in memory (tests): fake LLM and web app, in-memory runs/locks/persistence."""
    retriever = retriever or MemoryRetriever()
    store = MemoryRunStore()

    async def no_sleep(_s: float) -> None:
        return None

    deps = Deps(settings=settings, llm=llm, web=web, retriever=retriever, store=store, sleep=no_sleep, verify_delays=verify_delays, **crews)
    _crews(deps)
    _news(deps, MemoryNewsStore(), HashEmbeddings())
    runner = FlowRunner(SPECS, store, MemoryLocks(), lambda: MemoryFlowPersistence().bind(store), deps, timeout_s=settings.flow_timeout_s, policy=check_kickoff)
    deps.runner = runner
    guard = getattr(llm, "guard", BudgetGuard(MemorySpendStore(), settings.monthly_budget_usd, settings.ask_daily_budget_usd))
    ask = AskService(AskPipeline(deps), retriever, MemoryAskLog(), settings.ask_enabled, settings.ask_cache_ttl_s)
    c = Container(settings=settings, deps=deps, runner=runner, store=store, ask=ask, index=retriever, guard=guard, ready=True)
    deps.reindex = c.reindex
    return c


async def _gauges(c: Container) -> None:
    while True:
        try:
            st = await c.guard.status()
            if c.settings.monthly_budget_usd > 0:
                metrics.BUDGET_RATIO.set(st["month_spend_usd"] / c.settings.monthly_budget_usd)
            if c.settings.ask_daily_budget_usd > 0:
                metrics.ASK_BUDGET_RATIO.set(st["ask_day_spend_usd"] / c.settings.ask_daily_budget_usd)
            if c.pool is not None:
                async with c.pool.connection() as conn:
                    cur = await conn.execute("select count(*)::int as n from seo_drafts where status = 'ready_for_review'")
                    metrics.DRAFTS_AWAITING.set(int((await cur.fetchone())["n"]))
                    cur = await conn.execute("select flow_name, extract(epoch from max(finished_at))::float8 as t from seo_flow_runs where status = 'succeeded' group by flow_name")
                    for r in await cur.fetchall():
                        metrics.LAST_SUCCESS.labels(flow=metrics.flow_label(r["flow_name"])).set(float(r["t"] or 0))
            if isinstance(c.ask.log, PgAskLog):
                await c.ask.log.prune(c.settings.ask_retention_days)
        except Exception:  # noqa: S110 - gauges are best effort
            pass
        await asyncio.sleep(60)


async def startup(c: Container) -> None:
    try:
        await c.runner.recover()
    except Exception as e:  # e.g. the web app's migration 0016 not applied yet: start, report on /readyz
        c.reasons.append(f"run recovery failed: {type(e).__name__} (is migration 0016 applied?)")
        log.warning("run recovery failed", extra={"event": "recover_failed"})
    c.ready = True
    c.bg.append(asyncio.create_task(_gauges(c)))


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container
        owned = c is None
        if c is None:  # standalone (uvicorn app.main:create_app --factory); run() builds the container itself
            settings = load_settings()
            setup_tracing(settings.otel_endpoint, settings.service_name, VERSION)
            c = await build_container(settings)
            await startup(c)
        app.state.c = c
        yield
        if owned:
            for t in c.bg:
                t.cancel()
            if c.pool is not None:
                await c.pool.close()
            if c.sync_pool is not None:
                c.sync_pool.close()

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

    @app.exception_handler(ContentError)
    async def content_error(_r: Request, e: ContentError) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=400)

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

    @app.post("/flows/{flow_name}/kickoff", dependencies=[Depends(auth)])
    async def kickoff(flow_name: str, body: Kickoff, request: Request) -> JSONResponse:
        r = await ctr(request).runner.kickoff(flow_name, body.inputs, triggered_by=body.triggered_by, actor=body.actor, correlation_id=request.headers.get("x-correlation-id"))
        return JSONResponse(r, status_code=200 if r.get("existing") else 202)

    @app.get("/flows/runs", dependencies=[Depends(auth)])
    async def list_runs(request: Request, flow: str | None = None, status: str | None = None, draft_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        rows = await ctr(request).store.list_runs(flow=flow, status=status, draft_id=draft_id, limit=limit)
        return {"items": [view(r) for r in rows]}

    @app.get("/flows/runs/{run_id}", dependencies=[Depends(auth)])
    async def get_run(run_id: str, request: Request) -> dict[str, Any]:
        r = await ctr(request).store.get(run_id)
        if not r:
            raise HTTPException(404, "run not found")
        return view(r, with_state=True)

    @app.post("/flows/runs/{run_id}/cancel", dependencies=[Depends(auth)])
    async def cancel_run(run_id: str, request: Request) -> dict[str, Any]:
        return await ctr(request).runner.cancel(run_id)

    @app.post("/review/{draft_id}/decision", dependencies=[Depends(auth)])
    async def decision(draft_id: str, body: Decision, request: Request) -> JSONResponse:
        c = ctr(request)
        r = await review.decide(c.runner, c.deps.web, draft_id, body.decision, approval_id=body.approval_id, accepted_change_ids=body.accepted_change_ids, edits=body.edits, feedback=body.feedback, actor=body.actor)
        return JSONResponse(r, status_code=202 if body.decision != "reject" else 200)

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


async def serve() -> None:
    import uvicorn

    from .mcp import create_edge_app

    settings = load_settings()
    if settings.production and (len(settings.agent_token) < 32 or len(settings.callback_token) < 32):
        raise RuntimeError("SEO_AGENT_TOKEN and SEO_AGENT_CALLBACK_TOKEN must be set (≥ 32 characters)")
    setup_tracing(settings.otel_endpoint, settings.service_name, VERSION)
    c = await build_container(settings)
    await startup(c)
    internal = uvicorn.Server(uvicorn.Config(create_app(c), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), proxy_headers=False, access_log=False, lifespan="on"))  # noqa: S104 - ClusterIP
    # The edge sits behind Kong: its X-Forwarded-For is read explicitly (one trusted hop), not via proxy_headers.
    edge = uvicorn.Server(uvicorn.Config(create_edge_app(c), host="0.0.0.0", port=settings.edge_port, proxy_headers=False, access_log=False, lifespan="on", timeout_keep_alive=75))  # noqa: S104
    try:
        await asyncio.gather(internal.serve(), edge.serve())
    finally:
        for t in c.bg:
            t.cancel()
        await c.pool.close()
        c.sync_pool.close()


def run() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
    asyncio.run(serve())


if __name__ == "__main__":
    run()
