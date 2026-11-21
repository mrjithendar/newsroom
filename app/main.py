"""news-agent: the news desk's CrewAI Flows behind a small internal API (docs/ARCHITECTURE.md).

Split out of the seo-agent on 2026-11-21 (moks-seo-agent/app/main.py@bb9e553). One port, 8080, ClusterIP; the
NetworkPolicy admits only the web pods, the dispatcher CronJob and Prometheus. Every endpoint except /healthz, /readyz
and /metrics needs `Authorization: Bearer $NEWS_AGENT_TOKEN`:

    GET  /flows                         every news flow: input JSON Schema, schedulable, crews (CrewAI console)
    POST /flows/{flow_name}/validate    {inputs} → {inputs} as the flow would get them, or 422
    POST /flows/{flow_name}/kickoff     {inputs, triggered_by: cron|manual, actor?} → {run_id}
    POST /schedules/dispatch            start every news schedule that is due (its own CronJob, every minute)
    GET  /crews                         the news crew's editable configuration as the code defines it
    POST /crews/{crew}/validate         {config} → {config} as it would be stored (only what differs), or 422
    GET  /flows/runs                    ?flow=&status=&limit= (this service's runs only)
    GET  /flows/runs/{run_id}           status, current step, outputs, errors, cost
    POST /flows/runs/{run_id}/cancel
    GET  /healthz /readyz /metrics

Nothing here publishes: a news draft is published by the seo-agent's publish_flow after a person approves it in the
web app's review queue.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest
from pydantic import BaseModel, ConfigDict, Field

from . import metrics
from .config import Settings, load_settings
from .crews import config as crew_config
from .embeddings import HashEmbeddings
from .flows import CRON_FLOWS, OWNED_FLOWS, SPECS, check_kickoff
from .flows import catalogue as flow_catalogue
from .flows.base import Deps
from .llm import AnthropicLlm, BudgetGuard, MemorySpendStore, PgSpendStore
from .logs import configure_logging
from .news.store import MemoryNewsStore, PgNewsStore
from .run_history import MemoryHistorySink, PgHistorySink, RecordingLlm, RunHistory
from .run_history import install as install_run_log
from .run_history import uninstall as uninstall_run_log
from .runs import FlowRunner, MemoryFlowPersistence, MemoryLocks, MemoryRunStore, PgAdvisoryLocks, PgFlowPersistence, PgRunStore, RunError, view
from .schedules import MemoryScheduleStore, PgScheduleStore, SchedulesUnavailable
from .schedules import dispatch as dispatch_schedules
from .seo_agent import RemoteBudgetGuard, RemoteRetriever, SeoAgentClient
from .tracing import setup_tracing
from .webapp import HttpWebApp

log = logging.getLogger("news_agent")
VERSION = os.environ.get("GIT_SHA", "dev")


class Kickoff(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: dict[str, Any] = Field(default_factory=dict)
    triggered_by: Literal["cron", "manual"] = "manual"
    actor: str | None = Field(default=None, max_length=200)


class FlowInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: dict[str, Any] = Field(default_factory=dict)


class CrewConfigBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Container:
    settings: Settings
    deps: Deps
    runner: FlowRunner
    store: Any
    guard: BudgetGuard
    pool: Any = None
    sync_pool: Any = None
    ready: bool = False
    reasons: list[str] = field(default_factory=list)
    bg: list[asyncio.Task[Any]] = field(default_factory=list)
    history: RunHistory | None = None  # steps, log lines and model calls of every flow run (app/run_history.py)
    schedules: Any = field(default_factory=MemoryScheduleStore)  # seo_flow_schedules, the news rows (app/schedules.py)
    clock: Callable[[], Any] | None = None  # tests: the dispatcher's "now"
    seo_agent: SeoAgentClient | None = None


def _news(deps: Deps, store: Any, embeddings: Any) -> None:
    """News desk dependencies (app/flows/news.py). Tests may pre-set any of these in deps.extra."""

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embeddings.embed(texts)

    # Which model makes these vectors, and their size, travel with the function: stale stored vectors are re-embedded
    # and the cosine cutoffs follow the model (app/news/desk.py "The embeddings model", app/calibration.py). An embed
    # function put in its place without them (tests) keeps the behaviour from before.
    embed.model = embeddings.name  # type: ignore[attr-defined]
    embed.dim = embeddings.dim  # type: ignore[attr-defined]
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
    seo = SeoAgentClient(settings.seo_agent_url, settings.seo_agent_token)
    guard = RemoteBudgetGuard(PgSpendStore(pool), seo)
    web = HttpWebApp(settings.callback_url, settings.callback_token)
    history = RunHistory(PgHistorySink(pool), web.run_history_settings)
    llm = RecordingLlm(AnthropicLlm(settings.anthropic_api_key, guard, timeout_s=settings.llm_timeout_s), history)
    reasons: list[str] = []
    if not seo.configured:
        reasons.append("no seo-agent set up (SEO_AGENT_URL, SEO_AGENT_NEWS_TOKEN): no site search, and LLM calls are not checked against the one LLM budget")
    store = PgRunStore(pool, OWNED_FLOWS)
    deps = Deps(settings=settings, llm=llm, web=web, retriever=RemoteRetriever(seo), store=store, history=history)
    _news(deps, PgNewsStore(pool), make_embeddings(settings))
    runner = FlowRunner(SPECS, store, PgAdvisoryLocks(settings.database_url), lambda: PgFlowPersistence().bind(sync_pool), deps, timeout_s=settings.flow_timeout_s, policy=check_kickoff)
    deps.runner = runner
    return Container(settings=settings, deps=deps, runner=runner, store=store, guard=guard, pool=pool, sync_pool=sync_pool, reasons=reasons, history=history, schedules=PgScheduleStore(pool), seo_agent=seo)


def memory_container(settings: Settings, *, llm: Any, web: Any, retriever: Any = None, history: bool = False) -> Container:
    """Everything in memory (tests): fake LLM and web app, in-memory runs/locks/persistence. `history=True` records
    the run history in a MemoryHistorySink (container.history.sink) and wraps the model with RecordingLlm."""
    store = MemoryRunStore(flows=OWNED_FLOWS)
    run_history = RunHistory(MemoryHistorySink(runs=store.rows), getattr(web, "run_history_settings", None)) if history else None
    if run_history is not None:
        llm = RecordingLlm(llm, run_history)

    async def no_sleep(_s: float) -> None:
        return None

    deps = Deps(settings=settings, llm=llm, web=web, retriever=retriever, store=store, sleep=no_sleep, history=run_history)
    _news(deps, MemoryNewsStore(), HashEmbeddings())
    runner = FlowRunner(SPECS, store, MemoryLocks(), lambda: MemoryFlowPersistence().bind(store), deps, timeout_s=settings.flow_timeout_s, policy=check_kickoff)
    deps.runner = runner
    guard = getattr(llm, "guard", None) or BudgetGuard(MemorySpendStore())
    return Container(settings=settings, deps=deps, runner=runner, store=store, guard=guard, ready=True, history=run_history)


async def _news_gauges(conn: Any) -> None:
    """Restart-safe alert inputs for the news desk (like D-18): computed from the tables, not process counters."""
    try:
        cur = await conn.execute(
            """select j.id::text as job,
                 (select count(*) from (select status from news_runs r where r.job_id = j.id and r.kind = 'job' and not r.dry_run and r.status not in ('queued', 'running') order by started_at desc limit 5) x
                  where x.status = 'failed')::int as recent_failed,
                 (select status from news_runs r where r.job_id = j.id and r.kind = 'job' and not r.dry_run and r.status not in ('queued', 'running') order by started_at desc limit 1) as last_status,
                 (select status from news_runs r where r.job_id = j.id and r.kind = 'job' and not r.dry_run and r.status not in ('queued', 'running') order by started_at desc offset 1 limit 1) as prev_status,
                 extract(epoch from (select max(created_at) from news_articles a where a.job_id = j.id and a.status in ('ready_for_review', 'published')))::float8 as last_draft
               from news_jobs j where j.deleted_at is null and j.enabled limit 25"""
        )
        for r in await cur.fetchall():
            consecutive = 2 if r["last_status"] == "failed" and r["prev_status"] == "failed" else (1 if r["last_status"] == "failed" else 0)
            metrics.NEWS_JOB_CONSECUTIVE_FAILURES.labels(news_job=r["job"]).set(consecutive)
            metrics.NEWS_JOB_LAST_DRAFT.labels(news_job=r["job"]).set(float(r["last_draft"] or 0))
        cur = await conn.execute(
            "select count(*) filter (where status = 'failed_quality')::float8 / nullif(count(*) filter (where status in ('failed_quality', 'ready_for_review', 'published', 'rejected', 'incomplete')), 0) as ratio from news_articles where created_at > now() - interval '7 days'"
        )
        row = await cur.fetchone()
        metrics.NEWS_QUALITY_FAILURE_RATIO.set(float(row["ratio"] or 0))
        cur = await conn.execute(
            "select coalesce(max(extract(epoch from (now() - coalesce(last_success_at, created_at)))), 0)::float8 as s from news_sources where enabled and deleted_at is null and consecutive_errors > 0"
        )
        metrics.NEWS_SOURCE_FAILING_SINCE.set(float((await cur.fetchone())["s"] or 0))
    except Exception:  # noqa: S110 - the news tables may not exist yet (migration 0017 not applied)
        pass


async def _gauges(c: Container) -> None:
    while True:
        try:
            if c.pool is not None:
                async with c.pool.connection() as conn:
                    cur = await conn.execute(
                        "select flow_name, extract(epoch from max(finished_at))::float8 as t from seo_flow_runs where status = 'succeeded' and flow_name = any(%s) group by flow_name",
                        (sorted(OWNED_FLOWS),),
                    )
                    for r in await cur.fetchall():
                        metrics.LAST_SUCCESS.labels(flow=metrics.flow_label(r["flow_name"])).set(float(r["t"] or 0))
                    await _news_gauges(conn)
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
    if c.history is not None:  # started here, outside any run, so the writer's own log lines never enter a run's log
        install_run_log(c.history)
        c.bg.append(asyncio.create_task(c.history.run_forever()))


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
            await _shutdown(c)

    app = FastAPI(title="news-agent", version=VERSION, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

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
        return {"status": "ok", "version": VERSION, "agent": "news"}

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
        return JSONResponse({"status": "ready", "site_search": bool(getattr(c.deps.retriever, "available", False)), "notes": c.reasons})

    @app.get("/metrics")
    async def metrics_ep() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(metrics.REGISTRY).decode(), media_type="text/plain; version=0.0.4")

    # --- CrewAI console (D-98): the flow catalogue, Run with parameters, schedules, crew configuration ---

    @app.get("/flows", dependencies=[Depends(auth)])
    async def flows() -> dict[str, Any]:
        return {"items": flow_catalogue()}

    @app.post("/flows/{flow_name}/validate", dependencies=[Depends(auth)])
    async def validate_inputs(flow_name: str, body: FlowInputs) -> Any:
        spec = SPECS.get(flow_name)
        if spec is None:
            raise HTTPException(404, f"unknown flow {flow_name}")
        try:
            return {"inputs": spec.input_model.model_validate(body.inputs).model_dump(exclude_none=True)}
        except ValueError as e:
            return JSONResponse({"error": f"invalid inputs for {flow_name}: {str(e)[:500]}"}, status_code=422)

    @app.post("/schedules/dispatch", dependencies=[Depends(auth)])
    async def dispatch(request: Request) -> JSONResponse:
        from datetime import timedelta

        c = ctr(request)
        try:
            out = await dispatch_schedules(c.schedules, c.runner, schedulable=CRON_FLOWS, served=OWNED_FLOWS, now=c.clock() if c.clock else None, catchup=timedelta(minutes=c.settings.schedule_catchup_min))
        except SchedulesUnavailable as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        return JSONResponse(out)

    @app.get("/crews", dependencies=[Depends(auth)])
    async def crews() -> dict[str, Any]:
        return {"items": [crew_config.catalogue(c) for c in crew_config.registry().values()]}

    @app.post("/crews/{crew}/validate", dependencies=[Depends(auth)])
    async def validate_crew(crew: str, body: CrewConfigBody) -> JSONResponse:
        c = crew_config.registry().get(crew)
        if c is None:
            raise HTTPException(404, f"unknown crew {crew}")
        try:
            ov = crew_config.validate(c, body.config)
        except crew_config.OverlayError as e:
            return JSONResponse({"error": "the crew configuration is not valid", "issues": e.issues}, status_code=422)
        return JSONResponse({"config": crew_config.normalize(c, ov), "default_hash": crew_config.default_hash(c), "prompt_version": c.prompt_version})

    @app.post("/flows/{flow_name}/kickoff", dependencies=[Depends(auth)])
    async def kickoff(flow_name: str, body: Kickoff, request: Request) -> JSONResponse:
        r = await ctr(request).runner.kickoff(flow_name, body.inputs, triggered_by=body.triggered_by, actor=body.actor, correlation_id=request.headers.get("x-correlation-id"))
        return JSONResponse(r, status_code=200 if r.get("existing") else 202)

    @app.get("/flows/runs", dependencies=[Depends(auth)])
    async def list_runs(request: Request, flow: str | None = None, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        rows = await ctr(request).store.list_runs(flow=flow, status=status, limit=limit)
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

    return app


async def _shutdown(c: Container) -> None:
    for t in c.bg:
        t.cancel()
    await _close_history(c)
    if c.seo_agent is not None:
        await c.seo_agent.close()
    if c.pool is not None:
        await c.pool.close()
    if c.sync_pool is not None:
        c.sync_pool.close()


async def serve() -> None:
    import uvicorn

    settings = load_settings()
    if settings.production and (len(settings.agent_token) < 32 or len(settings.callback_token) < 32):
        raise RuntimeError("NEWS_AGENT_TOKEN and NEWS_AGENT_CALLBACK_TOKEN must be set (≥ 32 characters)")
    setup_tracing(settings.otel_endpoint, settings.service_name, VERSION)
    c = await build_container(settings)
    await startup(c)
    server = uvicorn.Server(uvicorn.Config(create_app(c), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), proxy_headers=False, access_log=False, lifespan="on"))  # noqa: S104 - ClusterIP
    try:
        await server.serve()
    finally:
        await _shutdown(c)


async def _close_history(c: Container) -> None:
    """Shutdown: stop copying records into run logs and write what is still queued."""
    if c.history is None:
        return
    uninstall_run_log()
    try:
        await asyncio.wait_for(c.history.close(), 10)
    except Exception:  # noqa: S110 - best effort at shutdown
        pass


def run() -> None:
    configure_logging()
    asyncio.run(serve())


if __name__ == "__main__":
    run()
