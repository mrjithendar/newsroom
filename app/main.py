"""seo-agent: CrewAI Flows behind a small internal API, plus the MCP edge (app/mcp) on a second port.

Internal API (port 8080, ClusterIP; NetworkPolicy admits only the web pods, the trigger CronJobs and Prometheus).
Every endpoint except /healthz, /readyz and /metrics needs `Authorization: Bearer $SEO_AGENT_TOKEN`:

    GET  /flows                         every flow: input JSON Schema, schedulable, crews (CrewAI console)
    POST /flows/{flow_name}/validate    {inputs} → {inputs} as the flow would get them, or 422
    POST /flows/{flow_name}/kickoff     {inputs, triggered_by: cron|manual, actor?} → {run_id}
    POST /schedules/dispatch            start every schedule that is due (one CronJob, every minute; app/schedules.py)
    GET  /crews                         each crew's editable configuration as the code defines it (app/crews/config.py)
    POST /crews/{crew}/validate         {config} → {config} as it would be stored (only what differs), or 422
    GET  /flows/runs                    ?flow=&status=&draft_id=&limit=
    GET  /flows/runs/{run_id}           status, current step, outputs, errors, cost
    POST /flows/runs/{run_id}/cancel
    POST /review/{draft_id}/decision    {approve|reject|regenerate, approval_id, accepted_change_ids, edits, feedback}
    POST /ask                           public RAG, proxied by the web app (rate-limited there)
    GET  /budget
    GET  /llm-budget?scope=day|month    the one LLM budget: spent, limit, remaining, forecast, spend by provider, model, purpose
    PUT  /llm-budget                    {daily_usd, monthly_usd, actor} → the budget, or 422 (app/llm_budgets.py)
    POST /llm-budget/reserve            {max_usd, purpose} → {id}: hold a call's upper bound against the budget, or 402
    POST /llm-budget/settle             {id} → let go of a hold (the caller has recorded its usage row)
    GET  /llm-budgets                   before the one budget: spend by model and provider, no limits (older web app)
    PUT  /llm-budgets                   409: replaced by /llm-budget
    GET  /llm-providers                 provider kinds, saved providers and their models; a key is never returned
    PUT  /llm-providers                 one provider (+ its models), validated, all or nothing (app/llm_providers)
    POST /llm-providers/{id}/discover   "Fetch models": the provider's own model list; nothing is saved or switched on
    GET  /llm-models                    models that are switched on, of providers that are on (for pickers)
    GET  /admin-settings                settings an admin may change: schema, effective values, source, restart pending
    PUT  /admin-settings/{group}        {values, actor} → one group, validated, all or nothing (app/admin_settings)
    GET  /healthz /readyz /metrics

publish_flow cannot be started through /flows/{name}/kickoff (403): only /review/{id}/decision with an approval.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import metrics
from .admin_settings import MemorySettingsStore, PgSettingsStore, SettingsInvalid, UnknownGroup
from .admin_settings import overview as settings_overview
from .admin_settings import save_group as save_settings_group
from .admin_settings.store import REFRESH_SECONDS as SETTINGS_REFRESH_SECONDS
from .admin_settings.store import apply_live as apply_live_settings
from .admin_settings.store import effective as effective_settings
from .ask import AskService, MemoryAskLog, PgAskLog
from .ask_runtime import PgAskRuntime, StaticAskRuntime
from .config import Settings, load_settings
from .crews import config as crew_config
from .crews.editorial import EditorialCrew
from .crews.optimization import OptimizationCrew
from .crews.promotion import PromotionCrew
from .crews.report import ReportCrew
from .embeddings import HashEmbeddings
from .flows import CRON_FLOWS, SPECS, check_kickoff
from .flows import catalogue as flow_catalogue
from .flows.base import Deps
from .index import MemoryRetriever, PgVectorIndex
from .lance_admin import LanceAdmin, LanceError
from .lance_admin import from_env as lance_from_env
from .lance_index import ArticleMirror, fetch_from
from .llm import AnthropicLlm, BudgetExceeded, BudgetGuard, MemorySpendStore, PgSpendStore
from .llm_budgets import REPLACED, BudgetIn, MemoryBudget, PgBudget, legacy_overview
from .llm_budgets import overview as llm_overview
from .llm_providers import Invalid as ProviderInvalid
from .llm_providers import MemoryProviderStore, PgProviderStore, ProviderIn, check_catalog, enabled_models, save_provider
from .llm_providers import overview as providers_overview
from .llm_providers.discover import CannotDiscover
from .llm_providers.discover import discover as discover_models
from .logs import configure_logging
from .news.store import MemoryNewsStore, PgNewsStore
from .run_history import MemoryHistorySink, PgHistorySink, RecordingLlm, RunHistory
from .run_history import install as install_run_log
from .run_history import uninstall as uninstall_run_log
from .runs import FlowRunner, MemoryFlowPersistence, MemoryLocks, MemoryRunStore, PgAdvisoryLocks, PgFlowPersistence, PgRunStore, RunError, view
from .schedules import MemoryScheduleStore, PgScheduleStore, SchedulesUnavailable
from .schedules import dispatch as dispatch_schedules
from .schemas import AskRequest
from .seo_ml.store import MemoryMlStore, PgMlStore
from .services import review
from .services.ask_pipeline import AskPipeline
from .services.content import ContentError
from .tracing import setup_tracing
from .webapp import HttpWebApp

log = logging.getLogger("seo_agent")
VERSION = os.environ.get("GIT_SHA", "dev")


class LanceKey(BaseModel):
    """One row of a LanceDB table, by a scalar key column and its value (/admin/lancedb)."""

    model_config = ConfigDict(extra="forbid")
    key_column: str = Field(min_length=1, max_length=128)
    key: str | int | float | bool
    actor: str | None = Field(default=None, max_length=200)


class LanceUpdate(LanceKey):
    values: dict[str, str | int | float | bool | None] = Field(min_length=1, max_length=50)


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
    lance: LanceAdmin = field(default_factory=lambda: lance_from_env(dict(os.environ)))  # /admin/lancedb (D-92)
    mirror: ArticleMirror | None = None  # the article index copied into LanceDB (D-93; app/lance_index.py)
    history: RunHistory | None = None  # steps, log lines and model calls of every flow run (app/run_history.py)
    schedules: Any = field(default_factory=MemoryScheduleStore)  # seo_flow_schedules (app/schedules.py)
    clock: Callable[[], Any] | None = None  # tests: the dispatcher's "now"
    budgets: Any = field(default_factory=MemoryBudget)  # the one LLM budget (app/llm_budgets.py)
    admin_settings: Any = field(default_factory=MemorySettingsStore)  # values saved from the admin (app/admin_settings)
    providers: Any = field(default_factory=MemoryProviderStore)  # LLM providers and models (app/llm_providers)

    async def reindex(self, article_ids: list[str] | None = None, full: bool = False, force: bool = False) -> dict[str, int]:
        async with self.reindex_lock:
            if not getattr(self.index, "available", False) or not hasattr(self.index, "reindex"):
                return {"articles": 0, "embedded": 0, "unchanged": 0, "deleted": 0}
            articles = await self.deps.web.published_articles()
            stats = await self.index.reindex(articles, full=full, only=set(article_ids) if article_ids else None, force=force)
            if self.mirror is not None:
                stats["lancedb_synced"] = int(bool((await self.mirror.after_reindex(stats)).get("synced")))
            return stats

    async def initial_mirror(self, attempts: int = 30, every_s: float = 60.0, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        """Fill LanceDB's article_chunks once after startup if it is missing, as soon as storage is configured."""
        for _ in range(attempts):
            if self.mirror is None:
                return
            async with self.reindex_lock:
                try:
                    out = await self.mirror.sync(only_if_missing=True)
                except Exception as e:  # R2 unreachable: try again later
                    log.warning("lancedb article index: startup sync failed", extra={"event": "lancedb_index_failed", "error": type(e).__name__})
                    out = {"synced": False, "reason": "failed"}
            if out.get("synced") or out.get("reason", "").startswith(("already", "off", "no chunks")):
                return
            await sleep(every_s)


def _crews(deps: Deps) -> None:
    s, llm, web, ret = deps.settings, deps.llm, deps.web, deps.retriever
    deps.optimization_crew = deps.optimization_crew or (lambda: OptimizationCrew(s, llm, web, ret))
    deps.editorial_crew = deps.editorial_crew or (lambda: EditorialCrew(s, llm, web, ret))
    deps.report_crew = deps.report_crew or (lambda: ReportCrew(s, llm, web, ret))
    deps.promotion_crew = deps.promotion_crew or (lambda: PromotionCrew(s, llm, web, ret))


def _ml(deps: Deps, store: Any, embeddings: Any) -> None:
    """v5 SEO-side ML (app/flows/ml.py). Queries are embedded as documents (no query prefix): they are clustered and
    classified against each other, not retrieved against passages."""

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embeddings.embed(texts)

    deps.extra.setdefault("ml_store", store)
    deps.extra.setdefault("ml_embed", embed)
    deps.extra.setdefault("ml_embed_model", getattr(embeddings, "name", "unknown"))


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
    # Values an admin saved win over the environment (app/admin_settings). Read before anything below is built, so
    # components that copy a value at start-up (timeouts) get the saved one. The connection itself always comes
    # from the environment: none of its settings can be saved from the admin.
    admin_settings = PgSettingsStore(pool)
    settings_note: str | None = None
    try:
        await admin_settings.ensure_schema()
        settings = effective_settings(await admin_settings.load(fresh=True))
    except Exception as e:  # the role cannot create the table, or the database is down: the deployment's values
        settings_note = f"admin settings table not available: {type(e).__name__}"
        log.warning("admin settings not loaded", extra={"event": "admin_settings_load_failed"})
    budgets = PgBudget(pool)
    guard = BudgetGuard(PgSpendStore(pool), None, settings.ask_daily_budget_usd, budget=budgets)
    web = HttpWebApp(settings.callback_url, settings.callback_token)
    history = RunHistory(PgHistorySink(pool), web.run_history_settings)
    llm = RecordingLlm(AnthropicLlm(settings.anthropic_api_key, guard, timeout_s=settings.llm_timeout_s), history)
    index = PgVectorIndex(pool, make_embeddings(settings))
    reasons: list[str] = [settings_note] if settings_note else []
    try:
        await index.ensure_schema()
        if not index.available:
            reasons.append(index.reason)
    except Exception as e:
        reasons.append(f"vector index: {type(e).__name__}")
    store = PgRunStore(pool)
    deps = Deps(settings=settings, llm=llm, web=web, retriever=index, store=store, history=history)
    _crews(deps)
    _news(deps, PgNewsStore(pool), index.emb)
    _ml(deps, PgMlStore(pool), index.emb)
    runner = FlowRunner(SPECS, store, PgAdvisoryLocks(settings.database_url), lambda: PgFlowPersistence().bind(sync_pool), deps, timeout_s=settings.flow_timeout_s, policy=check_kickoff)
    deps.runner = runner
    # The admin's Ask the Blog settings (web app → `setting` row "ask_blog") apply on top of the environment's switch.
    ask = AskService(AskPipeline(deps), index, PgAskLog(pool), settings.ask_enabled, settings.ask_cache_ttl_s, runtime=PgAskRuntime(pool), guard=guard)
    c = Container(settings=settings, deps=deps, runner=runner, store=store, ask=ask, index=index, guard=guard, pool=pool, sync_pool=sync_pool, reasons=reasons, history=history, schedules=PgScheduleStore(pool), budgets=budgets, admin_settings=admin_settings, providers=PgProviderStore(pool))
    c.lance = lance_from_env(dict(os.environ), resolve=web.lancedb_storage)
    c.mirror = ArticleMirror(c.lance, fetch_from(pool), enabled=settings.lancedb_article_index)
    deps.reindex = c.reindex
    return c


def memory_container(settings: Settings, *, llm: Any, web: Any, retriever: Any = None, verify_delays: tuple[float, ...] = (0, 0, 0), history: bool = False, **crews: Any) -> Container:
    """Everything in memory (tests): fake LLM and web app, in-memory runs/locks/persistence. `history=True` records
    the run history in a MemoryHistorySink (container.history.sink) and wraps the model with RecordingLlm."""
    retriever = retriever or MemoryRetriever()
    store = MemoryRunStore()
    run_history = RunHistory(MemoryHistorySink(runs=store.rows), getattr(web, "run_history_settings", None)) if history else None
    if run_history is not None:
        llm = RecordingLlm(llm, run_history)

    async def no_sleep(_s: float) -> None:
        return None

    deps = Deps(settings=settings, llm=llm, web=web, retriever=retriever, store=store, sleep=no_sleep, verify_delays=verify_delays, history=run_history, **crews)
    _crews(deps)
    _news(deps, MemoryNewsStore(), HashEmbeddings())
    _ml(deps, MemoryMlStore(), HashEmbeddings())
    runner = FlowRunner(SPECS, store, MemoryLocks(), lambda: MemoryFlowPersistence().bind(store), deps, timeout_s=settings.flow_timeout_s, policy=check_kickoff)
    deps.runner = runner
    guard = getattr(llm, "guard", BudgetGuard(MemorySpendStore(), None, settings.ask_daily_budget_usd))
    ask = AskService(AskPipeline(deps), retriever, MemoryAskLog(), settings.ask_enabled, settings.ask_cache_ttl_s, runtime=StaticAskRuntime(), guard=guard)
    budgets = guard.budget if isinstance(guard.budget, MemoryBudget) else MemoryBudget()
    guard.budget = budgets
    c = Container(settings=settings, deps=deps, runner=runner, store=store, ask=ask, index=retriever, guard=guard, ready=True, history=run_history, budgets=budgets)
    deps.reindex = c.reindex
    return c


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


async def _ml_gauges(conn: Any) -> None:
    """seo_ml_flow failures in a row (restart-safe, from seo_flow_runs) and which models are active (seo_ml_models)."""
    try:
        cur = await conn.execute("select status from seo_flow_runs where flow_name = 'seo_ml_flow' and status in ('succeeded', 'failed') order by started_at desc limit 5")
        n = 0
        for r in await cur.fetchall():
            if r["status"] != "failed":
                break
            n += 1
        metrics.ML_FLOW_CONSECUTIVE_FAILURES.set(n)
        cur = await conn.execute("select kind from seo_ml_models where active")
        active = {r["kind"] for r in await cur.fetchall()}
        for k in metrics.ML_MODELS:
            metrics.ML_MODEL_ACTIVE.labels(model=k).set(1 if k in active else 0)
        cur = await conn.execute("select (metrics ->> 'recall@3')::float8 as r from seo_rag_eval_runs where config = 'live' order by at desc limit 1")
        row = await cur.fetchone()
        if row and row["r"] is not None:
            metrics.RAG_EVAL_RECALL3.set(float(row["r"]))
    except Exception:  # noqa: S110 - migration 0019 may not be applied yet
        pass


async def _gauges(c: Container) -> None:
    while True:
        try:
            st = await c.guard.status()
            # The one LLM budget: the month's share when a monthly limit is set, else the day's, else 0 (no limit).
            ratios = [st[f"{u}_spend_usd"] / lim for u, lim in (("month", st["monthly_budget_usd"]), ("day", st["daily_budget_usd"])) if lim]
            metrics.BUDGET_RATIO.set(ratios[0] if ratios else 0.0)
            ask_budget = getattr(c.guard, "ask_daily_usd", c.settings.ask_daily_budget_usd)  # the admin's budget once a question has applied it
            if ask_budget > 0:
                metrics.ASK_BUDGET_RATIO.set(st["ask_day_spend_usd"] / ask_budget)
            if c.pool is not None:
                async with c.pool.connection() as conn:
                    cur = await conn.execute("select count(*)::int as n from seo_drafts where status = 'ready_for_review'")
                    metrics.DRAFTS_AWAITING.set(int((await cur.fetchone())["n"]))
                    cur = await conn.execute("select flow_name, extract(epoch from max(finished_at))::float8 as t from seo_flow_runs where status = 'succeeded' group by flow_name")
                    for r in await cur.fetchall():
                        metrics.LAST_SUCCESS.labels(flow=metrics.flow_label(r["flow_name"])).set(float(r["t"] or 0))
                    await _news_gauges(conn)
                    await _ml_gauges(conn)
            if isinstance(c.ask.log, PgAskLog):
                await c.ask.log.prune(c.settings.ask_retention_days)
        except Exception:  # noqa: S110 - gauges are best effort
            pass
        await asyncio.sleep(60)


async def _settings_refresh(c: Container) -> None:
    """Live settings saved through another replica (or straight in the table) reach this process within one period."""
    while True:
        await asyncio.sleep(SETTINGS_REFRESH_SECONDS)
        try:
            changed = apply_live_settings(c.settings, await c.admin_settings.load())
            if changed:
                log.info("admin settings applied", extra={"event": "admin_settings_applied", "settings": changed})
        except Exception:  # noqa: S110 - the next period tries again; the running values stay
            pass


async def startup(c: Container) -> None:
    mode = getattr(getattr(c.index, "config", None), "mode", "vector")
    for m in metrics.RAG_MODES:
        metrics.RAG_RETRIEVAL_MODE.labels(mode=m).set(1 if m == mode else 0)
    check_catalog()
    try:
        await c.providers.ensure_schema()
    except Exception as e:  # the role cannot create it, or the database is down: providers cannot be saved until it can
        c.reasons.append(f"LLM providers tables not available: {type(e).__name__}")
        log.warning("llm providers schema failed", extra={"event": "llm_providers_schema_failed"})
    try:
        await c.budgets.ensure_schema()
    except Exception as e:  # the role cannot create it, or the database is down: no LLM budgets until it can
        c.reasons.append(f"LLM budgets table not available: {type(e).__name__}")
        log.warning("llm budgets schema failed", extra={"event": "llm_budgets_schema_failed"})
    try:
        await c.runner.recover()
    except Exception as e:  # e.g. the web app's migration 0016 not applied yet: start, report on /readyz
        c.reasons.append(f"run recovery failed: {type(e).__name__} (is migration 0016 applied?)")
        log.warning("run recovery failed", extra={"event": "recover_failed"})
    c.ready = True
    c.bg.append(asyncio.create_task(_gauges(c)))
    c.bg.append(asyncio.create_task(_settings_refresh(c)))
    if c.history is not None:  # started here, outside any run, so the writer's own log lines never enter a run's log
        install_run_log(c.history)
        c.bg.append(asyncio.create_task(c.history.run_forever()))
    if c.mirror is not None:
        c.bg.append(asyncio.create_task(c.initial_mirror()))


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
            await _close_history(c)
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

    @app.exception_handler(LanceError)
    async def lance_error(_r: Request, e: LanceError) -> JSONResponse:
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
            out = await dispatch_schedules(c.schedules, c.runner, schedulable=CRON_FLOWS, now=c.clock() if c.clock else None, catchup=timedelta(minutes=c.settings.schedule_catchup_min))
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
        history = [t.model_dump() for t in body.history]
        if body.stream:
            return StreamingResponse(c.ask.stream(body.question, body.ip_hash, history, body.context_slug), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        return await c.ask.ask(body.question, body.ip_hash, history=history, context_slug=body.context_slug)

    # --- LanceDB browser for /admin/lancedb (D-92; app/lance_admin.py). One row per write, scalar columns only. ---

    @app.get("/lancedb", dependencies=[Depends(auth)])
    async def lance_overview(request: Request) -> dict[str, Any]:
        return await ctr(request).lance.run("overview")

    @app.get("/lancedb/tables/{name}", dependencies=[Depends(auth)])
    async def lance_browse(name: str, request: Request, offset: int = 0, limit: int = 25) -> dict[str, Any]:
        return await ctr(request).lance.run("browse", name, offset, limit)

    @app.post("/lancedb/tables/{name}/row", dependencies=[Depends(auth)])
    async def lance_row(name: str, body: LanceKey, request: Request) -> dict[str, Any]:
        return await ctr(request).lance.run("row", name, body.key_column, body.key)

    @app.post("/lancedb/tables/{name}/update", dependencies=[Depends(auth)])
    async def lance_update(name: str, body: LanceUpdate, request: Request) -> dict[str, Any]:
        return await ctr(request).lance.run("update", name, body.key_column, body.key, body.values, actor=body.actor)

    @app.post("/lancedb/tables/{name}/delete", dependencies=[Depends(auth)])
    async def lance_delete(name: str, body: LanceKey, request: Request) -> dict[str, Any]:
        return await ctr(request).lance.run("delete", name, body.key_column, body.key, actor=body.actor)

    @app.get("/budget", dependencies=[Depends(auth)])
    async def budget(request: Request) -> dict[str, Any]:
        return await ctr(request).guard.status()

    @app.get("/llm-budget", dependencies=[Depends(auth)])
    async def llm_budget(request: Request, scope: str = "month") -> Any:
        if scope not in ("day", "month"):
            return JSONResponse({"error": "scope is day or month"}, status_code=422)
        c = ctr(request)
        out = await llm_overview(await c.budgets.budget(), c.guard.store, c.guard.clock(), scope, c.guard.pending["total"])  # type: ignore[arg-type]
        now = c.guard.clock()
        out["ask_daily"] = {"limit_usd": c.guard.ask_daily_usd, "spent_usd": round(await c.guard.store.day_spend(now, "ask"), 6)}
        out["legacy_limits"] = await c.budgets.legacy()
        return out

    @app.put("/llm-budget", dependencies=[Depends(auth)])
    async def save_llm_budget(request: Request) -> Any:
        try:
            body = BudgetIn.model_validate(await request.json())
        except ValidationError as e:
            issues = [f"{'.'.join(str(p) for p in err['loc']) or 'body'}: {err['msg']}" for err in e.errors()][:30]
            return JSONResponse({"error": "the LLM budget is not valid", "issues": issues}, status_code=422)
        except ValueError:
            return JSONResponse({"error": "send a JSON object"}, status_code=422)
        c = ctr(request)
        saved = await c.budgets.save(body.budget(), body.actor)
        log.info("llm budget saved", extra={"event": "llm_budget_saved", "daily_usd": saved.daily_usd, "monthly_usd": saved.monthly_usd})
        return {"budget": saved.out()}

    @app.post("/llm-budget/reserve", dependencies=[Depends(auth)])
    async def reserve_llm_budget(request: Request) -> Any:
        """For LLM calls made outside this service (the web app's own): check the one budget and hold the call's
        upper bound until it is settled. 402 with the reason when it does not fit; the caller then does not call."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        asked = body if isinstance(body, dict) else {}
        max_usd, purpose = asked.get("max_usd"), asked.get("purpose")
        if isinstance(max_usd, bool) or not isinstance(max_usd, (int, float)) or not math.isfinite(max_usd) or not 0 <= max_usd <= 1000:
            return JSONResponse({"error": "max_usd is the call's upper bound in USD (0 to 1000)"}, status_code=422)
        if not isinstance(purpose, str) or not re.fullmatch(r"[a-z][a-z0-9_.\-]{1,60}", purpose):
            return JSONResponse({"error": "purpose names what the call is for (lower-case, e.g. ai_visibility)"}, status_code=422)
        try:
            return {"id": await ctr(request).guard.hold(float(max_usd), purpose)}
        except BudgetExceeded as e:
            return JSONResponse({"error": e.detail, "scope": e.scope}, status_code=402)

    @app.post("/llm-budget/settle", dependencies=[Depends(auth)])
    async def settle_llm_budget(request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            body = None
        key = body.get("id") if isinstance(body, dict) else None
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}", key):
            return JSONResponse({"error": "id is the hold's id from /llm-budget/reserve"}, status_code=422)
        return {"released": ctr(request).guard.settle(key)}

    @app.get("/llm-budgets", dependencies=[Depends(auth)])
    async def llm_budgets(request: Request) -> dict[str, Any]:
        c = ctr(request)
        return await legacy_overview(c.guard.store, c.guard.clock())

    @app.put("/llm-budgets", dependencies=[Depends(auth)])
    async def save_llm_budgets() -> Any:
        return JSONResponse({"error": REPLACED}, status_code=409)

    @app.get("/llm-providers", dependencies=[Depends(auth)])
    async def llm_providers(request: Request) -> dict[str, Any]:
        return await providers_overview(ctr(request).providers)

    @app.put("/llm-providers", dependencies=[Depends(auth)])
    async def save_llm_provider(request: Request) -> Any:
        """One provider and, when `models` is sent, all of its models. The key is write-only: it is sealed, never
        logged and never returned."""
        try:
            body = ProviderIn.model_validate(await request.json())
        except ValidationError as e:
            # Messages only, never the input: a rejected body may hold a key.
            issues = [f"{'.'.join(str(p) for p in err['loc']) or 'body'}: {err['msg']}" for err in e.errors(include_input=False)][:30]
            return JSONResponse({"error": "the provider is not valid", "issues": issues}, status_code=422)
        except ValueError:
            return JSONResponse({"error": "send a JSON object"}, status_code=422)
        c = ctr(request)
        try:
            changed = await save_provider(c.providers, body)
        except ProviderInvalid as e:
            return JSONResponse({"error": "the provider is not valid", "issues": e.issues[:30]}, status_code=422)
        log.info("llm provider saved", extra={"event": "llm_provider_saved", "provider": changed["id"], "enabled": changed["enabled"], "key_changed": changed["key_changed"], "models": changed["models"]})
        return {"saved": changed, **(await providers_overview(c.providers))}

    @app.post("/llm-providers/{provider_id}/discover", dependencies=[Depends(auth)])
    async def discover_llm_models(provider_id: str, request: Request) -> Any:
        try:
            return await discover_models(ctr(request).providers, provider_id)
        except LookupError:
            return JSONResponse({"error": "no such provider"}, status_code=404)
        except CannotDiscover as e:
            return JSONResponse({"error": str(e)}, status_code=e.status)

    @app.get("/llm-models", dependencies=[Depends(auth)])
    async def llm_models(request: Request) -> dict[str, Any]:
        return {"models": await enabled_models(ctr(request).providers)}

    @app.get("/admin-settings", dependencies=[Depends(auth)])
    async def admin_settings(request: Request) -> dict[str, Any]:
        c = ctr(request)
        return settings_overview(await c.admin_settings.load(fresh=True), c.settings)

    @app.put("/admin-settings/{group}", dependencies=[Depends(auth)])
    async def save_admin_settings(group: str, request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse({"error": "send a JSON object"}, status_code=422)
        c = ctr(request)
        try:
            return await save_settings_group(c.admin_settings, c.settings, group, body.get("values"), body.get("actor"))
        except UnknownGroup:
            return JSONResponse({"error": "no such settings group"}, status_code=404)
        except SettingsInvalid as e:
            return JSONResponse({"error": "the settings are not valid", "issues": e.issues}, status_code=422)

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
        await _close_history(c)
        await c.pool.close()
        c.sync_pool.close()


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
