"""News desk flows (docs/news/ARCHITECTURE.md). CrewAI Flows are the only orchestrator; ML is plain Python inside steps.

    news_dispatch_flow   CronJob every 15 min → which jobs are due (cron + time zone) → news_job_flow per due job;
                         daily housekeeping (retention, performance feedback, gauges)
    news_job_flow        ingest (shared) → select (job) → news_draft_flow × selected (concurrency 2) → notify
    news_draft_flow      one story or digest: crew → quality gates → web validation → cover → draft in review
    news_ingest_flow     "fetch now" / test sources outside a job
    news_image_flow      regenerate a draft's cover from the review page
    news_train_flow      weekly: classifier + ranker (promoted only when better)
    news_style_review_flow weekly: proposes a writing-profile version from the owner's edits (never applied)
    news_playground_flow profile studio A/B on a recent story; saves nothing unless the owner clicks "Save as draft"

None of these can publish: there is no call to publish_flow or the web app's publish callback anywhere below
(tests/test_news_flows.py checks this statically and at runtime).
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from crewai.flow import persist, start
from pydantic import BaseModel, ConfigDict, Field

from .. import metrics
from ..llm import BudgetExceeded
from ..news import desk as nd
from ..news.budget import NewsBudget, NewsLlm
from ..news.config import Job, load_news_settings
from ..news.cron import Cron, CronError
from ..news.runlog import RunRecorder
from ..runs import INSTANCE_ONLY
from ..state import UUID, RunState, _In
from .base import SeoFlow

# --- inputs ---------------------------------------------------------------------------------------------------------


class NewsJobInput(_In):
    job_id: str = Field(pattern=UUID)
    slot: str | None = Field(default=None, max_length=80, pattern=r"^[0-9A-Za-z:_.+-]+$")
    dry_run: bool = False
    llm: bool = True
    trigger: Literal["cron", "manual", "dry_run", "mcp", "retry"] = "manual"
    retry_of: str | None = Field(default=None, pattern=UUID)
    actor: str | None = Field(default=None, max_length=200)


class NewsDraftInput(_In):
    job_id: str = Field(pattern=UUID)
    cluster_ids: list[str] = Field(min_length=1, max_length=12)
    mode: Literal["single", "digest"] = "single"
    news_run_id: str = Field(pattern=UUID)
    dry_run: bool = False
    llm: bool = True
    score: float | None = None
    feedback: str | None = Field(default=None, max_length=2000)
    writing_profile_version: int | None = Field(default=None, ge=1)


class NewsDispatchInput(_In):
    slot: str | None = Field(default=None, max_length=40, pattern=r"^[0-9A-Za-z:_-]+$")


class NewsIngestInput(_In):
    source_ids: list[str] = Field(default_factory=list, max_length=200)
    force: bool = True
    actor: str | None = Field(default=None, max_length=200)


class NewsImageInput(_In):
    article_id: str = Field(pattern=UUID)
    prompt_tweak: str = Field(default="", max_length=400)
    provider: Literal["template", "gemini", "openai"] | None = None
    actor: str | None = Field(default=None, max_length=200)


class NewsTrainInput(_In):
    bootstrap: bool = False


class NewsStyleInput(_In):
    profile_key: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{1,40}$")
    days: int = Field(default=7, ge=1, le=60)


class NewsPlaygroundInput(_In):
    job_id: str = Field(pattern=UUID)
    cluster_id: str = Field(pattern=UUID)
    profile_key: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,40}$")
    version_a: int = Field(ge=1)
    version_b: int | None = Field(default=None, ge=1)


# --- states ---------------------------------------------------------------------------------------------------------


class NewsState(RunState):
    model_config = ConfigDict(extra="ignore")
    news_run_id: str = ""
    job_id: str = ""
    counters: dict[str, int] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    children: list[dict[str, Any]] = Field(default_factory=list)


class NewsJobState(NewsState):
    slot: str | None = None
    dry_run: bool = False
    llm: bool = True
    trigger: str = "manual"
    retry_of: str | None = None
    actor: str | None = None
    selected: list[dict[str, Any]] = Field(default_factory=list)


class NewsDraftState(NewsState):
    cluster_ids: list[str] = Field(default_factory=list)
    mode: str = "single"
    dry_run: bool = False
    llm: bool = True
    score: float | None = None
    feedback: str | None = None
    writing_profile_version: int | None = None


class NewsDispatchState(NewsState):
    slot: str | None = None


class NewsIngestState(NewsState):
    source_ids: list[str] = Field(default_factory=list)
    force: bool = True
    actor: str | None = None


class NewsImageState(NewsState):
    article_id: str = ""
    prompt_tweak: str = ""
    provider: str | None = None
    actor: str | None = None


class NewsTrainState(NewsState):
    bootstrap: bool = False


class NewsStyleState(NewsState):
    profile_key: str | None = None
    days: int = 7


class NewsPlaygroundState(NewsState):
    cluster_id: str = ""
    profile_key: str = ""
    version_a: int = 1
    version_b: int | None = None


# --- base -----------------------------------------------------------------------------------------------------------


class NewsHelpers:
    """Shared by every news flow (a mixin: CrewAI reads each flow's state type from its SeoFlow[State] parameter).
    Uses `self.state` and `self.deps` from SeoFlow."""

    if TYPE_CHECKING:
        state: Any
        deps: Any

    async def desk(self) -> nd.DeskDeps:
        x = self.deps.extra
        store = x["news_store"]
        news = load_news_settings(await store.settings_row(), x.get("env"))
        return nd.DeskDeps(store=store, web=self.deps.web, llm=self.deps.llm, embed=x["news_embed"], retriever=self.deps.retriever, settings=self.deps.settings, news=news, clock=x.get("clock") or (lambda: datetime.now(UTC)), crew_factory=x.get("news_crew"), extra=x)

    async def open_run(self, d: nd.DeskDeps, *, kind: str, job: Job | None, trigger: str, slot: str | None = None, dry_run: bool = False, llm: bool = True, parent: str | None = None, actor: str | None = None) -> tuple[RunRecorder, bool]:
        rid = self.state.id
        row, created = await d.store.create_run({"id": rid, "job_id": job.id if job else None, "job_version": job.version if job else None, "kind": kind, "flow_run_id": rid, "parent_run_id": parent, "trigger": trigger, "slot": slot, "status": "running", "dry_run": dry_run, "llm_enabled": llm, "correlation_id": rid, "actor": actor})
        self.state.news_run_id = str(row.get("id") or rid)
        return RunRecorder(d.store, self.state.news_run_id), created

    async def close_run(self, d: nd.DeskDeps, rec: RunRecorder, status: str, error: str | None = None, job: Job | None = None, started: float | None = None) -> None:
        rec.counters.update(self.state.counters)
        await rec.flush()
        run = await d.store.get_run(rec.run_id) or {}
        await d.store.update_run(rec.run_id, status=status, finished_at=d.clock(), counters=rec.counters, error_summary=(error or "")[:1000] or None)
        self.state.counters = rec.counters
        self.state.add_cost(0)  # cost lives on news_runs (summing seo_flow_runs too would double count)
        metrics.NEWS_COST.labels(kind="llm").inc(float(run.get("cost_usd") or 0))
        if job:
            metrics.NEWS_RUNS.labels(news_job=job.id[:36], status=status).inc()
            if started is not None:
                metrics.NEWS_RUN_DURATION.labels(news_job=job.id[:36]).observe(time.monotonic() - started)


# --- news_job_flow --------------------------------------------------------------------------------------------------


@persist(INSTANCE_ONLY)
class NewsJobFlow(NewsHelpers, SeoFlow[NewsJobState]):
    FLOW: ClassVar[str] = "news_job_flow"

    @start()
    async def run_job(self) -> None:
        s: NewsJobState = self.state
        started = time.monotonic()
        d = await self.desk()
        row = await d.store.get_job(s.job_id)
        if not row:
            s.outcome, s.errors = "failed", [*s.errors, "job not found"]
            return
        job = nd.as_job(row)
        trigger = "dry_run" if s.dry_run and s.trigger == "manual" else s.trigger
        rec, created = await self.open_run(d, kind="job", job=job, trigger=trigger, slot=s.slot, dry_run=s.dry_run, llm=s.llm, parent=s.retry_of, actor=s.actor)
        if not created:
            s.outcome = "skipped"
            s.result = {"reason": f"slot {s.slot} already ran (run {s.news_run_id})"}
            return
        if not d.news.enabled:
            rec.log("warning", "news desk is switched off (Newsroom → Settings, or NEWS_DESK_ENABLED=false in the deployment); nothing ran", outcome="disabled")
            await self.close_run(d, rec, "skipped", "news desk disabled", job, started)
            s.outcome = "skipped"
            return
        if s.trigger == "cron" and not job.enabled:
            await self.close_run(d, rec, "skipped", "job is paused", job, started)
            s.outcome = "skipped"
            return
        try:
            budget = NewsBudget(d.store, d.news, job, rec.run_id, d.clock, s.dry_run)
            ratios = await budget.ratios()
            for k, v in ratios.items():
                metrics.NEWS_BUDGET_RATIO.labels(scope=k).set(v)
            if not s.dry_run and (ratios.get("job_llm_day", 0) >= 1 or ratios.get("global_llm_day", 0) >= 1 or ratios.get("job_llm_month", 0) >= 1 or ratios.get("global_llm_month", 0) >= 1):
                raise BudgetExceeded("news", "an LLM budget cap is already reached")
            ingest_lock: asyncio.Lock = d.extra.setdefault("news_ingest_lock", asyncio.Lock())
            async with ingest_lock:  # ingestion is shared: one at a time; a source fetched within its interval is skipped
                # Ingest's Claude topic fallback is shared work: capped by the global news budget, booked to this run.
                ing_d = nd.with_news_llm(d, [rec.run_id], job=None, dry_run=s.dry_run, enabled=s.llm and not s.dry_run)
                ing = await nd.ingest(ing_d, rec, source_ids=job.config.source_ids, llm_ok=s.llm and not s.dry_run)
            if s.retry_of:
                selected = await self._retry_selection(d, rec, job)
            else:
                selected = await nd.select_for_job(d, rec, job, dry_run=s.dry_run)
            s.selected = [{"cluster_id": c.cluster_id, "score": c.score, "rank": c.rank, "title": c.title} for c in selected]
            singles = [c for c in selected if (c.rank or 0) < 100] if job.mode != "daily_digest" else []
            digest = [c for c in selected if (c.rank or 0) >= 100] if job.mode == "both" else (selected if job.mode == "daily_digest" else [])
            work = [{"cluster_ids": [c.cluster_id], "mode": "single", "score": c.score} for c in singles]
            if digest:
                work.append({"cluster_ids": [c.cluster_id for c in digest][: job.config.digest_size], "mode": "digest", "score": max(c.score for c in digest)})
            s.children = await self._fan_out(d, job, work)
            for ch in s.children:
                rec.count(ch.get("outcome") or "failed")
            rec.count("drafted", sum(1 for ch in s.children if ch.get("outcome") == "ready_for_review"))
            rec.count("failed", sum(1 for ch in s.children if ch.get("outcome") in ("failed", None)))
            await self._notify(d, rec, job)
            status = "paused_budget" if any(ch.get("outcome") == "paused_budget" for ch in s.children) else ("failed" if s.children and all(ch.get("outcome") in ("failed", None) for ch in s.children) else "succeeded")
            await self.close_run(d, rec, status, "; ".join(s.errors[-3:]) or None, job, started)
            await d.store.set_job_schedule(job.id, last_run_at=d.clock())
            s.result = {"new_items": ing["new"], "selected": len(selected), "children": len(s.children)}
            s.outcome = "paused_budget" if status == "paused_budget" else ("failed" if status == "failed" else "done")
        except BudgetExceeded as e:
            rec.log("warning", f"paused: {e.detail}", outcome="paused_budget")
            rec.count("skipped_budget")
            await self.close_run(d, rec, "paused_budget", e.detail, job, started)
            s.outcome = "paused_budget"
            s.errors.append(e.detail)
        except Exception as e:
            s.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            await self.close_run(d, rec, "failed", s.errors[-1], job, started)
            s.outcome = "failed"
            try:
                await d.web.news_notify({"kind": "job_failed", "jobId": job.id, "jobName": job.name, "runId": rec.run_id, "error": s.errors[-1]})
            except Exception:  # noqa: S110 - notification is best effort
                pass

    async def _retry_selection(self, d: nd.DeskDeps, rec: RunRecorder, job: Job) -> list[Any]:
        """Retry failed step: only the parent run's selected stories that did not reach review."""
        from ..news.select import Candidate

        async with rec.step("select") as o:
            parent = await d.store.candidates(self.state.retry_of or "")
            covered = await d.store.covered_clusters(job.id, d.clock() - timedelta(days=job.config.dedupe_window_days))
            out = [Candidate(cluster_id=str(c["cluster_id"]), title="", score=float(c["score"]), breakdown=c["breakdown"], features=c["features"], model_version=c["model_version"], topics=[], source_ids=[], text="", decision="selected", reason="retry of a failed draft", rank=c.get("rank")) for c in parent if c["decision"] == "selected" and str(c["cluster_id"]) not in covered]
            await d.store.save_candidates(rec.run_id, job.id, [{"cluster_id": c.cluster_id, "features": c.features, "score": c.score, "breakdown": c.breakdown, "model_version": c.model_version, "decision": "selected", "reason": c.reason, "rank": c.rank} for c in out])
            o["output"] = len(out)
        return out

    async def _fan_out(self, d: nd.DeskDeps, job: Job, work: list[dict[str, Any]]) -> list[dict[str, Any]]:
        s: NewsJobState = self.state
        runner = self.deps.runner

        async def one(w: dict[str, Any]) -> dict[str, Any]:
            started = await runner.kickoff("news_draft_flow", {"job_id": job.id, "cluster_ids": w["cluster_ids"], "mode": w["mode"], "news_run_id": s.news_run_id, "dry_run": s.dry_run, "llm": s.llm, "score": w["score"]}, triggered_by="flow", correlation_id=s.id)
            done = started if started.get("existing") else await runner.wait(started["run_id"])
            out = done.get("outputs") if isinstance(done.get("outputs"), dict) else {}
            return {"cluster_ids": w["cluster_ids"], "mode": w["mode"], "run_id": str(done.get("run_id") or started.get("run_id")), "status": done.get("status"), "outcome": (out or {}).get("outcome"), "result": (out or {}).get("result")}

        res = await nd.gather_limited(d.news.draft_concurrency, [one(w) for w in work])
        out = []
        for w, r in zip(work, res, strict=True):
            if isinstance(r, BaseException):
                s.errors.append(f"draft {w['cluster_ids'][0]}: {type(r).__name__}")
                out.append({"cluster_ids": w["cluster_ids"], "mode": w["mode"], "outcome": "failed"})
            else:
                out.append(r)
        return out

    async def _notify(self, d: nd.DeskDeps, rec: RunRecorder, job: Job) -> None:
        s: NewsJobState = self.state
        if s.dry_run or not job.config.notify or not s.children:
            return
        async with rec.step("notify") as o:
            ready = sum(1 for c in s.children if c.get("outcome") == "ready_for_review")
            failed_q = sum(1 for c in s.children if c.get("outcome") == "failed_quality")
            try:
                r = await d.web.news_notify({"kind": "run_summary", "jobId": job.id, "jobName": job.name, "runId": rec.run_id, "ready": ready, "failedQuality": failed_q, "other": len(s.children) - ready - failed_q})
                o["detail"] = {"sent": bool(r.get("sent"))}
            except Exception as e:
                o["detail"] = {"error": type(e).__name__}
            o["output"] = ready


# --- news_draft_flow ------------------------------------------------------------------------------------------------


@persist(INSTANCE_ONLY)
class NewsDraftFlow(NewsHelpers, SeoFlow[NewsDraftState]):
    FLOW: ClassVar[str] = "news_draft_flow"

    @start()
    async def draft(self) -> None:
        s: NewsDraftState = self.state
        d = await self.desk()
        row = await d.store.get_job(s.job_id)
        if not row:
            s.outcome = "failed"
            s.errors.append("job not found")
            return
        job = nd.as_job(row)
        if s.writing_profile_version:
            job.config.writing_profile_version = s.writing_profile_version
        rec = RunRecorder(d.store, s.news_run_id)  # steps and logs join the parent job run's timeline
        rec._seq = 100 + abs(hash(tuple(s.cluster_ids))) % 800
        try:
            r = await nd.draft_story(d, rec, job, s.cluster_ids, mode=s.mode, dry_run=s.dry_run, llm_enabled=s.llm, parent_run_id=s.news_run_id, score=s.score, feedback=s.feedback)
            s.result = {"article_id": r.article_id, "status": r.status, "reason": r.reason[:500], "draft_id": r.draft_id, "cost_usd": round(r.cost_usd, 6)}
            s.outcome = r.status
        except Exception as e:
            s.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            rec.log("error", f"draft failed: {s.errors[-1]}", cluster_ids=s.cluster_ids, outcome="failed")
            await rec.flush()
            metrics.NEWS_DRAFTS.labels(status="failed").inc()
            s.outcome = "failed"


# --- news_dispatch_flow ---------------------------------------------------------------------------------------------


@persist(INSTANCE_ONLY)
class NewsDispatchFlow(NewsHelpers, SeoFlow[NewsDispatchState]):
    FLOW: ClassVar[str] = "news_dispatch_flow"

    @start()
    async def dispatch(self) -> None:
        s = self.state
        d = await self.desk()
        now = d.clock()
        s.result = {"due": [], "skipped": [], "housekeeping": None}
        if not d.news.enabled:
            s.result["skipped"].append("news desk disabled")
            s.outcome = "skipped"
            return
        for row in await d.store.list_jobs(enabled_only=True):
            job = nd.as_job(row)
            try:
                cron = Cron(job.cron, job.timezone)
            except CronError as e:
                s.errors.append(f"{job.name}: {e}")
                continue
            due = cron.last_due(now, timedelta(minutes=16))
            await d.store.set_job_schedule(job.id, next_run_at=cron.next_after(now))
            if due is None:
                continue
            slot = due.strftime("%Y%m%dT%H%MZ")
            r = await self.deps.runner.kickoff("news_job_flow", {"job_id": job.id, "slot": slot, "trigger": "cron"}, triggered_by="flow", correlation_id=s.id)
            s.result["due"].append({"job": job.id, "slot": slot, "run_id": r.get("run_id"), "existing": r.get("existing")})
        if now.hour == 3 and now.minute < 15:  # once a day (UTC)
            s.result["housekeeping"] = await housekeeping(d)
        await gauges(d)
        s.outcome = "done"


async def housekeeping(d: nd.DeskDeps) -> dict[str, Any]:
    out: dict[str, Any] = {"pruned": await d.store.prune(d.news.log_retention_days, d.news.item_retention_days, d.clock())}
    try:
        out["performance"] = await d.web.news_collect_performance()
    except Exception as e:
        out["performance"] = {"error": type(e).__name__}
    return out


async def gauges(d: nd.DeskDeps) -> None:
    try:
        st = await NewsBudget(d.store, d.news, None, "", d.clock).ratios()
        for k, v in st.items():
            metrics.NEWS_BUDGET_RATIO.labels(scope=k).set(v)
        now = d.clock()
        worst = 0.0
        for src in await d.store.sources(enabled_only=True):
            if int(src.get("consecutive_errors") or 0) > 0:
                since = src.get("last_success_at") or src.get("created_at") or now
                worst = max(worst, (now - since).total_seconds())
        metrics.NEWS_SOURCE_FAILING_SINCE.set(worst)
        # The serving models' held-out metrics (restart-safe: read from news_models) and the score drift check.
        for kind in ("classifier", "ranker"):
            m = await d.store.active_model(kind)
            if m:
                nd.set_model_gauges(kind, m.get("metrics") or {})
        await nd.drift(d)
    except Exception:  # noqa: S110 - gauges are best effort
        pass


# --- the rest -------------------------------------------------------------------------------------------------------


@persist(INSTANCE_ONLY)
class NewsIngestFlow(NewsHelpers, SeoFlow[NewsIngestState]):
    FLOW: ClassVar[str] = "news_ingest_flow"

    @start()
    async def ingest(self) -> None:
        s = self.state
        d = await self.desk()
        rec, _ = await self.open_run(d, kind="ingest", job=None, trigger="manual", actor=s.actor)
        try:
            lock: asyncio.Lock = d.extra.setdefault("news_ingest_lock", asyncio.Lock())
            async with lock:
                ing_d = nd.with_news_llm(d, [rec.run_id], job=None, enabled=d.news.enabled)  # global news caps
                r = await nd.ingest(ing_d, rec, source_ids=list(s.source_ids) or None, force=s.force, llm_ok=d.news.enabled)
            s.result = {"new": r["new"], "clusters": len(r["clusters"])}
            await self.close_run(d, rec, "succeeded")
            s.outcome = "done"
        except Exception as e:
            s.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            await self.close_run(d, rec, "failed", s.errors[-1])
            s.outcome = "failed"


@persist(INSTANCE_ONLY)
class NewsImageFlow(NewsHelpers, SeoFlow[NewsImageState]):
    FLOW: ClassVar[str] = "news_image_flow"

    @start()
    async def regenerate(self) -> None:
        s = self.state
        d = await self.desk()
        a = await d.store.get_article(s.article_id)
        if not a or not a.get("draft_id") or a.get("status") != "ready_for_review":
            s.outcome, s.errors = "failed", [*s.errors, "only a draft waiting for review can get a new cover"]
            return
        job = nd.as_job(await d.store.get_job(str(a["job_id"])) or {"id": str(a["job_id"]), "name": "deleted job"})
        rec, _ = await self.open_run(d, kind="image", job=job, trigger="manual", actor=s.actor)
        try:
            from ..news.schemas import NewsDraft

            doc = a.get("generated_doc") or {}
            draft = NewsDraft.model_validate({**doc, **(doc.get("news") or {}), "headings": doc.get("outline") or []})
            ip = await d.store.image_profile(job.image_profile_key, a.get("image_profile_version")) or {"provider": "template", "template_fallback": True}
            clusters = [c for c in [await d.store.get_cluster(cid) for cid in (a.get("cluster_ids") or [])] if c]
            topics = {t.slug: t for t in await nd.load_topics(d)}
            budget = NewsBudget(d.store, d.news, job, rec.run_id, d.clock)
            async with rec.step("image") as o:
                img = await nd.make_cover(d, rec, budget, s.article_id, draft, clusters, topics, ip, job, dry_run=False, prompt_tweak=s.prompt_tweak, provider_override=s.provider)
                o["output"] = 1 if img else 0
            if img:
                await d.web.news_update_draft_image(str(a["draft_id"]), img)
            s.result = {"image": img}
            await self.close_run(d, rec, "succeeded", job=job)
            s.outcome = "done"
        except BudgetExceeded as e:
            await self.close_run(d, rec, "paused_budget", e.detail, job)
            s.outcome = "paused_budget"
        except Exception as e:
            s.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            await self.close_run(d, rec, "failed", s.errors[-1], job)
            s.outcome = "failed"


@persist(INSTANCE_ONLY)
class NewsTrainFlow(NewsHelpers, SeoFlow[NewsTrainState]):
    FLOW: ClassVar[str] = "news_train_flow"

    @start()
    async def train(self) -> None:
        s = self.state
        d = await self.desk()
        rec, _ = await self.open_run(d, kind="train", job=None, trigger="cron" if s.triggered_by == "cron" else "manual")
        budget = NewsBudget(d.store, d.news, None, rec.run_id, d.clock)
        d.llm = NewsLlm(self.deps.llm, budget, [rec.run_id], enabled=d.news.enabled)
        try:
            s.result = await nd.train_models(d, rec, bootstrap=s.bootstrap)
            await self.close_run(d, rec, "succeeded")
            s.outcome = "done"
        except BudgetExceeded as e:
            await self.close_run(d, rec, "paused_budget", e.detail)
            s.outcome = "paused_budget"
        except Exception as e:
            s.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            await self.close_run(d, rec, "failed", s.errors[-1])
            s.outcome = "failed"


@persist(INSTANCE_ONLY)
class NewsStyleReviewFlow(NewsHelpers, SeoFlow[NewsStyleState]):
    FLOW: ClassVar[str] = "news_style_review_flow"

    @start()
    async def review(self) -> None:
        s = self.state
        d = await self.desk()
        rec, _ = await self.open_run(d, kind="style_review", job=None, trigger="cron" if s.triggered_by == "cron" else "manual")
        budget = NewsBudget(d.store, d.news, None, rec.run_id, d.clock)
        d.llm = NewsLlm(self.deps.llm, budget, [rec.run_id], enabled=d.news.enabled)
        try:
            keys = [s.profile_key] if s.profile_key else sorted({nd.as_job(j).writing_profile_key for j in await d.store.list_jobs(enabled_only=False)})
            out = {}
            async with rec.step("style_review") as o:
                for k in keys:
                    out[k] = await nd.style_review(d, rec, k, s.days)
                o["output"] = sum(1 for v in out.values() if v.get("proposed"))
            s.result = out
            await self.close_run(d, rec, "succeeded")
            s.outcome = "done"
        except BudgetExceeded as e:
            await self.close_run(d, rec, "paused_budget", e.detail)
            s.outcome = "paused_budget"
        except Exception as e:
            s.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            await self.close_run(d, rec, "failed", s.errors[-1])
            s.outcome = "failed"


@persist(INSTANCE_ONLY)
class NewsPlaygroundFlow(NewsHelpers, SeoFlow[NewsPlaygroundState]):
    """Profile studio: runs the crew for one recent story with profile version A (and optionally B). Saves nothing
    to the review queue: the owner's "Save as draft" starts a normal news_draft_flow with the chosen version."""

    FLOW: ClassVar[str] = "news_playground_flow"

    @start()
    async def play(self) -> None:
        s = self.state
        d = await self.desk()
        job = nd.as_job(await d.store.get_job(s.job_id) or {"id": s.job_id, "name": "playground"})
        rec, _ = await self.open_run(d, kind="playground", job=job, trigger="manual")
        out: dict[str, Any] = {}
        try:
            for label, version in (("a", s.version_a), ("b", s.version_b)):
                if not version:
                    continue
                wp = await d.store.writing_profile(s.profile_key, version)
                if not wp:
                    out[label] = {"error": f"profile {s.profile_key} v{version} not found"}
                    continue
                items = (await d.store.items_by_cluster([s.cluster_id])).get(s.cluster_id, [])[:12]
                budget = NewsBudget(d.store, d.news, job, rec.run_id, d.clock)
                llm = NewsLlm(d.llm, budget, [rec.run_id])
                crew = (d.crew_factory or (lambda lm: nd.NewsDeskCrew(d.settings, lm, d.web, d.retriever)))(llm)
                cluster = await d.store.get_cluster(s.cluster_id) or {"title": "story"}
                r = await crew.run({"mode": "single", "title": cluster["title"], "cluster_id": s.cluster_id, "length_words": [job.config.length_min_words, job.config.length_max_words]}, items=items, writing_profile=wp, image_profile={}, allow_hosts=nd._allowed_hosts(items, d), max_tokens=job.config.budget.max_tokens_per_draft)
                out[label] = {"version": version, "title": r.draft.title, "body_mdx": r.draft.body_mdx, "dek": r.draft.dek, "tokens": llm.tokens, "cost_usd": round(llm.cost_usd, 4), "status": r.status}
            s.result = out
            await self.close_run(d, rec, "succeeded", job=job)
            s.outcome = "done"
        except BudgetExceeded as e:
            s.result = {**out, "error": e.detail}
            await self.close_run(d, rec, "paused_budget", e.detail, job)
            s.outcome = "paused_budget"
        except Exception as e:
            s.errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            await self.close_run(d, rec, "failed", s.errors[-1], job)
            s.outcome = "failed"


NEWS_INPUTS: dict[str, type[BaseModel]] = {
    "news_job_flow": NewsJobInput,
    "news_draft_flow": NewsDraftInput,
    "news_dispatch_flow": NewsDispatchInput,
    "news_ingest_flow": NewsIngestInput,
    "news_image_flow": NewsImageInput,
    "news_train_flow": NewsTrainInput,
    "news_style_review_flow": NewsStyleInput,
    "news_playground_flow": NewsPlaygroundInput,
}
