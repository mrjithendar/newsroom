"""Flow runs: the `seo_flow_runs` table, CrewAI's Postgres persistence, advisory locks, and the run manager.

- `seo_flow_runs` (web migration 0016) is the single record of every flow run: status (queued | running | succeeded
  | failed | cancelled | paused_budget), current step, the typed state CrewAI persists after each step, outputs,
  error, tokens, cost, trigger and correlation id. The admin UI, the MCP `list_flow_runs` tool and the alerts read it.
- `PgFlowPersistence` is a CrewAI `FlowPersistence` that writes the state into that row. CrewAI's default local SQLite
  persistence is never used: every flow class is decorated with `@persist(INSTANCE_ONLY)`, a backend that refuses
  to work, and every flow instance gets a `PgFlowPersistence` (an instance backend overrides the class one).
- One active run per flow + target: a partial unique index on (flow_name, target_id) for queued/running rows, and a
  Postgres session advisory lock held on a dedicated connection for the whole run (a crashed pod's connection
  closes, so its locks go with it).
- A run that dies mid-way is not resumed: on start-up, `running`/`queued` rows left by the previous process are
  marked failed ("interrupted by a restart"). A rerun starts from the beginning; every step is idempotent (upserts,
  one open draft per article, the approval is single-use).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from crewai.flow.persistence.base import FlowPersistence
from pydantic import BaseModel, ConfigDict, PrivateAttr

from . import metrics, tracing

log = logging.getLogger("seo_agent.runs")

ACTIVE = ("queued", "running")
TERMINAL = ("succeeded", "failed", "cancelled", "paused_budget")
# Business outcomes that make the run itself "failed" (the draft records the reason; alerts count these).
FAILED_OUTCOMES = {"generation_failed", "publish_failed", "verify_failed", "report_mismatch", "failed"}
# The flow run the current task belongs to (set by FlowRunner._drive and SeoFlow.step). Crews copy it into
# CallSpec.run_id so every model call can be attributed to its run.
CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("seo_flow_run_id", default=None)


def _jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


def _state_dict(state: Any) -> dict[str, Any]:
    return state.model_dump(mode="json") if isinstance(state, BaseModel) else _jsonable(dict(state))


class RunConflict(Exception):
    def __init__(self, existing: dict[str, Any]):
        super().__init__("an active run exists for this flow and target")
        self.existing = existing


class RunError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# --- Store --------------------------------------------------------------------------------------------------------

_COLS = ("flow_name", "target_id", "draft_id", "status", "current_step", "state_json", "inputs", "outputs", "error", "tokens", "cost_usd", "started_at", "finished_at", "triggered_by", "correlation_id", "actor")
_JSON = ("state_json", "inputs", "outputs")


class RunStore(Protocol):
    async def create(self, row: dict[str, Any]) -> dict[str, Any]: ...
    async def update(self, run_id: str, **fields: Any) -> None: ...
    async def get(self, run_id: str) -> dict[str, Any] | None: ...
    async def list_runs(self, *, flow: str | None = None, status: str | None = None, draft_id: str | None = None, limit: int = 20, before: str | None = None) -> list[dict[str, Any]]: ...
    async def count_status(self, status: str) -> int: ...
    async def sweep_orphans(self) -> list[dict[str, Any]]: ...
    async def recent_kickoffs(self, actor: str, since_s: int) -> int: ...


class PgRunStore:
    def __init__(self, pool: Any):
        self.pool = pool

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        f = {k: row[k] for k in _COLS if k in row}
        for k in _JSON:
            if k in f:
                f[k] = json.dumps(f[k], default=str)
        cols = ", ".join(f)
        vals = ", ".join(f"%({k})s" + ("::jsonb" if k in _JSON else "") for k in f)
        async with self.pool.connection() as c:
            # Column names come only from the fixed _COLS allowlist; values are bound parameters.
            cur = await c.execute(
                f"insert into seo_flow_runs (run_id, {cols}) values (%(run_id)s, {vals}) on conflict do nothing returning *",  # noqa: S608
                {"run_id": row["run_id"], **f},
            )
            got = await cur.fetchone()
            if got:
                return dict(got)
            cur = await c.execute(
                "select * from seo_flow_runs where flow_name = %s and target_id = %s and status in ('queued', 'running') order by started_at desc limit 1",
                (row["flow_name"], row["target_id"]),
            )
            existing = await cur.fetchone()
        raise RunConflict(dict(existing) if existing else {"flow_name": row["flow_name"], "target_id": row["target_id"]})

    async def update(self, run_id: str, **fields: Any) -> None:
        f = {k: v for k, v in fields.items() if k in _COLS}
        if not f:
            return
        for k in _JSON:
            if k in f and f[k] is not None:
                f[k] = json.dumps(f[k], default=str)
        sets = ", ".join(f"{k} = %({k})s" + ("::jsonb" if k in _JSON else "") for k in f)
        async with self.pool.connection() as c:
            await c.execute(f"update seo_flow_runs set {sets}, updated_at = now() where run_id = %(run_id)s", {"run_id": run_id, **f})  # noqa: S608

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as c:
            cur = await c.execute("select * from seo_flow_runs where run_id::text = %s", (run_id,))
            r = await cur.fetchone()
            return dict(r) if r else None

    async def list_runs(self, *, flow: str | None = None, status: str | None = None, draft_id: str | None = None, limit: int = 20, before: str | None = None) -> list[dict[str, Any]]:
        where: list[str] = ["true"]
        args: list[Any] = []
        for col, val in (("flow_name", flow), ("status", status), ("draft_id::text", draft_id)):
            if val:
                where.append(f"{col} = %s")
                args.append(val)
        if before:
            where.append("started_at < %s::timestamptz")
            args.append(before)
        args.append(max(1, min(limit, 50)))
        async with self.pool.connection() as c:
            cur = await c.execute(f"select * from seo_flow_runs where {' and '.join(where)} order by started_at desc limit %s", args)  # noqa: S608
            return [dict(r) for r in await cur.fetchall()]

    async def count_status(self, status: str) -> int:
        async with self.pool.connection() as c:
            cur = await c.execute("select count(*)::int as n from seo_flow_runs where status = %s", (status,))
            return int((await cur.fetchone())["n"])

    async def sweep_orphans(self) -> list[dict[str, Any]]:
        async with self.pool.connection() as c:
            cur = await c.execute(
                """update seo_flow_runs set status = 'failed', error = 'interrupted by a restart; rerun to start again',
                     finished_at = now(), updated_at = now() where status in ('queued', 'running') returning *"""
            )
            return [dict(r) for r in await cur.fetchall()]

    async def recent_kickoffs(self, actor: str, since_s: int) -> int:
        async with self.pool.connection() as c:
            cur = await c.execute("select count(*)::int as n from seo_flow_runs where actor = %s and started_at > now() - make_interval(secs => %s)", (actor, since_s))
            return int((await cur.fetchone())["n"])


@dataclass
class MemoryRunStore:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def create(self, row: dict[str, Any]) -> dict[str, Any]:
        for r in self.rows.values():
            if r["flow_name"] == row["flow_name"] and r["target_id"] == row["target_id"] and r["status"] in ACTIVE:
                raise RunConflict(r)
        now = datetime.now(UTC)
        r = {"tokens": 0, "cost_usd": 0.0, "started_at": now, "updated_at": now, "finished_at": None, "current_step": None, "error": None, "outputs": None, "state_json": None, **_jsonable(row)}
        r["started_at"] = now
        self.rows[row["run_id"]] = r
        return r

    async def update(self, run_id: str, **fields: Any) -> None:
        if run_id in self.rows:
            self.rows[run_id].update({k: (_jsonable(v) if k in _JSON else v) for k, v in fields.items() if k in _COLS})
            self.rows[run_id]["updated_at"] = datetime.now(UTC)

    async def get(self, run_id: str) -> dict[str, Any] | None:
        return self.rows.get(run_id)

    async def list_runs(self, *, flow: str | None = None, status: str | None = None, draft_id: str | None = None, limit: int = 20, before: str | None = None) -> list[dict[str, Any]]:
        out = [r for r in self.rows.values() if (not flow or r["flow_name"] == flow) and (not status or r["status"] == status) and (not draft_id or r.get("draft_id") == draft_id)]
        out.sort(key=lambda r: r["started_at"], reverse=True)
        if before:
            out = [r for r in out if r["started_at"].isoformat() < before]
        return out[: max(1, min(limit, 50))]

    async def count_status(self, status: str) -> int:
        return sum(1 for r in self.rows.values() if r["status"] == status)

    async def sweep_orphans(self) -> list[dict[str, Any]]:
        out = []
        for r in self.rows.values():
            if r["status"] in ACTIVE:
                r.update(status="failed", error="interrupted by a restart; rerun to start again")
                out.append(r)
        return out

    async def recent_kickoffs(self, actor: str, since_s: int) -> int:
        cutoff = datetime.now(UTC).timestamp() - since_s
        return sum(1 for r in self.rows.values() if r.get("actor") == actor and r["started_at"].timestamp() > cutoff)


# --- CrewAI persistence -------------------------------------------------------------------------------------------


class InstanceOnlyPersistence(FlowPersistence):
    """Class-level placeholder for `@persist(...)`: a flow must get its backend per instance (PgFlowPersistence or
    MemoryFlowPersistence). Without this, `@persist()` would create CrewAI's default SQLite store."""

    persistence_type: str = "InstanceOnlyPersistence"

    def init_db(self) -> None:
        return None

    def save_state(self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | BaseModel) -> None:
        raise RuntimeError("flow persistence must be set per instance (PgFlowPersistence); local SQLite is not allowed")

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        raise RuntimeError("flow persistence must be set per instance (PgFlowPersistence); local SQLite is not allowed")


INSTANCE_ONLY = InstanceOnlyPersistence()


class PgFlowPersistence(FlowPersistence):
    """CrewAI calls save_state synchronously after each flow method. It writes the state, the step name, tokens and
    cost into the run's seo_flow_runs row through a small *sync* psycopg pool (a few ms per step)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    persistence_type: str = "PgFlowPersistence"
    _pool: Any = PrivateAttr(default=None)

    def bind(self, pool: Any) -> PgFlowPersistence:
        self._pool = pool
        return self

    def init_db(self) -> None:
        return None  # the table is created by the web app's migration 0016

    def save_state(self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | BaseModel) -> None:
        st = _state_dict(state_data)
        with self._pool.connection() as c:
            c.execute(
                "update seo_flow_runs set state_json = %s::jsonb, current_step = %s, tokens = %s, cost_usd = %s, updated_at = now() where run_id::text = %s",
                (json.dumps(st, default=str), method_name, int(st.get("tokens") or 0), float(st.get("cost_usd") or 0), flow_uuid),
            )

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        with self._pool.connection() as c:
            r = c.execute("select state_json from seo_flow_runs where run_id::text = %s", (flow_uuid,)).fetchone()
        if not r:
            return None
        v = r["state_json"] if isinstance(r, dict) else r[0]
        return v if isinstance(v, dict) else (json.loads(v) if v else None)


class MemoryFlowPersistence(FlowPersistence):
    persistence_type: str = "MemoryFlowPersistence"
    _store: Any = PrivateAttr(default=None)
    _saved: list[tuple[str, str]] = PrivateAttr(default_factory=list)

    def bind(self, store: MemoryRunStore) -> MemoryFlowPersistence:
        self._store = store
        return self

    def init_db(self) -> None:
        return None

    def save_state(self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | BaseModel) -> None:
        st = _state_dict(state_data)
        self._saved.append((flow_uuid, method_name))
        r = self._store.rows.get(flow_uuid)
        if r is not None:
            r.update(state_json=st, current_step=method_name, tokens=int(st.get("tokens") or 0), cost_usd=float(st.get("cost_usd") or 0))

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        r = self._store.rows.get(flow_uuid)
        return r.get("state_json") if r else None


# --- Advisory locks -----------------------------------------------------------------------------------------------


class Locks(Protocol):
    async def acquire(self, key: str) -> Any | None: ...
    async def release(self, handle: Any) -> None: ...


class PgAdvisoryLocks:
    """Session-level pg_try_advisory_lock on a dedicated connection that lives exactly as long as the run.

    Through PgBouncer in session mode a client connection keeps one server connection, so lock and unlock happen
    on the same backend (the problem v1 D-02 avoided with leases came from a shared pool, which this does not use).
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    async def acquire(self, key: str) -> Any | None:
        import psycopg

        conn = await psycopg.AsyncConnection.connect(self.dsn, autocommit=True, prepare_threshold=None)
        try:
            cur = await conn.execute("select pg_try_advisory_lock(hashtextextended(%s, 0)) as ok", (f"seo_flow:{key}",))
            row = await cur.fetchone()
            if row and row[0]:
                return conn
        except Exception:
            await conn.close()
            raise
        await conn.close()
        return None

    async def release(self, handle: Any) -> None:
        try:
            await handle.close()  # closing the session releases its advisory locks
        except Exception:  # noqa: S110 - closing a dead connection is fine
            pass


class MemoryLocks:
    def __init__(self) -> None:
        self.held: set[str] = set()

    async def acquire(self, key: str) -> Any | None:
        if key in self.held:
            return None
        self.held.add(key)
        return key

    async def release(self, handle: Any) -> None:
        self.held.discard(handle)


# --- Run manager --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FlowSpec:
    name: str
    factory: Callable[..., Any]  # (deps, persistence) -> Flow instance
    input_model: type[BaseModel]
    target: Callable[[dict[str, Any]], str]
    draft_of: Callable[[dict[str, Any]], str | None] = lambda _i: None


class FlowRunner:
    """Creates runs, executes flows as background tasks in this (single-replica) process, cancels them, and sweeps
    orphans at start-up. `kickoff` is the only entry point: the HTTP API, the review decision, the CronJob trigger,
    weekly fan-out and the admin MCP tool all call it."""

    def __init__(self, specs: dict[str, FlowSpec], store: RunStore, locks: Locks, persistence: Callable[[], FlowPersistence], deps: Any, timeout_s: float = 1800, policy: Callable[[str, str], None] | None = None):
        self.specs = specs
        self.policy = policy
        self.store = store
        self.locks = locks
        self.persistence = persistence
        self.deps = deps
        self.timeout_s = timeout_s
        self.tasks: dict[str, asyncio.Task[None]] = {}

    async def kickoff(self, name: str, inputs: dict[str, Any] | None = None, *, triggered_by: str = "manual", actor: str | None = None, correlation_id: str | None = None) -> dict[str, Any]:
        spec = self.specs.get(name)
        if spec is None:
            raise RunError(404, f"unknown flow {name}")
        if self.policy is not None:
            self.policy(name, triggered_by)  # publish_flow: approval only (app/flows/__init__.py check_kickoff)
        try:
            clean = spec.input_model.model_validate(inputs or {}).model_dump(exclude_none=True)
        except ValueError as e:
            raise RunError(422, f"invalid inputs for {name}: {str(e)[:300]}") from e
        run_id = str(uuid.uuid4())
        row = {
            "run_id": run_id,
            "flow_name": name,
            "target_id": spec.target(clean)[:200],
            "draft_id": spec.draft_of(clean),
            "status": "queued",
            "inputs": clean,
            "triggered_by": triggered_by,
            "correlation_id": correlation_id or run_id,
            "actor": (actor or None) and str(actor)[:200],
        }
        try:
            await self.store.create(row)
        except RunConflict as c:
            return {"run_id": str(c.existing.get("run_id", "")), "status": c.existing.get("status", "running"), "existing": True}
        t = asyncio.create_task(self._drive(spec, run_id, clean, triggered_by), name=f"flow:{name}:{run_id}")
        self.tasks[run_id] = t
        t.add_done_callback(lambda _t: self.tasks.pop(run_id, None))
        return {"run_id": run_id, "status": "queued", "existing": False}

    async def cancel(self, run_id: str) -> dict[str, Any]:
        run = await self.store.get(run_id)
        if not run:
            raise RunError(404, "run not found")
        if run["status"] in TERMINAL:
            return {"run_id": run_id, "status": run["status"]}
        t = self.tasks.get(run_id)
        if t and not t.done():
            t.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(t), 5)
            except (asyncio.CancelledError, TimeoutError, Exception):  # noqa: S110 - the task records its own end
                pass
        run = await self.store.get(run_id) or {}
        if run.get("status") in ACTIVE:
            await self._finish(run_id, run.get("flow_name", "other"), "cancelled", None, error="cancelled")
        return {"run_id": run_id, "status": "cancelled"}

    async def cancel_for_draft(self, draft_id: str, flows: tuple[str, ...]) -> list[str]:
        out = []
        for r in await self.store.list_runs(draft_id=draft_id, limit=10):
            if r["status"] in ACTIVE and r["flow_name"] in flows:
                await self.cancel(str(r["run_id"]))
                out.append(str(r["run_id"]))
        return out

    async def wait(self, run_id: str, limit_s: float | None = None) -> dict[str, Any]:
        t = self.tasks.get(run_id)
        if t:
            try:
                await asyncio.wait_for(asyncio.shield(t), limit_s)
            except (asyncio.CancelledError, TimeoutError):
                pass
        return dict(await self.store.get(run_id) or {})

    async def recover(self) -> list[str]:
        """Start-up: runs the previous process left queued/running are failed (not resumed; see module docstring).
        A drafting run's draft is marked generation_failed so the review queue shows "regenerate"."""
        rows = await self.store.sweep_orphans()
        for r in rows:
            metrics.FLOW_RUNS.labels(flow=metrics.flow_label(r["flow_name"]), status="failed").inc()
            if r["flow_name"] in ("optimize_article_flow", "new_article_flow") and r.get("draft_id"):
                try:
                    await self.deps.web.set_draft_status(str(r["draft_id"]), "generation_failed", "Interrupted by a restart; regenerate to try again.")
                except Exception:  # noqa: S110 - best effort; the run row already says what happened
                    pass
        if rows:
            log.info("orphaned flow runs failed", extra={"event": "flow_runs_swept", "count": len(rows)})
        return [str(r["run_id"]) for r in rows]

    async def _finish(self, run_id: str, flow: str, status: str, state: Any, error: str | None = None) -> None:
        st = _state_dict(state) if state is not None else None
        fields: dict[str, Any] = {"status": status, "finished_at": datetime.now(UTC), "error": error}
        if st is not None:
            fields.update(state_json=st, outputs={k: st.get(k) for k in _OUTPUT_KEYS if st.get(k) not in (None, [], {}, "")}, tokens=int(st.get("tokens") or 0), cost_usd=float(st.get("cost_usd") or 0))
        await self.store.update(run_id, **fields)
        metrics.FLOW_RUNS.labels(flow=metrics.flow_label(flow), status=status).inc()
        outcome = (st or {}).get("outcome")
        tokens, cost = fields.get("tokens", 0), round(float(fields.get("cost_usd", 0.0)), 6)
        # The run's root span (_drive) and one log line carry the result, so Jaeger and Loki can be searched by it.
        tracing.annotate({"flow.status": status, "flow.outcome": outcome, "flow.tokens": tokens, "flow.cost_usd": cost}, error=status if status == "failed" else None)
        log.info(
            "flow run finished",
            extra={"event": "flow_finished", "flow": flow, "run_id": run_id, "status": status, "outcome": outcome, "tokens": tokens, "cost_usd": cost, "error": (error or "")[:300] or None},
        )
        history = getattr(self.deps, "history", None)
        if history is not None:  # steps left open end with the run; its last log lines are written now
            await history.run_finished(run_id, status)

    async def _drive(self, spec: FlowSpec, run_id: str, inputs: dict[str, Any], triggered_by: str) -> None:
        # This task belongs to this run from its first line: a child run started by another flow's step would otherwise
        # carry the parent's run id here (the task copies the caller's context) and log "started" into the parent's log.
        CURRENT_RUN_ID.set(run_id)
        # One root span per run (flow.<name>): its steps, crews, model calls, tool calls and web app calls nest under it.
        with tracing.span(f"flow.{spec.name}", {"flow": spec.name, "run_id": run_id, "triggered_by": triggered_by, "target_id": spec.target(inputs)}):
            log.info("flow run started", extra={"event": "flow_started", "flow": spec.name, "run_id": run_id, "triggered_by": triggered_by})
            await self._drive_run(spec, run_id, inputs, triggered_by)

    async def _drive_run(self, spec: FlowSpec, run_id: str, inputs: dict[str, Any], triggered_by: str) -> None:
        handle = None
        flow = None
        started = time.monotonic()
        try:
            handle = await self.locks.acquire(f"{spec.name}:{spec.target(inputs)}")
            if handle is None:
                await self._finish(run_id, spec.name, "cancelled", None, error="another run for this flow and target holds the lock")
                return
            await self.store.update(run_id, status="running")
            CURRENT_RUN_ID.set(run_id)  # this task's context only; CrewAI's step tasks inherit it
            flow = spec.factory(self.deps, self.persistence())
            await asyncio.wait_for(flow.kickoff_async(inputs={**inputs, "id": run_id, "triggered_by": triggered_by}), self.timeout_s)
            st = flow.state
            outcome = getattr(st, "outcome", None) or "done"
            status = "paused_budget" if outcome == "paused_budget" else ("failed" if outcome in FAILED_OUTCOMES else "succeeded")
            await self._finish(run_id, spec.name, status, st, error="; ".join(st.errors[-3:])[:1000] if status == "failed" and getattr(st, "errors", None) else None)
        except asyncio.CancelledError:
            # cancel()/cancel_for_draft: the CancelledError reaches the running step first; a crew in a worker thread
            # marks its CrewRun cancelled there (crews/common.run_crew), so it stops calling the model.
            await self._finish(run_id, spec.name, "cancelled", flow.state if flow else None, error="cancelled")
            raise
        except TimeoutError:
            # wait_for cancelled the flow task first (same path as above inside the step), then raised TimeoutError here.
            await self._finish(run_id, spec.name, "failed", flow.state if flow else None, error=f"flow wall-clock limit {self.timeout_s:.0f}s reached")
        except Exception as e:  # a step raised: the run is failed; a rerun starts from the beginning
            log.exception("flow run failed", extra={"event": "flow_failed", "flow": spec.name})
            await self._finish(run_id, spec.name, "failed", flow.state if flow else None, error=f"{type(e).__name__}: {str(e)[:500]}")
        finally:
            if handle is not None:
                await self.locks.release(handle)
            metrics.FLOW_DURATION.labels(flow=metrics.flow_label(spec.name)).observe(time.monotonic() - started)


_OUTPUT_KEYS = ("outcome", "draft_id", "article_id", "children", "results", "result", "publish_result", "verification", "submission_results", "reindex", "report_id", "check", "pdf", "notified", "errors")


def view(r: dict[str, Any], *, with_state: bool = False) -> dict[str, Any]:
    """The JSON shape of a run for the API, the admin UI and MCP. Never contains secrets (state holds none)."""

    def iso(v: Any) -> str | None:
        return v.isoformat() if hasattr(v, "isoformat") else (str(v) if v else None)

    out = {
        "run_id": str(r.get("run_id")),
        "flow": r.get("flow_name"),
        "target_id": r.get("target_id"),
        "draft_id": str(r["draft_id"]) if r.get("draft_id") else None,
        "status": r.get("status"),
        "current_step": r.get("current_step"),
        "outcome": (r.get("outputs") or {}).get("outcome") if isinstance(r.get("outputs"), dict) else None,
        "outputs": r.get("outputs") or {},
        "error": r.get("error"),
        "tokens": int(r.get("tokens") or 0),
        "cost_usd": round(float(r.get("cost_usd") or 0), 6),
        "triggered_by": r.get("triggered_by"),
        "correlation_id": r.get("correlation_id"),
        "started_at": iso(r.get("started_at")),
        "finished_at": iso(r.get("finished_at")),
    }
    if with_state:
        out["state"] = r.get("state_json") or {}
    return out
