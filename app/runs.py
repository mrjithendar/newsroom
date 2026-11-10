"""Run records (`seo_agent_runs`, created by the web app's migration 0015) and the run manager.

LangGraph's checkpoints (PostgresSaver tables) are the source of truth for graph state; `seo_agent_runs` is a small,
query-friendly summary per thread (status, current node, interrupt payload, cost) that the admin UI and the
`seo_graph_interrupts_pending` gauge read. thread_id = the draft id for optimize_article / new_article, so a run, its
draft and the audit log share one id.

Statuses: running | interrupted (waiting at human_review) | completed | failed | cancelled | paused_budget.
`outcome` holds the business result of a completed run (published, rejected, generation_failed, verify_failed, …).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from . import metrics

log = logging.getLogger("seo_agent.runs")

TERMINAL = {"completed", "failed", "cancelled", "paused_budget"}


class RunStore(Protocol):
    async def upsert(self, thread_id: str, **fields: Any) -> None: ...
    async def get(self, thread_id: str) -> dict[str, Any] | None: ...
    async def by_status(self, status: str) -> list[dict[str, Any]]: ...
    async def count_status(self, status: str) -> int: ...


_COLS = ("graph", "status", "current_node", "outcome", "draft_id", "article_id", "calendar_id", "parent_thread_id", "input", "interrupt", "error", "cost_usd", "result")


class PgRunStore:
    def __init__(self, pool: Any):
        self.pool = pool

    async def upsert(self, thread_id: str, **fields: Any) -> None:
        f = {k: v for k, v in fields.items() if k in _COLS}
        for k in ("input", "interrupt", "result"):
            if k in f and f[k] is not None:
                f[k] = json.dumps(f[k], default=str)
        cols = ", ".join(f)
        vals = ", ".join(f"%({k})s" + ("::jsonb" if k in ("input", "interrupt", "result") else "") for k in f)
        sets = ", ".join(f"{k} = excluded.{k}" for k in f)
        async with self.pool.connection() as c:
            await c.execute(
                # Column names come only from the fixed _COLS allowlist; values are bound parameters.
                f"insert into seo_agent_runs (thread_id, {cols}) values (%(thread_id)s, {vals}) "  # noqa: S608
                f"on conflict (thread_id) do update set {sets}, updated_at = now()",
                {"thread_id": thread_id, **f},
            )

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as c:
            cur = await c.execute("select * from seo_agent_runs where thread_id = %s", (thread_id,))
            return await cur.fetchone()

    async def by_status(self, status: str) -> list[dict[str, Any]]:
        async with self.pool.connection() as c:
            cur = await c.execute("select * from seo_agent_runs where status = %s order by created_at", (status,))
            return list(await cur.fetchall())

    async def count_status(self, status: str) -> int:
        async with self.pool.connection() as c:
            cur = await c.execute("select count(*)::int as n from seo_agent_runs where status = %s", (status,))
            return int((await cur.fetchone())["n"])


@dataclass
class MemoryRunStore:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def upsert(self, thread_id: str, **fields: Any) -> None:
        row = self.rows.setdefault(thread_id, {"thread_id": thread_id, "created_at": datetime.now(UTC), "cost_usd": 0.0})
        row.update({k: v for k, v in fields.items() if k in _COLS})
        row["updated_at"] = datetime.now(UTC)

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        return self.rows.get(thread_id)

    async def by_status(self, status: str) -> list[dict[str, Any]]:
        return [r for r in self.rows.values() if r.get("status") == status]

    async def count_status(self, status: str) -> int:
        return len(await self.by_status(status))


class RunError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


class RunManager:
    """Starts, resumes, cancels and recovers graph runs. Runs execute as background tasks in this process (one
    replica); a pod restart loses only in-memory tasks: `recover()` re-drives every `running` thread from its last
    checkpoint, and interrupted threads simply wait in Postgres for their resume."""

    def __init__(self, graphs: dict[str, Any], store: RunStore):
        self.graphs = graphs
        self.store = store
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    def _cfg(self, thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    async def start(self, graph: str, payload: dict[str, Any], thread_id: str, *, parent: str | None = None) -> dict[str, Any]:
        if graph not in self.graphs:
            raise RunError(400, f"unknown graph {graph}")
        lock = self.locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            existing = await self.store.get(thread_id)
            if existing and existing.get("status") not in TERMINAL:
                return {"thread_id": thread_id, "status": existing["status"], "existing": True}
            state = await self.graphs[graph].aget_state(self._cfg(thread_id))
            if existing and existing.get("status") in TERMINAL and state.values:
                # A finished thread is never restarted in place: its checkpoints are the run's history.
                raise RunError(409, f"thread {thread_id} already finished ({existing.get('outcome') or existing['status']})")
            await self.store.upsert(
                thread_id,
                graph=graph,
                status="running",
                current_node=None,
                outcome=None,
                error=None,
                interrupt=None,
                input=_jsonable(payload),
                draft_id=payload.get("draft_id"),
                article_id=payload.get("article_id"),
                calendar_id=payload.get("calendar_id"),
                parent_thread_id=parent,
            )
            self._spawn(graph, thread_id, {**payload, "thread_id": thread_id})
        return {"thread_id": thread_id, "status": "running", "existing": False}

    async def resume(self, thread_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        from langgraph.types import Command

        lock = self.locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            run = await self.store.get(thread_id)
            if not run:
                raise RunError(404, "run not found")
            if run["status"] != "interrupted":
                raise RunError(409, f"run is {run['status']}, not waiting for review")
            await self.store.upsert(thread_id, status="running", interrupt=None)
            self._spawn(run["graph"], thread_id, Command(resume=_jsonable(decision)))
        return {"thread_id": thread_id, "status": "running"}

    async def cancel(self, thread_id: str) -> dict[str, Any]:
        run = await self.store.get(thread_id)
        if not run:
            raise RunError(404, "run not found")
        if run["status"] in TERMINAL:
            return {"thread_id": thread_id, "status": run["status"]}
        t = self.tasks.pop(thread_id, None)
        if t and not t.done():
            t.cancel()
        await self.store.upsert(thread_id, status="cancelled", interrupt=None)
        metrics.GRAPH_RUNS.labels(graph=metrics.graph_label(run["graph"]), status="cancelled").inc()
        return {"thread_id": thread_id, "status": "cancelled"}

    async def recover(self) -> list[str]:
        """Re-drive runs that were mid-execution when the previous process stopped (resume from last checkpoint)."""
        out = []
        for r in await self.store.by_status("running"):
            if r["thread_id"] in self.tasks or r["graph"] not in self.graphs:
                continue
            snap = await self.graphs[r["graph"]].aget_state(self._cfg(r["thread_id"]))
            inp = r.get("input")
            if isinstance(inp, str):
                inp = json.loads(inp)
            # Died before the first checkpoint: start again from the recorded input. Otherwise continue from the
            # last checkpoint (input None), so completed nodes (model calls, publish) are not repeated.
            self._spawn(r["graph"], r["thread_id"], None if snap.values else {**(inp or {}), "thread_id": r["thread_id"]})
            out.append(r["thread_id"])
        if out:
            log.info("recovered runs", extra={"event": "runs_recovered", "count": len(out)})
        return out

    async def get(self, thread_id: str) -> dict[str, Any]:
        run = await self.store.get(thread_id)
        if not run:
            raise RunError(404, "run not found")
        snap = await self.graphs[run["graph"]].aget_state(self._cfg(thread_id))
        values = snap.values or {}
        return {
            "thread_id": thread_id,
            "graph": run["graph"],
            "status": run["status"],
            "outcome": run.get("outcome"),
            "current_node": run.get("current_node"),
            "next": list(snap.next or ()),
            "interrupt": run.get("interrupt"),
            "error": run.get("error"),
            "cost_usd": float(values.get("cost_usd") or run.get("cost_usd") or 0),
            "state": {k: values.get(k) for k in ("article_id", "draft_id", "recommendation_ids", "validation_report", "review_decision", "publish_result", "verification", "submission_results", "errors", "status", "children") if k in values},
        }

    async def wait(self, thread_id: str, limit_s: float | None = None) -> None:
        t = self.tasks.get(thread_id)
        if t:
            await asyncio.wait_for(asyncio.shield(t), limit_s)

    def _spawn(self, graph: str, thread_id: str, inp: Any) -> None:
        t = asyncio.create_task(self._drive(graph, thread_id, inp), name=f"run:{thread_id}")
        self.tasks[thread_id] = t
        t.add_done_callback(lambda _t: self.tasks.pop(thread_id, None) if self.tasks.get(thread_id) is _t else None)

    async def _drive(self, graph: str, thread_id: str, inp: Any) -> None:
        g = self.graphs[graph]
        cfg = self._cfg(thread_id)
        try:
            await g.ainvoke(inp, cfg)
            snap = await g.aget_state(cfg)
            interrupts = [i.value for i in (getattr(snap, "interrupts", None) or ())]
            if interrupts:
                await self.store.upsert(thread_id, status="interrupted", interrupt=_jsonable(interrupts[0]), current_node="human_review", cost_usd=float(snap.values.get("cost_usd", 0)))
                return
            values = snap.values or {}
            outcome = values.get("status") or "done"
            status = "paused_budget" if outcome == "paused_budget" else "completed"
            await self.store.upsert(thread_id, status=status, outcome=outcome, current_node=None, cost_usd=float(values.get("cost_usd", 0)), result=_jsonable({k: values.get(k) for k in ("publish_result", "verification", "submission_results", "errors", "children") if values.get(k) is not None}))
            label = outcome if status == "completed" and outcome in _OUTCOMES else status
            metrics.GRAPH_RUNS.labels(graph=metrics.graph_label(graph), status=label).inc()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # node retries are exhausted: record and keep the checkpoint for inspection/resume
            log.exception("graph run failed", extra={"event": "graph_failed", "graph": graph})
            await self.store.upsert(thread_id, status="failed", error=f"{type(e).__name__}: {str(e)[:500]}")
            metrics.GRAPH_RUNS.labels(graph=metrics.graph_label(graph), status="failed").inc()


_OUTCOMES = {"published", "rejected", "generation_failed", "verify_failed", "publish_failed", "no_changes", "done", "links_added"}
