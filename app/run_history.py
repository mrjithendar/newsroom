"""Run history for the admin's CrewAI console (/admin/crewai): the steps, log and model calls of every flow run.

The web app owns the tables (moks-cms migration 0036): `seo_flow_run_steps`, `seo_flow_run_logs` and
`seo_flow_run_calls`. Its run page reads them, streams the log while a run goes, and its daily retention job deletes old
lines and transcripts (Automation & AI → settings → Run history). Like GitHub Actions: numbered runs (a trigger on
seo_flow_runs gives #N), a step list, and a log grouped by step.

- Steps: `SeoFlow.step()` opens a row (running) and closes it (succeeded | failed | cancelled, with seconds and the
  error). When a run ends, the runner closes any step still open (a cancel or a timeout in the middle of a step).
- Log: `RunLogHandler`, on the root logger, copies every record that belongs to a flow run into a queue: the run is
  CURRENT_RUN_ID, or the `run_id` a crew worker thread passes with `extra=`. One background task writes the queue in
  batches (every half second). The records still go to stderr → Loki as before. The step of a line is the step the
  record names, else the step running in this task (CURRENT_STEP), else the run's latest open step: crew worker threads
  (run_in_executor) do not inherit context variables, so the run's open step is tracked here by run id.
- Model calls: `RecordingLlm` wraps the service's Llm. Each model call of a flow run (CallSpec.run_id) gets a log line
  with its agent, model, tokens, cost and time. While transcripts are on (the web app's setting, read through the
  callback API and kept 60 s; on when it cannot be read) the system prompt, prompt and answer are stored as well and the
  log line carries the transcript's id, which the run page opens.
- Never breaks a run: every write is best effort and only logged (to stderr, never into a run). The web app may be
  deployed after this service, or the agent's database role may not have the grants yet (scripts/db-role.sql): then the
  tables are missing or invisible and nothing is written (checked at start-up and again every 5 minutes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from .logs import _RECORD_ATTRS
from .runs import CURRENT_RUN_ID

log = logging.getLogger("seo_agent.run_history")  # never copied into a run's log (no recursion)
calls_log = logging.getLogger("seo_agent.llm")

# The flow step running in this task (SeoFlow.step); crew worker threads fall back to RunHistory's per-run map.
CURRENT_STEP: ContextVar[str | None] = ContextVar("seo_flow_step", default=None)

LEVELS = {logging.DEBUG: "debug", logging.INFO: "info", logging.WARNING: "warning", logging.ERROR: "error", logging.CRITICAL: "error"}
# Libraries whose INFO lines are detail, not news (one line per HTTP request): kept, at debug.
DEBUG_LOGGERS = ("httpx", "httpcore", "urllib3", "psycopg", "anthropic")
MAX_MESSAGE = 4000
MAX_CONTEXT = 16_000  # bytes of JSON per line; larger values are cut
MAX_EXC = 8000
MAX_LINES_PER_RUN = 20_000  # then one warning line, and the rest of the run goes to stderr only
MAX_QUEUE = 50_000
MAX_TEXT = 200_000  # per transcript field
BATCH = 2000
FLUSH_EVERY_S = 0.5
RECHECK_S = 300.0
SETTINGS_TTL_S = 60.0
_OWN = (log.name,)
_DROP_FIELDS = {"run_id", "step", "event", "span_id", "taskName", "color_message"}


def _now() -> datetime:
    return datetime.now(UTC)


def _cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _context(fields: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe extra fields of a record, at most MAX_CONTEXT bytes (long values are shortened first)."""
    ctx = json.loads(json.dumps(dict(fields), default=str))
    if len(json.dumps(ctx)) <= MAX_CONTEXT:
        return ctx
    short = {k: (_cut(v, 500) if isinstance(v, str) else (_cut(json.dumps(v), 500) if isinstance(v, (dict, list)) else v)) for k, v in ctx.items()}
    if len(json.dumps(short)) <= MAX_CONTEXT:
        return short
    return {"truncated": True, "keys": sorted(ctx)[:50]}


# --- Where it is written ----------------------------------------------------------------------------------------------


class HistorySink(Protocol):
    async def available(self) -> bool: ...
    async def open_step(self, run_id: str, seq: int, step: str) -> int | None: ...
    async def close_step(self, step_id: int, status: str, error: str | None, seconds: float) -> None: ...
    async def close_open_steps(self, run_id: str, status: str, error: str | None) -> None: ...
    async def write_logs(self, lines: list[dict[str, Any]]) -> None: ...
    async def write_call(self, row: dict[str, Any]) -> int | None: ...


class PgHistorySink:
    """Rows only for runs that exist: a line whose run id is not a flow run (a news run's own id, say) is skipped."""

    def __init__(self, pool: Any):
        self.pool = pool
        self._ok: bool | None = None
        self._checked = 0.0

    async def available(self) -> bool:
        if self._ok is not None and time.monotonic() - self._checked < RECHECK_S:
            return self._ok
        self._checked = time.monotonic()
        try:
            async with self.pool.connection() as c:
                # information_schema lists only tables this role may use: missing grants read as "not there yet".
                cur = await c.execute(
                    """select count(distinct table_name)::int as n from information_schema.tables
                       where table_name in ('seo_flow_run_steps', 'seo_flow_run_logs', 'seo_flow_run_calls') and table_schema = any(current_schemas(false))"""
                )
                ok = int((await cur.fetchone())["n"]) == 3
        except Exception as e:
            log.warning("run history check failed", extra={"event": "run_history_check_failed", "error": type(e).__name__})
            ok = False
        if ok != self._ok:
            log.info("run history " + ("on" if ok else "off: tables missing (moks-cms migration 0036) or not granted (scripts/db-role.sql)"), extra={"event": "run_history_available", "available": ok})
        self._ok = ok
        return ok

    async def open_step(self, run_id: str, seq: int, step: str) -> int | None:
        async with self.pool.connection() as c:
            cur = await c.execute(
                """insert into seo_flow_run_steps (run_id, seq, step) select %s::uuid, %s, %s
                   where exists (select 1 from seo_flow_runs where run_id = %s::uuid) returning id""",
                (run_id, seq, step, run_id),
            )
            row = await cur.fetchone()
            return int(row["id"]) if row else None

    async def close_step(self, step_id: int, status: str, error: str | None, seconds: float) -> None:
        async with self.pool.connection() as c:
            await c.execute(
                "update seo_flow_run_steps set status = %s, error = %s, finished_at = now(), detail = jsonb_build_object('seconds', %s::float8) where id = %s and status = 'running'",
                (status, error, round(seconds, 3), step_id),
            )

    async def close_open_steps(self, run_id: str, status: str, error: str | None) -> None:
        async with self.pool.connection() as c:
            await c.execute(
                "update seo_flow_run_steps set status = %s, error = coalesce(error, %s), finished_at = now() where run_id = %s::uuid and status = 'running'",
                (status, error, run_id),
            )

    async def write_logs(self, lines: list[dict[str, Any]]) -> None:
        async with self.pool.connection() as c, c.cursor() as cur:
            await cur.executemany(
                """insert into seo_flow_run_logs (run_id, step, level, logger, event, message, context, at)
                   select %(run_id)s::uuid, %(step)s, %(level)s, %(logger)s, %(event)s, %(message)s, %(context)s::jsonb, %(at)s
                   where exists (select 1 from seo_flow_runs where run_id = %(run_id)s::uuid)""",
                [{**ln, "context": json.dumps(ln["context"])} for ln in lines],
            )

    async def write_call(self, row: dict[str, Any]) -> int | None:
        async with self.pool.connection() as c:
            cur = await c.execute(
                """insert into seo_flow_run_calls (run_id, step, agent, purpose, model, outcome, input_tokens, output_tokens,
                     cost_usd, latency_ms, system, prompt, response, error)
                   select %(run_id)s::uuid, %(step)s, %(agent)s, %(purpose)s, %(model)s, %(outcome)s, %(input_tokens)s,
                     %(output_tokens)s, %(cost_usd)s, %(latency_ms)s, %(system)s, %(prompt)s, %(response)s, %(error)s
                   where exists (select 1 from seo_flow_runs where run_id = %(run_id)s::uuid) returning id""",
                row,
            )
            got = await cur.fetchone()
            return int(got["id"]) if got else None


@dataclass
class MemoryHistorySink:
    """Tests: the same rows in lists. `runs` (a MemoryRunStore's rows) plays the part of the run-exists check."""

    runs: Mapping[str, Any] | None = None
    ok: bool = True
    steps: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _exists(self, run_id: str) -> bool:
        return self.runs is None or run_id in self.runs

    async def available(self) -> bool:
        return self.ok

    async def open_step(self, run_id: str, seq: int, step: str) -> int | None:
        if not self._exists(run_id):
            return None
        self.steps.append({"id": len(self.steps) + 1, "run_id": run_id, "seq": seq, "step": step, "status": "running", "error": None, "seconds": None})
        return len(self.steps)

    async def close_step(self, step_id: int, status: str, error: str | None, seconds: float) -> None:
        s = self.steps[step_id - 1]
        if s["status"] == "running":
            s.update(status=status, error=error, seconds=round(seconds, 3))

    async def close_open_steps(self, run_id: str, status: str, error: str | None) -> None:
        for s in self.steps:
            if s["run_id"] == run_id and s["status"] == "running":
                s.update(status=status, error=s["error"] or error)

    async def write_logs(self, lines: list[dict[str, Any]]) -> None:
        self.logs.extend(dict(ln) for ln in lines if self._exists(ln["run_id"]))

    async def write_call(self, row: dict[str, Any]) -> int | None:
        if not self._exists(row["run_id"]):
            return None
        self.calls.append({**row, "id": len(self.calls) + 1})
        return len(self.calls)


# --- The recorder -----------------------------------------------------------------------------------------------------


class RunHistory:
    def __init__(self, sink: HistorySink, settings: Callable[[], Awaitable[Mapping[str, Any]]] | None = None):
        self.sink = sink
        self._settings = settings
        self._lock = threading.Lock()
        self._queue: deque[dict[str, Any]] = deque()
        self._dropped = 0
        self._lines: dict[str, int] = {}  # run id → lines queued (MAX_LINES_PER_RUN)
        self._open: dict[str, list[tuple[int | None, str]]] = {}  # run id → open steps (step row id, name), newest last
        self._seq: dict[str, int] = {}
        self._transcripts = True
        self._settings_at = 0.0
        self.enabled = True  # last known availability; lines are queued while unknown

    # steps ---------------------------------------------------------------------------------------------------------------

    def step_of(self, run_id: str) -> str | None:
        with self._lock:
            open_ = self._open.get(run_id)
            return open_[-1][1] if open_ else None

    async def step_started(self, run_id: str, step: str) -> int | None:
        with self._lock:
            seq = self._seq[run_id] = self._seq.get(run_id, 0) + 1
        step_id = None
        try:
            if await self._available():
                step_id = await self.sink.open_step(run_id, seq, step)
        except Exception as e:
            log.warning("run history: step not recorded", extra={"event": "run_history_write_failed", "what": "step", "error": type(e).__name__})
        with self._lock:
            self._open.setdefault(run_id, []).append((step_id, step))
        return step_id

    async def step_finished(self, run_id: str, step_id: int | None, step: str, status: str, error: str | None, seconds: float) -> None:
        with self._lock:
            open_ = self._open.get(run_id, [])
            for i in range(len(open_) - 1, -1, -1):
                if open_[i] == (step_id, step):
                    del open_[i]
                    break
        if step_id is None:
            return
        try:
            await self.sink.close_step(step_id, status, _cut(error, 1000) if error else None, seconds)
        except Exception as e:
            log.warning("run history: step not closed", extra={"event": "run_history_write_failed", "what": "step", "error": type(e).__name__})

    async def run_finished(self, run_id: str, status: str) -> None:
        """The run ended: steps still open end with it (cancelled, or failed for any other ending), and the log is
        written out now so the run page shows the last lines with the final status."""
        with self._lock:
            self._open.pop(run_id, None)
            self._seq.pop(run_id, None)
            self._lines.pop(run_id, None)
            if len(self._lines) > 10_000:  # counters of runs that logged after they ended
                self._lines.clear()
        try:
            if await self._available():
                ended = {"cancelled": "cancelled", "succeeded": "succeeded"}.get(status, "failed")
                await self.sink.close_open_steps(run_id, ended, None if ended == "succeeded" else "the run ended during this step")
        except Exception as e:
            log.warning("run history: open steps not closed", extra={"event": "run_history_write_failed", "what": "step", "error": type(e).__name__})
        await self.flush()

    # log lines -----------------------------------------------------------------------------------------------------------

    def add(self, run_id: str, record: logging.LogRecord) -> None:
        """Called by RunLogHandler on any thread. Builds the row now (the record may change later) and queues it."""
        step = getattr(record, "step", None)
        if not isinstance(step, str) or not step:
            step = CURRENT_STEP.get() or self.step_of(run_id)
        level = LEVELS.get(record.levelno, "info" if record.levelno < logging.ERROR else "error")
        if level == "info" and record.name.split(".")[0] in DEBUG_LOGGERS:
            level = "debug"
        fields = {k: v for k, v in record.__dict__.items() if k not in _RECORD_ATTRS and k not in _DROP_FIELDS}
        if record.exc_info and record.exc_info[0] is not None:
            fields["exc"] = _cut(logging.Formatter().formatException(record.exc_info), MAX_EXC)
        try:
            from .tracing import current_ids

            ids = current_ids()
            if ids:
                fields.setdefault("trace_id", ids[0])
        except Exception:  # noqa: S110 - a line without its trace id is fine
            pass
        event = getattr(record, "event", None)
        key = str(run_id)
        line: dict[str, Any] = {
            "run_id": key,
            "step": _cut(step, 200) if step else None,
            "level": level,
            "logger": _cut(record.name, 100),
            "event": _cut(str(event), 80) if event else None,
            "message": _cut(record.getMessage(), MAX_MESSAGE),
            "context": _context(fields),
            "at": datetime.fromtimestamp(record.created, UTC),
        }
        with self._lock:
            n = self._lines.get(key, 0) + 1
            self._lines[key] = n
            if n > MAX_LINES_PER_RUN + 1 or len(self._queue) >= MAX_QUEUE:
                self._dropped += 1
                return
            if n == MAX_LINES_PER_RUN + 1:
                line = {**line, "level": "warning", "event": "log_truncated", "message": f"This run wrote more than {MAX_LINES_PER_RUN} log lines; the rest are only in the service log (Loki).", "context": {}}
            self._queue.append(line)

    async def flush(self) -> int:
        with self._lock:
            batch = [self._queue.popleft() for _ in range(min(BATCH, len(self._queue)))]
            dropped, self._dropped = self._dropped, 0
        if dropped:
            log.warning("run history: log lines dropped", extra={"event": "run_history_dropped", "count": dropped})
        if not batch:
            return 0
        try:
            if not await self._available():
                return 0
            await self.sink.write_logs(batch)
        except Exception as e:
            log.warning("run history: log lines not written", extra={"event": "run_history_write_failed", "what": "logs", "count": len(batch), "error": type(e).__name__})
            return 0
        return len(batch)

    async def run_forever(self, every_s: float = FLUSH_EVERY_S) -> None:
        """Background task (started in app.main.startup, outside any run): writes the queued lines."""
        await self._available()
        while True:
            await asyncio.sleep(every_s)
            while await self.flush() == BATCH:
                pass

    async def close(self) -> None:
        while await self.flush():
            pass

    async def _available(self) -> bool:
        self.enabled = await self.sink.available()
        return self.enabled

    # model calls ---------------------------------------------------------------------------------------------------------

    async def transcripts_on(self) -> bool:
        if self._settings is None:
            return self._transcripts
        if time.monotonic() - self._settings_at < SETTINGS_TTL_S:
            return self._transcripts
        self._settings_at = time.monotonic()
        try:
            got = await asyncio.wait_for(self._settings(), 5)
            self._transcripts = bool(got.get("transcripts", True))
        except Exception as e:  # keep the last answer (on until the web app says otherwise)
            log.info("run history settings unavailable", extra={"event": "run_history_settings_unavailable", "error": type(e).__name__})
        return self._transcripts

    async def model_call(self, spec: Any, usage: Any, outcome: str, answer: str | None, error: BaseException | None, seconds: float) -> None:
        run_id = getattr(spec, "run_id", None)
        if not run_id:
            return  # not part of a flow run (Ask the Blog, MCP reads)
        step = CURRENT_STEP.get() or self.step_of(run_id)
        cost = float(usage.cost_usd()) if usage is not None else 0.0
        meta = {
            "agent": getattr(spec, "agent", None),
            "purpose": spec.purpose,
            "model": getattr(usage, "model", None) or spec.model,
            "outcome": outcome,
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cost_usd": round(cost, 6),
            "latency_ms": int(getattr(usage, "latency_ms", 0) or seconds * 1000),
        }
        err = f"{type(error).__name__}: {str(error)[:500]}" if error is not None else None
        call_id = None
        try:
            if await self._available() and await self.transcripts_on():
                images = len(getattr(spec, "images", None) or [])
                prompt = spec.user if not images else f"[{images} image(s) sent before this text]\n\n{spec.user}"
                call_id = await self.sink.write_call(
                    {
                        "run_id": str(run_id),
                        "step": step,
                        **meta,
                        "system": _cut("\n\n---\n\n".join(spec.cached_system), MAX_TEXT),
                        "prompt": _cut(prompt, MAX_TEXT),
                        "response": _cut(answer, MAX_TEXT) if answer is not None else None,
                        "error": err,
                    }
                )
        except Exception as e:
            log.warning("run history: transcript not written", extra={"event": "run_history_write_failed", "what": "call", "error": type(e).__name__})
        who = meta["agent"] or spec.purpose
        text = f"{who}: {meta['model']} · {meta['input_tokens']:,} in / {meta['output_tokens']:,} out · ${cost:.4f} · {seconds:.1f}s"
        calls_log.log(
            logging.INFO if outcome == "ok" else logging.WARNING,
            f"model call {text}" if outcome == "ok" else f"model call {outcome}: {text}",
            extra={"event": "llm_call", "run_id": str(run_id), "step": step, **meta, "seconds": round(seconds, 3), "call_id": call_id, "error": err},
        )


class RunLogHandler(logging.Handler):
    """Root-logger handler: a record that belongs to a flow run is also queued for that run's log."""

    def __init__(self, history: RunHistory):
        super().__init__(logging.DEBUG)
        self.history = history

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.name.startswith(_OWN) or not self.history.enabled:
                return
            run_id = CURRENT_RUN_ID.get() or getattr(record, "run_id", None)
            if run_id:
                self.history.add(str(run_id), record)
        except Exception:  # noqa: S110 - logging must never raise into the code that logged
            pass


def install(history: RunHistory, logger: logging.Logger | None = None) -> RunLogHandler:
    root = logger or logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, RunLogHandler):
            root.removeHandler(h)
    handler = RunLogHandler(history)
    root.addHandler(handler)
    return handler


def uninstall(logger: logging.Logger | None = None) -> None:
    root = logger or logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, RunLogHandler):
            root.removeHandler(h)


# --- Model calls --------------------------------------------------------------------------------------------------------


def _answer_text(out: Any) -> str:
    if isinstance(out, BaseModel):
        return out.model_dump_json(indent=2)
    return out if isinstance(out, str) else json.dumps(out, default=str)


class RecordingLlm:
    """The service's Llm with every flow-run model call recorded in the run history. Everything else (stream, guard,
    a test model's queue) is the wrapped Llm's own."""

    def __init__(self, inner: Any, history: RunHistory):
        self._inner = inner
        self._history = history

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def _record(self, spec: Any, usage: Any, outcome: str, answer: str | None, error: BaseException | None, started: float) -> None:
        try:
            await self._history.model_call(spec, usage, outcome, answer, error, time.monotonic() - started)
        except Exception:  # noqa: S110 - never fails the call it describes
            pass

    async def structured(self, spec: Any, schema: type[Any]) -> tuple[Any, Any]:
        started = time.monotonic()
        try:
            out, usage = await self._inner.structured(spec, schema)
        except Exception as e:
            await self._record(spec, getattr(e, "usage", None), _outcome(e), None, e, started)
            raise
        await self._record(spec, usage, "ok", _answer_text(out), None, started)
        return out, usage

    async def text(self, spec: Any) -> tuple[str, Any]:
        started = time.monotonic()
        try:
            out, usage = await self._inner.text(spec)
        except Exception as e:
            await self._record(spec, getattr(e, "usage", None), _outcome(e), None, e, started)
            raise
        await self._record(spec, usage, "ok", out, None, started)
        return out, usage

    def stream(self, spec: Any, **kw: Any) -> Any:
        return self._inner.stream(spec, **kw)


def _outcome(e: BaseException) -> str:
    name = type(e).__name__
    return {"Truncated": "truncated", "InvalidOutput": "invalid", "Refused": "refused", "BudgetExceeded": "budget", "TokenCapExceeded": "budget"}.get(name, "error")
