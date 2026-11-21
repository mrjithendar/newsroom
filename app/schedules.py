"""Flow schedules: moks-cms table seo_flow_schedules (migration 0037) and the one dispatcher that starts what is due.

Forked from moks-seo-agent/app/schedules.py@bb9e553 on 2026-11-21.

Until 2026-11-19 each scheduled flow had a Kubernetes CronJob of its own (15 in jithendar.com's kubernetes/). Now the
schedules are rows an admin edits in the CrewAI console (/admin/crewai/schedules, D-98), and one CronJob calls
POST /schedules/dispatch every minute (`python -m app.trigger dispatch`).

On each call, for every enabled schedule:
- `due` is the latest fire time of its cron, in its time zone, inside the catch-up window (SCHEDULE_CATCHUP_MINUTES,
  default 60, as the CronJobs' startingDeadlineSeconds was) and after `armed_at` (when its cron, time zone or switch
  last changed), so a new or edited schedule never fires a time that has already passed. Missed times are not run one
  by one: as with a CronJob under concurrencyPolicy Forbid, only the latest one runs.
- The slot (that time in UTC, `20261119T0610Z`) is claimed with one conditional UPDATE (`last_slot < slot`), so a
  second call, a retried CronJob or a second pod start it once.
- The flow starts like any other run: FlowRunner.kickoff(triggered_by="cron", actor="schedule:<name>") with the
  schedule's inputs, which the flow's input model checks. A run of the same flow and target that is still going
  counts as fired (`already_running`), as it did with the CronJobs.
- The outcome goes back on the row: last_status (started | already_running | failed), last_run_id, last_error. Those
  and last_slot / last_fired_at are the only columns the agent writes (scripts/db-role.sql).

Only flows in CRON_FLOWS can be scheduled: never publish_flow, the drafting flows or the per-story news flows.

Two services dispatch from this one table: the seo-agent and the news agent (moks-seo-agent, mrjithendar/newsroom),
each from its own CronJob. Each looks only at the rows whose flow it runs (`served`, = the keys of app/flows SPECS) and
leaves the others untouched (not claimed, no status written, counted as `skipped`), so neither marks the other's
schedules "cannot be scheduled". The slot claim still makes a time start once if both ever served the same flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from .news.cron import Cron, CronError
from .runs import RunError

log = logging.getLogger("news_agent.schedules")

SLOT_FORMAT = "%Y%m%dT%H%MZ"


class SchedulesUnavailable(Exception):
    """The schedules table is missing (moks-cms migration 0037 not applied yet)."""


class ScheduleStore(Protocol):
    async def enabled(self) -> list[dict[str, Any]]: ...
    async def claim(self, schedule_id: str, slot: str) -> bool: ...
    async def record(self, schedule_id: str, slot: str, status: str, run_id: str | None, error: str | None) -> None: ...


class PgScheduleStore:
    def __init__(self, pool: Any):
        self.pool = pool

    async def enabled(self) -> list[dict[str, Any]]:
        import psycopg

        try:
            async with self.pool.connection() as c:
                cur = await c.execute("select id::text as id, name, flow_name, cron, timezone, inputs, armed_at, last_slot from seo_flow_schedules where enabled order by name")
                return [dict(r) for r in await cur.fetchall()]
        except psycopg.errors.UndefinedTable as e:
            raise SchedulesUnavailable("table seo_flow_schedules is missing: apply moks-cms migration 0037 (the web app applies it when it starts)") from e

    async def claim(self, schedule_id: str, slot: str) -> bool:
        async with self.pool.connection() as c:
            cur = await c.execute(
                """update seo_flow_schedules set last_slot = %(slot)s, last_fired_at = now(), last_status = 'starting', last_run_id = null, last_error = null
                   where id = %(id)s::uuid and enabled and (last_slot is null or last_slot < %(slot)s) returning id""",
                {"id": schedule_id, "slot": slot},
            )
            return (await cur.fetchone()) is not None

    async def record(self, schedule_id: str, slot: str, status: str, run_id: str | None, error: str | None) -> None:
        async with self.pool.connection() as c:
            await c.execute(
                "update seo_flow_schedules set last_status = %(status)s, last_run_id = %(run)s::uuid, last_error = %(error)s where id = %(id)s::uuid and last_slot = %(slot)s",
                {"id": schedule_id, "slot": slot, "status": status, "run": run_id, "error": (error or None) and error[:1000]},
            )


@dataclass
class MemoryScheduleStore:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, **row: Any) -> dict[str, Any]:
        r = {"id": row.get("id") or f"s{len(self.rows) + 1}", "enabled": True, "timezone": "Etc/UTC", "inputs": {}, "armed_at": datetime(2000, 1, 1, tzinfo=UTC), "last_slot": None, "last_status": None, "last_run_id": None, "last_error": None, "last_fired_at": None, **row}
        self.rows[r["id"]] = r
        return r

    async def enabled(self) -> list[dict[str, Any]]:
        return [dict(r) for r in sorted(self.rows.values(), key=lambda r: r["name"]) if r["enabled"]]

    async def claim(self, schedule_id: str, slot: str) -> bool:
        r = self.rows.get(schedule_id)
        if not r or not r["enabled"] or (r["last_slot"] is not None and r["last_slot"] >= slot):
            return False
        r.update(last_slot=slot, last_fired_at=datetime.now(UTC), last_status="starting", last_run_id=None, last_error=None)
        return True

    async def record(self, schedule_id: str, slot: str, status: str, run_id: str | None, error: str | None) -> None:
        r = self.rows.get(schedule_id)
        if r and r["last_slot"] == slot:
            r.update(last_status=status, last_run_id=run_id, last_error=error)


def due_slot(row: dict[str, Any], now: datetime, catchup: timedelta) -> tuple[datetime | None, str | None]:
    """(the fire time to run now, or None; an error with the cron or time zone, or None)."""
    try:
        cron = Cron(str(row["cron"]), str(row.get("timezone") or "UTC"))
    except CronError as e:
        return None, str(e)
    due = cron.last_due(now, catchup)
    armed = row.get("armed_at")
    if due is None or (isinstance(armed, datetime) and due <= armed):
        return None, None
    return due, None


async def dispatch(store: ScheduleStore, runner: Any, *, schedulable: frozenset[str], served: frozenset[str] | None = None, now: datetime | None = None, catchup: timedelta = timedelta(minutes=60)) -> dict[str, Any]:
    """One tick: start every enabled schedule of this service that is due. Never raises for one schedule; each outcome
    is reported. `served`: the flows this service runs; a row for any other flow is the other service's (skipped
    before it is claimed). None = every row (tests)."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    rows = await store.enabled()
    mine = [r for r in rows if served is None or str(r["flow_name"]) in served]
    out: dict[str, Any] = {"at": now.isoformat(), "checked": len(mine), "skipped": len(rows) - len(mine), "fired": [], "errors": []}
    for row in mine:
        name, flow = str(row["name"]), str(row["flow_name"])
        due, error = due_slot(row, now, catchup)
        if error:
            out["errors"].append({"schedule": name, "error": error})
            continue
        if due is None:
            continue
        slot = due.astimezone(UTC).strftime(SLOT_FORMAT)
        if not await store.claim(str(row["id"]), slot):
            continue  # another call already started this time (or a later one)
        status, run_id, err = "failed", None, None
        if flow not in schedulable:
            err = f"{flow} cannot be scheduled"
        else:
            try:
                r = await runner.kickoff(flow, dict(row.get("inputs") or {}), triggered_by="cron", actor=f"schedule:{name}")
                run_id = str(r.get("run_id") or "") or None
                status = "already_running" if r.get("existing") else "started"
            except RunError as e:
                err = str(e)[:500]
            except Exception as e:  # the next schedule still runs; this one says why it did not
                err = f"{type(e).__name__}: {str(e)[:300]}"
        await store.record(str(row["id"]), slot, status, run_id, err)
        fired = {"schedule": name, "flow": flow, "slot": slot, "status": status, "run_id": run_id, "error": err}
        out["fired"].append(fired)
        log.info(f"schedule {name}: {status}", extra={"event": "schedule_fired", **fired})
    return out
