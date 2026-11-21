"""Two services share seo_flow_runs and seo_flow_schedules (the seo-agent and this one): each touches only its own
flows' rows (app/runs.py, app/schedules.py). And nothing here can publish or be started from MCP (app/flows)."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from app.flows import CRON_FLOWS, OWNED_FLOWS, SPECS, check_kickoff
from app.runs import MemoryRunStore, PgRunStore, RunError
from app.schedules import MemoryScheduleStore, dispatch


def seo_row(status: str = "running", flow: str = "gsc_sync_flow") -> dict[str, Any]:
    return {"run_id": str(uuid.uuid4()), "flow_name": flow, "target_id": "site", "status": status, "inputs": {}, "triggered_by": "cron"}


async def test_a_restart_fails_only_this_services_runs() -> None:
    store = MemoryRunStore(flows=OWNED_FLOWS)
    theirs = await store.create(seo_row())
    mine = await store.create({**seo_row(flow="news_job_flow"), "target_id": "job:1"})
    swept = await store.sweep_orphans()
    assert [r["run_id"] for r in swept] == [mine["run_id"]]
    assert store.rows[theirs["run_id"]]["status"] == "running"


async def test_lists_counts_and_reads_see_only_this_services_runs() -> None:
    store = MemoryRunStore(flows=OWNED_FLOWS)
    theirs = await store.create(seo_row(status="queued"))
    mine = await store.create({**seo_row(status="queued", flow="news_ingest_flow")})
    assert [r["run_id"] for r in await store.list_runs()] == [mine["run_id"]]
    assert await store.list_runs(flow="gsc_sync_flow") == []
    assert await store.count_status("queued") == 1
    assert await store.get(theirs["run_id"]) is None and await store.get(mine["run_id"])


async def test_another_services_run_cannot_be_cancelled_here() -> None:
    from app.main import memory_container
    from tests.conftest import FakeWebApp, settings

    c = memory_container(settings(), llm=None, web=FakeWebApp())
    theirs = await c.store.create(seo_row())
    with pytest.raises(RunError) as e:
        await c.runner.cancel(theirs["run_id"])
    assert e.value.status == 404 and c.store.rows[theirs["run_id"]]["status"] == "running"


async def test_the_dispatcher_leaves_the_other_services_schedules_alone() -> None:
    now = datetime(2026, 11, 21, 7, 0, 30, tzinfo=UTC)
    store = MemoryScheduleStore()
    seo = store.add(name="gsc-sync", flow_name="gsc_sync_flow", cron="0 7 * * *")
    news = store.add(name="news-train", flow_name="news_train_flow", cron="0 7 * * *")

    class Runner:
        def __init__(self) -> None:
            self.started: list[str] = []

        async def kickoff(self, name: str, inputs: dict[str, Any], **_: Any) -> dict[str, Any]:
            self.started.append(name)
            return {"run_id": str(uuid.uuid4())}

    r = Runner()
    out = await dispatch(store, r, schedulable=CRON_FLOWS, served=OWNED_FLOWS, now=now)
    assert r.started == ["news_train_flow"] and out["checked"] == 1 and out["skipped"] == 1
    assert store.rows[seo["id"]]["last_slot"] is None and store.rows[seo["id"]]["last_status"] is None  # not claimed
    assert store.rows[news["id"]]["last_status"] == "started"
    # The seo-agent's dispatcher (its own flows) then starts its row once; a second tick starts nothing.
    other = Runner()
    await dispatch(store, other, schedulable=frozenset({"gsc_sync_flow"}), served=frozenset({"gsc_sync_flow"}), now=now)
    await dispatch(store, other, schedulable=frozenset({"gsc_sync_flow"}), served=frozenset({"gsc_sync_flow"}), now=now + timedelta(seconds=20))
    assert other.started == ["gsc_sync_flow"]


def test_no_flow_can_be_started_by_an_approval_or_mcp() -> None:
    assert set(SPECS) == OWNED_FLOWS and "publish_flow" not in SPECS
    for name in SPECS:
        for trigger in ("approval", "mcp"):
            with pytest.raises(RunError):
                check_kickoff(name, trigger)
    with pytest.raises(RunError):
        check_kickoff("crew_test_flow", "cron")
    with pytest.raises(RunError):
        check_kickoff("news_draft_flow", "cron")
    check_kickoff("news_dispatch_flow", "cron")


@pytest.mark.postgres
async def test_postgres_scoping() -> None:
    url = os.environ.get("NEWS_AGENT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set NEWS_AGENT_TEST_DATABASE_URL to check the scoped run store on a real PostgreSQL")
    import psycopg
    from psycopg.conninfo import make_conninfo
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    database = f"scope_{uuid.uuid4().hex[:10]}"
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as admin:
        await admin.execute(f"create database {database}")
        try:
            pool = AsyncConnectionPool(make_conninfo(url, dbname=database), min_size=1, max_size=2, open=False, kwargs={"autocommit": True, "row_factory": dict_row})
            await pool.open()
            try:
                async with pool.connection() as c:
                    await c.execute(
                        """create table seo_flow_runs (run_id uuid primary key, flow_name text not null, target_id text not null, draft_id uuid, status text not null,
                           current_step text, state_json jsonb, inputs jsonb, outputs jsonb, error text, tokens integer not null default 0, cost_usd double precision not null default 0,
                           started_at timestamptz not null default now(), finished_at timestamptz, updated_at timestamptz not null default now(), triggered_by text, correlation_id text, actor text)"""
                    )
                mine, theirs = PgRunStore(pool, OWNED_FLOWS), PgRunStore(pool, frozenset({"gsc_sync_flow"}))
                a = await theirs.create(seo_row())
                b = await mine.create({**seo_row(flow="news_job_flow"), "target_id": "job:1"})
                assert [str(r["run_id"]) for r in await mine.sweep_orphans()] == [str(b["run_id"])]
                assert (await theirs.get(str(a["run_id"])))["status"] == "running"  # type: ignore[index]
                assert await mine.get(str(a["run_id"])) is None and await mine.list_runs(flow="gsc_sync_flow") == []
                assert [str(r["run_id"]) for r in await mine.list_runs()] == [str(b["run_id"])]
                assert await mine.count_status("running") == 0 and await theirs.count_status("running") == 1
            finally:
                await pool.close()
        finally:
            await admin.execute(f"drop database if exists {database} with (force)")


def test_the_default_news_schedules_are_flows_this_dispatcher_starts() -> None:
    """moks-cms migration 0037 seeds news-dispatch, news-train and news-style-review: each must be a flow this service
    schedules, with inputs the flow accepts and a cron the evaluator reads. The other twelve rows are the seo-agent's."""
    import json
    import re

    from app.news.cron import Cron
    from tests.moks_cms import www

    sql = www("src/db/migrations/0037_crewai_schedules_crews.sql").read_text()
    rows = re.findall(r"\('([a-z0-9-]+)', '([a-z0-9_]+_flow)', '(?:[^']|'')*', '([^']+)', '(\{[^']*\})'\)", sql)
    mine = [r for r in rows if r[1] in SPECS]
    assert {flow for _, flow, _, _ in mine} == CRON_FLOWS and len(mine) == 3
    for _name, flow, cron, inputs in mine:
        Cron(cron, "Etc/UTC")
        SPECS[flow].input_model.model_validate(json.loads(inputs))
