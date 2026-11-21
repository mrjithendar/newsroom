"""Run history (app/run_history.py): the steps, log lines and model calls the admin's CrewAI console shows per run."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Any

import pytest

from app import run_history as rh
from app.llm import BudgetGuard, CallSpec, FakeLlm, MemorySpendStore, Truncated, Usage
from app.main import memory_container
from app.run_history import MemoryHistorySink, RecordingLlm, RunHistory, install, uninstall
from tests.conftest import settings

RUN = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def info_level() -> Any:
    """The service logs at INFO (app/logs.py configure_logging); pytest leaves the root logger at WARNING."""
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.INFO)
    yield
    root.setLevel(before)


@pytest.fixture
def history() -> Any:
    h = RunHistory(MemoryHistorySink(runs={RUN: {}}))
    install(h)
    yield h
    uninstall()


async def kickoff(c: Any, name: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    r = await c.runner.kickoff(name, inputs or {}, triggered_by="manual")
    return await c.runner.wait(r["run_id"], 30)


async def test_a_news_flow_run_records_its_log() -> None:
    from tests.news_fixtures import NewsWeb, seed

    llm = FakeLlm(BudgetGuard(MemorySpendStore()))
    web = NewsWeb()
    c = memory_container(settings(), llm=llm, web=web, history=True)
    c.deps.extra["env"] = {"NEWS_DESK_ENABLED": "true"}
    seed(c.deps.extra["news_store"])
    install(c.history)
    try:
        r = await kickoff(c, "news_ingest_flow", {"force": True})
    finally:
        uninstall()
    run_id = str(r["run_id"])
    assert r["status"] == "succeeded", r
    lines = [ln for ln in c.history.sink.logs if ln["run_id"] == run_id]
    events = [ln["event"] for ln in lines]
    assert events[0] == "flow_started" and events[-1] == "flow_finished"
    by_event = {ln["event"]: ln for ln in lines}
    assert by_event["flow_finished"]["context"]["status"] == "succeeded" and "run_id" not in by_event["flow_finished"]["context"]


async def test_a_child_run_logs_into_its_own_run() -> None:
    """A flow that starts another (a news job's drafts) runs the kickoff with its own run id in context."""
    from app.runs import CURRENT_RUN_ID
    from tests.news_fixtures import NewsWeb, seed

    c = memory_container(settings(), llm=FakeLlm(BudgetGuard(MemorySpendStore())), web=NewsWeb(), history=True)
    c.deps.extra["env"] = {"NEWS_DESK_ENABLED": "true"}
    seed(c.deps.extra["news_store"])
    parent = str(uuid.uuid4())
    await c.store.create({"run_id": parent, "flow_name": "news_dispatch_flow", "target_id": "site", "status": "running", "inputs": {}, "triggered_by": "cron"})
    install(c.history)
    try:
        token = CURRENT_RUN_ID.set(parent)
        try:
            child = (await c.runner.kickoff("news_ingest_flow", {}, triggered_by="flow", correlation_id=parent))["run_id"]
        finally:
            CURRENT_RUN_ID.reset(token)
        await c.runner.wait(child, 30)
    finally:
        uninstall()
    lines = c.history.sink.logs
    assert [ln["event"] for ln in lines if ln["run_id"] == parent] == []
    assert next(ln["event"] for ln in lines if ln["run_id"] == child) == "flow_started"


async def test_crew_thread_lines_land_in_the_running_step(history: RunHistory) -> None:
    step_id = await history.step_started(RUN, "draft")
    logger = logging.getLogger("news_agent.crews")

    def worker() -> None:  # run_in_executor: no context variables, only the run id passed with extra=
        logger.info("crew task started: editorial.writer", extra={"event": "crew_task_started", "agent": "writer", "run_id": RUN})

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    await history.step_finished(RUN, step_id, "draft", "succeeded", None, 1.5)
    logger.info("after the step", extra={"run_id": RUN})
    await history.flush()
    lines = history.sink.logs
    assert [(ln["step"], ln["event"], ln["context"].get("agent")) for ln in lines] == [("draft", "crew_task_started", "writer"), (None, None, None)]
    assert history.sink.steps[0]["status"] == "succeeded" and history.sink.steps[0]["seconds"] == 1.5


async def test_lines_of_unknown_runs_and_own_warnings_are_not_kept(history: RunHistory) -> None:
    logging.getLogger("news_agent.news").info("news run line", extra={"run_id": str(uuid.uuid4())})  # a news_runs id
    logging.getLogger("news_agent.run_history").warning("own", extra={"run_id": RUN})
    logging.getLogger("httpx").info("HTTP Request: POST http://web/internal", extra={"run_id": RUN})
    await history.flush()
    assert [(ln["logger"], ln["level"]) for ln in history.sink.logs] == [("httpx", "debug")]


async def test_exceptions_and_long_values_are_kept_short(history: RunHistory) -> None:
    try:
        raise ValueError("bad answer")
    except ValueError:
        logging.getLogger("news_agent.flows").exception("step failed", extra={"run_id": RUN, "blob": "x" * 40_000})
    await history.flush()
    (line,) = history.sink.logs
    assert line["level"] == "error" and "ValueError: bad answer" in line["context"]["exc"]
    assert len(line["context"]["blob"]) <= 500


async def test_a_run_stops_logging_at_the_line_cap(history: RunHistory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rh, "MAX_LINES_PER_RUN", 3)
    for i in range(10):
        logging.getLogger("news_agent.flows").info(f"line {i}", extra={"run_id": RUN})
    await history.flush()
    assert [ln["message"] for ln in history.sink.logs][:3] == ["line 0", "line 1", "line 2"]
    assert len(history.sink.logs) == 4 and history.sink.logs[-1]["event"] == "log_truncated"


async def test_ending_a_run_closes_its_open_steps(history: RunHistory) -> None:
    await history.step_started(RUN, "draft")
    await history.run_finished(RUN, "cancelled")
    assert history.sink.steps[0]["status"] == "cancelled" and history.sink.steps[0]["error"] == "the run ended during this step"
    assert history.step_of(RUN) is None


async def test_without_the_tables_nothing_is_written_or_raised() -> None:
    sink = MemoryHistorySink(ok=False)
    h = RunHistory(sink)
    install(h)
    try:
        assert await h.step_started(RUN, "draft") is None
        logging.getLogger("news_agent.flows").info("x", extra={"run_id": RUN})
        await h.flush()
        await h.run_finished(RUN, "succeeded")
    finally:
        uninstall()
    assert sink.steps == [] and sink.logs == [] and h.enabled is False


def spec(**kw: Any) -> CallSpec:
    return CallSpec(model="claude-sonnet-5-5", purpose="crew:editorial", prompt_version="seo-crew-v6", cached_system=["rules", "agent prompt"], user="Write it.", run_id=RUN, agent="writer", **kw)


class Model:
    def __init__(self, answer: Any = "The article.", error: Exception | None = None):
        self.answer, self.error, self.guard = answer, error, "the guard"

    async def text(self, s: CallSpec) -> tuple[str, Usage]:
        if self.error is not None:
            raise self.error
        return self.answer, Usage(model=s.model, input_tokens=1200, output_tokens=300, latency_ms=2100)


async def test_model_calls_get_a_log_line_and_a_transcript(history: RunHistory) -> None:
    llm = RecordingLlm(Model(), history)
    assert llm.guard == "the guard"  # everything else is the wrapped model's
    await history.step_started(RUN, "draft")
    text, _ = await llm.text(spec())
    await history.flush()
    assert text == "The article."
    (call,) = history.sink.calls
    assert call["system"] == "rules\n\n---\n\nagent prompt" and call["prompt"] == "Write it." and call["response"] == "The article."
    assert (call["step"], call["agent"], call["input_tokens"], call["output_tokens"], call["outcome"]) == ("draft", "writer", 1200, 300, "ok")
    (line,) = [ln for ln in history.sink.logs if ln["event"] == "llm_call"]
    assert line["context"]["call_id"] == call["id"] and line["step"] == "draft" and line["message"].startswith("model call writer: claude-sonnet-5-5 · 1,200 in / 300 out")


async def test_a_failed_model_call_is_recorded_and_still_raises(history: RunHistory) -> None:
    usage = Usage(model="claude-sonnet-5-5", input_tokens=900, output_tokens=16_000)
    llm = RecordingLlm(Model(error=Truncated("output cut off at max_tokens=16000", usage)), history)
    with pytest.raises(Truncated):
        await llm.text(spec())
    await history.flush()
    (call,) = history.sink.calls
    assert call["outcome"] == "truncated" and call["response"] is None and "max_tokens" in call["error"] and call["output_tokens"] == 16_000
    (line,) = history.sink.logs
    assert line["level"] == "warning" and line["message"].startswith("model call truncated:")


async def test_transcripts_off_keeps_only_the_log_line() -> None:
    asked: list[int] = []

    async def settings_() -> dict[str, Any]:
        asked.append(1)
        return {"transcripts": False}

    h = RunHistory(MemoryHistorySink(runs={RUN: {}}), settings_)
    install(h)
    try:
        llm = RecordingLlm(Model(), h)
        await llm.text(spec())
        await llm.text(spec())
        await h.flush()
    finally:
        uninstall()
    assert h.sink.calls == [] and len(asked) == 1  # cached for a minute
    assert [ln["context"]["call_id"] for ln in h.sink.logs] == [None, None]


async def test_unreadable_settings_keep_transcripts_on() -> None:
    async def settings_() -> dict[str, Any]:
        raise RuntimeError("web app down")

    h = RunHistory(MemoryHistorySink(runs={RUN: {}}), settings_)
    assert await h.transcripts_on() is True


async def test_calls_outside_a_flow_run_are_not_recorded(history: RunHistory) -> None:
    await RecordingLlm(Model(), history).text(spec(draft_id=None).__class__(**{**spec().__dict__, "run_id": None}))
    await history.flush()
    assert history.sink.calls == [] and history.sink.logs == []


async def test_the_writer_runs_in_the_background(history: RunHistory) -> None:
    task = asyncio.create_task(history.run_forever(every_s=0.01))
    try:
        logging.getLogger("news_agent.flows").info("queued", extra={"run_id": RUN})
        for _ in range(100):
            if history.sink.logs:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
    assert [ln["message"] for ln in history.sink.logs] == ["queued"]
