"""Typed (Pydantic) state for every CrewAI Flow, plus the strict kickoff input model of each flow.

A Flow's state is what CrewAI persists after every step (app/runs.py → seo_flow_runs.state_json). Inputs are
validated against the flow's *Input* model (extra fields refused, bounded sizes) before a run is created, so the API,
the CronJob trigger and the admin MCP `kickoff_flow` tool all go through the same check.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

UUID = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
Trigger = Literal["cron", "manual", "approval", "flow", "mcp"]


class RunState(BaseModel):
    """Fields every flow state has. `id` is the run id (CrewAI uses it as the persistence key)."""

    model_config = ConfigDict(extra="ignore")
    id: str = ""
    triggered_by: str = "manual"
    outcome: str | None = None
    errors: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    tokens: int = 0

    def add_cost(self, usd: float, tokens: int = 0) -> None:
        self.cost_usd = round(self.cost_usd + float(usd or 0), 6)
        self.tokens += int(tokens or 0)


# --- Inputs (strict) ----------------------------------------------------------------------------------------------


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoInput(_In):
    pass


class JobInput(_In):
    slot: str | None = Field(default=None, max_length=40, pattern=r"^[0-9A-Za-z:_-]+$")


class WeeklyInput(_In):
    limit: int | None = Field(default=None, ge=1, le=25)


class OptimizeInput(_In):
    draft_id: str = Field(pattern=UUID)
    article_id: str = Field(pattern=UUID)
    recommendation_ids: list[str] = Field(default_factory=list, max_length=50)
    extra: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    trigger: str = Field(default="manual", max_length=20)
    feedback: str | None = Field(default=None, max_length=2000)


class LinkingInput(_In):
    article_ids: list[str] = Field(default_factory=list, max_length=50)


class NewArticleInput(_In):
    draft_id: str = Field(pattern=UUID)
    calendar_id: str = Field(pattern=UUID)
    article_id: str | None = Field(default=None, pattern=UUID)
    feedback: str | None = Field(default=None, max_length=2000)


class PublishInput(_In):
    draft_id: str = Field(pattern=UUID)
    approval_id: str = Field(min_length=1, max_length=80)
    accepted_change_ids: list[str] = Field(max_length=200)
    edits: dict[str, str] = Field(default_factory=dict)
    actor: str | None = Field(default=None, max_length=200)


class ReindexInput(_In):
    article_ids: list[str] | None = Field(default=None, max_length=200)
    full: bool = False


class ReportInput(_In):
    period: Literal["weekly", "monthly"] = "weekly"


# --- States -------------------------------------------------------------------------------------------------------


class JobState(RunState):
    job_type: str = ""
    slot: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)


class WeeklyState(RunState):
    limit: int | None = None
    prepared: dict[str, Any] = Field(default_factory=dict)
    children: list[dict[str, Any]] = Field(default_factory=list)


class OptimizeState(RunState):
    draft_id: str = ""
    article_id: str = ""
    recommendation_ids: list[str] = Field(default_factory=list)
    extra: list[dict[str, Any]] = Field(default_factory=list)
    trigger: str = "manual"
    feedback: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    related: list[dict[str, Any]] = Field(default_factory=list)
    plan: dict[str, Any] = Field(default_factory=dict)
    change_set: dict[str, Any] | None = None
    crew: dict[str, Any] = Field(default_factory=dict)
    changes: list[dict[str, Any]] = Field(default_factory=list)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0


class LinkingState(RunState):
    article_ids: list[str] = Field(default_factory=list)
    candidates: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)


class NewArticleState(RunState):
    draft_id: str = ""
    calendar_id: str = ""
    article_id: str | None = None
    feedback: str | None = None
    brief: dict[str, Any] = Field(default_factory=dict)
    document: dict[str, Any] | None = None
    crew: dict[str, Any] = Field(default_factory=dict)
    changes: list[dict[str, Any]] = Field(default_factory=list)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0


class PublishState(RunState):
    draft_id: str = ""
    article_id: str | None = None
    approval_id: str = ""
    accepted_change_ids: list[str] = Field(default_factory=list)
    edits: dict[str, str] = Field(default_factory=dict)
    actor: str | None = None
    publish_result: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    submission_results: dict[str, Any] = Field(default_factory=dict)
    reindex: dict[str, Any] = Field(default_factory=dict)
    measurement: dict[str, Any] = Field(default_factory=dict)


class ReindexState(RunState):
    article_ids: list[str] | None = None
    full: bool = False
    result: dict[str, Any] = Field(default_factory=dict)


class AskGapsState(RunState):
    result: dict[str, Any] = Field(default_factory=dict)


class ReportState(RunState):
    period: str = "weekly"
    report_id: str | None = None
    numbers: dict[str, Any] = Field(default_factory=dict)
    narrative: str = ""
    check: dict[str, Any] = Field(default_factory=dict)
    crew: dict[str, Any] = Field(default_factory=dict)
    pdf: dict[str, Any] = Field(default_factory=dict)
    notified: dict[str, Any] = Field(default_factory=dict)
