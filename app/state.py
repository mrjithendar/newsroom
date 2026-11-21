"""Typed (Pydantic) state for every CrewAI Flow, plus the strict kickoff input model of each flow.

Forked from moks-seo-agent/app/state.py@bb9e553 on 2026-11-21.

A Flow's state is what CrewAI persists after every step (app/runs.py → seo_flow_runs.state_json). Inputs are
validated against the flow's *Input* model (extra fields refused, bounded sizes) before a run is created, so the API,
the CronJob trigger and the admin MCP `kickoff_flow` tool all go through the same check.

`feedback` on the drafting inputs is the reviewer's regenerate feedback. When the run was started from the admin MCP
(`triggered_by="mcp"`), the crews treat it as untrusted data (a data block), never as instructions; the MCP tool also
refuses a draft that is not new or generation_failed (app/mcp/admin.py).
"""

from __future__ import annotations

from typing import Literal

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
