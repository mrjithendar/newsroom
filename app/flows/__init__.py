"""The news agent's flow catalogue (docs/ARCHITECTURE.md) and who may start what.

- `SPECS`: every flow this service runs, its strict input model, and its lock target (one active run per flow +
  target). These names are also what this service owns in the shared `seo_flow_runs` and `seo_flow_schedules` tables:
  it recovers, lists, cancels and dispatches only these (app/runs.py, app/schedules.py). The seo-agent does the same
  with its own names, so the two never touch each other's rows.
- No flow here can publish: publishing a news article is the seo-agent's `publish_flow`, started only by a human
  approval in the web app's review queue. `approval` and `mcp` triggers are refused for every flow (`check_kickoff`).
- `CRON_FLOWS`: what a schedule may start (app/schedules.py, the CrewAI console's Schedules).
- `crew_test_flow`: the console's Test run of the news crew's configuration (app/flows/crew_test.py); an admin only.
"""

from __future__ import annotations

from typing import Any

from ..runs import FlowSpec, RunError
from .crew_test import CrewTestFlow, CrewTestInput
from .news import (
    NEWS_INPUTS,
    NewsDispatchFlow,
    NewsDraftFlow,
    NewsImageFlow,
    NewsIngestFlow,
    NewsJobFlow,
    NewsPlaygroundFlow,
    NewsStyleReviewFlow,
    NewsTrainFlow,
)

AGENT = "news"  # GET /flows and GET /crews tag every entry with it: moks-cms merges the two agents' catalogues


def _site(_i: dict[str, Any]) -> str:
    return "site"


SPECS: dict[str, FlowSpec] = {
    # Lock targets: one active run per job; one per job + story for drafts.
    "news_dispatch_flow": FlowSpec("news_dispatch_flow", NewsDispatchFlow.create, NEWS_INPUTS["news_dispatch_flow"], _site),
    "news_job_flow": FlowSpec("news_job_flow", NewsJobFlow.create, NEWS_INPUTS["news_job_flow"], lambda i: f"job:{i['job_id']}"),
    "news_draft_flow": FlowSpec("news_draft_flow", NewsDraftFlow.create, NEWS_INPUTS["news_draft_flow"], lambda i: f"{i['job_id']}:{i['cluster_ids'][0]}:{i.get('mode', 'single')}"),
    "news_ingest_flow": FlowSpec("news_ingest_flow", NewsIngestFlow.create, NEWS_INPUTS["news_ingest_flow"], _site),
    "news_image_flow": FlowSpec("news_image_flow", NewsImageFlow.create, NEWS_INPUTS["news_image_flow"], lambda i: str(i["article_id"])),
    "news_train_flow": FlowSpec("news_train_flow", NewsTrainFlow.create, NEWS_INPUTS["news_train_flow"], _site),
    "news_style_review_flow": FlowSpec("news_style_review_flow", NewsStyleReviewFlow.create, NEWS_INPUTS["news_style_review_flow"], _site),
    "news_playground_flow": FlowSpec("news_playground_flow", NewsPlaygroundFlow.create, NEWS_INPUTS["news_playground_flow"], lambda i: f"{i['job_id']}:{i['cluster_id']}"),
    # CrewAI console (D-98): the news crew's configuration test run on an earlier run's inputs. One at a time.
    "crew_test_flow": FlowSpec("crew_test_flow", CrewTestFlow.create, CrewTestInput, lambda i: f"crew:{i['crew']}"),
}
NEWS_FLOWS = frozenset(n for n in SPECS if n.startswith("news_"))
OWNED_FLOWS = frozenset(SPECS)

CRON_FLOWS = frozenset({"news_dispatch_flow", "news_train_flow", "news_style_review_flow"})
MANUAL_ONLY = frozenset({"crew_test_flow"})  # started by a person in the console: never by a schedule or a flow


def check_kickoff(name: str, triggered_by: str) -> None:
    if name not in SPECS:
        raise RunError(404, f"unknown flow {name}")
    if triggered_by == "approval":
        raise RunError(400, "no news flow is started by an approval (publishing is the seo-agent's publish_flow)")
    if triggered_by == "mcp":
        raise RunError(403, f"{name} cannot be started from MCP")
    if name in MANUAL_ONLY and triggered_by != "manual":
        raise RunError(403, f"{name} starts only from the CrewAI console")
    if triggered_by == "cron" and name not in CRON_FLOWS:
        raise RunError(403, f"{name} cannot be scheduled")


def catalogue() -> list[dict[str, Any]]:
    """GET /flows: every flow with its input model as JSON Schema (the console's Run with parameters form), whether a
    schedule may start it, which crews it runs (a crew's test run replays these flows' runs) and the owning agent."""
    from ..crews.config import registry

    crews: dict[str, list[str]] = {}
    for c in registry().values():
        for f in c.replay:
            crews.setdefault(f, []).append(c.crew)
    return [
        {
            "name": name,
            "agent": AGENT,
            "inputs_schema": spec.input_model.model_json_schema(),
            "approval_only": False,
            "schedulable": name in CRON_FLOWS,
            "mcp": False,
            "crews": crews.get(name, []),
        }
        for name, spec in SPECS.items()
    ]
