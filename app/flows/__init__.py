"""The flow catalogue (docs/seo/FLOWS.md) and who may start what.

- `SPECS`: every flow, its strict input model, and its lock target (one active run per flow + target).
- `publish_flow` is started only by a human approval (`triggered_by="approval"`, set solely by the review decision
  endpoint). The runner refuses it for every other trigger: the generic kickoff API, CronJobs, child runs of other
  flows and MCP (`check_kickoff`).
- `MCP_KICKOFF_ALLOWLIST`: what the admin MCP `kickoff_flow` tool may start. Never publish_flow.
"""

from __future__ import annotations

from typing import Any

from ..runs import FlowSpec, RunError
from ..state import JobInput, LinkingInput, NewArticleInput, NoInput, OptimizeInput, PublishInput, ReindexInput, ReportInput, WeeklyInput
from .jobs import Ga4SyncFlow, GscSyncFlow, TechAuditFlow
from .maintenance import AskGapsFlow, InternalLinkingFlow, ReindexFlow
from .new_article import NewArticleFlow
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
from .optimize_article import OptimizeArticleFlow
from .publish import PublishFlow
from .report import ReportFlow
from .weekly import WeeklyRecommendationsFlow


def _site(_i: dict[str, Any]) -> str:
    return "site"


def _draft(i: dict[str, Any]) -> str:
    return str(i["draft_id"])


def _draft_of(i: dict[str, Any]) -> str | None:
    return str(i["draft_id"]) if i.get("draft_id") else None


SPECS: dict[str, FlowSpec] = {
    "gsc_sync_flow": FlowSpec("gsc_sync_flow", GscSyncFlow.create, JobInput, _site),
    "ga4_sync_flow": FlowSpec("ga4_sync_flow", Ga4SyncFlow.create, JobInput, _site),
    "tech_audit_flow": FlowSpec("tech_audit_flow", TechAuditFlow.create, JobInput, _site),
    "weekly_recommendations_flow": FlowSpec("weekly_recommendations_flow", WeeklyRecommendationsFlow.create, WeeklyInput, _site),
    "optimize_article_flow": FlowSpec("optimize_article_flow", OptimizeArticleFlow.create, OptimizeInput, _draft, _draft_of),
    "internal_linking_flow": FlowSpec("internal_linking_flow", InternalLinkingFlow.create, LinkingInput, _site),
    "new_article_flow": FlowSpec("new_article_flow", NewArticleFlow.create, NewArticleInput, _draft, _draft_of),
    "publish_flow": FlowSpec("publish_flow", PublishFlow.create, PublishInput, _draft, _draft_of),
    "reindex_flow": FlowSpec("reindex_flow", ReindexFlow.create, ReindexInput, _site),
    "ask_gaps_flow": FlowSpec("ask_gaps_flow", AskGapsFlow.create, NoInput, _site),
    "report_flow": FlowSpec("report_flow", ReportFlow.create, ReportInput, lambda i: str(i.get("period", "weekly"))),
    # News desk (docs/news/ARCHITECTURE.md). Lock targets: one active run per job; one per job + story for drafts.
    "news_dispatch_flow": FlowSpec("news_dispatch_flow", NewsDispatchFlow.create, NEWS_INPUTS["news_dispatch_flow"], _site),
    "news_job_flow": FlowSpec("news_job_flow", NewsJobFlow.create, NEWS_INPUTS["news_job_flow"], lambda i: f"job:{i['job_id']}"),
    "news_draft_flow": FlowSpec("news_draft_flow", NewsDraftFlow.create, NEWS_INPUTS["news_draft_flow"], lambda i: f"{i['job_id']}:{i['cluster_ids'][0]}:{i.get('mode', 'single')}"),
    "news_ingest_flow": FlowSpec("news_ingest_flow", NewsIngestFlow.create, NEWS_INPUTS["news_ingest_flow"], _site),
    "news_image_flow": FlowSpec("news_image_flow", NewsImageFlow.create, NEWS_INPUTS["news_image_flow"], lambda i: str(i["article_id"])),
    "news_train_flow": FlowSpec("news_train_flow", NewsTrainFlow.create, NEWS_INPUTS["news_train_flow"], _site),
    "news_style_review_flow": FlowSpec("news_style_review_flow", NewsStyleReviewFlow.create, NEWS_INPUTS["news_style_review_flow"], _site),
    "news_playground_flow": FlowSpec("news_playground_flow", NewsPlaygroundFlow.create, NEWS_INPUTS["news_playground_flow"], lambda i: f"{i['job_id']}:{i['cluster_id']}"),
}
NEWS_FLOWS = frozenset(n for n in SPECS if n.startswith("news_"))

APPROVAL_ONLY = frozenset({"publish_flow"})
DRAFTING_FLOWS = ("optimize_article_flow", "new_article_flow")
MCP_KICKOFF_ALLOWLIST = frozenset({"news_job_flow", "optimize_article_flow", "new_article_flow", "internal_linking_flow", "tech_audit_flow", "report_flow", "gsc_sync_flow", "ga4_sync_flow"})
CRON_FLOWS = frozenset({"news_dispatch_flow", "news_train_flow", "news_style_review_flow", "gsc_sync_flow", "ga4_sync_flow", "tech_audit_flow", "weekly_recommendations_flow", "internal_linking_flow", "reindex_flow", "ask_gaps_flow", "report_flow"})


def check_kickoff(name: str, triggered_by: str) -> None:
    if name not in SPECS:
        raise RunError(404, f"unknown flow {name}")
    if name in APPROVAL_ONLY and triggered_by != "approval":
        raise RunError(403, f"{name} starts only from a human approval in the review UI")
    if triggered_by == "approval" and name not in APPROVAL_ONLY:
        raise RunError(400, "only publish_flow is started by an approval")
    if triggered_by == "mcp" and name not in MCP_KICKOFF_ALLOWLIST:
        raise RunError(403, f"{name} cannot be started from MCP")
