"""Prometheus metrics. Labels are low-cardinality by construction: flow/step/crew/status/purpose/server/tool come from
fixed sets, never from article ids, questions, URLs or client ids (the *_label helpers collapse anything else to
"other")."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

FLOWS = {
    "gsc_sync_flow",
    "ga4_sync_flow",
    "tech_audit_flow",
    "weekly_recommendations_flow",
    "optimize_article_flow",
    "internal_linking_flow",
    "new_article_flow",
    "publish_flow",
    "reindex_flow",
    "ask_gaps_flow",
    "report_flow",
}
CREWS = {"optimization", "editorial", "report"}
PURPOSES = {"draft", "regenerate", "crew", "report", "ask_grade", "ask_answer", "classify", "other"}
MCP_SERVERS = {"public", "admin"}
_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800)

FLOW_RUNS = Counter("seo_flow_runs_total", "Flow runs by final status.", ["flow", "status"], registry=REGISTRY)
FLOW_DURATION = Histogram("seo_flow_duration_seconds", "Flow run wall-clock duration.", ["flow"], buckets=_BUCKETS, registry=REGISTRY)
STEP_DURATION = Histogram("seo_flow_step_duration_seconds", "Flow step duration.", ["flow", "step"], buckets=_BUCKETS, registry=REGISTRY)
CREW_RUNS = Counter("seo_crew_runs_total", "Crew runs by result.", ["crew", "status"], registry=REGISTRY)
LLM_TOKENS = Counter("seo_llm_tokens_total", "LLM tokens (input + output + cache).", ["model", "purpose"], registry=REGISTRY)
LLM_COST = Counter("seo_llm_cost_usd_total", "Estimated LLM cost in USD.", ["purpose"], registry=REGISTRY)
DRAFTS_AWAITING = Gauge("seo_drafts_awaiting_review", "Drafts in ready_for_review (from the web app).", registry=REGISTRY)
ASK_REQUESTS = Counter("seo_ask_requests_total", "Ask-the-Blog requests by outcome.", ["outcome"], registry=REGISTRY)
REINDEX_CHUNKS = Counter("seo_embeddings_reindex_chunks_total", "Chunks (re)embedded.", registry=REGISTRY)
BUDGET_BLOCKED = Counter("seo_llm_budget_blocked_total", "Calls refused by a budget cap.", ["scope"], registry=REGISTRY)
BUDGET_RATIO = Gauge("seo_llm_budget_used_ratio", "Month-to-date LLM spend / monthly budget (shared with v2).", registry=REGISTRY)
ASK_BUDGET_RATIO = Gauge("seo_ask_budget_used_ratio", "Today's Ask-the-Blog spend / daily cap.", registry=REGISTRY)
LAST_SUCCESS = Gauge("seo_flow_last_success_timestamp_seconds", "Last succeeded run per flow (from seo_flow_runs).", ["flow"], registry=REGISTRY)
MCP_REQUESTS = Counter("seo_mcp_requests_total", "MCP tool/resource/prompt calls.", ["server", "tool", "status"], registry=REGISTRY)
MCP_AUTH_FAILURES = Counter("seo_mcp_auth_failures_total", "Rejected MCP bearer tokens (missing, invalid, expired, revoked, wrong audience).", ["server"], registry=REGISTRY)
MCP_DURATION = Histogram("seo_mcp_request_duration_seconds", "MCP call duration.", ["server", "tool"], buckets=_BUCKETS, registry=REGISTRY)

ASK_OUTCOMES = {"answered", "fallback", "refused", "cached", "rate_limited", "disabled", "budget", "error"}

# MCP tool names are a fixed registry (app/mcp); anything else is "other".
MCP_TOOLS: set[str] = set()


def purpose_label(purpose: str) -> str:
    return purpose if purpose in PURPOSES else "other"


def flow_label(flow: str) -> str:
    return flow if flow in FLOWS else "other"


def crew_label(crew: str) -> str:
    return crew if crew in CREWS else "other"


def mcp_tool_label(tool: str) -> str:
    return tool if tool in MCP_TOOLS else "other"
