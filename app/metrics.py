"""Prometheus metrics. Labels are low-cardinality by construction: graph/node/status/purpose come from fixed sets,
never from article ids, questions or URLs (purpose_label collapses anything unknown to "other")."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

GRAPHS = {"optimize_article", "weekly_recommendations", "internal_linking", "new_article", "ask_blog"}
PURPOSES = {"draft", "regenerate", "crew", "ask_grade", "ask_answer", "classify", "other"}

GRAPH_RUNS = Counter("seo_graph_runs_total", "Graph runs by final status.", ["graph", "status"], registry=REGISTRY)
NODE_DURATION = Histogram(
    "seo_graph_node_duration_seconds",
    "Graph node duration.",
    ["graph", "node"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)
INTERRUPTS_PENDING = Gauge("seo_graph_interrupts_pending", "Runs waiting at human_review (from seo_agent_runs).", registry=REGISTRY)
CREW_RUNS = Counter("seo_crew_runs_total", "Editorial crew runs.", ["status"], registry=REGISTRY)
LLM_TOKENS = Counter("seo_llm_tokens_total", "LLM tokens (input + output + cache).", ["model", "purpose"], registry=REGISTRY)
LLM_COST = Counter("seo_llm_cost_usd_total", "Estimated LLM cost in USD.", ["purpose"], registry=REGISTRY)
ASK_REQUESTS = Counter("seo_ask_requests_total", "Ask-the-Blog requests by outcome.", ["outcome"], registry=REGISTRY)
REINDEX_CHUNKS = Counter("seo_embeddings_reindex_chunks_total", "Chunks (re)embedded.", registry=REGISTRY)
BUDGET_BLOCKED = Counter("seo_llm_budget_blocked_total", "Calls refused by a budget cap.", ["scope"], registry=REGISTRY)
BUDGET_RATIO = Gauge("seo_llm_budget_used_ratio", "Month-to-date LLM spend / monthly budget (shared with v2).", registry=REGISTRY)
ASK_BUDGET_RATIO = Gauge("seo_ask_budget_used_ratio", "Today's Ask-the-Blog spend / daily cap.", registry=REGISTRY)

ASK_OUTCOMES = {"answered", "fallback", "refused", "cached", "rate_limited", "disabled", "budget", "error"}


def purpose_label(purpose: str) -> str:
    return purpose if purpose in PURPOSES else "other"


def graph_label(graph: str) -> str:
    return graph if graph in GRAPHS else "other"
