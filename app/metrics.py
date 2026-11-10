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
    "news_dispatch_flow",
    "news_job_flow",
    "news_draft_flow",
    "news_ingest_flow",
    "news_image_flow",
    "news_train_flow",
    "news_style_review_flow",
    "news_playground_flow",
    "seo_ml_flow",
}
CREWS = {"optimization", "editorial", "report", "news"}
PURPOSES = {"draft", "regenerate", "crew", "report", "ask_grade", "ask_answer", "classify", "ml_label", "news.crew", "news.classify", "news.vision", "news.style", "news.playground", "other"}
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
# v5 RAG quality (docs/seo/RAG.md). Labels: `mode` ∈ {vector, hybrid}.
RAG_CHUNK_TOKENS = Histogram(
    "seo_rag_chunk_tokens", "Tokens of each embedded chunk input (title + heading path + text), at reindex.", buckets=(32, 64, 128, 192, 256, 320, 384, 448, 480, 512, 1024), registry=REGISTRY
)
RAG_CHUNKS_TRUNCATED = Counter("seo_rag_chunks_truncated_total", "Embedded inputs longer than the model's 512 tokens (their tail is not embedded). Must stay 0.", registry=REGISTRY)
RAG_RETRIEVAL_MODE = Gauge("seo_rag_retrieval_mode", "1 for the retrieval mode in use (RAG_MODE).", ["mode"], registry=REGISTRY)
RAG_EVAL_RECALL3 = Gauge("seo_rag_eval_recall_at_3", "Recall@3 of the latest retrieval evaluation run against the live index.", registry=REGISTRY)
RAG_MODES = {"vector", "hybrid"}

# v5 SEO-side ML (docs/seo/ML.md). `model` ∈ ML_MODELS, `status` ∈ ML_RUN_STATUSES, `metric` ∈ the fixed names in
# app/seo_ml/train.py EXPORTED: all bounded.
ML_MODELS = {"decay", "query_clusters", "intent", "priority", "related"}
ML_RUN_STATUSES = {"activated", "rule_kept", "needs_review", "fallback_to_rule", "failed"}
ML_TRAIN_RUNS = Counter("seo_ml_train_runs_total", "seo_ml_flow model trainings by outcome (fallback_to_rule = an active model lost to the rule).", ["model", "status"], registry=REGISTRY)
ML_MODEL_ACTIVE = Gauge("seo_ml_model_active", "1 when the model's latest version is active (the web app uses it in auto mode).", ["model"], registry=REGISTRY)
ML_METRIC = Gauge("seo_ml_metric", "Held-out metric of the latest version (model) and of the rule on the same data.", ["model", "metric"], registry=REGISTRY)
ML_FLOW_CONSECUTIVE_FAILURES = Gauge("seo_ml_flow_consecutive_failures", "Failed seo_ml_flow runs in a row (from seo_flow_runs).", registry=REGISTRY)

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


# --- News desk (docs/news/RUNBOOK.md "Metrics"). `news_job` is the news job id (not `job`, which Prometheus sets to the
# scrape job): bounded because the web app refuses more than
# MAX_NEWS_JOBS (25) jobs; every other label comes from a fixed set below.
MAX_NEWS_JOBS = 25
NEWS_SOURCE_TYPES = {"rss", "atom", "json_api", "github_releases", "sitemap", "html_allowlisted"}
NEWS_DECISIONS = {"selected", "rejected"}
NEWS_DRAFT_STATUSES = {"ready_for_review", "failed_quality", "incomplete", "paused_budget", "dry_run", "failed", "duplicate"}
NEWS_GATES = {"source_similarity", "grounding", "quotes", "value", "style", "seo", "duplicate", "web_validation"}
NEWS_IMAGE_PROVIDERS = {"template", "gemini", "openai"}
NEWS_COST_KINDS = {"llm", "image"}
NEWS_MODELS = {"classifier", "ranker"}

NEWS_RUNS = Counter("news_runs_total", "News job runs by final status.", ["news_job", "status"], registry=REGISTRY)
NEWS_RUN_DURATION = Histogram("news_run_duration_seconds", "News job run duration.", ["news_job"], buckets=_BUCKETS, registry=REGISTRY)
NEWS_ITEMS_FETCHED = Counter("news_items_fetched_total", "Feed items fetched (before dedupe).", ["source_type"], registry=REGISTRY)
NEWS_SOURCE_ERRORS = Counter("news_source_errors_total", "Source fetch/parse failures.", ["source_type"], registry=REGISTRY)
NEWS_CANDIDATES = Counter("news_candidates_total", "Scored candidates by decision.", ["decision"], registry=REGISTRY)
NEWS_DRAFTS = Counter("news_drafts_total", "News drafts by outcome.", ["status"], registry=REGISTRY)
NEWS_QUALITY_FAILURES = Counter("news_quality_failures_total", "Drafts failing a quality gate (one count per failed gate).", ["gate"], registry=REGISTRY)
NEWS_IMAGES = Counter("news_images_generated_total", "Cover images by provider and outcome.", ["provider", "status"], registry=REGISTRY)
NEWS_COST = Counter("news_cost_usd_total", "News desk spend in USD.", ["kind"], registry=REGISTRY)
NEWS_MODEL_AUC = Gauge("news_model_auc", "Held-out AUC (ranker) / macro F1 (classifier) of the active model.", ["model"], registry=REGISTRY)
NEWS_BUDGET_RATIO = Gauge("news_budget_used_ratio", "News spend / cap for the current period.", ["scope"], registry=REGISTRY)
NEWS_SOURCE_FAILING_SINCE = Gauge("news_source_failing_seconds", "Seconds since the longest-failing enabled source last succeeded (0 = all healthy).", registry=REGISTRY)
NEWS_JOB_CONSECUTIVE_FAILURES = Gauge("news_job_consecutive_failures", "Consecutive failed runs per job.", ["news_job"], registry=REGISTRY)
NEWS_JOB_LAST_DRAFT = Gauge("news_job_last_draft_timestamp_seconds", "Last draft that reached review, per job.", ["news_job"], registry=REGISTRY)
NEWS_QUALITY_FAILURE_RATIO = Gauge("news_quality_failure_ratio_7d", "Share of drafts failing quality over 7 days.", registry=REGISTRY)


def one_of(value: str, allowed: set[str]) -> str:
    return value if value in allowed else "other"
