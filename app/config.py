"""seo-agent settings, from environment variables only (docs/seo/AGENTS.md lists them all).

Secrets (ANTHROPIC_API_KEY, SEO_AGENT_TOKEN, SEO_AGENT_CALLBACK_TOKEN, DATABASE_URL) come from Kubernetes Secrets and
are never logged, returned by an endpoint or put into a prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Third-party telemetry off before crewai is imported anywhere (it reads these at import/first use). OTEL_SDK_DISABLED
# is deliberately NOT set: it would also switch off this service's own OpenTelemetry tracing to the cluster collector.
for _k, _v in {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_DISABLE_VERSION_CHECK": "true",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
}.items():
    os.environ.setdefault(_k, _v)

DEFAULT_DOCS_ALLOWLIST = (
    "kubernetes.io",
    "docs.aws.amazon.com",
    "developer.hashicorp.com",
    "docs.python.org",
    "langchain-ai.github.io",
    "docs.crewai.com",
    "modelcontextprotocol.io",
)


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, ""))
        return v if v >= 0 else default
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, ""))
        return v if v >= 0 else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = ""
    # Next.js admin API → seo-agent (bearer). Required: the service refuses to start without it in production.
    agent_token: str = ""
    # seo-agent → Next.js internal callback API (bearer), a different secret so one leaked direction is not both.
    callback_token: str = ""
    callback_url: str = "http://jithendar-com-internal.jithendar:9465"
    anthropic_api_key: str = ""
    draft_model: str = "claude-sonnet-5-5"
    classify_model: str = "claude-haiku-5-5"
    monthly_budget_usd: float = 10.0
    ask_daily_budget_usd: float = 1.0
    ask_enabled: bool = False
    crew_max_tokens_per_run: int = 150_000  # ≈ a normal 4-agent run is ~80k (docs/seo/COSTS.md)
    crew_timeout_s: int = 600
    crew_max_iter: int = 4
    fanout_concurrency: int = 2
    weekly_max_articles: int = 10
    embeddings_provider: str = "onnx"
    embeddings_model_dir: str = "/models/bge-small-en-v1.5"
    ollama_url: str = ""
    ollama_model: str = "nomic-embed-text"
    embeddings_api_url: str = ""
    embeddings_api_key: str = ""
    docs_allowlist: tuple[str, ...] = field(default=DEFAULT_DOCS_ALLOWLIST)
    ask_cache_ttl_s: int = 900
    ask_top_k: int = 6
    ask_retention_days: int = 30
    production: bool = False
    otel_endpoint: str = ""
    service_name: str = "seo-agent"


def load_settings() -> Settings:
    env = os.environ
    docs = tuple(h.strip().lower() for h in env.get("SEO_AGENT_DOCS_ALLOWLIST", "").split(",") if h.strip())
    return Settings(
        database_url=env.get("DATABASE_URL", ""),
        agent_token=env.get("SEO_AGENT_TOKEN", ""),
        callback_token=env.get("SEO_AGENT_CALLBACK_TOKEN", ""),
        callback_url=env.get("SEO_AGENT_CALLBACK_URL", Settings.callback_url).rstrip("/"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        draft_model=env.get("SEO_DRAFT_MODEL", "").strip() or Settings.draft_model,
        classify_model=env.get("SEO_CLASSIFY_MODEL", "").strip() or Settings.classify_model,
        monthly_budget_usd=_float("SEO_LLM_MONTHLY_BUDGET_USD", Settings.monthly_budget_usd),
        ask_daily_budget_usd=_float("ASK_BLOG_DAILY_BUDGET_USD", Settings.ask_daily_budget_usd),
        ask_enabled=_bool("ASK_BLOG_ENABLED", False),
        crew_max_tokens_per_run=_int("CREW_MAX_TOKENS_PER_RUN", Settings.crew_max_tokens_per_run),
        crew_timeout_s=_int("CREW_TIMEOUT_SECONDS", Settings.crew_timeout_s),
        crew_max_iter=_int("CREW_MAX_ITER", Settings.crew_max_iter),
        fanout_concurrency=max(1, _int("SEO_AGENT_FANOUT_CONCURRENCY", Settings.fanout_concurrency)),
        weekly_max_articles=_int("SEO_AGENT_WEEKLY_MAX_ARTICLES", Settings.weekly_max_articles),
        embeddings_provider=env.get("EMBEDDINGS_PROVIDER", "onnx").strip().lower() or "onnx",
        embeddings_model_dir=env.get("EMBEDDINGS_MODEL_DIR", Settings.embeddings_model_dir),
        ollama_url=env.get("OLLAMA_URL", "").rstrip("/"),
        ollama_model=env.get("OLLAMA_EMBED_MODEL", Settings.ollama_model),
        embeddings_api_url=env.get("EMBEDDINGS_API_URL", ""),
        embeddings_api_key=env.get("EMBEDDINGS_API_KEY", ""),
        docs_allowlist=docs or DEFAULT_DOCS_ALLOWLIST,
        ask_cache_ttl_s=_int("ASK_BLOG_CACHE_TTL_SECONDS", Settings.ask_cache_ttl_s),
        ask_top_k=max(1, min(12, _int("ASK_BLOG_TOP_K", Settings.ask_top_k))),
        ask_retention_days=_int("ASK_BLOG_RETENTION_DAYS", Settings.ask_retention_days),
        production=env.get("APP_ENV", "") == "production",
        otel_endpoint=env.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        service_name=env.get("OTEL_SERVICE_NAME", "seo-agent"),
    )
