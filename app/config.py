"""seo-agent settings, from environment variables only (docs/seo/AGENTS.md lists them all).

Secrets (ANTHROPIC_API_KEY, SEO_AGENT_TOKEN, SEO_AGENT_CALLBACK_TOKEN, DATABASE_URL, SEO_AGENT_DATABASE_URL) come from
Kubernetes Secrets and are never logged, returned by an endpoint or put into a prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Third-party telemetry off before crewai is imported anywhere (crewai_core 1.15.30 reads these at import/first use).
# OTEL_SDK_DISABLED is deliberately NOT set: it would also switch off this service's own OpenTelemetry tracing.
# CREWAI_STORAGE_DIR points CrewAI's local storage at an emptyDir: nothing may write SQLite there (the flows use the
# Postgres persistence, app/runs.py, and a test asserts no .db file appears), but if a code path ever tried, it must
# not land in the read-only image.
for _k, _v in {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_DISABLE_VERSION_CHECK": "true",
    "CREWAI_STORAGE_DIR": "/tmp/crewai",  # noqa: S108 - emptyDir in the pod
}.items():
    os.environ.setdefault(_k, _v)

# Official documentation only (the host or a subdomain). Keep equal to services/www/src/lib/seo-agent/config.ts.
DEFAULT_DOCS_ALLOWLIST = (
    "kubernetes.io",
    "docs.aws.amazon.com",
    "developer.hashicorp.com",
    "docs.python.org",
    "docs.crewai.com",
    "modelcontextprotocol.io",
    "minikube.sigs.k8s.io",
    "helm.sh",
    "docs.docker.com",
    "istio.io",
    "developer.konghq.com",
    "docs.konghq.com",
    "argo-cd.readthedocs.io",
    "karpenter.sh",
    "keda.sh",
    "velero.io",
    "docs.cilium.io",
    "linkerd.io",
    "prometheus.io",
    "grafana.com",
    "opentelemetry.io",
    "docs.github.com",
    "docs.gitlab.com",
    "docs.ansible.com",
    "developers.cloudflare.com",
    "cloud.google.com",
    "learn.microsoft.com",
    "www.postgresql.org",
    "redis.io",
    "nextjs.org",
    "react.dev",
    "nodejs.org",
    "developer.mozilla.org",
    "go.dev",
    "fastapi.tiangolo.com",
    "docs.pydantic.dev",
    "docs.claude.com",
    "platform.claude.com",
    "docs.langchain.com",
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
    monthly_budget_usd: float = 15.0  # v5: COSTS.md estimates the admin month at $8–10.5
    ask_daily_budget_usd: float = 1.0
    ask_enabled: bool = False
    # D-93: copy the article index into LanceDB on R2 after each reindex (/admin/lancedb → article_chunks).
    lancedb_article_index: bool = True
    crew_max_tokens_per_run: int = 150_000  # ≈ a normal 4-agent run is ~80k (docs/seo/COSTS.md)
    crew_timeout_s: int = 600
    crew_max_iter: int = 4
    crew_max_rpm: int = 20
    # Per-agent model overrides, "<crew>.<agent>" → model, e.g. {"optimization.analyst": "claude-haiku-5-5"}.
    crew_models: dict[str, str] = field(default_factory=dict)
    # Per-agent max_iter overrides, "<crew>.<agent>" → turns (1–20), e.g. {"editorial.researcher": 10}; others use
    # crew_max_iter.
    crew_max_iters: dict[str, int] = field(default_factory=dict)
    # Per-request timeout of the Anthropic client. A 16k-token article body on a larger model can take minutes.
    llm_timeout_s: float = 120.0
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
    # Cosine pre-filter for retrieved chunks (bge-small). v5: calibrated on the eval set = the highest floor that loses
    # no expected answer (RAG.md "Score calibration"); was 0.15. Other embedding providers need their own value.
    ask_min_score: float = 0.66
    ask_retention_days: int = 30
    production: bool = False
    otel_endpoint: str = ""
    service_name: str = "seo-agent"
    # v4: MCP (docs/seo/MCP.md). Both servers are off until switched on; the admin one only after OAuth is verified.
    site_url: str = "https://jithendar.com"
    public_mcp_enabled: bool = False
    admin_mcp_enabled: bool = False
    edge_port: int = 8090  # MCP listener (Kong → here only); the internal API stays on PORT (8080)
    mcp_allowed_hosts: tuple[str, ...] = ("jithendar.com", "www.jithendar.com")
    public_mcp_rpm_per_ip: int = 30
    public_mcp_daily_calls: int = 5000
    admin_mcp_rpm_per_token: int = 120
    admin_kickoffs_per_hour: int = 10
    flow_timeout_s: int = 1800  # wall clock per flow run (crews have their own, shorter limit)

    @property
    def admin_mcp_resource(self) -> str:
        """RFC 8707 resource indicator / token audience of the admin MCP server."""
        return f"{self.site_url.rstrip('/')}/mcp/admin"


def _models(raw: str) -> dict[str, str]:
    """SEO_CREW_MODELS: JSON object of "<crew>.<agent>" → Claude model id; anything malformed is ignored."""
    import json
    import re

    try:
        v = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}
    if not isinstance(v, dict):
        return {}
    return {k: m for k, m in v.items() if isinstance(k, str) and isinstance(m, str) and re.fullmatch(r"[a-z_]+\.[a-z_]+", k) and re.fullmatch(r"claude-[a-z0-9-]{3,40}", m)}


def _max_iters(raw: str) -> dict[str, int]:
    """SEO_CREW_MAX_ITERS: JSON object of "<crew>.<agent>" → max_iter (1–20); anything malformed is ignored."""
    import json
    import re

    try:
        v = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}
    if not isinstance(v, dict):
        return {}
    return {k: n for k, n in v.items() if isinstance(k, str) and re.fullmatch(r"[a-z_]+\.[a-z_]+", k) and isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 20}


def load_settings() -> Settings:
    env = os.environ
    docs = tuple(h.strip().lower() for h in env.get("SEO_AGENT_DOCS_ALLOWLIST", "").split(",") if h.strip())
    return Settings(
        # Its own, narrower database role when one is configured (scripts/db-role.sql); otherwise the web app's.
        database_url=env.get("SEO_AGENT_DATABASE_URL", "").strip() or env.get("DATABASE_URL", ""),
        agent_token=env.get("SEO_AGENT_TOKEN", ""),
        callback_token=env.get("SEO_AGENT_CALLBACK_TOKEN", ""),
        callback_url=env.get("SEO_AGENT_CALLBACK_URL", Settings.callback_url).rstrip("/"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        draft_model=env.get("SEO_DRAFT_MODEL", "").strip() or Settings.draft_model,
        classify_model=env.get("SEO_CLASSIFY_MODEL", "").strip() or Settings.classify_model,
        monthly_budget_usd=_float("SEO_LLM_MONTHLY_BUDGET_USD", Settings.monthly_budget_usd),
        ask_daily_budget_usd=_float("ASK_BLOG_DAILY_BUDGET_USD", Settings.ask_daily_budget_usd),
        ask_enabled=_bool("ASK_BLOG_ENABLED", False),
        lancedb_article_index=_bool("LANCEDB_ARTICLE_INDEX", True),
        crew_max_tokens_per_run=_int("CREW_MAX_TOKENS_PER_RUN", Settings.crew_max_tokens_per_run),
        crew_timeout_s=_int("CREW_TIMEOUT_SECONDS", Settings.crew_timeout_s),
        crew_max_iter=_int("CREW_MAX_ITER", Settings.crew_max_iter),
        crew_max_rpm=max(1, _int("CREW_MAX_RPM", Settings.crew_max_rpm)),
        crew_models=_models(env.get("SEO_CREW_MODELS", "")),
        crew_max_iters=_max_iters(env.get("SEO_CREW_MAX_ITERS", "")),
        llm_timeout_s=max(30.0, min(900.0, _float("SEO_LLM_TIMEOUT_SECONDS", Settings.llm_timeout_s))),
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
        ask_min_score=max(0.0, min(0.95, _float("ASK_BLOG_MIN_SCORE", Settings.ask_min_score))),
        ask_retention_days=_int("ASK_BLOG_RETENTION_DAYS", Settings.ask_retention_days),
        production=env.get("APP_ENV", "") == "production",
        otel_endpoint=env.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        service_name=env.get("OTEL_SERVICE_NAME", "seo-agent"),
        site_url=(env.get("SITE_URL", "").strip() or Settings.site_url).rstrip("/"),
        public_mcp_enabled=_bool("PUBLIC_MCP_ENABLED", False),
        admin_mcp_enabled=_bool("ADMIN_MCP_ENABLED", False),
        edge_port=_int("SEO_AGENT_EDGE_PORT", Settings.edge_port),
        mcp_allowed_hosts=tuple(h.strip().lower() for h in env.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()) or Settings.mcp_allowed_hosts,
        public_mcp_rpm_per_ip=max(1, _int("PUBLIC_MCP_RPM_PER_IP", Settings.public_mcp_rpm_per_ip)),
        public_mcp_daily_calls=_int("PUBLIC_MCP_DAILY_CALLS", Settings.public_mcp_daily_calls),
        admin_mcp_rpm_per_token=max(1, _int("ADMIN_MCP_RPM_PER_TOKEN", Settings.admin_mcp_rpm_per_token)),
        admin_kickoffs_per_hour=max(1, _int("ADMIN_MCP_KICKOFFS_PER_HOUR", Settings.admin_kickoffs_per_hour)),
        flow_timeout_s=max(60, _int("SEO_FLOW_TIMEOUT_SECONDS", Settings.flow_timeout_s)),
    )
