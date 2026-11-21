"""news-agent settings, from environment variables only (README.md lists them all).

Forked from moks-seo-agent/app/config.py@bb9e553 on 2026-11-21 and cut down to what the news desk reads. The news
desk's own settings (sources, caps, models, quality thresholds, the on/off switch) are the Newsroom settings row in
the web app's database (app/news/config.py), edited in Admin → Newsroom → Settings, not environment variables.

Secrets (ANTHROPIC_API_KEY, NEWS_AGENT_TOKEN, NEWS_AGENT_CALLBACK_TOKEN, SEO_AGENT_NEWS_TOKEN, DATABASE_URL,
NEWS_AGENT_DATABASE_URL, OPENAI_API_KEY) come from Kubernetes Secrets and are never logged, returned by an endpoint or
put into a prompt.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

# Third-party telemetry off before crewai is imported anywhere (crewai_core 1.15.30 reads these at import/first use).
# OTEL_SDK_DISABLED is deliberately NOT set: it would also switch off this service's own OpenTelemetry tracing.
# CREWAI_STORAGE_DIR points CrewAI's local storage at an emptyDir: nothing may write SQLite there (the flows use the
# Postgres persistence, app/runs.py), but if a code path ever tried, it must not land in the read-only image.
for _k, _v in {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_DISABLE_VERSION_CHECK": "true",
    "CREWAI_STORAGE_DIR": "/tmp/crewai",  # noqa: S108 - emptyDir in the pod
}.items():
    os.environ.setdefault(_k, _v)

# Official documentation hosts a news researcher may also fetch as primary sources (the host or a subdomain), on top
# of the story's own source hosts and the Newsroom's primary-source hosts. Same list as moks-seo-agent/app/config.py.
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
    "prometheus.github.io",
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
    "man7.org",
    "www.freedesktop.org",
    "systemd.io",
    "docs.kernel.org",
    "www.gnu.org",
)


def _float(name: str, default: float, env: Mapping[str, str] | None = None) -> float:
    try:
        v = float((os.environ if env is None else env).get(name, ""))
        return v if v >= 0 else default
    except ValueError:
        return default


def _int(name: str, default: int, env: Mapping[str, str] | None = None) -> int:
    try:
        v = int((os.environ if env is None else env).get(name, ""))
        return v if v >= 0 else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str = ""
    # Web app admin API → news-agent (bearer). Required: the service refuses to start without it in production.
    agent_token: str = ""
    # news-agent → web app internal callback API (bearer). Its own secret: the web app accepts it on the news routes only.
    callback_token: str = ""
    callback_url: str = "http://jithendar-com-internal.jithendar:9465"
    # news-agent → seo-agent: site search and the one LLM budget (app/seo_agent.py). Empty = neither.
    seo_agent_url: str = ""
    seo_agent_token: str = ""
    anthropic_api_key: str = ""
    # The crews' default model when neither the job, the crew configuration nor NEWS_CREW_MODELS names one.
    draft_model: str = "claude-sonnet-5-5"
    classify_model: str = "claude-haiku-5-5"
    crew_max_tokens_per_run: int = 150_000
    crew_timeout_s: int = 600
    crew_max_iter: int = 4
    crew_max_rpm: int = 20
    # Per-agent model overrides, "news.<agent>" → model, e.g. {"news.analyst": "claude-haiku-5-5"}.
    crew_models: dict[str, str] = field(default_factory=dict)
    # Per-agent max_iter overrides, "news.<agent>" → turns (1–20); others use crew_max_iter.
    crew_max_iters: dict[str, int] = field(default_factory=dict)
    # Per-request timeout of the Anthropic client.
    llm_timeout_s: float = 120.0
    embeddings_provider: str = "onnx"
    embeddings_model_dir: str = "/models/bge-small-en-v1.5"
    ollama_url: str = ""
    ollama_model: str = "nomic-embed-text"
    ollama_embed_dim: int = 0  # 0 = the model's known size (embeddings.OLLAMA_MODELS)
    embeddings_timeout_s: float = 300.0
    embeddings_api_url: str = ""
    embeddings_api_key: str = ""
    docs_allowlist: tuple[str, ...] = field(default=DEFAULT_DOCS_ALLOWLIST)
    production: bool = False
    otel_endpoint: str = ""
    service_name: str = "news-agent"
    flow_timeout_s: int = 1800  # wall clock per flow run (crews have their own, shorter limit)
    # Schedules (app/schedules.py): how late a missed fire time still runs, e.g. after the agent was down.
    schedule_catchup_min: int = 60


def _models(raw: str) -> dict[str, str]:
    """NEWS_CREW_MODELS: JSON object of "<crew>.<agent>" → Claude model id; anything malformed is ignored."""
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
    """NEWS_CREW_MAX_ITERS: JSON object of "<crew>.<agent>" → max_iter (1–20); anything malformed is ignored."""
    import json
    import re

    try:
        v = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}
    if not isinstance(v, dict):
        return {}
    return {k: n for k, n in v.items() if isinstance(k, str) and re.fullmatch(r"[a-z_]+\.[a-z_]+", k) and isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 20}


def embeddings_model_name(provider: str, model_dir: str, ollama_model: str) -> str:
    """What make_embeddings(...).name will be, without loading the model (app/calibration.py keys on it)."""
    if provider == "onnx":
        return f"onnx:{os.path.basename(model_dir.rstrip('/'))}"
    if provider == "ollama":
        return f"ollama:{ollama_model}"
    return provider


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    docs = tuple(h.strip().lower() for h in env.get("NEWS_AGENT_DOCS_ALLOWLIST", "").split(",") if h.strip())
    return Settings(
        # Its own, narrower database role when one is configured (scripts/db-role.sql); otherwise the web app's.
        database_url=env.get("NEWS_AGENT_DATABASE_URL", "").strip() or env.get("DATABASE_URL", ""),
        agent_token=env.get("NEWS_AGENT_TOKEN", ""),
        callback_token=env.get("NEWS_AGENT_CALLBACK_TOKEN", ""),
        callback_url=env.get("NEWS_AGENT_CALLBACK_URL", Settings.callback_url).rstrip("/"),
        seo_agent_url=env.get("SEO_AGENT_URL", "").strip().rstrip("/"),
        seo_agent_token=env.get("SEO_AGENT_NEWS_TOKEN", "").strip(),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        draft_model=env.get("NEWS_DRAFT_MODEL", "").strip() or Settings.draft_model,
        classify_model=env.get("NEWS_CLASSIFY_MODEL", "").strip() or Settings.classify_model,
        crew_max_tokens_per_run=_int("CREW_MAX_TOKENS_PER_RUN", Settings.crew_max_tokens_per_run, env),
        crew_timeout_s=_int("CREW_TIMEOUT_SECONDS", Settings.crew_timeout_s, env),
        crew_max_iter=_int("CREW_MAX_ITER", Settings.crew_max_iter, env),
        crew_max_rpm=max(1, _int("CREW_MAX_RPM", Settings.crew_max_rpm, env)),
        crew_models=_models(env.get("NEWS_CREW_MODELS", "")),
        crew_max_iters=_max_iters(env.get("NEWS_CREW_MAX_ITERS", "")),
        llm_timeout_s=max(30.0, min(900.0, _float("NEWS_LLM_TIMEOUT_SECONDS", Settings.llm_timeout_s, env))),
        embeddings_provider=env.get("EMBEDDINGS_PROVIDER", "onnx").strip().lower() or "onnx",
        embeddings_model_dir=env.get("EMBEDDINGS_MODEL_DIR", Settings.embeddings_model_dir),
        ollama_url=env.get("OLLAMA_URL", "").rstrip("/"),
        ollama_model=env.get("OLLAMA_EMBED_MODEL", Settings.ollama_model),
        ollama_embed_dim=max(0, _int("OLLAMA_EMBED_DIM", 0, env)),
        embeddings_timeout_s=max(10.0, min(1800.0, _float("EMBEDDINGS_TIMEOUT_SECONDS", Settings.embeddings_timeout_s, env))),
        embeddings_api_url=env.get("EMBEDDINGS_API_URL", ""),
        embeddings_api_key=env.get("EMBEDDINGS_API_KEY", ""),
        docs_allowlist=docs or DEFAULT_DOCS_ALLOWLIST,
        production=env.get("APP_ENV", "") == "production",
        otel_endpoint=env.get("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
        service_name=env.get("OTEL_SERVICE_NAME", "news-agent"),
        flow_timeout_s=max(60, _int("NEWS_FLOW_TIMEOUT_SECONDS", Settings.flow_timeout_s, env)),
        schedule_catchup_min=max(2, min(1440, _int("SCHEDULE_CATCHUP_MINUTES", Settings.schedule_catchup_min, env))),
    )
