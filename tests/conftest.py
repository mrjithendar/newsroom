"""Shared fixtures: settings, a deterministic FakeLlm behind a local BudgetGuard, a FakeWebApp with the callback routes
the news agent may call, and an in-memory site search. The news desk's own fakes are in tests/news_fixtures.py; the
real-Postgres tests in test_news_postgres.py and test_postgres_scope.py."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from app.config import Settings
from app.llm import BudgetGuard, FakeLlm, MemorySpendStore
from app.seo_agent import Hit

os.environ.setdefault("SEO_LLM_PRICES_JSON", "")


def settings(**kw: Any) -> Settings:
    base = Settings(agent_token="t" * 40, callback_token="c" * 40, anthropic_api_key="sk-ant-test", crew_timeout_s=20, crew_max_tokens_per_run=50_000, crew_max_iter=3)
    return replace(base, **kw)


@dataclass
class FakeWebApp:
    """In-memory stand-in for the web app's internal API: the routes the news agent's token may call (app/webapp.py
    CALLBACK_ROUTES). tests/news_fixtures.py NewsWeb adds the news desk's own."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)  # GET /skills (D-96); none unless a test sets them
    skills_error: Exception | None = None
    crew_config_rows: list[dict[str, Any]] = field(default_factory=list)  # GET /crew-configs
    transcripts: bool = True  # GET /run-history/settings

    async def gsc_queries(self, *, topic: str | None = None, article_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        self.calls.append(("gsc", (topic, article_id)))
        return [{"query": "cert-manager renew", "clicks": 12, "impressions": 400, "ctr": 0.03, "position": 4.1}, {"query": "jane@example.com", "clicks": 1, "impressions": 1, "ctr": 1, "position": 1}]

    async def agent_skills(self) -> list[dict[str, Any]]:
        if self.skills_error is not None:
            raise self.skills_error
        return [dict(s) for s in self.skills]

    async def crew_configs(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.crew_config_rows]

    async def run_history_settings(self) -> dict[str, Any]:
        return {"transcripts": self.transcripts}


class MemoryRetriever:
    """In-memory site search over fixture hits (word overlap, not cosine)."""

    def __init__(self, hits: list[Hit] | None = None, available: bool = True):
        self.hits = hits or []
        self.available = available
        self.queries: list[str] = []

    async def search(self, query: str, k: int = 6, **_: Any) -> list[Hit]:
        self.queries.append(query)
        words = {w for w in query.lower().split() if len(w) > 2}
        scored = []
        for h in self.hits:
            text = f"{h.title} {h.heading_path} {h.content}".lower()
            s = sum(1 for w in words if w in text) / max(1, len(words))
            scored.append(Hit(**{**h.__dict__, "score": s}))
        return sorted(scored, key=lambda h: -h.score)[:k]


@pytest.fixture
def spend() -> MemorySpendStore:
    return MemorySpendStore()


@pytest.fixture
def guard(spend: MemorySpendStore) -> BudgetGuard:
    return BudgetGuard(spend, monthly_usd=10.0)


@pytest.fixture
def llm(guard: BudgetGuard) -> FakeLlm:
    return FakeLlm(guard)


@pytest.fixture
def web() -> FakeWebApp:
    return FakeWebApp()


def hits() -> list[Hit]:
    return [
        Hit("a1", "Ingress and TLS on Kubernetes", "/blog/ingress-tls", "Ingress and TLS › cert-manager issuers", "cert-manager-issuers", "Create a ClusterIssuer for Let's Encrypt with the ACME HTTP-01 solver; cert-manager then renews certificates 30 days before expiry.", 0.9),
        Hit("a2", "Istio service mesh on minikube", "/blog/istio-minikube", "Istio › mTLS", "mtls", "PeerAuthentication STRICT enforces mutual TLS between sidecars in the namespace.", 0.8),
        Hit("a3", "Terraform state on S3", "/blog/terraform-s3-state", "Remote state › Locking", "locking", "Use a DynamoDB table or S3 native locking to prevent concurrent applies.", 0.7),
    ]


@pytest.fixture
def retriever() -> MemoryRetriever:
    return MemoryRetriever(hits())
