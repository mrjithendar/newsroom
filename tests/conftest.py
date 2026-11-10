"""Shared fixtures: settings, a deterministic FakeLlm and FakeWebApp that behaves like the web app's callback API
(including the approval gate), and in-memory containers. Postgres-backed tests are in test_postgres.py."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from app.config import Settings
from app.index import Hit, MemoryRetriever
from app.llm import BudgetGuard, FakeLlm, MemorySpendStore
from app.schemas import ModelChange, ModelChangeSet
from app.webapp import WebAppError

ARTICLE = "11111111-1111-4111-8111-111111111111"
DRAFT = "22222222-2222-4222-8222-222222222222"


def settings(**kw: Any) -> Settings:
    base = Settings(agent_token="t" * 40, callback_token="c" * 40, anthropic_api_key="sk-ant-test", monthly_budget_usd=10.0, ask_daily_budget_usd=1.0, ask_enabled=True, fanout_concurrency=2, crew_timeout_s=20, crew_max_tokens_per_run=50_000, crew_max_iter=3)
    return replace(base, **kw)


def good_changeset() -> ModelChangeSet:
    return ModelChangeSet(
        summary="Tighten the search title and meta description.",
        changes=[
            ModelChange(type="title", block_id="", before="Ingress and TLS on Kubernetes", after="Kubernetes ingress TLS with cert-manager", rationale="Low CTR at position 4.", evidence_ref=["rec:r1"]),
            ModelChange(type="meta_description", block_id="", before="old", after="How to terminate TLS at a Kubernetes ingress with cert-manager: issuers, certificates and the failure modes to check first.", rationale="Meta under 120 chars.", evidence_ref=["rec:r1"]),
        ],
    )


@dataclass
class FakeWebApp:
    """In-memory stand-in for the web app's internal API. Mirrors its contract and its approval gate."""

    article_status: str = "published"
    validate_errors: list[list[str]] = field(default_factory=list)  # queued: one list per validate call
    verify_states: list[str] = field(default_factory=lambda: ["passed"])
    article_body: str = "Intro paragraph.\n\n## Setup\n\nInstall cert-manager."
    weekly_items: list[dict[str, Any]] = field(default_factory=list)
    drafts: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    published: list[str] = field(default_factory=list)
    submitted: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    links_planned: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)
    docs: dict[str, str] = field(default_factory=dict)
    fail_publish_transient: int = 0

    def approve(self, draft_id: str, approval_id: str = "appr-1") -> dict[str, Any]:
        """What the admin approve route does before resuming: record an approval for the accepted changes."""
        accepted = [c["id"] for c in self.drafts[draft_id]["changes"]]
        self.approvals[approval_id] = {"draft": draft_id, "accepted": accepted, "used": False}
        return {"decision": "approve", "approval_id": approval_id, "accepted_change_ids": accepted, "actor": "admin@example.com"}

    async def article_context(self, article_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("context", (article_id, body)))
        return {
            "status": self.article_status,
            "collection": "blog",
            "prompt": {"user": f'Draft changes.\n\n<data name="article_blocks">\n{self.article_body}\n</data>', "rules": "RULES", "styleGuide": "STYLE", "maxTokens": 4096},
            "recommendations": [{"id": "r1", "category": "low_ctr"}],
            "modelCategories": ["title_meta"],
            "categories": ["title_meta"],
            "linksAllowed": False,
            "draftingEnabled": True,
            "ga4": {"engagementRate": 0.61},
            "metrics": {"clicks28d": 40, "impressions28d": 2100},
        }

    async def validate(self, article_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("validate", body))
        errs = self.validate_errors.pop(0) if self.validate_errors else []
        if body.get("kind") == "new_article":
            return {"errors": errs, "flags": [], "changes": [{"id": "c1", "type": "new_article", "decision": "pending"}]}
        out = body.get("output") or {"changes": []}
        changes = [{"id": f"c{i + 1}", "type": c["type"], "before": c["before"], "after": c["after"], "decision": "pending", "flags": []} for i, c in enumerate(out.get("changes", []))]
        return {"errors": errs, "flags": [], "changes": changes}

    async def save_draft(self, draft_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("save", draft_id))
        self.drafts[draft_id] = {**body, "status": "ready_for_review"}
        return {"ok": True}

    async def set_draft_status(self, draft_id: str, status: str, reason: str | None = None) -> dict[str, Any]:
        self.calls.append(("status", (draft_id, status, reason)))
        self.drafts.setdefault(draft_id, {})["status"] = status
        return {"ok": True}

    async def publish(self, draft_id: str, approval_id: str, accepted_change_ids: list[str]) -> dict[str, Any]:
        self.calls.append(("publish", (draft_id, approval_id)))
        if self.fail_publish_transient > 0:
            self.fail_publish_transient -= 1
            raise WebAppError(503, "temporarily unavailable")
        a = self.approvals.get(approval_id)
        if not a or a["draft"] != draft_id or a["used"]:
            raise WebAppError(403, "No valid admin approval for this draft.", "approval_required")
        if sorted(a["accepted"]) != sorted(accepted_change_ids):
            raise WebAppError(409, "Accepted changes differ from the approval.", "approval_mismatch")
        a["used"] = True
        self.published.append(draft_id)
        return {"status": "published", "version": 8, "substantive": False}

    async def verify(self, draft_id: str, attempt: int, final: bool) -> dict[str, Any]:
        self.calls.append(("verify", (attempt, final)))
        state = self.verify_states[min(attempt - 1, len(self.verify_states) - 1)]
        if state != "passed" and final:
            self.alerts.append(draft_id)
        return {"state": state if state == "passed" else ("failed" if final else "retrying"), "attempts": attempt, "checks": []}

    async def submit(self, draft_id: str) -> dict[str, Any]:
        self.submitted.append(draft_id)
        return {"indexnow": "queued", "sitemaps": ["queued"]}

    async def read_article(self, ref: str) -> dict[str, Any]:
        self.calls.append(("read_article", ref))
        return {"title": "Ingress and TLS on Kubernetes", "path": "/blog/ingress-tls", "body": self.article_body}

    async def published_articles(self) -> list[dict[str, Any]]:
        return [{"id": ARTICLE, "title": "Ingress and TLS on Kubernetes", "path": "/blog/ingress-tls", "body": self.article_body}]

    async def gsc_queries(self, *, topic: str | None = None, article_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        self.calls.append(("gsc", topic or article_id))
        return [{"query": "kubernetes ingress tls", "impressions": 900, "clicks": 30, "position": 6.2}, {"query": "email me at bob@example.com", "impressions": 3, "clicks": 0, "position": 40}]

    async def fetch_docs(self, url: str) -> dict[str, Any]:
        self.calls.append(("fetch_docs", url))
        if url in self.docs:
            return {"status": 200, "finalUrl": url, "title": "Docs", "body": self.docs[url]}
        return {"status": 404, "error": "not found"}

    async def weekly_candidates(self, limit: int) -> dict[str, Any]:
        return {"items": self.weekly_items[:limit], "fresh": True}

    async def plan_links(self, article_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        self.links_planned.append((article_id, candidates))
        return {"added": min(3, len(candidates)), "draftId": DRAFT}

    async def calendar_brief(self, calendar_id: str) -> dict[str, Any]:
        return {"postId": ARTICLE, "title": "Rotating cert-manager certificates", "gap": "Asked on site 12 times", "outline": ["Why rotate", "How", "Verify"], "styleGuide": "STYLE", "supportingQueries": ["cert-manager renew"]}


@pytest.fixture
def spend() -> MemorySpendStore:
    return MemorySpendStore()


@pytest.fixture
def guard(spend: MemorySpendStore) -> BudgetGuard:
    return BudgetGuard(spend, monthly_usd=10.0, ask_daily_usd=1.0)


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


os.environ.setdefault("SEO_LLM_PRICES_JSON", "")
