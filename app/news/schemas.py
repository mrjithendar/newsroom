"""Structured output of the News Desk Crew and the fact sheet it is checked against.

`NewsDraft.to_document()` is what the web app stores on the draft (seo_drafts.document). It is a superset of the
existing new-article document (title, slug_suggestion, meta_description, outline, body_mdx, sources, fact_check_flags,
internal_links, incomplete), so v4's review page, validators and publishNewArticle keep working; the news-specific part
lives under `news`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Claim(_M):
    id: str = Field(default="", max_length=12)
    claim: str = Field(max_length=600)
    source_url: str = Field(max_length=2000)
    snippet: str = Field(default="", max_length=600)
    kind: Literal["fact", "number", "version", "date", "quote", "cve", "command"] = "fact"


class FactSheet(_M):
    """Researcher output: every claim the article may make, each with the source URL and the supporting snippet."""

    claims: list[Claim] = Field(default_factory=list, max_length=80)
    primary_sources: list[str] = Field(default_factory=list, max_length=20)
    open_questions: list[str] = Field(default_factory=list, max_length=20)


class Analysis(_M):
    why_it_matters: str = Field(default="", max_length=3000)
    who_is_affected: str = Field(default="", max_length=1500)
    what_to_do: list[str] = Field(default_factory=list, max_length=10)
    migration_notes: str = Field(default="", max_length=2000)
    commands: list[dict[str, str]] = Field(default_factory=list, max_length=6)  # {lang, code, source_url}


class NewsSourceRef(_M):
    title: str = Field(max_length=300)
    publisher: str = Field(default="", max_length=120)
    url: str = Field(max_length=2000)
    accessed_at: str = Field(default="", max_length=40)


class FactFlag(_M):
    claim: str = Field(max_length=600)
    reason: str = Field(max_length=600)
    severity: Literal["error", "warning"] = "warning"


class FactCheckReport(_M):
    checked: int = 0
    removed: list[str] = Field(default_factory=list, max_length=40)
    flagged: list[FactFlag] = Field(default_factory=list, max_length=40)


class InternalLink(_M):
    target_path: str = Field(max_length=300)
    anchor: str = Field(max_length=120)
    reason: str = Field(default="", max_length=300)


class ImageBrief(_M):
    subject: str = Field(default="", max_length=400)
    composition: str = Field(default="", max_length=400)
    style_notes: str = Field(default="", max_length=400)
    avoid: list[str] = Field(default_factory=list, max_length=12)


class DigestItem(_M):
    cluster_id: str = Field(default="", max_length=40)
    title: str = Field(max_length=200)
    summary: str = Field(max_length=1200)
    why_it_matters: str = Field(default="", max_length=800)
    sources: list[str] = Field(default_factory=list, max_length=8)


class NewsDraft(_M):
    title: str = Field(max_length=110)
    dek: str = Field(default="", max_length=320)
    meta_description: str = Field(default="", max_length=200)
    slug_suggestion: str = Field(default="", max_length=90)
    body_mdx: str = Field(default="", max_length=60_000)
    key_takeaways: list[str] = Field(default_factory=list, max_length=6)
    why_it_matters: str = Field(default="", max_length=3000)
    headings: list[str] = Field(default_factory=list, max_length=20)
    sources: list[NewsSourceRef] = Field(default_factory=list, max_length=20)
    fact_check_report: FactCheckReport = Field(default_factory=FactCheckReport)
    categories: list[str] = Field(default_factory=list, max_length=3)
    tags: list[str] = Field(default_factory=list, max_length=10)
    internal_links: list[InternalLink] = Field(default_factory=list, max_length=6)
    schema_jsonld: dict[str, Any] = Field(default_factory=dict)
    image_brief: ImageBrief = Field(default_factory=ImageBrief)
    reading_time_minutes: int = Field(default=0, ge=0, le=60)
    confidence: float = Field(default=0.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list, max_length=20)
    digest_items: list[DigestItem] = Field(default_factory=list, max_length=12)
    incomplete: bool = False
    incomplete_reason: str | None = None

    def to_document(self, news: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": self.title,
            "slug_suggestion": self.slug_suggestion,
            "meta_description": self.meta_description,
            "outline": self.headings,
            "body_mdx": self.body_mdx,
            "code_blocks": [],
            "sources": [{"url": s.url, "title": s.title, "used_for": s.publisher} for s in self.sources],
            "fact_check_flags": [f.model_dump() for f in self.fact_check_report.flagged],
            "internal_links": [{"target_path": link.target_path, "anchor": link.anchor, "reason": link.reason} for link in self.internal_links],
            "incomplete": self.incomplete,
            "incomplete_reason": self.incomplete_reason,
            "news": {
                "dek": self.dek,
                "key_takeaways": self.key_takeaways,
                "why_it_matters": self.why_it_matters,
                "sources": [s.model_dump() for s in self.sources],
                "fact_check_report": self.fact_check_report.model_dump(),
                "categories": self.categories,
                "tags": self.tags,
                "schema_jsonld": self.schema_jsonld,
                "image_brief": self.image_brief.model_dump(),
                "reading_time_minutes": self.reading_time_minutes,
                "confidence": self.confidence,
                "warnings": self.warnings,
                "digest_items": [d.model_dump() for d in self.digest_items],
                **news,
            },
        }


class TopicLabels(_M):
    """Claude's topic labels (bootstrap and low-confidence fallback)."""

    items: list[dict[str, Any]] = Field(default_factory=list, max_length=50)  # {i: int, topics: [slug]}


class ImageVerdict(_M):
    """Vision check of one generated image against the brief and the safety rules."""

    safe: bool
    faces: bool = False
    logos_or_trademarks: bool = False
    text_artifacts: bool = False
    other_issues: list[str] = Field(default_factory=list, max_length=8)
    brief_match: float = Field(default=0.0, ge=0, le=1)
    alt_text: str = Field(default="", max_length=200)
    decorative: bool = False


class StyleProposal(_M):
    explanation: str = Field(max_length=3000)
    system_style: str = Field(default="", max_length=6000)
    structure_template: str = Field(default="", max_length=6000)
    tone: str = Field(default="", max_length=1000)
    banned_phrases: list[str] = Field(default_factory=list, max_length=60)
    required_sections: list[str] = Field(default_factory=list, max_length=12)
    examples: list[dict[str, str]] = Field(default_factory=list, max_length=6)  # {before, after, why}
