"""News job config and global news settings. The same shapes are validated by zod in moks-cms/src/lib/news/schema.ts;
tests/test_news_config.py and the web test news-desk.test.ts both parse docs/news/fixtures/job-config.json so the two
stay in step.

The kill switch is NEWS_DESK_ENABLED (env, default false). The admin Settings page can pause the desk (setting row
`news`.paused) but cannot switch it on when the environment says off.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["single_stories", "daily_digest", "both"]
FEATURES = ("credibility", "source_count", "burst", "recency", "topic_match", "audience", "novelty", "severity", "keyword_overlap")
DEFAULT_WEIGHTS: dict[str, float] = {
    "credibility": 0.14,
    "source_count": 0.12,
    "burst": 0.10,
    "recency": 0.16,
    "topic_match": 0.14,
    "audience": 0.10,
    "novelty": 0.12,
    "severity": 0.06,
    "keyword_overlap": 0.06,
}


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_daily_usd: float = Field(default=1.0, ge=0, le=100)
    llm_monthly_usd: float = Field(default=15.0, ge=0, le=1000)
    image_daily_usd: float = Field(default=0.5, ge=0, le=100)
    image_monthly_usd: float = Field(default=5.0, ge=0, le=1000)
    max_tokens_per_draft: int = Field(default=120_000, ge=10_000, le=400_000)


class Diversity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_per_topic_per_day: int = Field(default=2, ge=1, le=20)
    max_per_source_per_day: int = Field(default=2, ge=1, le=20)


class ImageOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    provider: Literal["template", "gemini", "openai"] | None = None  # None = the image profile's provider
    variants: int | None = Field(default=None, ge=1, le=4)


class JobConfig(BaseModel):
    """Everything a job can change. Unknown keys are refused so a typo never silently does nothing."""

    model_config = ConfigDict(extra="forbid")
    source_ids: list[str] = Field(default_factory=list, max_length=200)
    include_keywords: list[str] = Field(default_factory=list, max_length=100)
    exclude_keywords: list[str] = Field(default_factory=list, max_length=100)
    include_entities: list[str] = Field(default_factory=list, max_length=100)
    exclude_entities: list[str] = Field(default_factory=list, max_length=100)
    topics: list[str] = Field(default_factory=list, max_length=40)
    max_articles_per_run: int = Field(default=3, ge=0, le=20)
    max_articles_per_day: int = Field(default=5, ge=0, le=40)
    min_score: float = Field(default=0.45, ge=0, le=1)
    dedupe_window_days: int = Field(default=7, ge=1, le=60)
    language: str = Field(default="en", pattern=r"^[a-z]{2}$")
    length_min_words: int = Field(default=350, ge=100, le=5000)
    length_max_words: int = Field(default=1200, ge=150, le=8000)
    author: str | None = Field(default=None, max_length=80)
    default_categories: list[str] = Field(default_factory=list, max_length=5)
    writing_profile_version: int | None = Field(default=None, ge=1)
    image_profile_version: int | None = Field(default=None, ge=1)
    models: dict[str, str] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)
    diversity: Diversity = Field(default_factory=Diversity)
    image: ImageOptions = Field(default_factory=ImageOptions)
    digest_size: int = Field(default=5, ge=2, le=12)
    recency_half_life_hours: float = Field(default=36, ge=1, le=720)
    max_age_hours: float = Field(default=96, ge=1, le=2160)
    weights: dict[str, float] = Field(default_factory=dict)
    expiry_days: int | None = Field(default=None, ge=1, le=3650)
    publishing_slots: list[str] = Field(default_factory=list, max_length=12)
    notify: bool = True

    @field_validator("weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k in v if k not in FEATURES]
        if bad:
            raise ValueError(f"unknown score features: {', '.join(bad)}")
        if any(not 0 <= w <= 1 for w in v.values()):
            raise ValueError("weights must be between 0 and 1")
        return v

    @field_validator("models")
    @classmethod
    def _models(cls, v: dict[str, str]) -> dict[str, str]:
        import re

        for k, m in v.items():
            if k not in ("researcher", "analyst", "writer", "fact_checker", "seo_editor", "art_director") or not re.fullmatch(r"claude-[a-z0-9-]{3,40}", m):
                raise ValueError(f"invalid model override {k}={m}")
        return v

    @field_validator("publishing_slots")
    @classmethod
    def _slots(cls, v: list[str]) -> list[str]:
        import re

        for s in v:
            if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", s):
                raise ValueError(f"publishing slot {s!r} is not HH:MM")
        return sorted(set(v))

    def effective_weights(self) -> dict[str, float]:
        w = {**DEFAULT_WEIGHTS, **self.weights}
        total = sum(w.values()) or 1.0
        return {k: round(v / total, 6) for k, v in w.items()}


class Job(BaseModel):
    """A news_jobs row as the flows use it."""

    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    enabled: bool = False
    cron: str = "0 7 * * *"
    timezone: str = "Asia/Kolkata"
    mode: Mode = "single_stories"
    writing_profile_key: str = "news-default"
    image_profile_key: str = "cover-template"
    review_priority: int = 50
    version: int = 1
    config: JobConfig = Field(default_factory=JobConfig)


class Quality(BaseModel):
    """Quality-gate thresholds (docs/news/QUALITY_GATES.md). Global, editable on the Settings page."""

    model_config = ConfigDict(extra="forbid")
    ngram: int = Field(default=8, ge=5, le=15)
    max_ngram_overlap: float = Field(default=0.12, ge=0, le=1)  # share of a paragraph's 8-grams found in one source
    max_shared_run_words: int = Field(default=25, ge=10, le=80)  # longest verbatim run outside a quote
    max_paragraph_cosine: float = Field(default=0.93, ge=0.5, le=1)
    max_quote_words: int = Field(default=25, ge=5, le=60)
    min_analysis_share: float = Field(default=0.25, ge=0, le=1)
    readability_min: float = Field(default=25, ge=-50, le=100)
    readability_max: float = Field(default=75, ge=0, le=120)
    title_max: int = Field(default=60, ge=30, le=110)
    meta_min: int = Field(default=110, ge=50, le=160)
    meta_max: int = Field(default=160, ge=80, le=200)
    max_internal_links: int = Field(default=3, ge=0, le=10)


class NewsSettings(BaseModel):
    """Setting row `news` (web admin → Settings), plus the environment switches."""

    model_config = ConfigDict(extra="ignore")
    env_enabled: bool = False
    paused: bool = False
    llm_daily_usd: float = 2.0
    llm_monthly_usd: float = 20.0
    image_daily_usd: float = 1.0
    image_monthly_usd: float = 10.0
    max_drafts_per_day: int = 10
    image_provider: Literal["template", "gemini", "openai"] = "template"
    writer_model: str = "claude-sonnet-5-5"
    fast_model: str = "claude-haiku-5-5"
    vision_model: str = "claude-haiku-5-5"
    classifier_threshold: float = 0.55
    # Probability recorded for a topic Claude assigned (it returns labels, not probabilities). It feeds topic_match, so
    # keep it near classifier_threshold: high enough to pass, not so high that Claude's picks outrank the model's.
    claude_topic_p: float = Field(default=0.6, ge=0, le=1)
    ranker_min_feedback: int = 100
    # Opt-in exploration (docs/news/ML.md "Selection"): share of below-threshold candidates made eligible anyway.
    explore_rate: float = Field(default=0.0, ge=0, le=0.2)
    # Items naming more CVE ids than this (advisory roundups) do not use the shared-CVE clustering rule.
    cve_roundup_max: int = Field(default=5, ge=1, le=100)
    # Texts shorter than this many words skip MinHash near-dedupe (exact and semantic dedupe still apply).
    near_dup_min_tokens: int = Field(default=30, ge=0, le=500)
    log_retention_days: int = 90
    item_retention_days: int = 180
    draft_concurrency: int = 2
    max_items_per_source: int = 50
    quality: Quality = Field(default_factory=Quality)
    image_prices: dict[str, float] = Field(default_factory=lambda: {"template": 0.0, "gemini": 0.04, "openai": 0.04})
    paywalled_hosts: list[str] = Field(default_factory=list)  # never fetched (enforced by the web app's fetcher too)
    primary_source_hosts: list[str] = Field(default_factory=list)  # extra hosts the Researcher may fetch

    @property
    def enabled(self) -> bool:
        return self.env_enabled and not self.paused


def env_enabled(env: dict[str, str] | None = None) -> bool:
    v = (env if env is not None else os.environ).get("NEWS_DESK_ENABLED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def load_news_settings(row: dict[str, Any] | None, env: dict[str, str] | None = None) -> NewsSettings:
    e = env if env is not None else dict(os.environ)
    data = dict(row or {})
    provider = e.get("NEWS_IMAGE_PROVIDER", "").strip().lower()
    if provider in ("template", "gemini", "openai"):
        data["image_provider"] = provider
    try:
        s = NewsSettings.model_validate(data)
    except ValueError:
        s = NewsSettings()
    return s.model_copy(update={"env_enabled": env_enabled(e)})
