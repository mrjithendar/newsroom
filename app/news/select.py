"""Selection (docs/news/ML.md "Selection"): job filters → threshold → caps → diversity. Every candidate gets exactly one
decision and a human-readable reason; the order of the checks below is the order the reasons are reported in."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import JobConfig


@dataclass
class Candidate:
    cluster_id: str
    title: str
    score: float
    breakdown: dict[str, Any]
    features: dict[str, float]
    model_version: str
    topics: list[str]
    source_ids: list[str]
    text: str  # title + summaries, for keyword filters
    entities: set[str] = field(default_factory=set)
    age_hours: float = 0.0
    language: str | None = "en"
    site_match: dict[str, Any] | None = None
    covered: bool = False  # drafted/published/selected for this job within the dedupe window
    decision: str = "rejected"
    reason: str = ""
    rank: int | None = None


@dataclass
class Counts:
    """Already-used capacity (in the job's time zone "today"), read inside the same transaction as the reservation."""

    job_today: int = 0
    global_today: int = 0
    per_topic: dict[str, int] = field(default_factory=dict)
    per_source: dict[str, int] = field(default_factory=dict)


def _hit(term: str, text: str) -> bool:
    return bool(re.search(rf"(?<![\w-]){re.escape(term.lower())}(?![\w-])", text))


def _filters(c: Candidate, cfg: JobConfig) -> str | None:
    text = c.text.lower()
    ents = {e.split(":", 1)[-1] for e in c.entities}
    if c.language and cfg.language and c.language != cfg.language:
        return f"language {c.language} (job is {cfg.language})"
    if c.age_hours > cfg.max_age_hours:
        return f"too old ({c.age_hours:.0f} h > {cfg.max_age_hours:.0f} h)"
    for k in cfg.exclude_keywords:
        if _hit(k, text):
            return f"excluded keyword “{k}”"
    for e in cfg.exclude_entities:
        if e.lower() in ents:
            return f"excluded entity “{e}”"
    if cfg.include_keywords or cfg.include_entities:
        if not any(_hit(k, text) for k in cfg.include_keywords) and not any(e.lower() in ents for e in cfg.include_entities):
            return "no include keyword or entity matched"
    if cfg.topics and not set(cfg.topics) & set(c.topics):
        return f"off-topic for this job (topics: {', '.join(c.topics) or 'none'})"
    if c.covered:
        return f"already covered within {cfg.dedupe_window_days} days"
    if c.site_match and c.site_match.get("update_candidate"):
        return f"an existing article covers this ({c.site_match.get('path')}); suggested as an update instead"
    if c.score < cfg.min_score:
        return f"score {c.score:.2f} below the job's threshold {cfg.min_score:.2f}"
    return None


def select(cands: list[Candidate], cfg: JobConfig, counts: Counts, *, global_cap: int, mode: str = "single_stories") -> list[Candidate]:
    """Mutates and returns the candidates (sorted by score) with decision/reason/rank set."""
    cands.sort(key=lambda c: (-c.score, c.cluster_id))
    eligible: list[Candidate] = []
    for c in cands:
        why = _filters(c, cfg)
        if why:
            c.decision, c.reason = "rejected", why
        else:
            eligible.append(c)
    if mode == "daily_digest":
        return _digest(cands, eligible, cfg, counts, global_cap)
    per_topic = dict(counts.per_topic)
    per_source = dict(counts.per_source)
    taken = 0
    for c in eligible:
        if taken >= cfg.max_articles_per_run:
            c.reason = f"per-run cap ({cfg.max_articles_per_run}) reached"
            continue
        if counts.job_today + taken >= cfg.max_articles_per_day:
            c.reason = f"daily cap for this job ({cfg.max_articles_per_day}) reached"
            continue
        if counts.global_today + taken >= global_cap:
            c.reason = f"global daily cap ({global_cap}) reached"
            continue
        main_topic = c.topics[0] if c.topics else "uncategorised"
        if per_topic.get(main_topic, 0) >= cfg.diversity.max_per_topic_per_day:
            c.reason = f"diversity: {cfg.diversity.max_per_topic_per_day} stories on {main_topic} today"
            continue
        crowded = [s for s in c.source_ids if per_source.get(s, 0) >= cfg.diversity.max_per_source_per_day]
        if crowded and len(crowded) == len(c.source_ids):
            c.reason = f"diversity: {cfg.diversity.max_per_source_per_day} stories from this source today"
            continue
        taken += 1
        c.decision, c.rank = "selected", taken
        c.reason = f"selected: score {c.score:.2f}, rank {taken}"
        per_topic[main_topic] = per_topic.get(main_topic, 0) + 1
        for s in c.source_ids:
            per_source[s] = per_source.get(s, 0) + 1
    return cands


def _digest(cands: list[Candidate], eligible: list[Candidate], cfg: JobConfig, counts: Counts, global_cap: int) -> list[Candidate]:
    if counts.job_today >= cfg.max_articles_per_day or counts.global_today >= global_cap or cfg.max_articles_per_run < 1:
        for c in eligible:
            c.reason = "daily cap reached (a digest counts as one article)"
        return cands
    if len(eligible) < 2:
        for c in eligible:
            c.reason = "too few stories for a digest (needs at least 2)"
        return cands
    for i, c in enumerate(eligible):
        if i < cfg.digest_size:
            c.decision, c.rank, c.reason = "selected", i + 1, f"selected for the digest: score {c.score:.2f}, position {i + 1}"
        else:
            c.reason = f"digest is full ({cfg.digest_size} stories)"
    return cands
