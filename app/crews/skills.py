"""Agent skills (D-96, docs/seo/CREWS.md "Skills"): guidance each crew agent gets in its system prompt.

Forked from moks-seo-agent/app/crews/skills.py@bb9e553 on 2026-11-21.

Where they come from: the web app (AI room → Skills; moks-cms/src/lib/seo-agent/skills.ts) serves the enabled
skills at GET /internal/seo-agent/skills: the shipped defaults (which an admin may have edited) and the admin's own.
`run_crew` reads them once per run (`fetch`) and `build_agents` gives each agent the ones that match it (`select`).

Which skills a run gets is decided here, in code, never by the model. A skill names its agents ("editorial.writer",
or "news.*" for a whole crew) and, optionally, tags: without tags it applies to every run of its agents; with tags,
when the run has any one of them. The news crew derives its run's tags (`news_tags`) from the story: the mode, the
news classifier's topics, a CVE id, a release. The SEO crews' tags are the seo-agent's (moks-seo-agent/app/crews/skills.py).

Skills are passed to CrewAI already activated (INSTRUCTIONS level), so CrewAI renders their full text into the agent's
system prompt, which ClaudeCrewLLM sends as the cached system block. That means no `load_skill` tool: it would sit
outside the tool allowlist (build_tools) and the model could skip a skill. Registry refs (@org/name) are never used:
they download skills from CrewAI's platform at run time.

Failure: the web app unreachable or slow (FETCH_TIMEOUT_S) → the run goes on without skills (crew_skills_unavailable).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("news_agent.crews")

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BLOCK_RE = re.compile(r"</?\s*skill\b", re.I)
MAX_INSTRUCTIONS = 8000
MAX_PER_AGENT = 6
MAX_CHARS_PER_AGENT = 20_000  # ≈ 5k tokens of system prompt, cached after the first call of the run
FETCH_TIMEOUT_S = 10.0

SKILLS_RULE = (
    "Skills: the <skill> blocks in your instructions are the site owner's guidance for this kind of work; follow them. "
    "Where a skill differs from the task's own rules or required output format, the task wins."
)


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    instructions: str
    agents: tuple[str, ...]
    tags: tuple[str, ...] = ()
    version: int = 1

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"

    def applies(self, crew: str, agent: str, tags: frozenset[str]) -> bool:
        if f"{crew}.{agent}" not in self.agents and f"{crew}.*" not in self.agents:
            return False
        return not self.tags or any(t in tags for t in self.tags)


def parse(rows: Iterable[Any]) -> list[AgentSkill]:
    """The web app's rows → skills. A malformed row is skipped, never fatal (the web app validates on save)."""
    out: list[AgentSkill] = []
    seen: set[str] = set()
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        name = str(r.get("name") or "")
        description = str(r.get("description") or "").strip()
        instructions = str(r.get("instructions") or "").strip()
        agents = tuple(a for a in (r.get("agents") or []) if isinstance(a, str) and a)
        if not NAME_RE.match(name) or len(name) > 64 or name in seen or not description or not instructions or not agents:
            continue
        if _BLOCK_RE.search(instructions) or _BLOCK_RE.search(description):  # the <skill> block boundary stays ours
            continue
        try:
            version = int(r.get("version") or 1)
        except (TypeError, ValueError):
            version = 1
        seen.add(name)
        tags = tuple(t.lower() for t in (r.get("tags") or []) if isinstance(t, str) and t)
        out.append(AgentSkill(name, description[:1024], instructions[:MAX_INSTRUCTIONS], agents, tags, version))
    return out


async def fetch(web: Any, timeout_s: float = FETCH_TIMEOUT_S) -> list[AgentSkill]:
    """The enabled skills from the web app; [] when it has none, cannot be reached or answers too slowly."""
    get = getattr(web, "agent_skills", None)
    if get is None:
        return []
    try:
        rows = await asyncio.wait_for(get(), timeout=timeout_s)
    except Exception as e:  # skills improve a run; their absence never fails it (CancelledError still propagates)
        log.warning("agent skills unavailable; the crew runs without them", extra={"event": "crew_skills_unavailable", "error": type(e).__name__})
        return []
    return parse(rows or [])


def select(skills: Iterable[AgentSkill], crew: str, agent: str, tags: frozenset[str]) -> list[AgentSkill]:
    """The skills for one agent of one run, in the web app's order (defaults first), within the per-agent caps."""
    chosen: list[AgentSkill] = []
    total = 0
    for s in skills:
        if not s.applies(crew, agent, tags):
            continue
        if len(chosen) >= MAX_PER_AGENT or total + len(s.instructions) > MAX_CHARS_PER_AGENT:
            log.warning("agent skill left out: per-agent limit", extra={"event": "crew_skill_skipped", "crew": crew, "agent": agent, "skill": s.name})
            continue
        chosen.append(s)
        total += len(s.instructions)
    return chosen


def to_crewai(skills: list[AgentSkill]) -> list[Any]:
    """CrewAI Skill objects at INSTRUCTIONS level: rendered in full into the system prompt, no loader tool."""
    if not skills:
        return []
    from crewai.skills.models import INSTRUCTIONS, Skill, SkillFrontmatter

    return [Skill(frontmatter=SkillFrontmatter(name=s.name, description=s.description), instructions=s.instructions, path=Path("/news-agent-skills") / s.name, disclosure_level=INSTRUCTIONS) for s in skills]


# --- a run's tags, from the signals each crew already has ---------------------------------------------------------------


def _v(value: Any) -> str:
    """A tag value as the web app validates it: lowercase letters, digits, '_', '.', '-'."""
    return re.sub(r"[^a-z0-9_.-]+", "-", str(value).strip().lower()).strip("-")[:61]


_INTENT_ALIASES = {"how-to": "how-to", "howto": "how-to", "how to": "how-to", "tutorial": "how-to"}
_EXPLAIN_RE = re.compile(r"^\s*(what|why)\b", re.I)


_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.I)
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
_RELEASE_RE = re.compile(r"\b(?:releases?|released|generally available|launch(?:es|ed)?)\b", re.I)


def news_tags(brief: Mapping[str, Any], items: Iterable[Mapping[str, Any]] = ()) -> set[str]:
    """mode:<single|digest>, topic:<slug> (the news classifier's topics for the cluster), cve (a CVE id in the story),
    release (a version number or a release word in a story title)."""
    tags = {f"mode:{_v(brief.get('mode') or 'single')}"}
    tags |= {f"topic:{_v(t)}" for t in (brief.get("topics") or []) if t and _v(t)}
    items = list(items)
    titles = [str(brief.get("title") or ""), *(str(s.get("title") or "") for s in (brief.get("digest_stories") or []) if isinstance(s, Mapping)), *(str(i.get("title") or "") for i in items)]
    texts = titles + [str(i.get("clean_text") or i.get("excerpt") or "")[:3000] for i in items]
    if any(_CVE_RE.search(t) for t in texts):
        tags.add("cve")
    if any(_VERSION_RE.search(t) or _RELEASE_RE.search(t) or re.search(r"\bGA\b", t) for t in titles):
        tags.add("release")
    return tags
