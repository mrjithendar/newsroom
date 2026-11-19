"""Crew configuration: the code's defaults, the admin's versioned overlay on them, and the test-run override.

What an admin can change from the CrewAI console (moks-cms /admin/crewai/crews, D-98), per crew:
- each agent's role, goal, backstory, model, output-token limit, iteration limit and tools (only tools this crew's
  agents have in code: the allowlist in app/tools stays the boundary);
- each task's instructions and, where the code does not parse it, its expected output.

What stays in code (guardrails): the untrusted-data rule and the skills rule in every system prompt, the data blocks
a task gets, each task's output contract (JSON shapes, the NEEDS_FACT_CHECK marker, the news crew's rules), the
sequential process, no delegation, memory or code execution, the token and wall-clock caps, and which flows run which
crew. A crew's structure (its agents and tasks, their order and context) is code too.

Where a crew's configuration comes from, per run:
1. a test run's override (`crew_test_flow`, CREW_OVERRIDE): the edited, unsaved configuration, like Jenkins Replay;
2. otherwise the crew's active version, served by the web app (GET /internal/seo-agent/crew-configs, read before every
   crew run like the skills; moks-cms table seo_crew_configs);
3. otherwise, or when the web app cannot be reached, the code's defaults.

An overlay holds only what differs from the defaults (`normalize`). An overlay that no longer fits the code (an agent
or task that was removed, a tool no longer allowed) is applied as far as it fits, and the rest is ignored with a
warning: a stored version never stops a crew. `default_hash` changes whenever a default changes, so the console can
say "the code's defaults changed since this version was saved".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..llm import PRICES
from .common import AgentDef

log = logging.getLogger("seo_agent.crews")

FETCH_TIMEOUT_S = 5.0
CREW_IDS = ("editorial", "optimization", "report", "promotion", "news")
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
MODEL_RE = r"^claude-[a-z0-9][a-z0-9.-]{2,60}$"

# A test run's configuration, per crew (crew_test_flow sets it for its own task only).
CREW_OVERRIDE: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar("seo_crew_override", default=None)


@dataclass(frozen=True)
class TaskDef:
    """One task of a crew. `instructions` and `expected_output` are the editable defaults; `variables` are the
    placeholders the instructions may use ({name}); `contract` is the code-owned text the crew always adds (shown in
    the editor, not editable). `output_editable=False`: the code parses the answer, so its expected output stays."""

    key: str
    agent: str
    label: str
    instructions: str
    expected_output: str
    variables: tuple[str, ...] = ()
    contract: str = ""
    output_editable: bool = True


@dataclass(frozen=True)
class CrewDef:
    crew: str
    label: str
    description: str
    prompt_version: str
    agents: tuple[AgentDef, ...]
    tasks: tuple[TaskDef, ...]
    replay: tuple[str, ...]  # flows whose runs a test run can replay (crew_test_flow)
    tools: tuple[str, ...] = ()  # tools an agent of this crew may be given; default: every tool its agents have in code
    default_models: str = "draft"  # which setting gives the crew's default model: draft | classify | profile

    def allowed_tools(self) -> tuple[str, ...]:
        return self.tools or tuple(dict.fromkeys(t for a in self.agents for t in a.tools))

    def agent(self, key: str) -> AgentDef:
        return next(a for a in self.agents if a.key == key)

    def task(self, key: str) -> TaskDef:
        return next(t for t in self.tasks if t.key == key)


# --- the overlay ------------------------------------------------------------------------------------------------------


class AgentOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str | None = Field(default=None, min_length=1, max_length=200)
    goal: str | None = Field(default=None, min_length=1, max_length=2000)
    backstory: str | None = Field(default=None, min_length=1, max_length=4000)
    model: str | None = Field(default=None, pattern=MODEL_RE)
    max_tokens: int | None = Field(default=None, ge=256, le=64_000)
    max_iter: int | None = Field(default=None, ge=1, le=25)
    tools: list[str] | None = Field(default=None, max_length=12)


class TaskOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str | None = Field(default=None, min_length=1, max_length=12_000)
    expected_output: str | None = Field(default=None, min_length=1, max_length=2000)


class CrewOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agents: dict[str, AgentOverride] = Field(default_factory=dict, max_length=12)
    tasks: dict[str, TaskOverride] = Field(default_factory=dict, max_length=12)


class OverlayError(ValueError):
    def __init__(self, issues: list[str]):
        super().__init__("; ".join(issues)[:500])
        self.issues = issues


def models() -> list[str]:
    """The models an agent can be set to: the ones this service has prices for (a cost is never guessed)."""
    return sorted(PRICES)


def validate(crew: CrewDef, data: Any) -> CrewOverlay:
    """Strict: the shape, then every agent, task, tool, model and placeholder against the crew's code. Raises
    OverlayError with readable issues ("agents.writer.tools: …")."""
    try:
        ov = CrewOverlay.model_validate(data or {})
    except ValidationError as e:
        raise OverlayError([f"{'.'.join(str(p) for p in err['loc']) or 'config'}: {err['msg']}" for err in e.errors()][:20]) from e
    issues: list[str] = []
    agents = {a.key for a in crew.agents}
    tasks = {t.key: t for t in crew.tasks}
    allowed = set(crew.allowed_tools())
    known_models = set(models())
    for key, a in ov.agents.items():
        if key not in agents:
            issues.append(f"agents.{key}: the {crew.crew} crew has no such agent ({', '.join(sorted(agents))})")
            continue
        if a.tools is not None:
            bad = [t for t in a.tools if t not in allowed]
            if bad:
                issues.append(f"agents.{key}.tools: not allowed for this crew: {', '.join(bad)} (allowed: {', '.join(sorted(allowed)) or 'none'})")
            if len(set(a.tools)) != len(a.tools):
                issues.append(f"agents.{key}.tools: a tool is listed twice")
        if a.model is not None and a.model not in known_models:
            issues.append(f"agents.{key}.model: unknown model {a.model} (known: {', '.join(sorted(known_models))})")
    for key, t in ov.tasks.items():
        task = tasks.get(key)
        if task is None:
            issues.append(f"tasks.{key}: the {crew.crew} crew has no such task ({', '.join(tasks)})")
            continue
        if t.expected_output is not None and not task.output_editable:
            issues.append(f"tasks.{key}.expected_output: fixed by the code (it parses this answer)")
        if t.instructions is not None:
            unknown = sorted({m for m in PLACEHOLDER.findall(t.instructions) if m not in task.variables})
            if unknown:
                hint = f"; this task's placeholders: {', '.join('{' + v + '}' for v in task.variables)}" if task.variables else "; this task has no placeholders"
                issues.append(f"tasks.{key}.instructions: unknown placeholder {', '.join('{' + u + '}' for u in unknown)}{hint}")
    if issues:
        raise OverlayError(issues)
    return ov


def normalize(crew: CrewDef, ov: CrewOverlay) -> dict[str, Any]:
    """The overlay as stored: only the fields that differ from the code's defaults (text compared after trimming)."""
    out: dict[str, Any] = {"agents": {}, "tasks": {}}
    for key, a in ov.agents.items():
        d = crew.agent(key)
        diff: dict[str, Any] = {}
        for f in ("role", "goal", "backstory"):
            v = getattr(a, f)
            if v is not None and v.strip() != getattr(d, f).strip():
                diff[f] = v.strip()
        if a.model is not None:
            diff["model"] = a.model
        if a.max_tokens is not None and a.max_tokens != d.max_tokens:
            diff["max_tokens"] = a.max_tokens
        if a.max_iter is not None:
            diff["max_iter"] = a.max_iter
        if a.tools is not None and list(a.tools) != list(d.tools):
            diff["tools"] = list(a.tools)
        if diff:
            out["agents"][key] = diff
    for key, t in ov.tasks.items():
        d2 = crew.task(key)
        diff = {}
        if t.instructions is not None and t.instructions.strip() != d2.instructions.strip():
            diff["instructions"] = t.instructions.strip()
        if t.expected_output is not None and t.expected_output.strip() != d2.expected_output.strip():
            diff["expected_output"] = t.expected_output.strip()
        if diff:
            out["tasks"][key] = diff
    return {k: v for k, v in out.items() if v}


def lenient(crew: CrewDef, data: Any) -> dict[str, Any]:
    """A stored overlay as far as it still fits the code: each agent and task entry is validated on its own and
    dropped (with a warning) when it does not. Used at run time; the console saves only overlays `validate` accepts."""
    if not isinstance(data, Mapping):
        return {}
    kept: dict[str, Any] = {"agents": {}, "tasks": {}}
    for part in ("agents", "tasks"):
        for key, value in (data.get(part) or {}).items() if isinstance(data.get(part), Mapping) else []:
            try:
                validate(crew, {part: {key: value}})
            except OverlayError as e:
                log.warning("crew configuration entry ignored", extra={"event": "crew_config_ignored", "crew": crew.crew, "entry": f"{part}.{key}", "error": str(e)[:300]})
                continue
            kept[part][key] = value
    return {k: v for k, v in kept.items() if v}


# --- the defaults' fingerprint and the catalogue ----------------------------------------------------------------------


def defaults(crew: CrewDef) -> dict[str, Any]:
    return {
        "agents": [{"key": a.key, "role": a.role, "goal": a.goal, "backstory": a.backstory, "tools": list(a.tools), "max_tokens": a.max_tokens} for a in crew.agents],
        "tasks": [{"key": t.key, "agent": t.agent, "label": t.label, "instructions": t.instructions, "expected_output": t.expected_output, "variables": list(t.variables), "contract": t.contract, "output_editable": t.output_editable} for t in crew.tasks],
    }


def default_hash(crew: CrewDef) -> str:
    return hashlib.sha256(json.dumps(defaults(crew), sort_keys=True).encode()).hexdigest()[:16]


def catalogue(crew: CrewDef) -> dict[str, Any]:
    """GET /crews: what the console's crew editor shows and diffs against."""
    return {
        "crew": crew.crew,
        "label": crew.label,
        "description": crew.description,
        "prompt_version": crew.prompt_version,
        "default_hash": default_hash(crew),
        "tools": list(crew.allowed_tools()),
        "models": models(),
        "replay_flows": list(crew.replay),
        **defaults(crew),
    }


def registry() -> dict[str, CrewDef]:
    """Every crew's definition, from the crew modules (imported here, not at module load: they import this one)."""
    from ..news import crew as news
    from . import editorial, optimization, promotion, report

    return {c.crew: c for c in (editorial.CREW, optimization.CREW, report.CREW, promotion.CREW, news.CREW)}


# --- the configuration a run uses ---------------------------------------------------------------------------------------


def render(template: str, values: Mapping[str, Any]) -> str:
    """Replace the task's placeholders ({name}); any other brace (JSON in an example) is left as it is."""
    return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)


@dataclass
class Effective:
    """The configuration one crew run uses. `source`: code | version | test."""

    crew: CrewDef
    overlay: dict[str, Any] = field(default_factory=dict)
    source: str = "code"
    version: int | None = None

    @property
    def prompt_version(self) -> str:
        """Written with every model call (seo_llm_usage, transcripts): which prompts produced the answer."""
        if self.source == "test":
            return f"{self.crew.prompt_version}+test" + (f"-v{self.version}" if self.version else "")
        if self.source == "version" and self.version:
            return f"{self.crew.prompt_version}+v{self.version}"
        return self.crew.prompt_version

    def agents(self, keys: tuple[str, ...] | None = None) -> list[AgentDef]:
        out = []
        for a in self.crew.agents:
            if keys is not None and a.key not in keys:
                continue
            o = (self.overlay.get("agents") or {}).get(a.key) or {}
            out.append(
                replace(
                    a,
                    role=o.get("role") or a.role,
                    goal=o.get("goal") or a.goal,
                    backstory=o.get("backstory") or a.backstory,
                    tools=list(o["tools"]) if isinstance(o.get("tools"), list) else list(a.tools),
                    max_tokens=o.get("max_tokens") or a.max_tokens,
                    model=o.get("model") or a.model,
                    max_iter=o.get("max_iter") or a.max_iter,
                )
            )
        return out

    def instructions(self, task: str, **values: Any) -> str:
        t = self.crew.task(task)
        o = (self.overlay.get("tasks") or {}).get(task) or {}
        return render(o.get("instructions") or t.instructions, values)

    def expected_output(self, task: str) -> str:
        t = self.crew.task(task)
        o = (self.overlay.get("tasks") or {}).get(task) or {}
        return (o.get("expected_output") if t.output_editable else None) or t.expected_output

    def summary(self) -> dict[str, Any]:
        return {"crew": self.crew.crew, "source": self.source, "version": self.version, "prompt_version": self.prompt_version}


def code_defaults(crew: str) -> Effective:
    return Effective(registry()[crew])


async def active_versions(web: Any, timeout_s: float = FETCH_TIMEOUT_S) -> dict[str, dict[str, Any]]:
    """{crew: {version, config}} of the active versions, from the web app; {} when it has none or cannot be reached."""
    get = getattr(web, "crew_configs", None)
    if get is None:
        return {}
    try:
        rows = await asyncio.wait_for(get(), timeout=timeout_s)
    except Exception as e:  # a version improves a run; its absence never fails it (CancelledError still propagates)
        log.warning("crew configuration unavailable; the crew runs with the code's defaults", extra={"event": "crew_config_unavailable", "error": type(e).__name__})
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in rows or []:
        if isinstance(r, Mapping) and isinstance(r.get("crew"), str) and isinstance(r.get("config"), Mapping):
            try:
                out[r["crew"]] = {"version": int(r.get("version") or 0) or None, "config": dict(r["config"])}
            except (TypeError, ValueError):
                continue
    return out


async def load(crew: str, web: Any) -> Effective:
    """The configuration for one run of `crew`: a test run's override, else the active version, else the code."""
    c = registry()[crew]
    override = CREW_OVERRIDE.get()
    if override is not None and crew in override:
        o = override[crew]
        return Effective(c, lenient(c, o.get("config")), "test", o.get("version"))
    row = (await active_versions(web)).get(crew)
    if row:
        ov = lenient(c, row["config"])
        log.info("crew configuration version applied", extra={"event": "crew_config", "crew": crew, "version": row["version"]})
        return Effective(c, ov, "version", row["version"])
    return Effective(c)
