"""The registry of admin-editable settings: which environment variables an admin may set from the web app.

One module per group in `groups/`. A field keeps its environment-variable
name as its key, so the precedence is easy to state: the admin's saved value, then the deployment's environment
variable, then the default in app/config.py. `attr` names the Settings attribute the variable feeds; it is how the
store reports the effective value and applies a live change.

`apply`:
- "live": every reader takes it from the shared Settings object at use time, so a save applies at once in this
  process and within REFRESH_SECONDS in any other.
- "restart": a component copied the value when it was built (a client's timeout, the flow runner's limit). A save is
  stored and reported as "restart pending" until the running value matches.

Never register (bootstrap and lock-out risk; checked by a test): DATABASE_URL, NEWS_AGENT_DATABASE_URL,
NEWS_AGENT_TOKEN, NEWS_AGENT_CALLBACK_TOKEN, NEWS_AGENT_CALLBACK_URL, NEWS_AGENT_SETTINGS_KEY, SEO_AGENT_URL,
SEO_AGENT_NEWS_TOKEN, NEWS_DESK_ENABLED (the deployment's emergency switch), OTEL_*.

Forked from moks-seo-agent/app/admin_settings/registry.py@6d6464b on 2026-11-21.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

FieldType = Literal["text", "integer", "number", "boolean", "select", "list", "secret"]
Apply = Literal["live", "restart"]

NEVER = (
    "DATABASE_URL",
    "NEWS_AGENT_DATABASE_URL",
    "NEWS_AGENT_TOKEN",
    "NEWS_AGENT_CALLBACK_TOKEN",
    "NEWS_AGENT_CALLBACK_URL",
    "NEWS_AGENT_SETTINGS_KEY",
    "SEO_AGENT_URL",
    "SEO_AGENT_NEWS_TOKEN",
    "NEWS_DESK_ENABLED",
)
NEVER_PREFIXES = ("OTEL_",)
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,60}$")
GROUP_ID = re.compile(r"^[a-z][a-z0-9-]{1,40}$")
MAX_TEXT, MAX_LIST = 2000, 300


@dataclass(frozen=True)
class Field:
    env: str
    attr: str
    label: str
    hint: str
    type: FieldType
    apply: Apply = "live"
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    options: tuple[str, ...] = ()
    pattern: str | None = None  # text: the whole value; list: each item
    item_max: int = 200  # list: the longest item

    @property
    def secret(self) -> bool:
        return self.type == "secret"


@dataclass(frozen=True)
class Group:
    id: str
    title: str
    description: str
    fields: tuple[Field, ...] = field(default_factory=tuple)

    def field(self, env: str) -> Field | None:
        return next((f for f in self.fields if f.env == env), None)


def never(env: str) -> bool:
    return env in NEVER or env.startswith(NEVER_PREFIXES)


def check_registry(groups: list[Group]) -> None:
    """Raises on a registry that could not work: run at import, so a bad group fails the tests and the start-up."""
    seen_groups: set[str] = set()
    seen_env: set[str] = set()
    for g in groups:
        if not GROUP_ID.fullmatch(g.id) or g.id in seen_groups:
            raise ValueError(f"settings group id {g.id!r} is invalid or used twice")
        seen_groups.add(g.id)
        for f in g.fields:
            if not ENV_NAME.fullmatch(f.env) or f.env in seen_env:
                raise ValueError(f"{g.id}: {f.env!r} is not an environment-variable name or is registered twice")
            if never(f.env):
                raise ValueError(f"{g.id}: {f.env} must stay in the deployment (bootstrap or lock-out risk)")
            if f.type == "select" and not f.options:
                raise ValueError(f"{g.id}: {f.env} is a select without options")
            if f.min is not None and f.max is not None and f.min > f.max:
                raise ValueError(f"{g.id}: {f.env} has min above max")
            seen_env.add(f.env)


class Invalid(Exception):
    """One field's value is not acceptable; the message is for the admin."""


def clean(f: Field, value: Any) -> Any:
    """The stored form of an admin's value for `f`, or Invalid. None and "" are handled by the caller (they clear)."""
    if f.type == "boolean":
        if not isinstance(value, bool):
            raise Invalid("send true or false")
        return value
    if f.type in ("integer", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
            raise Invalid("send a number")
        if f.type == "integer":
            if float(value) != int(value):
                raise Invalid("send a whole number")
            value = int(value)
        if f.min is not None and value < f.min:
            raise Invalid(f"at least {f.min:g}")
        if f.max is not None and value > f.max:
            raise Invalid(f"at most {f.max:g}")
        return value
    if f.type == "select":
        if value not in f.options:
            raise Invalid("choose one of: " + ", ".join(f.options))
        return value
    if f.type == "list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise Invalid("send a list of text values")
        items = list(dict.fromkeys(v.strip() for v in value if v.strip()))
        if len(items) > MAX_LIST:
            raise Invalid(f"at most {MAX_LIST} entries")
        for v in items:
            if len(v) > f.item_max or "," in v or any(ord(ch) < 32 for ch in v):
                raise Invalid(f"{v[:40]!r} is too long or has a comma or control character")
            if f.pattern and not re.fullmatch(f.pattern, v):
                raise Invalid(f"{v[:40]!r} is not valid here")
        return items
    if not isinstance(value, str):
        raise Invalid("send text")
    value = value.strip()
    if len(value) > MAX_TEXT or any(ord(ch) < 32 for ch in value):
        raise Invalid(f"at most {MAX_TEXT} characters, on one line")
    if f.pattern and not re.fullmatch(f.pattern, value):
        raise Invalid("not valid here")
    return value


def to_env(f: Field, value: Any) -> str:
    """A stored value as the environment-variable text app/config.py parses."""
    if f.type == "boolean":
        return "true" if value else "false"
    if f.type == "list":
        return ",".join(value)
    return str(value)


def shown(value: Any) -> Any:
    """A Settings attribute as JSON (tuples become lists)."""
    return list(value) if isinstance(value, (tuple, list, set, frozenset)) else value
