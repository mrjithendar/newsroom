"""The admin-editable settings store: values an admin saved from the web app, laid over the deployment's environment.

- Stored in the agent's own table `news_agent_settings` (key, value jsonb, updated_by, updated_at), created at start-up
  by the agent's role: no web app migration. The key is the environment-variable name.
- Precedence: the admin's saved value, then the environment variable, then the default in app/config.py. Clearing a
  field ("use the deployment's value") deletes its row.
- Set through the internal API: `GET /admin-settings` (schema, effective values, where each comes from, restart
  pending) and `PUT /admin-settings/{group}` (validated against the registry, all or nothing).
- Secrets are sealed (secrets.py) and never returned: the API says only whether one is set.

A database error before the first successful read means the deployment's values; after one, an outage keeps the last
known values, so it never silently reverts a saved setting.

Forked from moks-seo-agent/app/admin_settings/store.py@6d6464b on 2026-11-21.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, Protocol

from ..config import Settings, load_settings
from . import secrets
from .groups import GROUPS
from .registry import Field, Group, Invalid, clean, shown, to_env

log = logging.getLogger("news_agent.admin_settings")

TABLE = "news_agent_settings"
DDL = f"""create table if not exists {TABLE} (
  key text primary key,
  value jsonb not null,
  updated_by text,
  updated_at timestamptz not null default now()
)"""
REFRESH_SECONDS = 30.0
MAX_SECRET = 4000
MAX_ACTOR = 200


@dataclass(frozen=True)
class Stored:
    value: Any
    updated_by: str | None = None
    updated_at: str | None = None


class SettingsInvalid(Exception):
    def __init__(self, issues: list[str]):
        super().__init__("the settings are not valid")
        self.issues = issues


class UnknownGroup(Exception):
    pass


class SettingsStore(Protocol):
    async def ensure_schema(self) -> None: ...
    async def load(self, fresh: bool = False) -> dict[str, Stored]: ...
    async def write(self, values: dict[str, Any], clear: list[str], actor: str | None) -> dict[str, Stored]: ...


@dataclass
class MemorySettingsStore:
    """Tests and the in-memory mode without a database."""

    rows: dict[str, Stored] | None = None

    async def ensure_schema(self) -> None:
        return None

    async def load(self, fresh: bool = False) -> dict[str, Stored]:
        return dict(self.rows or {})

    async def write(self, values: dict[str, Any], clear: list[str], actor: str | None) -> dict[str, Stored]:
        rows = dict(self.rows or {})
        for k in clear:
            rows.pop(k, None)
        for k, v in values.items():
            rows[k] = Stored(v, actor, None)
        self.rows = rows
        return dict(rows)


class PgSettingsStore:
    def __init__(self, pool: Any, ttl_s: float = REFRESH_SECONDS):
        self.pool, self.ttl = pool, ttl_s
        self._at = 0.0
        self._value: dict[str, Stored] | None = None

    async def ensure_schema(self) -> None:
        async with self.pool.connection() as c:
            await c.execute(DDL)

    async def load(self, fresh: bool = False) -> dict[str, Stored]:
        now = time.monotonic()
        if not fresh and self._value is not None and now - self._at < self.ttl:
            return dict(self._value)
        try:
            async with self.pool.connection() as c:
                cur = await c.execute(f"select key, value, updated_by, updated_at from {TABLE}")  # noqa: S608 - constant name
                self._value = {
                    r["key"]: Stored(r["value"], r["updated_by"], r["updated_at"].isoformat() if r["updated_at"] else None) for r in await cur.fetchall()
                }
        except Exception:
            log.warning("could not read the admin settings; keeping the last known ones", exc_info=True)
            if self._value is None:
                return {}  # never read successfully: the deployment's values, and try again on the next call
        self._at = now
        return dict(self._value)

    async def write(self, values: dict[str, Any], clear: list[str], actor: str | None) -> dict[str, Stored]:
        """Every change of one save in one transaction."""
        from psycopg.types.json import Jsonb

        async with self.pool.connection() as c, c.transaction():
            for k in clear:
                await c.execute(f"delete from {TABLE} where key = %s", (k,))  # noqa: S608 - constant name
            for k, v in values.items():
                await c.execute(
                    f"insert into {TABLE} (key, value, updated_by) values (%s, %s, %s) "  # noqa: S608 - constant name
                    "on conflict (key) do update set value = excluded.value, updated_by = excluded.updated_by, updated_at = now()",
                    (k, Jsonb(v), actor),
                )
        return await self.load(fresh=True)


# --- Precedence -------------------------------------------------------------------------------------------------------


def group_of(group_id: str) -> Group:
    g = next((g for g in GROUPS if g.id == group_id), None)
    if g is None:
        raise UnknownGroup(group_id)
    return g


def all_fields() -> list[Field]:
    return [f for g in GROUPS for f in g.fields]


def overlay(env: Mapping[str, str], stored: Mapping[str, Stored]) -> dict[str, str]:
    """The environment with the admin's saved values laid over it. A stored value that no longer fits its field (the
    registry changed) or a secret that cannot be unsealed is skipped, so the deployment's value applies."""
    out = dict(env)
    for f in all_fields():
        row = stored.get(f.env)
        if row is None:
            continue
        try:
            out[f.env] = secrets.unseal(row.value, env) if f.secret else to_env(f, clean(f, row.value))
        except (Invalid, secrets.NoKey, secrets.Unreadable):
            log.warning("a saved setting is not usable; the deployment's value applies", extra={"event": "admin_setting_skipped", "setting": f.env})
    return out


def effective(stored: Mapping[str, Stored], env: Mapping[str, str] | None = None) -> Settings:
    return load_settings(overlay(os.environ if env is None else env, stored))


def apply_live(running: Settings, stored: Mapping[str, Stored], env: Mapping[str, str] | None = None) -> list[str]:
    """Copies the effective value of every live field onto the shared Settings object and returns the changed names.

    Settings is a frozen dataclass that crews, flows and the container hold by reference; frozen keeps code from
    assigning to it by accident. This is the one place that changes it, one attribute at a time (each assignment is
    atomic), and only for fields the registry marks live.
    """
    want = effective(stored, env)
    changed = []
    for f in all_fields():
        if f.apply != "live":
            continue
        new = getattr(want, f.attr)
        if getattr(running, f.attr) != new:
            object.__setattr__(running, f.attr, new)
            changed.append(f.env)
    return changed


# --- GET /admin-settings ------------------------------------------------------------------------------------------------


def _default(attr: str) -> Any:
    f = next(f for f in fields(Settings) if f.name == attr)
    if f.default is not MISSING:
        return f.default
    return f.default_factory() if f.default_factory is not MISSING else None


def overview(stored: Mapping[str, Stored], running: Settings, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    want, deployed = effective(stored, env), load_settings(env)
    can_seal = secrets.available(env)
    groups = []
    for g in GROUPS:
        rows = []
        for f in g.fields:
            row, in_env = stored.get(f.env), bool(env.get(f.env, "").strip())
            source = "admin" if row is not None else "deployment" if in_env else "default"
            out: dict[str, Any] = {
                "env": f.env,
                "label": f.label,
                "hint": f.hint,
                "type": f.type,
                "apply": f.apply,
                "unit": f.unit,
                "min": f.min,
                "max": f.max,
                "options": list(f.options),
                "source": source,
                "updated_by": row.updated_by if row else None,
                "updated_at": row.updated_at if row else None,
            }
            if f.secret:
                out |= {"set": row is not None, "deployment_set": in_env, "editable": can_seal, "restart_pending": False}
            else:
                out |= {
                    "value": shown(getattr(want, f.attr)),
                    "default": shown(_default(f.attr)),
                    "deployment": shown(getattr(deployed, f.attr)) if in_env else None,
                    "restart_pending": f.apply == "restart" and getattr(want, f.attr) != getattr(running, f.attr),
                }
            rows.append(out)
        groups.append({"id": g.id, "title": g.title, "description": g.description, "fields": rows, "restart_pending": any(r["restart_pending"] for r in rows)})
    return {"groups": groups, "secrets_key_set": can_seal, "refresh_seconds": REFRESH_SECONDS, "restart_pending": any(g["restart_pending"] for g in groups)}


# --- PUT /admin-settings/{group} ----------------------------------------------------------------------------------------


def plan_save(group: Group, values: Any, env: Mapping[str, str] | None = None) -> tuple[dict[str, Any], list[str]]:
    """Validates one group's values. Returns (values to store, keys to clear) or raises SettingsInvalid: nothing is
    stored unless every field is acceptable. A field that is not sent is left as it is; null, "" or an empty list
    clears it (the deployment's value, then the default, applies)."""
    if not isinstance(values, dict):
        raise SettingsInvalid(["values: send an object of setting name to value"])
    issues: list[str] = []
    keep: dict[str, Any] = {}
    clear: list[str] = []
    for key, value in values.items():
        f = group.field(key) if isinstance(key, str) else None
        if f is None:
            issues.append(f"{str(key)[:60]}: not a setting of {group.title}")
            continue
        if value is None or value == "" or value == []:
            clear.append(f.env)
            continue
        try:
            if f.secret:
                if not isinstance(value, str) or len(value) > MAX_SECRET or any(ord(ch) < 32 for ch in value):
                    raise Invalid(f"send the secret as text (at most {MAX_SECRET} characters)")
                keep[f.env] = secrets.seal(value.strip(), env)
            else:
                keep[f.env] = clean(f, value)
        except Invalid as e:
            issues.append(f"{f.label}: {e}")
        except secrets.NoKey:
            issues.append(f"{f.label}: Requires manual configuration: set {secrets.KEY_ENV} in the news agent's Secret (news-agent-env) and restart it")
    if issues:
        raise SettingsInvalid(issues[:30])
    return keep, clear


async def save_group(store: SettingsStore, running: Settings, group_id: str, values: Any, actor: Any, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    group = group_of(group_id)
    keep, clear = plan_save(group, values, env)
    who = actor.strip()[:MAX_ACTOR] if isinstance(actor, str) and actor.strip() else None
    stored = await store.write(keep, clear, who)
    changed = apply_live(running, stored, env)
    # Names only: a value could be a secret, and the web app's audit log records who saved what.
    log.info(
        "admin settings saved",
        extra={"event": "admin_settings_saved", "group": group.id, "saved": sorted(keep), "cleared": sorted(clear), "applied_now": changed},
    )
    return overview(stored, running, env)
