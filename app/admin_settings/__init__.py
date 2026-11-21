"""Settings an admin can change from the web app (Admin → Newsroom → Settings → News agent) instead of the
deployment's environment. See store.py for precedence and the API, registry.py for the field model, groups/ for the
fields.

Forked from moks-seo-agent/app/admin_settings@6d6464b on 2026-11-21: the same store, registry and sealing, with this
service's own table (news_agent_settings), key (NEWS_AGENT_SETTINGS_KEY) and groups."""

from __future__ import annotations

from .groups import GROUPS
from .store import (
    REFRESH_SECONDS,
    TABLE,
    MemorySettingsStore,
    PgSettingsStore,
    SettingsInvalid,
    Stored,
    UnknownGroup,
    apply_live,
    effective,
    overview,
    save_group,
)

__all__ = [
    "GROUPS",
    "REFRESH_SECONDS",
    "TABLE",
    "MemorySettingsStore",
    "PgSettingsStore",
    "SettingsInvalid",
    "Stored",
    "UnknownGroup",
    "apply_live",
    "effective",
    "overview",
    "save_group",
]
