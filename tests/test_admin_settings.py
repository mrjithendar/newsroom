"""The news agent's admin-editable settings (app/admin_settings, forked from the seo-agent's): its own table and key,
the Claude key sealed, nothing that could lock the service out, and the API the web app's Newsroom → Settings uses."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.admin_settings import GROUPS, TABLE, MemorySettingsStore, SettingsInvalid, overview, save_group
from app.admin_settings import secrets as sec
from app.admin_settings.registry import NEVER, never
from app.admin_settings.store import effective
from app.config import load_settings
from app.main import create_app, memory_container
from tests.conftest import settings

KEY = {"NEWS_AGENT_SETTINGS_KEY": "k" * 40}
AUTH = {"Authorization": "Bearer " + "t" * 40}


def test_own_table_and_key_and_nothing_that_locks_the_service_out() -> None:
    assert TABLE == "news_agent_settings" and sec.KEY_ENV == "NEWS_AGENT_SETTINGS_KEY"
    for env in ("NEWS_AGENT_TOKEN", "NEWS_AGENT_CALLBACK_TOKEN", "NEWS_AGENT_SETTINGS_KEY", "SEO_AGENT_NEWS_TOKEN", "SEO_AGENT_URL", "DATABASE_URL", "NEWS_AGENT_DATABASE_URL", "NEWS_DESK_ENABLED", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        assert never(env), env
    registered = {f.env for g in GROUPS for f in g.fields}
    assert not registered & set(NEVER)
    assert {g.id for g in GROUPS} == {"news-agent-llm", "news-agent-limits", "embeddings"}
    attrs = set(load_settings({}).__dataclass_fields__)
    assert all(f.attr in attrs for g in GROUPS for f in g.fields)


async def test_the_claude_key_is_sealed_never_returned_and_wins_over_the_environment() -> None:
    store = MemorySettingsStore()
    running = load_settings({**KEY, "ANTHROPIC_API_KEY": "sk-ant-from-secret"})
    view = await save_group(store, running, "news-agent-llm", {"ANTHROPIC_API_KEY": "sk-ant-saved-here"}, "admin@example.com", env={**KEY, "ANTHROPIC_API_KEY": "sk-ant-from-secret"})
    row = (await store.load())["ANTHROPIC_API_KEY"]
    assert sec.is_sealed(row.value) and "sk-ant-saved-here" not in str(row.value)
    field = next(f for g in view["groups"] for f in g["fields"] if f["env"] == "ANTHROPIC_API_KEY")
    assert field["set"] and "value" not in field and "sk-ant" not in str(view)
    assert effective(await store.load(), {**KEY, "ANTHROPIC_API_KEY": "sk-ant-from-secret"}).anthropic_api_key == "sk-ant-saved-here"
    # Without the key, a secret cannot be saved (and the API says what to set).
    with pytest.raises(SettingsInvalid) as e:
        await save_group(MemorySettingsStore(), load_settings({}), "news-agent-llm", {"ANTHROPIC_API_KEY": "sk-ant-x"}, "a", env={})
    assert "NEWS_AGENT_SETTINGS_KEY" in e.value.issues[0]


async def test_limits_apply_live_and_bad_values_are_refused() -> None:
    store = MemorySettingsStore()
    running = load_settings({})
    await save_group(store, running, "news-agent-limits", {"CREW_MAX_ITER": 7}, "a", env={})
    assert running.crew_max_iter == 7  # live: the shared Settings object changed
    with pytest.raises(SettingsInvalid):
        await save_group(store, running, "news-agent-limits", {"CREW_MAX_ITER": 50}, "a", env={})
    with pytest.raises(SettingsInvalid):
        await save_group(store, running, "news-agent-llm", {"NEWS_DRAFT_MODEL": "gpt-5"}, "a", env={})
    view = overview(await store.load(), running, env={})
    f = next(f for g in view["groups"] for f in g["fields"] if f["env"] == "CREW_MAX_ITER")
    assert f["source"] == "admin" and f["value"] == 7


def test_the_api(llm: Any) -> None:
    from tests.news_fixtures import NewsWeb

    c = memory_container(settings(), llm=llm, web=NewsWeb())
    with TestClient(create_app(c)) as tc:
        assert tc.get("/admin-settings").status_code == 401
        view = tc.get("/admin-settings", headers=AUTH).json()
        assert [g["id"] for g in view["groups"]] == ["news-agent-llm", "news-agent-limits", "embeddings"]
        ok = tc.put("/admin-settings/news-agent-limits", json={"values": {"CREW_MAX_RPM": 30}, "actor": "admin"}, headers=AUTH)
        assert ok.status_code == 200 and c.settings.crew_max_rpm == 30
        assert tc.put("/admin-settings/news-agent-limits", json={"values": {"CREW_MAX_RPM": 0}}, headers=AUTH).status_code == 422
        assert tc.put("/admin-settings/nope", json={"values": {}}, headers=AUTH).status_code == 404
