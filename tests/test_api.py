"""The internal API (app/main.py): what the web app's CrewAI console and Newsroom call."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app, memory_container
from tests.conftest import settings
from tests.news_fixtures import JOB, NewsWeb, seed

AUTH = {"Authorization": "Bearer " + "t" * 40}


def client(llm: Any) -> tuple[TestClient, Any]:
    c = memory_container(settings(), llm=llm, web=NewsWeb())
    seed(c.deps.extra["news_store"])
    return TestClient(create_app(c)), c


def test_health_is_open_and_says_which_agent(llm: Any) -> None:
    tc, _ = client(llm)
    with tc:
        assert tc.get("/healthz").json()["agent"] == "news"
        assert tc.get("/flows").status_code == 401
        assert tc.get("/flows", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_catalogues_tag_every_entry_with_the_agent(llm: Any) -> None:
    tc, _ = client(llm)
    with tc:
        flows = tc.get("/flows", headers=AUTH).json()["items"]
        crews = tc.get("/crews", headers=AUTH).json()["items"]
    assert {f["name"] for f in flows} >= {"news_job_flow", "news_dispatch_flow", "crew_test_flow"} and all(f["agent"] == "news" for f in flows)
    assert not any(f["approval_only"] or f["mcp"] for f in flows)
    assert {f["name"] for f in flows if f["schedulable"]} == {"news_dispatch_flow", "news_train_flow", "news_style_review_flow"}
    assert [c["crew"] for c in crews] == ["news"] and crews[0]["agent"] == "news"


def test_kickoff_validate_and_unknown_flows(llm: Any) -> None:
    tc, _ = client(llm)
    with tc:
        assert tc.post("/flows/news_job_flow/validate", json={"inputs": {"job_id": JOB}}, headers=AUTH).json()["inputs"]["job_id"] == JOB
        assert tc.post("/flows/news_job_flow/validate", json={"inputs": {"job_id": "x"}}, headers=AUTH).status_code == 422
        assert tc.post("/flows/publish_flow/kickoff", json={}, headers=AUTH).status_code == 404
        assert tc.post("/flows/gsc_sync_flow/kickoff", json={}, headers=AUTH).status_code == 404
        r = tc.post("/flows/news_ingest_flow/kickoff", json={"inputs": {}}, headers=AUTH)
        assert r.status_code == 202
        run = tc.get(f"/flows/runs/{r.json()['run_id']}", headers=AUTH).json()
        assert run["flow"] == "news_ingest_flow"
        assert tc.post("/crews/editorial/validate", json={"config": {}}, headers=AUTH).status_code == 404


def test_settings_come_from_the_news_agents_own_environment() -> None:
    s = load_settings({"NEWS_AGENT_TOKEN": "a", "NEWS_AGENT_CALLBACK_TOKEN": "b", "NEWS_AGENT_CALLBACK_URL": "http://web:9465/", "SEO_AGENT_URL": "http://seo-agent:8080/", "SEO_AGENT_NEWS_TOKEN": " n ", "SEO_AGENT_TOKEN": "never-read", "NEWS_CREW_MODELS": '{"news.analyst": "claude-haiku-5-5"}'})
    assert (s.agent_token, s.callback_token, s.callback_url, s.seo_agent_url, s.seo_agent_token) == ("a", "b", "http://web:9465", "http://seo-agent:8080", "n")
    assert s.crew_models == {"news.analyst": "claude-haiku-5-5"} and s.service_name == "news-agent"
