"""PgNewsStore against a real PostgreSQL using the web app's actual migration 0017 (not a copy): unique constraints
(item per source, slot per job, one active draft per story per job, one active model), counts, spend, retention, and
the rollback script. Needs NEWS_AGENT_TEST_DATABASE_URL (or testcontainers); skipped otherwise. No pgvector needed."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.news.store import DuplicateArticle, PgNewsStore
from tests.moks_cms import www

pytestmark = pytest.mark.postgres
BASE = """
create table category (slug text primary key, label text not null);
create table setting (key text primary key, value jsonb not null, updated_at timestamptz not null default now(), updated_by text);
create table post (id uuid primary key, collection text not null default 'news', slug text not null default 's', title text not null default 't', body text not null default '', status text not null default 'draft', deleted_at timestamptz);
create table schema_migrations (name text primary key);
"""


def statements(path: Path) -> list[str]:
    return [s.strip() for s in path.read_text().split("--> statement-breakpoint") if s.strip()]


@pytest.fixture
async def store() -> AsyncIterator[PgNewsStore]:
    url = os.environ.get("NEWS_AGENT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set NEWS_AGENT_TEST_DATABASE_URL to run the news Postgres tests")
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    schema = f"n_{uuid.uuid4().hex[:10]}"
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as c:
        await c.execute(f"create schema {schema}")
        await c.execute(f"set search_path to {schema}")
        for s in [*BASE.split(";"), *statements(www("src/db/migrations/0017_news_desk.sql"))]:
            if s.strip():
                await c.execute(s)
    pool = AsyncConnectionPool(url, min_size=1, max_size=3, open=False, kwargs={"autocommit": True, "row_factory": dict_row, "options": f"-c search_path={schema}"})
    await pool.open()
    try:
        yield PgNewsStore(pool)
    finally:
        await pool.close()
        async with await psycopg.AsyncConnection.connect(url, autocommit=True) as c:
            await c.execute(f"drop schema {schema} cascade")


async def seed(s: PgNewsStore) -> tuple[str, str]:
    job, src = str(uuid.uuid4()), str(uuid.uuid4())
    async with s.pool.connection() as c:
        await c.execute("insert into news_jobs (id, name, config) values (%s, 'Job', '{}'::jsonb)", (job,))
        await c.execute("insert into news_sources (id, name, type, url, enabled) values (%s, 'Src', 'rss', 'https://e.org/f', true)", (src,))
        await c.execute("insert into setting (key, value) values ('news', '{\"paused\": false}')")
        await c.execute("insert into news_writing_profiles (key, version, name) values ('news-default', 1, 'Default')")
    return job, src


def item(src: str, n: int) -> dict:
    return {"source_id": src, "url": f"https://e.org/{n}", "url_hash": f"h{n}", "title": f"T{n}", "published_at": datetime.now(UTC), "excerpt": "x", "clean_text": "x", "content_hash": f"c{n}", "simhash": -5, "minhash": [1, 2, 3], "embedding": [0.1, 0.2], "entities": {"product": ["Kubernetes"]}, "language": "en", "fetch_run_id": None}


async def test_contract(store: PgNewsStore) -> None:
    job, src = await seed(store)
    assert (await store.settings_row()) == {"paused": False}
    assert (await store.get_job(job))["name"] == "Job" and (await store.writing_profile("news-default"))["version"] == 1
    a = await store.insert_item(item(src, 1))
    assert a and await store.insert_item(item(src, 1)) is None  # (source, url) unique
    cid = str(uuid.uuid4())
    now = datetime.now(UTC)
    row = {"id": cid, "title": "T", "centroid": [0.1, 0.2], "entities": {"set": ["product:kubernetes"]}, "cve_ids": [], "severity": None, "first_seen": now, "last_seen": now, "item_count": 1, "source_count": 1, "burst": 0.3, "status": "new", "topics": [{"slug": "kubernetes", "p": 0.8}], "site_match": None}
    await store.save_cluster(row)
    await store.update_item(a, cluster_id=cid)
    await store.save_cluster({**row, "status": "drafted"})
    await store.save_cluster({**row, "status": "new"})
    assert (await store.get_cluster(cid))["status"] == "drafted"  # drafted is sticky
    assert list((await store.items_by_cluster([cid]))[cid][0].keys()) and (await store.recent_items(now - timedelta(hours=1)))[0]["minhash"] == [1, 2, 3]

    rid = str(uuid.uuid4())
    _r, created = await store.create_run({"id": rid, "job_id": job, "kind": "job", "trigger": "cron", "slot": "S1"})
    assert created
    again, created2 = await store.create_run({"id": str(uuid.uuid4()), "job_id": job, "kind": "job", "trigger": "cron", "slot": "S1"})
    assert not created2 and again["id"] == rid  # one run per slot
    await store.add_run_cost(rid, usd=0.25, tokens=1000, image_usd=0.04)
    assert (await store.spend(job, now - timedelta(days=1))) == {"llm": 0.25, "image": 0.04}
    step = await store.start_step(rid, "fetch", 1, 3)
    await store.finish_step(step, "succeeded", output_count=2, detail={"x": 1})
    assert (await store.steps(rid))[0]["output_count"] == 2
    await store.add_logs([{"run_id": rid, "step": "fetch", "level": "info", "message": "m", "context": {"a": 1}, "at": now}])
    assert (await store.logs(rid))[0]["context"] == {"a": 1}

    await store.save_candidates(rid, job, [{"cluster_id": cid, "features": {"recency": 1}, "score": 0.7, "breakdown": {"model": "v1"}, "model_version": "v1", "decision": "selected", "reason": "ok", "rank": 1}])
    aid = await store.create_article({"job_id": job, "run_id": rid, "cluster_id": cid, "status": "drafting"})
    with pytest.raises(DuplicateArticle):
        await store.create_article({"job_id": job, "run_id": rid, "cluster_id": cid, "status": "drafting"})
    await store.update_article(aid, status="ready_for_review", quality={"passed": True}, generated_doc={"title": "x"})
    assert cid in await store.covered_clusters(job, now - timedelta(days=7))
    counts = await store.counts_since(job, now - timedelta(days=1))
    assert counts.job_today == 1 and counts.per_topic == {"kubernetes": 1} and counts.per_source == {src: 1}
    assert (await store.open_article_for_cluster(cid, exclude_job=str(uuid.uuid4())))["id"] == aid
    iid = await store.add_image({"article_id": aid, "provider": "template", "safety": {"safe": True}})
    await store.update_image(iid, selected=True, alt="alt")
    assert (await store.images(aid))[0]["selected"]

    async with store.pool.connection() as c:
        await c.execute("update news_articles set review_outcome = 'approved' where id = %s", (aid,))
    tr = (await store.training_rows())[0]
    assert tr["features"] == {"recency": 1} and tr["job_id"] == job
    rc = await store.recent_candidates(10)
    assert rc[0]["run_id"] == rid and rc[0]["job_id"] == job and rc[0]["features"] == {"recency": 1}
    await store.add_labels([{"item_id": a, "topics": ["kubernetes"], "source": "human"}])
    lr = (await store.label_rows())[0]
    assert lr["item_id"] == a and lr["cluster_id"] == cid and lr["source"] == "human"

    await store.save_model("ranker", "v1", metrics={"auc": 0.7}, n_samples=10, artifact=b"x", active=True)
    await store.save_model("ranker", "v2", metrics={"auc": 0.8}, n_samples=10, artifact=b"y", active=True)
    assert (await store.active_model("ranker"))["version"] == "v2"  # exactly one active per kind
    await store.save_model("ranker", "v3", metrics={}, n_samples=10, artifact=None, active=False)
    assert not await store.activate_model("ranker", "v3")  # no artifact: cannot serve
    assert await store.activate_model("ranker", "v1")  # rollback
    assert (await store.active_model("ranker"))["version"] == "v1"
    models = await store.list_models("ranker")
    assert {m["version"]: m["active"] for m in models} == {"v1": True, "v2": False, "v3": False} and not next(m for m in models if m["version"] == "v3")["has_artifact"]

    prop = await store.add_profile_version("writing", {"key": "news-default", "name": "Default", "system_style": "s", "structure_template": "", "tone": "", "audience": "", "banned_phrases": [], "required_sections": [], "examples": [], "explanation": "e", "created_by": "flow"})
    assert prop["version"] == 2 and prop["status"] == "proposed" and (await store.writing_profile("news-default"))["version"] == 1

    old = now - timedelta(days=200)
    async with store.pool.connection() as c:
        await c.execute("update news_run_logs set at = %s", (old,))
    assert (await store.prune(90, 180, now))["logs"] == 1


async def test_rollback_then_up_again(store: PgNewsStore) -> None:
    async with store.pool.connection() as c:
        for s in statements(www("src/db/rollback/0017_news_desk.down.sql")):
            await c.execute(s)
        cur = await c.execute("select count(*)::int as n from information_schema.tables where table_schema = current_schema() and table_name like 'news_%%'")
        assert (await cur.fetchone())["n"] == 0
        for s in statements(www("src/db/migrations/0017_news_desk.sql")):
            await c.execute(s)
        cur = await c.execute("select count(*)::int as n from information_schema.tables where table_schema = current_schema() and table_name like 'news_%%'")
        assert (await cur.fetchone())["n"] == 17
