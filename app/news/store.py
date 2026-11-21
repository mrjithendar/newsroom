"""Persistence for the news desk: PostgreSQL (the web app's migration 0017 creates the tables) and an in-memory twin for
the flow tests. Both implement `NewsStore`; tests/test_news_postgres.py runs the shared contract against a real
PostgreSQL when one is available (SEO_AGENT_TEST_DATABASE_URL or a local server), like tests/test_postgres.py.

The agent reads config tables (jobs, sources, profiles, topics, the `news` setting) and writes run/ML/draft-tracking
tables. It never writes `post` or `seo_drafts`: drafts go through the web app's callback API.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from .select import Counts

ACTIVE_ARTICLE = ("drafting", "ready_for_review")
COUNTED_ARTICLE = ("drafting", "ready_for_review", "published", "incomplete")


class DuplicateArticle(Exception):
    pass


def _j(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _row(r: Any) -> dict[str, Any]:
    d = dict(r)
    for k, v in d.items():
        if isinstance(v, uuid.UUID):
            d[k] = str(v)
        elif isinstance(v, list) and v and isinstance(v[0], uuid.UUID):
            d[k] = [str(x) for x in v]
    return d


class NewsStore(Protocol):
    async def settings_row(self) -> dict[str, Any] | None: ...
    async def get_job(self, job_id: str) -> dict[str, Any] | None: ...
    async def list_jobs(self, enabled_only: bool = True) -> list[dict[str, Any]]: ...
    async def set_job_schedule(self, job_id: str, *, last_run_at: datetime | None = None, next_run_at: datetime | None = None) -> None: ...
    async def writing_profile(self, key: str, version: int | None = None) -> dict[str, Any] | None: ...
    async def image_profile(self, key: str, version: int | None = None) -> dict[str, Any] | None: ...
    async def add_profile_version(self, kind: str, row: dict[str, Any]) -> dict[str, Any]: ...
    async def sources(self, ids: list[str] | None = None, enabled_only: bool = True) -> list[dict[str, Any]]: ...
    async def update_source(self, source_id: str, **fields: Any) -> None: ...
    async def topics(self) -> list[dict[str, Any]]: ...
    async def recent_items(self, since: datetime) -> list[dict[str, Any]]: ...
    async def insert_item(self, row: dict[str, Any]) -> str | None: ...
    async def update_item(self, item_id: str, **fields: Any) -> None: ...
    async def items_by_cluster(self, cluster_ids: list[str]) -> dict[str, list[dict[str, Any]]]: ...
    async def recent_clusters(self, since: datetime) -> list[dict[str, Any]]: ...
    async def get_cluster(self, cluster_id: str) -> dict[str, Any] | None: ...
    async def save_cluster(self, row: dict[str, Any]) -> None: ...
    async def create_run(self, row: dict[str, Any]) -> tuple[dict[str, Any], bool]: ...
    async def update_run(self, run_id: str, **fields: Any) -> None: ...
    async def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    async def add_run_cost(self, run_id: str, *, usd: float = 0.0, tokens: int = 0, image_usd: float = 0.0) -> None: ...
    async def start_step(self, run_id: str, step: str, seq: int, input_count: int | None = None) -> int: ...
    async def finish_step(self, step_id: int, status: str, *, output_count: int | None = None, error: str | None = None, detail: dict[str, Any] | None = None) -> None: ...
    async def steps(self, run_id: str) -> list[dict[str, Any]]: ...
    async def add_logs(self, rows: list[dict[str, Any]]) -> None: ...
    async def logs(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]: ...
    async def save_candidates(self, run_id: str, job_id: str | None, rows: list[dict[str, Any]]) -> None: ...
    async def candidates(self, run_id: str) -> list[dict[str, Any]]: ...
    async def covered_clusters(self, job_id: str, since: datetime) -> set[str]: ...
    async def counts_since(self, job_id: str, since: datetime) -> Counts: ...
    async def create_article(self, row: dict[str, Any]) -> str: ...
    async def update_article(self, article_id: str, **fields: Any) -> None: ...
    async def get_article(self, article_id: str) -> dict[str, Any] | None: ...
    async def open_article_for_cluster(self, cluster_id: str, exclude_job: str | None = None) -> dict[str, Any] | None: ...
    async def add_image(self, row: dict[str, Any]) -> str: ...
    async def update_image(self, image_id: str, **fields: Any) -> None: ...
    async def images(self, article_id: str) -> list[dict[str, Any]]: ...
    async def spend(self, job_id: str | None, since: datetime) -> dict[str, float]: ...
    async def training_rows(self) -> list[dict[str, Any]]: ...
    async def label_rows(self) -> list[dict[str, Any]]: ...
    async def unlabeled_items(self, limit: int) -> list[dict[str, Any]]: ...
    async def add_labels(self, rows: list[dict[str, Any]]) -> None: ...
    async def save_model(self, kind: str, version: str, *, metrics: dict[str, Any], n_samples: int, artifact: bytes | None, active: bool, notes: str | None = None) -> None: ...
    async def active_model(self, kind: str) -> dict[str, Any] | None: ...
    async def list_models(self, kind: str | None, limit: int = 20) -> list[dict[str, Any]]: ...
    async def activate_model(self, kind: str, version: str) -> bool: ...
    async def recent_candidates(self, limit: int) -> list[dict[str, Any]]: ...
    async def edited_articles(self, since: datetime, limit: int) -> list[dict[str, Any]]: ...
    async def add_feedback(self, row: dict[str, Any]) -> None: ...
    async def prune(self, log_days: int, item_days: int, now: datetime) -> dict[str, int]: ...
    # read models for the admin/public MCP tools
    async def list_runs(self, job_id: str | None, limit: int) -> list[dict[str, Any]]: ...
    async def list_articles(self, status: str | None, limit: int) -> list[dict[str, Any]]: ...
    async def mark_rejected(self, draft_id: str, reason: str, actor: str) -> dict[str, Any] | None: ...
    async def published_news(self, query: str, since: datetime | None, limit: int) -> list[dict[str, Any]]: ...


# --- PostgreSQL -----------------------------------------------------------------------------------------------------

_JOB_COLS = "id, name, description, enabled, cron, timezone, mode, writing_profile_key, image_profile_key, review_priority, config, version, next_run_at, last_run_at"
_RUN_UPDATABLE = {"status", "finished_at", "counters", "tokens", "cost_usd", "image_cost_usd", "error_summary", "flow_run_id", "job_version", "llm_enabled"}
_ARTICLE_UPDATABLE = {"status", "draft_id", "post_id", "sources", "fact_sheet", "fact_check", "similarity", "quality", "generated_doc", "tokens", "cost_usd", "score", "writing_profile_version", "image_profile_version", "published_at", "expires_after_days"}
_ARTICLE_JSON = {"sources", "fact_sheet", "fact_check", "similarity", "quality", "generated_doc"}


class PgNewsStore:
    def __init__(self, pool: Any):
        self.pool = pool

    async def _all(self, q: str, args: Any = None) -> list[dict[str, Any]]:
        async with self.pool.connection() as c:
            cur = await c.execute(q, args)
            return [_row(r) for r in await cur.fetchall()]

    async def _one(self, q: str, args: Any = None) -> dict[str, Any] | None:
        rows = await self._all(q, args)
        return rows[0] if rows else None

    async def _exec(self, q: str, args: Any = None) -> None:
        async with self.pool.connection() as c:
            await c.execute(q, args)

    async def settings_row(self) -> dict[str, Any] | None:
        r = await self._one("select value from setting where key = 'news'")
        return _j(r["value"]) if r else None

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        return await self._one(f"select {_JOB_COLS} from news_jobs where id::text = %s and deleted_at is null", (job_id,))  # noqa: S608 - fixed column list

    async def list_jobs(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        return await self._all(f"select {_JOB_COLS} from news_jobs where deleted_at is null {'and enabled' if enabled_only else ''} order by name")  # noqa: S608

    async def set_job_schedule(self, job_id: str, *, last_run_at: datetime | None = None, next_run_at: datetime | None = None) -> None:
        await self._exec("update news_jobs set last_run_at = coalesce(%s, last_run_at), next_run_at = coalesce(%s, next_run_at) where id::text = %s", (last_run_at, next_run_at, job_id))

    async def _profile(self, table: str, key: str, version: int | None) -> dict[str, Any] | None:
        if version:
            return await self._one(f"select * from {table} where key = %s and version = %s", (key, version))  # noqa: S608 - table is one of two literals
        return await self._one(f"select * from {table} where key = %s and status = 'active'", (key,))  # noqa: S608

    async def writing_profile(self, key: str, version: int | None = None) -> dict[str, Any] | None:
        return await self._profile("news_writing_profiles", key, version)

    async def image_profile(self, key: str, version: int | None = None) -> dict[str, Any] | None:
        return await self._profile("news_image_profiles", key, version)

    async def add_profile_version(self, kind: str, row: dict[str, Any]) -> dict[str, Any]:
        if kind != "writing":
            raise ValueError("only writing-profile proposals are created by the agent")
        cols = ("key", "name", "system_style", "structure_template", "tone", "audience", "banned_phrases", "required_sections", "examples", "temperature", "model", "explanation", "created_by")
        vals = {c: row.get(c) for c in cols}
        vals["examples"] = json.dumps(vals.get("examples") or [])
        r = await self._one(
            f"""insert into news_writing_profiles (version, status, {", ".join(cols)})
                select coalesce(max(version), 0) + 1, 'proposed', {", ".join(f"%({c})s" + ("::jsonb" if c == "examples" else "") for c in cols)}
                from news_writing_profiles where key = %(key)s returning *""",  # noqa: S608 - fixed column list
            vals,
        )
        return r or {}

    async def sources(self, ids: list[str] | None = None, enabled_only: bool = True) -> list[dict[str, Any]]:
        where = ["deleted_at is null"]
        args: list[Any] = []
        if enabled_only:
            where.append("enabled")
        if ids:
            where.append("id::text = any(%s)")
            args.append(ids)
        return await self._all(f"select * from news_sources where {' and '.join(where)} order by name", args)  # noqa: S608

    async def update_source(self, source_id: str, **fields: Any) -> None:
        allowed = {"etag", "last_modified", "last_fetch_at", "last_success_at", "last_status", "last_http_status", "error_count", "consecutive_errors", "fetch_count", "avg_items", "items_total"}
        f = {k: v for k, v in fields.items() if k in allowed}
        if f:
            sets = ", ".join(f"{k} = %({k})s" for k in f)
            await self._exec(f"update news_sources set {sets}, updated_at = now() where id::text = %(id)s", {**f, "id": source_id})  # noqa: S608

    async def topics(self) -> list[dict[str, Any]]:
        return await self._all("select * from news_topics where enabled order by sort, slug")

    async def recent_items(self, since: datetime) -> list[dict[str, Any]]:
        return await self._all(
            "select id, source_id, url, url_hash, content_hash, minhash, embedding, cluster_id, title, published_at, fetched_at, entities, excerpt, language from news_items where fetched_at >= %s order by fetched_at",
            (since,),
        )

    async def insert_item(self, row: dict[str, Any]) -> str | None:
        r = await self._one(
            """insert into news_items (source_id, url, url_hash, title, published_at, excerpt, clean_text, content_hash, simhash, minhash, embedding, entities, language, fetch_run_id)
               values (%(source_id)s, %(url)s, %(url_hash)s, %(title)s, %(published_at)s, %(excerpt)s, %(clean_text)s, %(content_hash)s, %(simhash)s, %(minhash)s, %(embedding)s, %(entities)s::jsonb, %(language)s, %(fetch_run_id)s)
               on conflict (source_id, url_hash) do nothing returning id""",
            {**row, "entities": json.dumps(row.get("entities") or {})},
        )
        return r["id"] if r else None

    async def update_item(self, item_id: str, **fields: Any) -> None:
        f = {k: v for k, v in fields.items() if k in {"cluster_id", "duplicate_of", "dedupe_reason", "embedding"}}
        if f:
            await self._exec(f"update news_items set {', '.join(f'{k} = %({k})s' for k in f)} where id::text = %(id)s", {**f, "id": item_id})  # noqa: S608

    async def items_by_cluster(self, cluster_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not cluster_ids:
            return {}
        rows = await self._all(
            """select i.id, i.cluster_id, i.source_id, i.url, i.title, i.published_at, i.fetched_at, i.excerpt, i.clean_text, i.entities, i.language, i.duplicate_of, i.embedding,
                      s.name as source_name, s.credibility, s.url as source_url, s.type as source_type
               from news_items i join news_sources s on s.id = i.source_id where i.cluster_id::text = any(%s) order by i.fetched_at""",
            (cluster_ids,),
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            out.setdefault(r["cluster_id"], []).append(r)
        return out

    async def recent_clusters(self, since: datetime) -> list[dict[str, Any]]:
        return await self._all("select * from news_clusters where last_seen >= %s order by last_seen desc", (since,))

    async def get_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        return await self._one("select * from news_clusters where id::text = %s", (cluster_id,))

    async def save_cluster(self, row: dict[str, Any]) -> None:
        await self._exec(
            """insert into news_clusters (id, title, centroid, entities, cve_ids, severity, first_seen, last_seen, item_count, source_count, burst, status, topics, site_match, updated_at)
               values (%(id)s, %(title)s, %(centroid)s, %(entities)s::jsonb, %(cve_ids)s, %(severity)s, %(first_seen)s, %(last_seen)s, %(item_count)s, %(source_count)s, %(burst)s, %(status)s, %(topics)s::jsonb, %(site_match)s::jsonb, now())
               on conflict (id) do update set title = excluded.title, centroid = excluded.centroid, entities = excluded.entities, cve_ids = excluded.cve_ids,
                 severity = excluded.severity, first_seen = excluded.first_seen, last_seen = excluded.last_seen, item_count = excluded.item_count,
                 source_count = excluded.source_count, burst = excluded.burst,
                 status = case when news_clusters.status in ('drafted', 'covered') then news_clusters.status else excluded.status end,
                 topics = excluded.topics, site_match = coalesce(excluded.site_match, news_clusters.site_match), updated_at = now()""",
            {**row, "entities": json.dumps(row.get("entities") or {}), "topics": json.dumps(row.get("topics") or []), "site_match": json.dumps(row["site_match"]) if row.get("site_match") is not None else None},
        )

    async def create_run(self, row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        r = await self._one(
            """insert into news_runs (id, job_id, job_version, kind, flow_run_id, parent_run_id, trigger, slot, status, dry_run, llm_enabled, correlation_id, actor)
               values (%(id)s, %(job_id)s, %(job_version)s, %(kind)s, %(flow_run_id)s, %(parent_run_id)s, %(trigger)s, %(slot)s, %(status)s, %(dry_run)s, %(llm_enabled)s, %(correlation_id)s, %(actor)s)
               on conflict do nothing returning *""",
            {"job_version": None, "flow_run_id": None, "parent_run_id": None, "slot": None, "dry_run": False, "llm_enabled": True, "correlation_id": None, "actor": None, "status": "running", **row},
        )
        if r:
            return r, True
        existing = await self._one("select * from news_runs where job_id::text = %s and slot = %s", (row.get("job_id"), row.get("slot")))
        return existing or {}, False

    async def update_run(self, run_id: str, **fields: Any) -> None:
        f = {k: (json.dumps(v) if k == "counters" else v) for k, v in fields.items() if k in _RUN_UPDATABLE}
        if f:
            sets = ", ".join(f"{k} = %({k})s" + ("::jsonb" if k == "counters" else "") for k in f)
            await self._exec(f"update news_runs set {sets} where id::text = %(id)s", {**f, "id": run_id})  # noqa: S608

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        return await self._one("select * from news_runs where id::text = %s", (run_id,))

    async def add_run_cost(self, run_id: str, *, usd: float = 0.0, tokens: int = 0, image_usd: float = 0.0) -> None:
        await self._exec("update news_runs set cost_usd = cost_usd + %s, tokens = tokens + %s, image_cost_usd = image_cost_usd + %s where id::text = %s", (usd, tokens, image_usd, run_id))

    async def start_step(self, run_id: str, step: str, seq: int, input_count: int | None = None) -> int:
        r = await self._one("insert into news_run_steps (run_id, step, seq, input_count) values (%s, %s, %s, %s) returning id", (run_id, step, seq, input_count))
        return int(r["id"]) if r else 0

    async def finish_step(self, step_id: int, status: str, *, output_count: int | None = None, error: str | None = None, detail: dict[str, Any] | None = None) -> None:
        await self._exec("update news_run_steps set status = %s, finished_at = now(), output_count = %s, error = %s, detail = %s::jsonb where id = %s", (status, output_count, error, json.dumps(detail or {}, default=str), step_id))

    async def steps(self, run_id: str) -> list[dict[str, Any]]:
        return await self._all("select * from news_run_steps where run_id::text = %s order by seq, id", (run_id,))

    async def add_logs(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        async with self.pool.connection() as c, c.cursor() as cur:
            await cur.executemany(
                "insert into news_run_logs (run_id, step, level, message, context, at) values (%(run_id)s, %(step)s, %(level)s, %(message)s, %(context)s::jsonb, %(at)s)",
                [{**r, "context": json.dumps(r.get("context") or {}, default=str)} for r in rows],
            )

    async def logs(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return await self._all("select * from news_run_logs where run_id::text = %s and id > %s order by id limit 500", (run_id, after_id))

    async def save_candidates(self, run_id: str, job_id: str | None, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        async with self.pool.connection() as c, c.cursor() as cur:
            await cur.executemany(
                """insert into news_candidates (run_id, job_id, cluster_id, features, score, breakdown, model_version, decision, reason, rank)
                   values (%(run_id)s, %(job_id)s, %(cluster_id)s, %(features)s::jsonb, %(score)s, %(breakdown)s::jsonb, %(model_version)s, %(decision)s, %(reason)s, %(rank)s)
                   on conflict (run_id, cluster_id) do update set features = excluded.features, score = excluded.score, breakdown = excluded.breakdown,
                     model_version = excluded.model_version, decision = excluded.decision, reason = excluded.reason, rank = excluded.rank""",
                [{**r, "run_id": run_id, "job_id": job_id, "features": json.dumps(r["features"]), "breakdown": json.dumps(r["breakdown"], default=str)} for r in rows],
            )

    async def candidates(self, run_id: str) -> list[dict[str, Any]]:
        return await self._all("select * from news_candidates where run_id::text = %s order by score desc", (run_id,))

    async def recent_candidates(self, limit: int) -> list[dict[str, Any]]:
        return await self._all(
            "select run_id::text as run_id, job_id::text as job_id, features, score, model_version, decision, created_at from news_candidates order by created_at desc, id desc limit %s",
            (limit,),
        )

    async def covered_clusters(self, job_id: str, since: datetime) -> set[str]:
        rows = await self._all(
            """select cluster_id::text as c from news_articles where cluster_id is not null and created_at >= %(since)s
                 and status in ('drafting', 'ready_for_review', 'published', 'incomplete') and (job_id::text = %(job)s or status = 'published')
               union select unnest(cluster_ids)::text from news_articles where created_at >= %(since)s and status in ('ready_for_review', 'published') and job_id::text = %(job)s
               union select id::text from news_clusters where status in ('covered', 'drafted') and last_seen >= %(since)s""",
            {"since": since, "job": job_id},
        )
        return {r["c"] for r in rows if r["c"]}

    async def counts_since(self, job_id: str, since: datetime) -> Counts:
        rows = await self._all(
            "select job_id::text as job, count(*)::int as n from news_articles where created_at >= %s and status = any(%s) group by job_id",
            (since, list(COUNTED_ARTICLE)),
        )
        c = Counts(job_today=sum(r["n"] for r in rows if r["job"] == job_id), global_today=sum(r["n"] for r in rows))
        sel = await self._all(
            """select k.cluster_id::text as cluster_id, cl.topics from news_candidates k join news_runs r on r.id = k.run_id join news_clusters cl on cl.id = k.cluster_id
               where k.job_id::text = %s and k.decision = 'selected' and k.created_at >= %s and not r.dry_run""",
            (job_id, since),
        )
        ids = [s["cluster_id"] for s in sel]
        by = await self.items_by_cluster(ids) if ids else {}
        for s in sel:
            t = _j(s["topics"]) or []
            main = (t[0]["slug"] if t and isinstance(t[0], dict) else (t[0] if t else "uncategorised"))
            c.per_topic[main] = c.per_topic.get(main, 0) + 1
            for src in {i["source_id"] for i in by.get(s["cluster_id"], [])}:
                c.per_source[src] = c.per_source.get(src, 0) + 1
        return c

    async def create_article(self, row: dict[str, Any]) -> str:
        import psycopg

        cols = ("job_id", "job_version", "run_id", "cluster_id", "cluster_ids", "mode", "status", "writing_profile_key", "writing_profile_version", "image_profile_key", "image_profile_version", "score", "expires_after_days")
        vals = {c: row.get(c) for c in cols}
        vals["cluster_ids"] = row.get("cluster_ids") or []
        vals["status"] = row.get("status") or "drafting"
        vals["mode"] = row.get("mode") or "single"
        try:
            r = await self._one(f"insert into news_articles ({', '.join(cols)}) values ({', '.join(f'%({c})s' + ('::uuid[]' if c == 'cluster_ids' else '') for c in cols)}) returning id", vals)  # noqa: S608
        except psycopg.errors.UniqueViolation as e:
            raise DuplicateArticle("an active draft for this story already exists for this job") from e
        return str(r["id"]) if r else ""

    async def update_article(self, article_id: str, **fields: Any) -> None:
        f = {k: (json.dumps(v, default=str) if k in _ARTICLE_JSON else v) for k, v in fields.items() if k in _ARTICLE_UPDATABLE}
        if f:
            sets = ", ".join(f"{k} = %({k})s" + ("::jsonb" if k in _ARTICLE_JSON else "") for k in f)
            await self._exec(f"update news_articles set {sets}, updated_at = now() where id::text = %(id)s", {**f, "id": article_id})  # noqa: S608

    async def get_article(self, article_id: str) -> dict[str, Any] | None:
        return await self._one("select * from news_articles where id::text = %s", (article_id,))

    async def open_article_for_cluster(self, cluster_id: str, exclude_job: str | None = None) -> dict[str, Any] | None:
        return await self._one(
            "select id, job_id, status, draft_id, post_id from news_articles where cluster_id::text = %s and status in ('drafting', 'ready_for_review', 'published') and (%s::text is null or job_id::text <> %s or status = 'published') order by created_at desc limit 1",
            (cluster_id, exclude_job, exclude_job),
        )

    async def add_image(self, row: dict[str, Any]) -> str:
        cols = ("article_id", "variant", "provider", "model", "prompt", "negative_prompt", "seed", "width", "height", "sizes", "media_id", "src", "alt", "decorative", "ai_generated", "safety", "score", "selected", "status", "cost_usd")
        vals = {c: row.get(c) for c in cols}
        vals["sizes"] = json.dumps(row.get("sizes") or {})
        vals["safety"] = json.dumps(row.get("safety") or {})
        for k, d in (("variant", 1), ("decorative", False), ("ai_generated", False), ("selected", False), ("status", "ok"), ("cost_usd", 0.0)):
            vals[k] = d if vals.get(k) is None else vals[k]
        r = await self._one(f"insert into news_images ({', '.join(cols)}) values ({', '.join(f'%({c})s' + ('::jsonb' if c in ('sizes', 'safety') else '') for c in cols)}) returning id", vals)  # noqa: S608
        return str(r["id"]) if r else ""

    async def update_image(self, image_id: str, **fields: Any) -> None:
        f = {k: (json.dumps(v) if k in ("sizes", "safety") else v) for k, v in fields.items() if k in {"media_id", "src", "alt", "decorative", "selected", "status", "sizes", "safety", "score", "width", "height"}}
        if f:
            sets = ", ".join(f"{k} = %({k})s" + ("::jsonb" if k in ("sizes", "safety") else "") for k in f)
            await self._exec(f"update news_images set {sets} where id::text = %(id)s", {**f, "id": image_id})  # noqa: S608

    async def images(self, article_id: str) -> list[dict[str, Any]]:
        return await self._all("select * from news_images where article_id::text = %s order by created_at, variant", (article_id,))

    async def spend(self, job_id: str | None, since: datetime) -> dict[str, float]:
        r = await self._one(
            "select coalesce(sum(cost_usd), 0)::float8 as llm, coalesce(sum(image_cost_usd), 0)::float8 as image from news_runs where started_at >= %s and (%s::text is null or job_id::text = %s)",
            (since, job_id, job_id),
        )
        return {"llm": float(r["llm"]) if r else 0.0, "image": float(r["image"]) if r else 0.0}

    async def training_rows(self) -> list[dict[str, Any]]:
        return await self._all(
            """select k.run_id::text as run_id, k.job_id::text as job_id, k.created_at as at, k.features, a.review_outcome, a.reject_category,
                      (select max((f.payload->>'clicks')::float8) from news_feedback f where f.article_id = a.id and f.kind = 'performance') as clicks
               from news_articles a join news_candidates k on k.run_id = a.run_id and k.cluster_id = a.cluster_id
               where a.review_outcome is not null"""
        )

    async def label_rows(self) -> list[dict[str, Any]]:
        return await self._all(
            """select distinct on (coalesce(l.item_id, l.cluster_id)) l.topics, l.source, i.id::text as item_id, i.cluster_id::text as cluster_id, i.title, i.clean_text, i.excerpt, i.embedding, i.entities
               from news_labels l join news_items i on i.id = l.item_id order by coalesce(l.item_id, l.cluster_id), (l.source = 'human') desc, l.at desc"""
        )

    async def unlabeled_items(self, limit: int) -> list[dict[str, Any]]:
        return await self._all(
            "select i.id, i.title, i.excerpt from news_items i where i.duplicate_of is null and not exists (select 1 from news_labels l where l.item_id = i.id) order by i.fetched_at desc limit %s",
            (limit,),
        )

    async def add_labels(self, rows: list[dict[str, Any]]) -> None:
        async with self.pool.connection() as c, c.cursor() as cur:
            await cur.executemany("insert into news_labels (item_id, topics, source) values (%(item_id)s, %(topics)s, %(source)s) on conflict do nothing", rows)

    async def save_model(self, kind: str, version: str, *, metrics: dict[str, Any], n_samples: int, artifact: bytes | None, active: bool, notes: str | None = None) -> None:
        async with self.pool.connection() as c, c.transaction():
            if active:
                await c.execute("update news_models set active = false where kind = %s and active", (kind,))
            await c.execute(
                "insert into news_models (kind, version, active, metrics, n_samples, artifact, notes) values (%s, %s, %s, %s::jsonb, %s, %s, %s) on conflict (kind, version) do nothing",
                (kind, version, active, json.dumps(metrics, default=str), n_samples, artifact, notes),
            )

    async def active_model(self, kind: str) -> dict[str, Any] | None:
        return await self._one("select version, artifact, metrics, trained_at from news_models where kind = %s and active", (kind,))

    async def list_models(self, kind: str | None, limit: int = 20) -> list[dict[str, Any]]:
        return await self._all(
            "select kind, version, active, metrics, n_samples, notes, trained_at, artifact is not null as has_artifact from news_models where (%s::text is null or kind = %s) order by trained_at desc limit %s",
            (kind, kind, limit),
        )

    async def activate_model(self, kind: str, version: str) -> bool:
        """Rollback: make a stored version (with an artifact) the one active model of its kind, in one transaction."""
        async with self.pool.connection() as c, c.transaction():
            cur = await c.execute("select 1 from news_models where kind = %s and version = %s and artifact is not null", (kind, version))
            if await cur.fetchone() is None:
                return False
            await c.execute("update news_models set active = false where kind = %s and active", (kind,))
            await c.execute("update news_models set active = true where kind = %s and version = %s", (kind, version))
            return True

    async def edited_articles(self, since: datetime, limit: int) -> list[dict[str, Any]]:
        return await self._all(
            """select a.id, a.generated_doc, a.edit_distance, p.body as published_body, p.title as published_title
               from news_articles a join post p on p.id = a.post_id where a.status = 'published' and a.published_at >= %s and a.generated_doc is not null
               order by a.published_at desc limit %s""",
            (since, limit),
        )

    async def add_feedback(self, row: dict[str, Any]) -> None:
        await self._exec(
            "insert into news_feedback (article_id, cluster_id, job_id, kind, reason_category, reason, payload, actor) values (%(article_id)s, %(cluster_id)s, %(job_id)s, %(kind)s, %(reason_category)s, %(reason)s, %(payload)s::jsonb, %(actor)s) on conflict do nothing",
            {"cluster_id": None, "job_id": None, "reason_category": None, "reason": None, "actor": None, **row, "payload": json.dumps(row.get("payload") or {})},
        )

    async def prune(self, log_days: int, item_days: int, now: datetime) -> dict[str, int]:
        out = {}
        async with self.pool.connection() as c:
            cur = await c.execute("delete from news_run_logs where at < %s", (now - timedelta(days=log_days),))
            out["logs"] = cur.rowcount
            cur = await c.execute(
                "delete from news_items i where fetched_at < %s and not exists (select 1 from news_articles a where a.cluster_id = i.cluster_id) and not exists (select 1 from news_labels l where l.item_id = i.id)",
                (now - timedelta(days=item_days),),
            )
            out["items"] = cur.rowcount
            cur = await c.execute("delete from news_candidates k where created_at < %s and not exists (select 1 from news_articles a where a.run_id = k.run_id and a.cluster_id = k.cluster_id)", (now - timedelta(days=item_days),))
            out["candidates"] = cur.rowcount
        return out

    async def list_runs(self, job_id: str | None, limit: int) -> list[dict[str, Any]]:
        return await self._all(
            "select r.id, r.job_id, j.name as job_name, r.kind, r.trigger, r.status, r.dry_run, r.counters, r.cost_usd, r.image_cost_usd, r.error_summary, r.started_at, r.finished_at from news_runs r left join news_jobs j on j.id = r.job_id where (%s::text is null or r.job_id::text = %s) order by r.started_at desc limit %s",
            (job_id, job_id, max(1, min(limit, 50))),
        )

    async def list_articles(self, status: str | None, limit: int) -> list[dict[str, Any]]:
        return await self._all(
            "select a.id, a.job_id, j.name as job_name, a.status, a.mode, a.draft_id, a.score, a.cost_usd, a.created_at, a.generated_doc->>'title' as title, a.quality->'failed' as failed_gates from news_articles a left join news_jobs j on j.id = a.job_id where (%s::text is null or a.status = %s) order by a.created_at desc limit %s",
            (status, status, max(1, min(limit, 50))),
        )

    async def mark_rejected(self, draft_id: str, reason: str, actor: str) -> dict[str, Any] | None:
        a = await self._one("update news_articles set status = 'rejected', review_outcome = 'rejected', reject_category = 'other', reject_reason = %s, reviewed_at = now(), updated_at = now() where draft_id::text = %s and status = 'ready_for_review' returning id, cluster_id, job_id", (reason[:2000], draft_id))
        if a:
            await self.add_feedback({"article_id": a["id"], "cluster_id": a.get("cluster_id"), "job_id": a.get("job_id"), "kind": "reject", "reason_category": "other", "reason": reason[:2000], "actor": actor})
        return a

    async def published_news(self, query: str, since: datetime | None, limit: int) -> list[dict[str, Any]]:
        q = f"%{query.replace('%', '').replace('_', ' ')[:120]}%"
        return await self._all(
            """select title, '/' || collection || '/' || slug as path, description, published_at, updated_at from post
               where collection = 'news' and status = 'published' and deleted_at is null and noindex = false and (%(since)s::timestamptz is null or published_at >= %(since)s)
                 and (title ilike %(q)s or description ilike %(q)s or body ilike %(q)s) order by published_at desc limit %(n)s""",
            {"q": q, "since": since, "n": max(1, min(limit, 20))},
        )


# --- In memory (tests) ----------------------------------------------------------------------------------------------


@dataclass
class MemoryNewsStore:
    settings: dict[str, Any] | None = None
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    writing: list[dict[str, Any]] = field(default_factory=list)
    imaging: list[dict[str, Any]] = field(default_factory=list)
    srcs: dict[str, dict[str, Any]] = field(default_factory=dict)
    topic_rows: list[dict[str, Any]] = field(default_factory=list)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    clusters: dict[str, dict[str, Any]] = field(default_factory=dict)
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    step_rows: list[dict[str, Any]] = field(default_factory=list)
    log_rows: list[dict[str, Any]] = field(default_factory=list)
    cand_rows: list[dict[str, Any]] = field(default_factory=list)
    articles: dict[str, dict[str, Any]] = field(default_factory=dict)
    image_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    labels: list[dict[str, Any]] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    posts: dict[str, dict[str, Any]] = field(default_factory=dict)
    clock: Any = None

    def now(self) -> datetime:
        return self.clock() if self.clock else datetime.now(UTC)

    async def settings_row(self) -> dict[str, Any] | None:
        return self.settings

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        j = self.jobs.get(job_id)
        return dict(j) if j and not j.get("deleted_at") else None

    async def list_jobs(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        return [dict(j) for j in self.jobs.values() if not j.get("deleted_at") and (j.get("enabled") or not enabled_only)]

    async def set_job_schedule(self, job_id: str, *, last_run_at: datetime | None = None, next_run_at: datetime | None = None) -> None:
        j = self.jobs[job_id]
        j["last_run_at"] = last_run_at or j.get("last_run_at")
        j["next_run_at"] = next_run_at or j.get("next_run_at")

    @staticmethod
    def _pick(rows: list[dict[str, Any]], key: str, version: int | None) -> dict[str, Any] | None:
        for r in rows:
            if r["key"] == key and ((version and r["version"] == version) or (not version and r.get("status", "active") == "active")):
                return dict(r)
        return None

    async def writing_profile(self, key: str, version: int | None = None) -> dict[str, Any] | None:
        return self._pick(self.writing, key, version)

    async def image_profile(self, key: str, version: int | None = None) -> dict[str, Any] | None:
        return self._pick(self.imaging, key, version)

    async def add_profile_version(self, kind: str, row: dict[str, Any]) -> dict[str, Any]:
        v = max([r["version"] for r in self.writing if r["key"] == row["key"]] or [0]) + 1
        r = {**row, "version": v, "status": "proposed", "id": str(uuid.uuid4())}
        self.writing.append(r)
        return r

    async def sources(self, ids: list[str] | None = None, enabled_only: bool = True) -> list[dict[str, Any]]:
        return [dict(s) for s in self.srcs.values() if (not enabled_only or s.get("enabled")) and (not ids or s["id"] in ids)]

    async def update_source(self, source_id: str, **fields: Any) -> None:
        self.srcs[source_id].update(fields)

    async def topics(self) -> list[dict[str, Any]]:
        return [dict(t) for t in self.topic_rows if t.get("enabled", True)]

    async def recent_items(self, since: datetime) -> list[dict[str, Any]]:
        return [dict(i) for i in self.items.values() if i["fetched_at"] >= since]

    async def insert_item(self, row: dict[str, Any]) -> str | None:
        if any(i["source_id"] == row["source_id"] and i["url_hash"] == row["url_hash"] for i in self.items.values()):
            return None
        iid = str(uuid.uuid4())
        self.items[iid] = {"id": iid, "fetched_at": self.now(), "cluster_id": None, "duplicate_of": None, "dedupe_reason": None, **row}
        return iid

    async def update_item(self, item_id: str, **fields: Any) -> None:
        # Like news_items_cluster_id_fkey in Postgres: an item can only point at a cluster that is already saved.
        cid = fields.get("cluster_id")
        if cid is not None and cid not in self.clusters:
            raise ValueError(f"news_items.cluster_id {cid} is not in news_clusters")
        self.items[item_id].update(fields)

    async def items_by_cluster(self, cluster_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for i in sorted(self.items.values(), key=lambda x: x["fetched_at"]):
            if i.get("cluster_id") in cluster_ids:
                s = self.srcs.get(i["source_id"], {})
                out.setdefault(i["cluster_id"], []).append({**i, "source_name": s.get("name"), "credibility": s.get("credibility", 0.7), "source_url": s.get("url"), "source_type": s.get("type")})
        return out

    async def recent_clusters(self, since: datetime) -> list[dict[str, Any]]:
        return [dict(c) for c in self.clusters.values() if c["last_seen"] >= since]

    async def get_cluster(self, cluster_id: str) -> dict[str, Any] | None:
        c = self.clusters.get(cluster_id)
        return dict(c) if c else None

    async def save_cluster(self, row: dict[str, Any]) -> None:
        old = self.clusters.get(row["id"])
        if old and old.get("status") in ("drafted", "covered"):
            row = {**row, "status": old["status"]}
        if old and row.get("site_match") is None:
            row = {**row, "site_match": old.get("site_match")}
        self.clusters[row["id"]] = {**(old or {}), **row}

    async def create_run(self, row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if row.get("slot"):
            for r in self.runs.values():
                if r.get("job_id") == row.get("job_id") and r.get("slot") == row["slot"]:
                    return dict(r), False
        r = {"status": "running", "counters": {}, "tokens": 0, "cost_usd": 0.0, "image_cost_usd": 0.0, "started_at": self.now(), "finished_at": None, "dry_run": False, "llm_enabled": True, **row}
        self.runs[row["id"]] = r
        return dict(r), True

    async def update_run(self, run_id: str, **fields: Any) -> None:
        self.runs[run_id].update({k: v for k, v in fields.items() if k in _RUN_UPDATABLE})

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        r = self.runs.get(run_id)
        return dict(r) if r else None

    async def add_run_cost(self, run_id: str, *, usd: float = 0.0, tokens: int = 0, image_usd: float = 0.0) -> None:
        r = self.runs.get(run_id)
        if r:
            r["cost_usd"] += usd
            r["tokens"] += tokens
            r["image_cost_usd"] += image_usd

    async def start_step(self, run_id: str, step: str, seq: int, input_count: int | None = None) -> int:
        self.step_rows.append({"id": len(self.step_rows) + 1, "run_id": run_id, "step": step, "seq": seq, "status": "running", "input_count": input_count, "started_at": self.now(), "finished_at": None, "output_count": None, "error": None, "detail": {}})
        return len(self.step_rows)

    async def finish_step(self, step_id: int, status: str, *, output_count: int | None = None, error: str | None = None, detail: dict[str, Any] | None = None) -> None:
        self.step_rows[step_id - 1].update(status=status, output_count=output_count, error=error, detail=detail or {}, finished_at=self.now())

    async def steps(self, run_id: str) -> list[dict[str, Any]]:
        return [dict(s) for s in self.step_rows if s["run_id"] == run_id]

    async def add_logs(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            self.log_rows.append({"id": len(self.log_rows) + 1, **r})

    async def logs(self, run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return [dict(r) for r in self.log_rows if r["run_id"] == run_id and r["id"] > after_id]

    async def save_candidates(self, run_id: str, job_id: str | None, rows: list[dict[str, Any]]) -> None:
        self.cand_rows = [c for c in self.cand_rows if not (c["run_id"] == run_id and c["cluster_id"] in {r["cluster_id"] for r in rows})]
        for r in rows:
            self.cand_rows.append({**r, "run_id": run_id, "job_id": job_id, "created_at": self.now()})

    async def candidates(self, run_id: str) -> list[dict[str, Any]]:
        return sorted((dict(c) for c in self.cand_rows if c["run_id"] == run_id), key=lambda c: -c["score"])

    async def recent_candidates(self, limit: int) -> list[dict[str, Any]]:
        return [dict(c) for c in reversed(self.cand_rows)][:limit]

    async def covered_clusters(self, job_id: str, since: datetime) -> set[str]:
        out = set()
        for a in self.articles.values():
            if a["created_at"] < since:
                continue
            if a.get("cluster_id") and a["status"] in COUNTED_ARTICLE and (a.get("job_id") == job_id or a["status"] == "published"):
                out.add(a["cluster_id"])
            if a.get("job_id") == job_id and a["status"] in ("ready_for_review", "published"):
                out.update(a.get("cluster_ids") or [])
        out.update(c["id"] for c in self.clusters.values() if c.get("status") in ("covered", "drafted") and c["last_seen"] >= since)
        return out

    async def counts_since(self, job_id: str, since: datetime) -> Counts:
        arts = [a for a in self.articles.values() if a["created_at"] >= since and a["status"] in COUNTED_ARTICLE]
        c = Counts(job_today=sum(1 for a in arts if a.get("job_id") == job_id), global_today=len(arts))
        for k in self.cand_rows:
            run = self.runs.get(k["run_id"], {})
            if k.get("job_id") != job_id or k["decision"] != "selected" or k["created_at"] < since or run.get("dry_run"):
                continue
            t = (self.clusters.get(k["cluster_id"]) or {}).get("topics") or []
            main = t[0]["slug"] if t and isinstance(t[0], dict) else "uncategorised"
            c.per_topic[main] = c.per_topic.get(main, 0) + 1
            for src in {i["source_id"] for i in self.items.values() if i.get("cluster_id") == k["cluster_id"]}:
                c.per_source[src] = c.per_source.get(src, 0) + 1
        return c

    async def create_article(self, row: dict[str, Any]) -> str:
        if row.get("cluster_id") and (row.get("status") or "drafting") in ACTIVE_ARTICLE:
            for a in self.articles.values():
                if a.get("job_id") == row.get("job_id") and a.get("cluster_id") == row["cluster_id"] and a["status"] in ACTIVE_ARTICLE:
                    raise DuplicateArticle("an active draft for this story already exists for this job")
        aid = str(uuid.uuid4())
        self.articles[aid] = {"id": aid, "status": "drafting", "mode": "single", "cluster_ids": [], "created_at": self.now(), "updated_at": self.now(), "cost_usd": 0.0, "tokens": 0, "review_outcome": None, "draft_id": None, "post_id": None, **row}
        return aid

    async def update_article(self, article_id: str, **fields: Any) -> None:
        self.articles[article_id].update({k: v for k, v in fields.items() if k in _ARTICLE_UPDATABLE}, updated_at=self.now())

    async def get_article(self, article_id: str) -> dict[str, Any] | None:
        a = self.articles.get(article_id)
        return dict(a) if a else None

    async def open_article_for_cluster(self, cluster_id: str, exclude_job: str | None = None) -> dict[str, Any] | None:
        for a in sorted(self.articles.values(), key=lambda x: x["created_at"], reverse=True):
            if a.get("cluster_id") == cluster_id and a["status"] in ("drafting", "ready_for_review", "published") and (exclude_job is None or a.get("job_id") != exclude_job or a["status"] == "published"):
                return dict(a)
        return None

    async def add_image(self, row: dict[str, Any]) -> str:
        iid = str(uuid.uuid4())
        self.image_rows[iid] = {"id": iid, "created_at": self.now(), "selected": False, "status": "ok", "cost_usd": 0.0, **row}
        return iid

    async def update_image(self, image_id: str, **fields: Any) -> None:
        self.image_rows[image_id].update(fields)

    async def images(self, article_id: str) -> list[dict[str, Any]]:
        return [dict(i) for i in self.image_rows.values() if i["article_id"] == article_id]

    async def spend(self, job_id: str | None, since: datetime) -> dict[str, float]:
        rs = [r for r in self.runs.values() if r["started_at"] >= since and (job_id is None or r.get("job_id") == job_id)]
        return {"llm": sum(r["cost_usd"] for r in rs), "image": sum(r["image_cost_usd"] for r in rs)}

    async def training_rows(self) -> list[dict[str, Any]]:
        out = []
        for a in self.articles.values():
            if not a.get("review_outcome"):
                continue
            k = next((c for c in self.cand_rows if c["run_id"] == a.get("run_id") and c["cluster_id"] == a.get("cluster_id")), None)
            if k:
                clicks = max([float(f["payload"].get("clicks", 0)) for f in self.feedback if f.get("article_id") == a["id"] and f["kind"] == "performance"] or [0.0])
                out.append({"run_id": k["run_id"], "job_id": k.get("job_id"), "at": k["created_at"], "features": k["features"], "review_outcome": a["review_outcome"], "reject_category": a.get("reject_category"), "clicks": clicks or None})
        return out

    async def label_rows(self) -> list[dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for lb in self.labels:
            key = str(lb.get("item_id") or lb.get("cluster_id"))
            if key not in best or lb["source"] == "human":
                best[key] = lb
        out = []
        for key, lb in best.items():
            i = self.items.get(key)
            if i:
                out.append({"topics": lb["topics"], "source": lb["source"], "item_id": key, "cluster_id": i.get("cluster_id"), "title": i["title"], "clean_text": i.get("clean_text", ""), "excerpt": i.get("excerpt", ""), "embedding": i.get("embedding"), "entities": i.get("entities", {})})
        return out

    async def unlabeled_items(self, limit: int) -> list[dict[str, Any]]:
        done = {lb.get("item_id") for lb in self.labels}
        rows = [i for i in self.items.values() if i["id"] not in done and not i.get("duplicate_of")]
        return [{"id": i["id"], "title": i["title"], "excerpt": i.get("excerpt", "")} for i in rows[:limit]]

    async def add_labels(self, rows: list[dict[str, Any]]) -> None:
        self.labels.extend({**r, "at": self.now()} for r in rows)

    async def save_model(self, kind: str, version: str, *, metrics: dict[str, Any], n_samples: int, artifact: bytes | None, active: bool, notes: str | None = None) -> None:
        if active:
            for m in self.models:
                if m["kind"] == kind:
                    m["active"] = False
        self.models.append({"kind": kind, "version": version, "metrics": metrics, "n_samples": n_samples, "artifact": artifact, "active": active, "notes": notes, "trained_at": self.now()})

    async def active_model(self, kind: str) -> dict[str, Any] | None:
        return next((dict(m) for m in reversed(self.models) if m["kind"] == kind and m["active"]), None)

    async def list_models(self, kind: str | None, limit: int = 20) -> list[dict[str, Any]]:
        rows = [{**{k: v for k, v in m.items() if k != "artifact"}, "has_artifact": m.get("artifact") is not None} for m in reversed(self.models) if kind is None or m["kind"] == kind]
        return rows[:limit]

    async def activate_model(self, kind: str, version: str) -> bool:
        target = next((m for m in self.models if m["kind"] == kind and m["version"] == version and m.get("artifact") is not None), None)
        if target is None:
            return False
        for m in self.models:
            if m["kind"] == kind:
                m["active"] = m is target
        return True

    async def edited_articles(self, since: datetime, limit: int) -> list[dict[str, Any]]:
        out = []
        for a in self.articles.values():
            p = self.posts.get(a.get("post_id") or "")
            if a["status"] == "published" and a.get("generated_doc") and p and (a.get("published_at") or a["created_at"]) >= since:
                out.append({"id": a["id"], "generated_doc": a["generated_doc"], "edit_distance": a.get("edit_distance"), "published_body": p["body"], "published_title": p["title"]})
        return out[:limit]

    async def add_feedback(self, row: dict[str, Any]) -> None:
        self.feedback.append({"at": self.now(), **row})

    async def prune(self, log_days: int, item_days: int, now: datetime) -> dict[str, int]:
        before = len(self.log_rows)
        self.log_rows = [r for r in self.log_rows if r["at"] >= now - timedelta(days=log_days)]
        return {"logs": before - len(self.log_rows), "items": 0, "candidates": 0}

    async def list_runs(self, job_id: str | None, limit: int) -> list[dict[str, Any]]:
        rows = [r for r in self.runs.values() if job_id is None or r.get("job_id") == job_id]
        return sorted((dict(r) for r in rows), key=lambda r: r["started_at"], reverse=True)[:limit]

    async def list_articles(self, status: str | None, limit: int) -> list[dict[str, Any]]:
        rows = [dict(a, title=(a.get("generated_doc") or {}).get("title")) for a in self.articles.values() if status is None or a["status"] == status]
        return sorted(rows, key=lambda a: a["created_at"], reverse=True)[:limit]

    async def mark_rejected(self, draft_id: str, reason: str, actor: str) -> dict[str, Any] | None:
        for a in self.articles.values():
            if a.get("draft_id") == draft_id and a["status"] == "ready_for_review":
                a.update(status="rejected", review_outcome="rejected", reject_category="other", reject_reason=reason)
                await self.add_feedback({"article_id": a["id"], "kind": "reject", "reason_category": "other", "reason": reason, "actor": actor})
                return dict(a)
        return None

    async def published_news(self, query: str, since: datetime | None, limit: int) -> list[dict[str, Any]]:
        q = query.lower()
        out = [p for p in self.posts.values() if p.get("collection") == "news" and p.get("status") == "published" and (since is None or p["published_at"] >= since) and (q in p["title"].lower() or q in p.get("body", "").lower())]
        return sorted(out, key=lambda p: p["published_at"], reverse=True)[:limit]
