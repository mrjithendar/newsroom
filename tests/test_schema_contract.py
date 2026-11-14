"""Contract: the web app tables that the Postgres tests declare by hand are still what the web app's migrations create.

The agent reads and writes the web app's database. Its tests declare the few columns they need inline, so a migration
in services/www that renames, drops or retypes one of those columns would leave these tests green while the agent
fails in production. Here every migration is applied to a scratch database, each hand-written `create table` found in
the test sources is created beside it, and each of its columns has to exist in the real table with the same type.

Needs SEO_AGENT_TEST_DATABASE_URL (the CI service container); skipped without it.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

TESTS = Path(__file__).resolve().parent
MIGRATIONS = TESTS.parents[1] / "www" / "src" / "db" / "migrations"
# Created by the agent itself (app/index.py: the vector width follows the embedding model), not by a migration.
AGENT_OWNED = {"seo_article_chunks"}


def hand_written_tables() -> list[tuple[str, str, str]]:
    """(file, table, statement) for every `create table` in the other test modules."""
    found = []
    for path in sorted(TESTS.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        src = path.read_text()
        for m in re.finditer(r"create table (?:if not exists )?(\w+) \(", src):
            depth, end = 1, m.end()
            while depth:
                depth += {"(": 1, ")": -1}.get(src[end], 0)
                end += 1
            found.append((path.name, m.group(1), src[m.start() : end]))
    return found


def migration_statements() -> list[str]:
    """Every statement of every migration, in the order the web app's migrator applies them (by file name)."""
    files = sorted(MIGRATIONS.glob("*.sql"))
    assert len(files) >= 30, "could not find the web app's migrations"
    return [stmt for path in files for stmt in path.read_text().split("--> statement-breakpoint") if stmt.strip()]


async def columns(c: Any, schema: str, table: str) -> dict[str, str]:
    cur = await c.execute("select column_name, udt_name from information_schema.columns where table_schema = %s and table_name = %s", (schema, table))
    return {str(name): str(kind) for name, kind in await cur.fetchall()}


def test_the_hand_written_tables_are_found() -> None:
    names = {table for _, table, _ in hand_written_tables()}
    assert {"post", "seo_flow_runs", "seo_llm_usage", "setting"} <= names
    assert all(stmt.endswith(")") and stmt.count("(") == stmt.count(")") for _, _, stmt in hand_written_tables())


@pytest.mark.postgres
async def test_hand_written_tables_match_the_web_app_migrations() -> None:
    url = os.environ.get("SEO_AGENT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set SEO_AGENT_TEST_DATABASE_URL to check the test tables against the web app's migrations")
    import psycopg
    from psycopg.conninfo import make_conninfo

    tables, migrations = hand_written_tables(), migration_statements()
    # A database of its own: the migrations name the `public` schema explicitly ("public"."user"), and the other
    # Postgres tests share this server's default database.
    database = f"contract_{uuid.uuid4().hex[:10]}"
    problems: list[str] = []
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as admin:
        await admin.execute(f"create database {database}")
        try:
            async with await psycopg.AsyncConnection.connect(make_conninfo(url, dbname=database), autocommit=True) as c:
                # The migrator's own ledger (services/www/src/db/migrate.ts), then every migration in its order.
                await c.execute("create table schema_migrations (name text primary key, applied_at timestamptz not null default now())")
                for stmt in migrations:
                    await c.execute(stmt)
                for i, (file, table, stmt) in enumerate(tables):
                    if table in AGENT_OWNED:
                        continue
                    have = await columns(c, "public", table)
                    if not have:
                        problems.append(f"{file}: no migration creates {table}")
                        continue
                    # A schema of its own, so the copy does not collide with the real table; `public` stays on the
                    # path, so its foreign keys resolve.
                    copy = f"copy_{i}"
                    await c.execute(f"create schema {copy}")
                    await c.execute(f"set search_path to {copy}, public")
                    await c.execute(stmt)
                    for name, kind in (await columns(c, copy, table)).items():
                        if name not in have:
                            problems.append(f"{file}: {table}.{name} is not in the migrations")
                        elif have[name] != kind:
                            problems.append(f"{file}: {table}.{name} is {kind} in the test and {have[name]} in the migrations")
        finally:
            await admin.execute(f"drop database if exists {database} with (force)")
    assert not problems, "\n".join(problems)
