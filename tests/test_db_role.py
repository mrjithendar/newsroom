"""The agent's own database role (scripts/db-role.sql) grants what the SQL in app/ needs, and nothing more.

The requirement is read from the code: every string literal in app/ is scanned for `insert into`, `update … set`,
`delete from`, `from` and `join` followed by the name of a table that exists. The script is then run on a real
PostgreSQL that has every web app migration applied, and each table is checked privilege by privilege. So a new
statement fails this test until the script grants what it needs, and a grant that nothing uses fails it too.

The scan cannot see a table name that is passed around as a value; those are listed in DYNAMIC, and a test makes sure
the list is complete.
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.moks_cms import www

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
SCRIPT = ROOT / "scripts" / "db-role.sql"
MIGRATIONS = "src/db/migrations"
ROLE = "seo_agent"
PRIVILEGES = ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE")

# Table names the code hands to a statement as a value, and what that statement does with them.
DYNAMIC: dict[str, set[str]] = {
    # seo_ml/store.py replace_outputs(): delete every row, insert the new ones
    "seo_ml_decay_signals": {"SELECT", "INSERT", "DELETE"},
    "seo_ml_query_clusters": {"SELECT", "INSERT", "DELETE"},
    "seo_ml_intent_predictions": {"SELECT", "INSERT", "DELETE"},
    "seo_ml_related_articles": {"SELECT", "INSERT", "DELETE"},
    # news/store.py _profile(): select one row by key
    "news_image_profiles": {"SELECT"},
    "news_writing_profiles": {"SELECT"},
}
# The same word as a table, used for something else (mcp/common.py: a key whose value is redacted).
NOT_A_TABLE = {"api_key"}
# Granted on these columns only, never on the table.
BY_COLUMN: dict[str, dict[str, set[str]]] = {
    "user": {"SELECT": {"id", "role", "banned"}},
    "seo_mcp_grants": {"UPDATE": {"last_used_at"}},
}
# Created by the agent itself (app/index.py) and owned by its role.
OWNED = ("seo_article_chunks", "seo_article_chunks_next")
CHUNK_COLUMNS = "(id bigserial primary key, article_id uuid not null references post(id) on delete cascade, chunk_index integer not null)"


def literals(path: Path) -> Iterator[str]:
    """Every string in a module; an f-string with `{}` where its expressions were."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.JoinedStr):
            yield "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}" for v in node.values)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def needs(text: str, tables: set[str]) -> dict[str, set[str]]:
    """Privileges the SQL in `text` needs, for names that are tables."""
    out: dict[str, set[str]] = {}
    sql = " ".join(text.lower().split())
    name = r'"?([a-z_][a-z0-9_]*)"?'

    def add(table: str, *privileges: str) -> None:
        if table in tables:
            out.setdefault(table, set()).update(privileges)

    for m in re.finditer(rf"\binsert into {name}", sql):
        following = sql[m.end() :]
        end = re.search(r"\binsert into\b", following)
        statement = following[: end.start()] if end else following
        add(m.group(1), "INSERT")
        if "do update" in statement:
            add(m.group(1), "UPDATE", "SELECT")
        if "returning" in statement or "on conflict (" in statement:
            add(m.group(1), "SELECT")
    for m in re.finditer(rf"\bupdate {name} (?:as \w+ |\w+ )?set\b", sql):
        add(m.group(1), "UPDATE", "SELECT")
    for m in re.finditer(rf"\bdelete from {name}", sql):
        add(m.group(1), "DELETE", "SELECT")
    for m in re.finditer(rf"\b(?:from|join) {name}", sql):
        add(m.group(1), "SELECT")
    return out


def required(tables: set[str]) -> dict[str, set[str]]:
    """Table → privileges, from the code and DYNAMIC."""
    out = {table: set(privileges) for table, privileges in DYNAMIC.items()}
    for path in sorted(APP.rglob("*.py")):
        for text in literals(path):
            for table, privileges in needs(text, tables).items():
                out.setdefault(table, set()).update(privileges)
    return out


def migrated_tables() -> set[str]:
    names: set[str] = set()
    for path in www(MIGRATIONS).glob("*.sql"):
        names.update(re.findall(r'CREATE TABLE (?:IF NOT EXISTS )?"?([a-z_0-9]+)"?', path.read_text(), re.IGNORECASE))
    assert len(names) > 80, "could not read the web app's migrations"
    return names


def migration_statements() -> list[str]:
    return [stmt for path in sorted(www(MIGRATIONS).glob("*.sql")) for stmt in path.read_text().split("--> statement-breakpoint") if stmt.strip()]


def script_text() -> str:
    return SCRIPT.read_text()


def test_the_scan_reads_what_a_statement_needs() -> None:
    tables = {"post", "seo_ask_gaps", "seo_flow_runs", "audit_log", "user"}
    assert needs("select p.id from post p join seo_flow_runs r on r.target_id = p.id::text", tables) == {"post": {"SELECT"}, "seo_flow_runs": {"SELECT"}}
    assert needs("insert into audit_log (actor) values (%s)", tables) == {"audit_log": {"INSERT"}}
    assert needs("insert into seo_ask_gaps (normalized) values (%s)\n on conflict (normalized) do update set asked = seo_ask_gaps.asked + 1", tables) == {"seo_ask_gaps": {"INSERT", "UPDATE", "SELECT"}}
    assert needs("insert into seo_flow_runs (run_id) values (%s) on conflict do nothing returning *", tables) == {"seo_flow_runs": {"INSERT", "SELECT"}}
    assert needs("update seo_flow_runs set status = %s where run_id = %s", tables) == {"seo_flow_runs": {"UPDATE", "SELECT"}}
    assert needs('delete from seo_flow_runs where finished_at < now(); select 1 from "user"', tables) == {"seo_flow_runs": {"DELETE", "SELECT"}, "user": {"SELECT"}}
    # Prose and other names are not tables.
    assert needs("Read the article from the web app and update it.", tables) == {}


def test_table_names_passed_as_values_are_all_listed() -> None:
    tables = migrated_tables()
    # One-word table names ("post", "user", "page") are everyday words in the code; statements that use them are
    # found by the scan like any other.
    as_values = {text for path in APP.rglob("*.py") for text in literals(path) if text in tables and "_" in text}
    assert as_values == set(DYNAMIC) | NOT_A_TABLE, "a table name is used as a value: say in DYNAMIC what is done with it"
    assert set(DYNAMIC) <= tables


def test_the_code_needs_nothing_from_the_sign_in_tables() -> None:
    wanted = required(migrated_tables())
    assert wanted.get("post") == {"SELECT"}
    assert wanted.get("audit_log") == {"INSERT"}
    assert not {"session", "account", "verification", "api_key", "seo_mcp_clients", "seo_mcp_codes", "page", "site_config", "media"} & set(wanted)
    # What the column grants in the script rest on.
    assert wanted.get("user") == {"SELECT"} and wanted.get("seo_mcp_grants") == {"SELECT", "UPDATE"}


def test_the_agents_own_database_url_is_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import load_settings

    monkeypatch.setenv("DATABASE_URL", "postgresql://site@db/app")
    monkeypatch.delenv("SEO_AGENT_DATABASE_URL", raising=False)
    assert load_settings().database_url == "postgresql://site@db/app"
    monkeypatch.setenv("SEO_AGENT_DATABASE_URL", "  ")  # an empty Secret value falls back, it does not break the start
    assert load_settings().database_url == "postgresql://site@db/app"
    monkeypatch.setenv("SEO_AGENT_DATABASE_URL", "postgresql://seo_agent@db/app")
    assert load_settings().database_url == "postgresql://seo_agent@db/app"


async def one(c: Any, query: str, *args: object) -> Any:
    row = await (await c.execute(query, args)).fetchone()
    return row[0] if row else None


@pytest.mark.postgres
async def test_the_role_can_do_what_the_code_does_and_nothing_else() -> None:
    url = os.environ.get("SEO_AGENT_TEST_DATABASE_URL")
    if not url:
        pytest.skip("set SEO_AGENT_TEST_DATABASE_URL to check scripts/db-role.sql on a real PostgreSQL")
    import psycopg
    from psycopg.conninfo import make_conninfo
    from psycopg.errors import InsufficientPrivilege

    script, migrations = script_text(), migration_statements()
    # A database of its own: the migrations name the `public` schema, and the other Postgres tests share the default one.
    database = f"role_{uuid.uuid4().hex[:10]}"
    problems: list[str] = []
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as admin:
        existed = await one(admin, "select 1 from pg_roles where rolname = %s", ROLE)
        await admin.execute(f"create database {database}")
        try:
            async with await psycopg.AsyncConnection.connect(make_conninfo(url, dbname=database), autocommit=True) as c:
                await c.execute("create table schema_migrations (name text primary key, applied_at timestamptz not null default now())")
                for stmt in migrations:
                    await c.execute(stmt)
                # As in production today: the index table exists and belongs to the web app's role.
                await c.execute(f"create table {OWNED[0]} {CHUNK_COLUMNS}")
                await c.execute(script)
                await c.execute(script)  # safe to run again

                tables = {r[0] for r in await (await c.execute("select tablename from pg_tables where schemaname = 'public'")).fetchall()}
                wanted = required(tables)
                for table, columns in BY_COLUMN.items():
                    wanted[table] = wanted.get(table, set()) - set(columns)
                for table in sorted(tables - set(OWNED)):
                    for privilege in PRIVILEGES:
                        has = await one(c, "select has_table_privilege(%s::name, %s::text, %s::text)", ROLE, f'public."{table}"', privilege)
                        if has != (privilege in wanted.get(table, set())):
                            problems.append(f"{table}: {privilege} is {'granted but not used by app/' if has else 'used by app/ but not granted'}")
                for table, columns in BY_COLUMN.items():
                    for privilege, allowed in columns.items():
                        names = [r[0] for r in await (await c.execute("select column_name from information_schema.columns where table_schema = 'public' and table_name = %s", (table,))).fetchall()]
                        assert allowed <= set(names), f"{table} has no column {allowed - set(names)}"
                        for column in names:
                            has = await one(c, "select has_column_privilege(%s::name, %s::text, %s::text, %s::text)", ROLE, f'public."{table}"', column, privilege)
                            if has != (column in allowed):
                                problems.append(f"{table}.{column}: {privilege} is {'granted' if has else 'missing'}")
                assert await one(c, "select tableowner from pg_tables where schemaname = 'public' and tablename = %s", OWNED[0]) == ROLE

                # And as the role itself: what the agent does at start-up and during a rebuild works, the rest does not.
                await c.execute(f"set role {ROLE}")
                await c.execute(f"alter table {OWNED[0]} add column if not exists title text not null default ''")
                await c.execute(f"create table {OWNED[1]} {CHUNK_COLUMNS}")
                await c.execute(f"drop table {OWNED[1]}")
                await c.execute("select id, title from post limit 1")
                await c.execute('select id, role, banned from "user" limit 1')
                await c.execute("insert into audit_log (actor, action, severity) values ('mcp:test', 'seo.mcp.test', 'info')")
                for forbidden in (
                    "select 1 from session",
                    "select 1 from account",
                    'select email from "user"',
                    "update post set title = 'changed'",
                    "delete from audit_log",
                    "select 1 from audit_log",
                    "update seo_mcp_grants set revoked_at = null",
                    "insert into seo_mcp_grants (client_id, user_id, scopes, resource) values ('c', 'u', '{}', 'r')",
                    "update setting set value = '{}'::jsonb",
                ):
                    with pytest.raises(InsufficientPrivilege):
                        await c.execute(forbidden)
                await c.execute("reset role")
        finally:
            await admin.execute(f"drop database if exists {database} with (force)")
            if not existed:  # roles belong to the server, not to the database that was just dropped
                with contextlib.suppress(psycopg.Error):
                    await admin.execute(f"drop role if exists {ROLE}")
    assert not problems, "\n".join(problems)
