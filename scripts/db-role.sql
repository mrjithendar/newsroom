-- A database role of its own for the news agent: `news_agent`, with only what the news desk's SQL needs.
--
-- Why. Until this exists the agent connects with the web app's role (DATABASE_URL from the same Secret), which owns
-- every table. A prompt injection that reached SQL, or a leaked agent credential, could then read sessions and
-- password hashes, rewrite posts or make itself an admin. With this role it can read published posts, the settings
-- and review drafts, write the news desk's own tables and its flow runs, book its LLM usage, and nothing else: no
-- sessions, accounts or API keys, no writes to posts, pages, users, settings or the SEO tables.
--
-- How (nothing changes until you do this; README.md "Database role"):
--   1. Run this file as a superuser in the site's database. It can be run again at any time:
--        psql "$SUPERUSER_URL" -v ON_ERROR_STOP=1 -f scripts/db-role.sql
--   2. ALTER ROLE news_agent LOGIN PASSWORD '…';
--   3. Create Secret jithendar/news-agent-db-credentials with key `uri` and restart deploy/news-agent.
--
-- Keep it in step with the code: tests/test_db_role.py runs this file on a real PostgreSQL with every web app
-- migration applied and compares the result, privilege by privilege, with the SQL in app/. A new statement that
-- needs a privilege this file does not grant fails that test, and so does a grant nothing uses.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'news_agent') THEN
    CREATE ROLE news_agent NOLOGIN;
  END IF;
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO news_agent', current_database());
END $$;

-- Start from nothing on every table, so that a line removed from this file is also removed from the database on the
-- next run.
DO $$
DECLARE rel record;
BEGIN
  FOR rel IN
    SELECT c.oid::regclass AS name, c.relkind AS kind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm', 'S') AND c.relowner <> 'news_agent'::regrole
  LOOP
    EXECUTE format('REVOKE ALL ON %s %s FROM news_agent', CASE WHEN rel.kind = 'S' THEN 'SEQUENCE' ELSE 'TABLE' END, rel.name);
  END LOOP;
END $$;

-- CREATE: the agent creates its one table itself at start-up (news_agent_settings, below). Sequences: ids of the rows
-- it inserts.
GRANT USAGE, CREATE ON SCHEMA public TO news_agent;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO news_agent;

-- Read only: published posts (duplicates, performance), the Newsroom settings row, review drafts (usage attribution),
-- in-flight LLM batches (spend), and what the admin configures for the desk.
GRANT SELECT ON
  post, setting, seo_drafts, seo_llm_batches,
  news_topics, news_image_profiles
  TO news_agent;

-- The desk's configuration: the agent updates only its bookkeeping columns' rows (schedule times, source health).
GRANT SELECT, UPDATE ON news_jobs, news_sources TO news_agent;

-- Append only.
GRANT SELECT, INSERT ON
  seo_llm_usage,
  news_feedback, news_labels, news_writing_profiles
  TO news_agent;
-- Run logs: appended while a run goes, pruned by the retention housekeeping.
GRANT SELECT, INSERT, DELETE ON news_run_logs TO news_agent;

-- The news desk's working tables and its flow runs.
GRANT SELECT, INSERT, UPDATE ON
  seo_flow_runs,
  news_articles, news_clusters, news_images, news_models, news_run_steps, news_runs
  TO news_agent;
GRANT SELECT, INSERT, UPDATE, DELETE ON news_candidates, news_items TO news_agent;

-- Schedules (shared with the seo-agent; each dispatches only its own flows): it reads them and records a firing.
GRANT SELECT ON seo_flow_schedules TO news_agent;
GRANT UPDATE (last_slot, last_fired_at, last_status, last_run_id, last_error) ON seo_flow_schedules TO news_agent;

-- Run history for the CrewAI console: steps, log lines and model calls of its runs.
GRANT SELECT, INSERT, UPDATE ON seo_flow_run_steps TO news_agent;
GRANT INSERT ON seo_flow_run_logs TO news_agent;
GRANT SELECT, INSERT ON seo_flow_run_calls TO news_agent;

-- The agent's own table: the settings an admin saved (app/admin_settings, Newsroom → Settings → News agent). The agent
-- creates it at start-up; if it already exists (created while the agent still used the web app's role), it is handed
-- over. The site's role keeps full access to it and to whatever the agent creates later.
DO $$
DECLARE
  site_role name;
BEGIN
  SELECT tableowner INTO STRICT site_role FROM pg_tables WHERE schemaname = 'public' AND tablename = 'post';
  IF to_regclass('public.news_agent_settings') IS NOT NULL THEN
    ALTER TABLE public.news_agent_settings OWNER TO news_agent;
    EXECUTE format('GRANT ALL ON public.news_agent_settings TO %I', site_role);
  END IF;
  EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE news_agent IN SCHEMA public GRANT ALL ON TABLES TO %I', site_role);
END $$;
