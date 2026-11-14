-- A database role of its own for the SEO agent: `seo_agent`, with only what the agent's SQL needs.
--
-- Why. The agent connects with the web app's role (DATABASE_URL from the same Secret), which owns every table. A
-- prompt injection that reached SQL, or a leaked agent credential, could read sessions and password hashes, rewrite
-- posts or make itself an admin. With this role it can read published content and SEO data, write its own working
-- tables, append to the audit log, and nothing else: no sessions, accounts or API keys, no writes to posts, pages,
-- users or settings.
--
-- How (docs/seo/SECURITY.md has the whole runbook; nothing changes until you do this):
--   1. Run this file as a superuser in the site's database. It can be run again at any time:
--        psql "$SUPERUSER_URL" -v ON_ERROR_STOP=1 -f services/seo-agent/scripts/db-role.sql
--   2. ALTER ROLE seo_agent LOGIN PASSWORD '…';
--   3. Create Secret jithendar/seo-agent-db-credentials with key `uri` and restart deploy/seo-agent.
--
-- Keep it in step with the code: tests/test_db_role.py runs this file on a real PostgreSQL with every web app
-- migration applied and compares the result, privilege by privilege, with the SQL in app/. A new statement that
-- needs a privilege this file does not grant fails that test, and so does a grant nothing uses.

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'seo_agent') THEN
    CREATE ROLE seo_agent NOLOGIN;
  END IF;
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO seo_agent', current_database());
END $$;

-- Start from nothing on everything the role does not own, so that a line removed from this file is also removed
-- from the database on the next run.
DO $$
DECLARE rel record;
BEGIN
  FOR rel IN
    SELECT c.oid::regclass AS name, c.relkind AS kind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm', 'S') AND c.relowner <> 'seo_agent'::regrole
  LOOP
    EXECUTE format('REVOKE ALL ON %s %s FROM seo_agent', CASE WHEN rel.kind = 'S' THEN 'SEQUENCE' ELSE 'TABLE' END, rel.name);
  END LOOP;
END $$;

-- CREATE: the agent builds its own search index tables (below). Sequences: ids of the rows it inserts.
GRANT USAGE, CREATE ON SCHEMA public TO seo_agent;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO seo_agent;

-- Read only: published content, SEO measurements, and what the web app decides.
GRANT SELECT ON
  post, setting,
  seo_drafts, seo_llm_batches, seo_url_map,
  seo_gsc_daily_totals, seo_gsc_page_daily, seo_gsc_query_daily, seo_ga4_page_pairs,
  seo_recommendations, seo_recommendation_evidence, seo_query_intents, seo_ml_overrides,
  seo_mcp_tokens,
  news_topics, news_image_profiles
  TO seo_agent;

-- The admin MCP server checks a token's owner: whether that user is still an admin and not banned. Three columns, not
-- the table (no email addresses, no names).
GRANT SELECT (id, role, banned) ON "user" TO seo_agent;
-- …and notes when a grant was last used. It cannot create, widen or un-revoke a grant.
GRANT SELECT ON seo_mcp_grants TO seo_agent;
GRANT UPDATE (last_used_at) ON seo_mcp_grants TO seo_agent;

-- Append only: MCP tool calls are written to the site's audit log and can never be changed or removed from here.
GRANT INSERT ON audit_log TO seo_agent;

-- Configured by an admin in the web app; the agent records run times and fetch state.
GRANT SELECT, UPDATE ON news_jobs, news_sources TO seo_agent;

-- The agent's working tables.
GRANT SELECT, INSERT ON
  seo_llm_usage, seo_rag_eval_runs, seo_ml_labels, seo_ml_query_embeddings,
  news_feedback, news_labels, news_writing_profiles
  TO seo_agent;
GRANT SELECT, INSERT, DELETE ON
  seo_ask_log, seo_ml_decay_signals, seo_ml_intent_predictions, seo_ml_query_clusters, seo_ml_related_articles,
  news_run_logs
  TO seo_agent;
GRANT SELECT, INSERT, UPDATE ON
  seo_flow_runs, seo_ask_gaps, seo_ml_models, seo_ml_weight_proposals,
  news_articles, news_clusters, news_images, news_models, news_run_steps, news_runs
  TO seo_agent;
GRANT SELECT, INSERT, UPDATE, DELETE ON news_candidates, news_items TO seo_agent;

-- The search index. The agent creates seo_article_chunks itself (the vector width follows the embedding model),
-- adds columns to it at start-up and replaces it after a model change, all of which need ownership. Its rows point
-- at post(id). Tables that exist already are handed over; the site's role keeps full access to them and to
-- whatever the agent creates later.
GRANT REFERENCES (id) ON post TO seo_agent;
DO $$
DECLARE
  site_role name;
  tbl text;
BEGIN
  SELECT tableowner INTO STRICT site_role FROM pg_tables WHERE schemaname = 'public' AND tablename = 'post';
  FOREACH tbl IN ARRAY ARRAY['seo_article_chunks', 'seo_article_chunks_next'] LOOP
    IF to_regclass(format('public.%I', tbl)) IS NOT NULL THEN
      EXECUTE format('ALTER TABLE public.%I OWNER TO seo_agent', tbl);
      EXECUTE format('GRANT ALL ON public.%I TO %I', tbl, site_role);
    END IF;
  END LOOP;
  EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE seo_agent IN SCHEMA public GRANT ALL ON TABLES TO %I', site_role);
  EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE seo_agent IN SCHEMA public GRANT ALL ON SEQUENCES TO %I', site_role);
END $$;
