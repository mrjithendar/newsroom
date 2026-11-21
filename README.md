# newsroom (news-agent)

The news desk's engine for [moks-cms](https://github.com/mrjithendar/moks-cms): configurable news jobs that find
stories in trusted sources, rank them, have a crew of agents research, write, fact-check and edit each one, check the
result in code, make a cover, and put the draft in the web app's review queue. **Nothing publishes from here**: a
person approves a draft in the web app, and the seo-agent's `publish_flow` publishes it.

It was split out of [moks-seo-agent](https://github.com/mrjithendar/moks-seo-agent) with its history on 2026-11-21, so
old commit messages refer to the seo-agent and to moks-cms pull requests. The image is
`registry.jithendar.com/jithendar/news-agent:latest`; the Deployment, Service and CronJob are called `news-agent`.

```
moks-cms (web app)                     news-agent (this repository)                 seo-agent
  Newsroom admin, review queue  ──────▶ POST /flows/news_*/kickoff  (NEWS_AGENT_TOKEN)
  callbacks :9465 /news/*, …   ◀────── fetch, drafts, media, notify (NEWS_AGENT_CALLBACK_TOKEN)
                                        site search, the one LLM budget ─────────────▶ POST /search, /llm-budget/*
                                                                                         (SEO_AGENT_NEWS_TOKEN)
Shared PostgreSQL: news_* tables (moks-cms migrations), seo_flow_runs + run history, seo_flow_schedules, seo_llm_usage
```

| Path | What |
|---|---|
| `app/` | The service: internal API on port 8080 (`app/main.py`), the news flows (`app/flows/news.py`), the news desk (`app/news/`), the crew plumbing (`app/crews/`) |
| `app/seo_agent.py` | The two things it asks the seo-agent for: site search and the one LLM budget |
| `tests/` | pytest; `tests/moks_cms.py` finds the moks-cms checkout for the contract tests |
| `scripts/` | The database role (`db-role.sql`), CI helpers |
| `docs/` | Architecture, ML, quality gates, images, costs, Prometheus alerts. The admin's docs (jobs, sources, set-up, runbook) are in moks-cms `docs/news/` |
| `Dockerfile` | The image: locked dependencies and the bge-small-en-v1.5 ONNX model (checksums verified in the build) |

The core (`app/llm.py`, `runs.py`, `run_history.py`, `schedules.py`, `crews/`, `flows/base.py`, …) is a fork of the
seo-agent's at bb9e553 (2026-11-21), cut down to what the news desk uses; each forked file says so at the top. There is
no shared package (it would need a credential between the two private repositories), so a fix to that code is made in
both repositories. A path written as `moks-cms/<path>` or `moks-seo-agent/<path>` in code comments means that file there.

## What it shares, and how it stays out of the seo-agent's way

- **Flow runs** (`seo_flow_runs`, `seo_flow_run_steps/logs/calls`): both agents write them, so the CrewAI console shows
  every run. Each agent reads, counts, recovers after a restart and cancels **only its own flows** (`app/runs.py`).
- **Schedules** (`seo_flow_schedules`, edited in the CrewAI console): each agent has its own dispatcher CronJob and
  claims only the rows of its own flows (`app/schedules.py`). The news rows are `news-dispatch`, `news-train` and
  `news-style-review`.
- **The one LLM budget** (Admin → AI & SEO → Settings → LLM budget) is the seo-agent's. Before every model call this
  service holds the call's upper bound there and lets go once its usage row is in `seo_llm_usage` (purpose `news.*`).
  Not set up → no budget check (the Anthropic Console workspace's limit is then the only one); set up but not
  answering → the call is refused. Image spend keeps the news desk's own caps (Newsroom → Settings).
- **Site search**: the blog's index stays in the seo-agent; `POST /search` there. When it can't be reached, site match
  is skipped and the run goes on.

## Environment

| Variable | What |
|---|---|
| `NEWS_AGENT_TOKEN` | Bearer the web app (and the dispatcher CronJob) sends. ≥ 32 characters in production. |
| `NEWS_AGENT_CALLBACK_TOKEN`, `NEWS_AGENT_CALLBACK_URL` | This service → the web app's callback API (default `http://jithendar-com-internal.jithendar:9465`) |
| `SEO_AGENT_URL`, `SEO_AGENT_NEWS_TOKEN` | This service → the seo-agent (site search, LLM budget). Both empty: neither. |
| `DATABASE_URL`, `NEWS_AGENT_DATABASE_URL` | The shared database; the second (its own role, `scripts/db-role.sql`) wins when set |
| `ANTHROPIC_API_KEY` | Claude. Use a key of its own Anthropic Console workspace, with that workspace's spend limit. |
| `OPENAI_API_KEY` | Only for the `openai` cover provider |
| `NEWS_DESK_ENABLED` | `false` forces the desk off whatever the admin switch says; unset lets the admin decide |
| `NEWS_IMAGE_PROVIDER`, `NEWS_OPENAI_IMAGE_MODEL` | Cover provider override and the OpenAI image model |
| `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL_DIR`, `OLLAMA_URL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_EMBED_DIM`, `EMBEDDINGS_API_URL`, `EMBEDDINGS_API_KEY`, `EMBEDDINGS_TIMEOUT_SECONDS` | The embeddings the desk clusters and dedupes with (default: the ONNX model in the image) |
| `NEWS_DRAFT_MODEL`, `NEWS_CLASSIFY_MODEL`, `NEWS_CREW_MODELS`, `NEWS_CREW_MAX_ITERS` | Crew defaults; jobs and the crew configuration in the console override them |
| `CREW_MAX_TOKENS_PER_RUN`, `CREW_TIMEOUT_SECONDS`, `CREW_MAX_ITER`, `CREW_MAX_RPM`, `NEWS_LLM_TIMEOUT_SECONDS`, `NEWS_FLOW_TIMEOUT_SECONDS`, `SCHEDULE_CATCHUP_MINUTES` | Limits |
| `NEWS_AGENT_DOCS_ALLOWLIST` | Extra documentation hosts a researcher may fetch |
| `APP_ENV`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `LOG_LEVEL` | Runtime |

Everything else the desk does (sources, jobs, profiles, topics, caps, models, thresholds) is set in the web app's
Admin → Newsroom and read from the database.

## Local development

```bash
uv sync --frozen
uv run ruff check app tests
uv run mypy app
uv run pytest -q -rs
```

The contract tests read moks-cms (its migrations, the callback server's route table and the news agent's allowed
routes, the skills list). They find it at `$MOKS_CMS_DIR` or in `../moks-cms`, and are skipped otherwise; CI has no
moks-cms checkout (no cross-repository credentials, by choice), so run them locally when a change touches the callbacks,
the news tables or the skills. `NEWS_AGENT_TEST_DATABASE_URL` (a PostgreSQL superuser URL) runs the real-database tests:
the news store against every migration, the run-store scoping and the database role. On a Mac without Homebrew's
`libomp` the LightGBM ranker tests fail to load the library; CI (Linux) runs them.

## Database role

`scripts/db-role.sql` creates `news_agent` with only what the SQL in `app/` needs (the news tables, its flow runs and
run history, the schedules' firing columns, LLM usage rows; read-only posts, settings and review drafts). Run it as a
superuser in the site's database, give the role a password, put its URL in Secret `news-agent-db-credentials` (key
`uri`) and restart `deploy/news-agent`. `tests/test_db_role.py` keeps the script and the code in step.

## CI and release

`.github/workflows/ci.yml` is one job on the mk4 runners: ruff, mypy, pytest with PostgreSQL, pip-audit; the image is
built and smoke-tested when its inputs change, and on a merge into `master` it is pushed as `news-agent:latest`. Nothing
deploys from here and no site is told: jithendar.com's manifests run `:latest`, so a release reaches the site with
`kubectl -n jithendar rollout restart deploy/news-agent` (or its deploy workflow). The repository needs only the
`HARBOR_USERNAME` / `HARBOR_PASSWORD` secrets.
