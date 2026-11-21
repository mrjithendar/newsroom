# moks-seo-agent

The SEO agent service of [moks-cms](https://github.com/moks-tech/moks-cms). It's written in Python and runs CrewAI
Flows and Crews as the only job engine. It also hosts the MCP servers, pgvector retrieval, Ask the Blog and report
PDFs. It reads and writes the web app's PostgreSQL database and calls the web app's internal callback API. It was
split out of moks-cms (`services/seo-agent`) with its history on 2026-11-18, so `moks-tech/moks-cms#N` in old commit
messages refers to moks-cms pull requests.

| Path | What |
|---|---|
| `app/` | The service: internal API on port 8080 (`app/main.py`), MCP edge on port 8090 (`app/mcp`), flows, crews, tools |
| `tests/` | pytest; `tests/moks_cms.py` finds the moks-cms checkout for the contract tests |
| `scripts/` | RAG eval and calibration, the database role (`db-role.sql`), CI helpers |
| `Dockerfile` | The image: locked dependencies and the bge-small-en-v1.5 ONNX model (checksums verified in the build) |

The design docs are in moks-cms under `docs/seo/` (FLOWS, CREWS, RAG, MCP, ML, OPERATIONS, SECURITY, MANUAL_SETUP,
DECISIONS) and `docs/news/`. A path written as `moks-cms/<path>` in code comments means that file in moks-cms.

## Local development

```bash
uv sync --frozen
uv run ruff check app tests
uv run mypy app
uv run pytest -q -rs
```

Some tests check that the agent and the web app still agree. They read moks-cms's migrations, the callback server's
route table and a few shared constants. They find moks-cms at `$MOKS_CMS_DIR`, or in `../moks-cms` next to this
checkout, and are skipped when it's in neither place. CI has no moks-cms checkout (no cross-repository credentials, by
choice), so run them locally before merging a change that touches the database, the callback API or shared constants.

The Postgres tests need `SEO_AGENT_TEST_DATABASE_URL` (pgvector, PostgreSQL 18) or Docker for testcontainers. The CronJob test reads jithendar.com's
`kubernetes/seo-agent-*.yaml` (`$SEO_AGENT_K8S_DIR` or `../jithendar.com`): one dispatcher CronJob, every minute.

## The LLM budget

One budget for everything (`app/llm_budgets.py`): an optional daily limit and an optional monthly limit in USD, by UTC
day and month. It counts every provider, model and purpose together (drafts, crews, the news desk, Ask the Blog, and
the web app's own calls). An empty limit is no limit, and **nothing is preset**: until a budget is saved, only the
provider's own spend limit stops spending.

- `GET /llm-budget?scope=day|month`: spent, limit, remaining, calls in flight, a month-end forecast, and the period's
  spend by provider, by model and by purpose (for reading; they limit nothing). `PUT /llm-budget`
  `{daily_usd, monthly_usd, actor}` saves it.
- `BudgetGuard` checks it before every call and holds the call's upper-bound estimate while it runs, so concurrent
  calls cannot overspend together.
- The web app's own LLM calls use `POST /llm-budget/reserve` `{max_usd, purpose}` (402 when it does not fit) and
  `POST /llm-budget/settle` `{id}`. A hold that is never settled lapses after 15 minutes.
- Still apart: Ask the Blog's own daily limit (`ASK_BLOG_DAILY_BUDGET_USD`, a guard against readers running up spend)
  and, until the news desk moves over, the news desk's own caps.
- Gone: `SEO_LLM_MONTHLY_BUDGET_USD` (ignored if still set) and the limits per model and per provider. Their rows in
  `seo_agent_llm_budgets` are left in place and shown read-only as `legacy_limits`; `GET /llm-budgets` still answers
  an older web app with spend and no limits, and `PUT /llm-budgets` answers 409.


## LLM providers and models

`app/llm_providers/`: which providers exist (`catalog.py`) and what an admin has set up (`store.py`, tables
`seo_agent_llm_providers` and `seo_agent_llm_models`, created at start-up).

- **Catalog.** Anthropic, OpenAI, Google Gemini, Azure OpenAI, AWS Bedrock, Mistral, xAI and "Custom
  OpenAI-compatible". Each entry was checked against the provider's own docs and cites the page and the date. The
  catalog holds no model list and no price. Named in the plan and not yet checked, so reachable only through
  "Custom" for now: DeepSeek, Groq, Cohere, Perplexity, Together AI, Fireworks AI, OpenRouter, Ollama.
- **Keys.** Sealed with `SEO_AGENT_SETTINGS_KEY` (AES-256-GCM). Write-only: a key is never returned, logged or sent
  to the browser; the API says `saved`, `env`, `none` or `not_needed`. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the
  environment are used when no key is saved.
- **Prices.** Never invented. A model cannot be switched on until its input and output prices (USD per million
  tokens) are entered, unless it is a Claude model with a built-in price (`app/llm.py` `PRICES`) or it is marked free.
- **API.** `GET /llm-providers`, `PUT /llm-providers` (one provider and, when `models` is sent, all of its models;
  validated, all or nothing), `GET /llm-models` (models that are on, for pickers).
- **Not yet.** Nothing calls a provider from these tables: every job still calls Anthropic as before. The adapters,
  "Fetch models", "Test", and the model for each job come in later changes; until then Anthropic cannot be switched
  off here.

## LLM adapters

`app/llm_adapters/`: thin `httpx` clients for providers other than Anthropic (no LiteLLM, no vendor SDKs). An adapter
knows one API shape and nothing about the budget, prices or usage rows; the caller books what it returns.

- `openai_compat.py`: Chat Completions for OpenAI and every compatible server (`Authorization: Bearer`, Azure's
  `api-key` header, or no key for a local server). Structured answers use `response_format` `json_schema` where the
  endpoint supports it, else JSON mode, else plain text; every answer is validated against the Pydantic model here
  and gets one repair request before it is refused. Retries 408/409/429/5xx and connection errors twice. Errors
  carry the provider's message and code, never the key or the prompt. `list_models()` reads `GET /models`.
- Not yet: images, streaming, tools; and **nothing calls an adapter yet**. Wiring behind the budget guard, with the
  admin's prices, is a later change.

## Settings an admin can change

Some settings can be changed from the web app (Admin → AI & SEO → Settings) instead of the deployment's environment.
They live in `app/admin_settings/`:

- **Precedence**: the admin's saved value, then the environment variable, then the default in `app/config.py`. With
  nothing saved, the agent behaves exactly as before.
- **Storage**: the agent's own table `seo_agent_settings` (key = the environment-variable name), created at start-up
  like `seo_agent_llm_budgets`. No moks-cms migration.
- **API** (internal token): `GET /admin-settings` returns every group with its fields, the effective value, where it
  comes from (`admin`, `deployment`, `default`) and whether a restart is pending. `PUT /admin-settings/{group}` takes
  `{"values": {"ENV_NAME": value}, "actor": "..."}`; a field that is not sent stays as it is, and `null`, `""` or `[]`
  clears it. One invalid field refuses the whole save (422 with `issues`).
- **Live or restart**: a `live` field applies at once in this process and within 30 seconds in any other. A `restart`
  field is stored and reported as "restart pending" until the agent restarts.
- **Secrets** are sealed with `SEO_AGENT_SETTINGS_KEY` (32 characters or more; `openssl rand -base64 32`) using the
  same scheme as moks-cms `src/lib/secrets.ts`, with its own key. They are never returned: the API says only whether
  one is set. Without the key, secret fields cannot be saved ("Requires manual configuration"). Rotating the key
  makes stored secrets unreadable.
- **Never registered** (the agent could not start, or the admin could lock itself out): `DATABASE_URL`,
  `SEO_AGENT_DATABASE_URL`, `SEO_AGENT_TOKEN`, `SEO_AGENT_CALLBACK_TOKEN`, `SEO_AGENT_CALLBACK_URL`,
  `SEO_AGENT_SETTINGS_KEY`, `OTEL_*`, `SITE_URL`. The registry refuses them and a test checks it.

### How to add a settings group

1. Add `app/admin_settings/groups/<group>.py` with one `GROUP = Group(id, title, description, fields=(...))`. Each
   `Field` names the environment variable, the `Settings` attribute it feeds (`attr`), a label, a plain hint, a type
   (`text`, `integer`, `number`, `boolean`, `select`, `list`, `secret`), and where they apply a unit, `min`/`max`,
   `options` or a `pattern`. Ranges should match the clamps in `app/config.py`: they refuse a value, they do not set
   one.
2. Append the group to `GROUPS` in `app/admin_settings/groups/__init__.py` (append only: one module per group so
   parallel changes do not collide).
3. Decide `apply` by reading the code: `live` only if every reader takes the value from the shared `Settings` object
   at the time it uses it (`deps.settings.x`, `c.settings.x`). If anything copies it when it is built (a client's
   timeout, a pool size), use `restart`.
4. Nothing else changes: `app/config.py` already parses the variable, and the store lays the saved value over the
   environment before `load_settings` runs.
5. Add the group's fields to `tests/test_admin_settings.py` (which are live, which need a restart).

## CI and release

- `.github/workflows/ci.yml` is one job on one mk4 runner. On every run it runs lint, types, tests (without the
  moks-cms contract tests), the RAG eval smoke and pip-audit.
- A pull request also builds and smoke-tests the image only when it changes an input of the image beyond the Python
  code (`Dockerfile`, `.dockerignore`, `pyproject.toml`, `uv.lock`, the image scripts or size baseline, `ci.yml`). A
  manual run always does.
- When a pull request is merged, the same job runs the checks on the merge commit, then builds the arm64 image,
  smoke-tests it and pushes `registry.jithendar.com/jithendar/seo-agent:latest`. Merges release one at a time. That's
  all: it tells no site and deploys nothing, so each site pulls and restarts the agent itself.
- `.github/workflows/rag-eval.yml` is the manual retrieval evaluation with the production embedding model.

Repository settings (private repository on GitHub Free, so these are set at repository level):

| Name | Kind | What |
|---|---|---|
| `HARBOR_USERNAME`, `HARBOR_PASSWORD` | Actions secret | The Harbor robot that pushes to project `jithendar` |
