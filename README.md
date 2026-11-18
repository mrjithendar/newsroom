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
`kubernetes/seo-agent-*-triggers.yaml` (`$SEO_AGENT_K8S_DIR` or `../jithendar.com`).

## CI and release

- `.github/workflows/ci.yml` runs on pull requests. It runs lint, types, tests (without the moks-cms contract tests)
  and pip-audit, then builds the image without pushing it and smoke-tests it.
- When a pull request is merged, it builds the arm64 image once, smoke-tests it and pushes
  `registry.jithendar.com/jithendar/seo-agent:latest`. That's all: it tells no site and deploys nothing, so each site
  pulls and restarts the agent itself.
- `.github/workflows/rag-eval.yml` is the manual retrieval evaluation with the production embedding model.

Repository settings (private repository on GitHub Free, so these are set at repository level):

| Name | Kind | What |
|---|---|---|
| `HARBOR_USERNAME`, `HARBOR_PASSWORD` | Actions secret | The Harbor robot that pushes to project `jithendar` |
