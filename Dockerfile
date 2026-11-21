# news-agent (docs/ARCHITECTURE.md). Multi-stage, non-root, CPU-only: no torch, no CUDA; ONNX Runtime runs the
# embeddings the news desk clusters and dedupes with. The one system library is libgomp1 (OpenMP), which the LightGBM
# wheel links against for the ranker (docs/ML.md); scikit-learn/SciPy wheels bundle their own.
# The embeddings model (BAAI/bge-small-en-v1.5, ONNX export, MIT) is baked in at a pinned revision and verified by
# sha256, so pods never download anything at start-up and the image is reproducible.
ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS build
WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --no-cache-dir "uv==0.13.5"
COPY pyproject.toml uv.lock ./
# Exactly the locked versions, production dependencies only.
RUN uv sync --frozen --no-dev --no-install-project
# chromadb 1.5.9 (crewai pins ~=1.5.9 and imports it at load; no fixed release exists) ships a Chroma server whose
# advisories (GHSA-f4j7-r4q5-qw2c, GHSA-2wm9-hf6c-p5cr, GHSA-xph7-9rjv-w5fr, GHSA-36p7-vc44-83pf: code injection
# through the collections API, missing tenant checks, SimpleRBAC) all need that server running. This service only
# imports chromadb's client-side types, so the server is removed from the image: the Rust bindings (the `chroma run`
# server, 50 MB), the Python server, CLI and app, the basic-auth and RBAC providers, and the `chroma` command.
# moks-seo-agent docs/seo/DECISIONS.md D-53 (the same removal; this service imports none of them either).
RUN sp="$(echo /srv/.venv/lib/python3.*/site-packages)" \
 && rm -rf "$sp/chromadb_rust_bindings" "$sp/chromadb/server" "$sp/chromadb/cli" "$sp/chromadb/app.py" \
           "$sp/chromadb/auth/basic_authn" "$sp/chromadb/auth/simple_rbac_authz" /srv/.venv/bin/chroma \
 && CREWAI_TRACING_ENABLED=false /srv/.venv/bin/python -c "import crewai, chromadb.config" \
 && ! /srv/.venv/bin/python -c "import chromadb.server" 2>/dev/null

ARG MODEL_REPO=BAAI/bge-small-en-v1.5
ARG MODEL_REV=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
ARG MODEL_SHA256=828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35
ARG TOKENIZER_SHA256=d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
RUN mkdir -p /models/bge-small-en-v1.5 \
 && python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/onnx/model.onnx', '/models/bge-small-en-v1.5/model.onnx'); u.urlretrieve('https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/tokenizer.json', '/models/bge-small-en-v1.5/tokenizer.json')" \
 && echo "${MODEL_SHA256}  /models/bge-small-en-v1.5/model.onnx" | sha256sum -c - \
 && echo "${TOKENIZER_SHA256}  /models/bge-small-en-v1.5/tokenizer.json" | sha256sum -c -

FROM python:${PYTHON_VERSION}-slim
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /srv/.venv /srv/.venv
COPY --from=build /models /models
COPY app ./app
ENV PATH=/srv/.venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/tmp XDG_DATA_HOME=/tmp XDG_CACHE_HOME=/tmp \
    EMBEDDINGS_MODEL_DIR=/models/bge-small-en-v1.5 \
    TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 OMP_NUM_THREADS=2 MALLOC_ARENA_MAX=2 \
    CREWAI_DISABLE_TELEMETRY=true CREWAI_DISABLE_TRACKING=true CREWAI_TRACING_ENABLED=false CREWAI_DISABLE_VERSION_CHECK=true \
    CREWAI_STORAGE_DIR=/tmp/crewai
USER 10001:10001
# 8080 internal API (web app, the dispatcher CronJob, Prometheus). No MCP edge.
EXPOSE 8080
CMD ["python", "-m", "app.main"]
