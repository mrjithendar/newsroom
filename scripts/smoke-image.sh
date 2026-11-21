#!/usr/bin/env bash
# Start the news-agent image against a database without the web app's tables and check that it answers on its port
# (internal API 8080), keeps its guarded endpoints closed, refuses what no news flow may do, and can embed text with
# the bundled ONNX model. Never push an image that fails this.
#
#   scripts/smoke-image.sh <image> <database url>
#
# <database url> is as seen from inside the container (CI: the job's PostgreSQL service via host.docker.internal).
set -euo pipefail
IMAGE="${1:?usage: $0 <image> <database url>}"
DATABASE_URL="${2:?usage: $0 <image> <database url>}"
name="news-agent-smoke-${GITHUB_RUN_ID:-local}-$$"

docker run -d --name "$name" -e NEWS_AGENT_TOKEN=ci -e DATABASE_URL="$DATABASE_URL" \
  --add-host host.docker.internal:host-gateway -p 8080:8080 "$IMAGE" >/dev/null
trap 'docker logs "$name" 2>&1 | tail -50; docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
# A fresh database without the web app's tables: the service must still start and say so on /readyz.
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8080/healthz && break; sleep 2; done
curl -fsS http://127.0.0.1:8080/healthz | grep -q '"agent":"news"'
curl -sS http://127.0.0.1:8080/readyz | tee /dev/stderr | grep -q '"status"'
curl -fsS http://127.0.0.1:8080/metrics | grep -q seo_flow_runs_total
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8080/flows/news_job_flow/kickoff -H 'Content-Type: application/json' -d '{}')" = 401
# No publishing and no SEO flows here, even with the service token.
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8080/flows/publish_flow/kickoff -H 'Authorization: Bearer ci' -H 'Content-Type: application/json' -d '{"inputs":{}}')" = 404
curl -fsS http://127.0.0.1:8080/flows -H 'Authorization: Bearer ci' | grep -q '"agent":"news"'
docker exec "$name" python -c "from app.embeddings import OnnxEmbeddings; import asyncio; e=OnnxEmbeddings('/models/bge-small-en-v1.5'); print(len(asyncio.run(e.embed(['ok']))[0]))" | grep -qx 384
trap - EXIT
docker rm -f "$name" >/dev/null
echo "Image healthy: $IMAGE"
