#!/usr/bin/env bash
# Start the seo-agent image against a database without the web app's tables and check that it answers on both ports
# (internal API 8080, MCP edge 8090), keeps its guarded endpoints closed, and can render a PDF and embed text with the
# bundled ONNX model. Never push an image that fails this.
#
#   scripts/smoke-image.sh <image> <database url>
#
# <database url> is as seen from inside the container (CI: the job's PostgreSQL service via host.docker.internal).
set -euo pipefail
IMAGE="${1:?usage: $0 <image> <database url>}"
DATABASE_URL="${2:?usage: $0 <image> <database url>}"
name="seo-agent-smoke-${GITHUB_RUN_ID:-local}-$$"

docker run -d --name "$name" -e SEO_AGENT_TOKEN=ci -e DATABASE_URL="$DATABASE_URL" \
  --add-host host.docker.internal:host-gateway -p 8080:8080 -p 8090:8090 "$IMAGE" >/dev/null
trap 'docker logs "$name" 2>&1 | tail -50; docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
# A fresh database without the web app's tables: the service must still start and say so on /readyz.
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8080/healthz && break; sleep 2; done
curl -sS http://127.0.0.1:8080/readyz | tee /dev/stderr | grep -q '"status"'
curl -fsS http://127.0.0.1:8080/metrics | grep -q seo_flow_runs_total
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8080/flows/gsc_sync_flow/kickoff -H 'Content-Type: application/json' -d '{}')" = 401
# MCP edge: up, and both servers switched off by default (404) until PUBLIC_/ADMIN_MCP_ENABLED.
curl -fsS http://127.0.0.1:8090/healthz | grep -q '"public":false'
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8090/mcp/public -H 'Host: jithendar.com' -H 'Content-Type: application/json' -d '{}')" = 404
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8090/mcp/admin -H 'Host: jithendar.com' -H 'Content-Type: application/json' -d '{}')" = 404
# The publish path is not reachable by the generic kickoff, even with the service token.
test "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8080/flows/publish_flow/kickoff -H 'Authorization: Bearer ci' -H 'Content-Type: application/json' -d '{"inputs":{"draft_id":"00000000-0000-4000-8000-000000000000","approval_id":"x","accepted_change_ids":["c1"]}}')" = 403
docker exec "$name" python -c "from app.reports_pdf import render; assert render({'type':'weekly','period':{'start':'2026-11-02','end':'2026-11-08'}}).startswith(b'%PDF-')"
docker exec "$name" python -c "from app.embeddings import OnnxEmbeddings; import asyncio; e=OnnxEmbeddings('/models/bge-small-en-v1.5'); print(len(asyncio.run(e.embed(['ok']))[0]))" | grep -qx 384
trap - EXIT
docker rm -f "$name" >/dev/null
echo "Image healthy: $IMAGE"
