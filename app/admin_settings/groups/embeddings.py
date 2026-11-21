"""Settings group "Embeddings": how the embeddings backend is reached.

Only the connection lives here. Which provider and which model embed the articles (EMBEDDINGS_PROVIDER,
EMBEDDINGS_MODEL_DIR, OLLAMA_EMBED_MODEL, OLLAMA_EMBED_DIM) is deliberately not editable yet: changing one makes
the stored news item vectors stale until they are embedded again, and that needs a guarded change (cost shown,
confirm, re-embed, the old index serving until the new one passes the calibration gates). Until that exists they
stay in the deployment.

The embeddings client is built once at start-up, so every field here applies on restart.

Forked from moks-seo-agent/app/admin_settings/groups/embeddings.py@6d6464b on 2026-11-21.
"""

from __future__ import annotations

from ..registry import Field, Group

URL = r"https?://[^\s/@?#]+(?::\d{1,5})?(/[^\s?#]*)?"

GROUP = Group(
    id="embeddings",
    title="Embeddings",
    description="Where the embeddings backend is and how long a request may take. The provider and the model stay in the deployment for now: changing them needs every article to be embedded again.",
    fields=(
        Field("OLLAMA_URL", "ollama_url", "Ollama address", "Used when the embeddings provider is Ollama. For example http://ollama.ai.svc.cluster.local:11434.", "text", apply="restart", pattern=URL),
        Field("EMBEDDINGS_API_URL", "embeddings_api_url", "Embeddings API address", "Used when the embeddings provider is an HTTP API.", "text", apply="restart", pattern=URL),
        Field("EMBEDDINGS_API_KEY", "embeddings_api_key", "Embeddings API key", "Sent to the embeddings API. Stored sealed; it is never shown again.", "secret", apply="restart"),
        Field(
            "EMBEDDINGS_TIMEOUT_SECONDS",
            "embeddings_timeout_s",
            "Embeddings request time limit",
            "How long one embeddings request may take. A large model on a CPU needs minutes for a batch.",
            "number",
            apply="restart",
            unit="seconds",
            min=10,
            max=1800,
        ),
    ),
)
