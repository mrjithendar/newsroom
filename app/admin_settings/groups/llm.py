"""Settings group "news-agent-llm": the Claude key and the crews' default models.

The key: the news agent's own (ideally of its own Anthropic Console workspace, with that workspace's spend limit).
Saved here it is sealed with NEWS_AGENT_SETTINGS_KEY and wins over ANTHROPIC_API_KEY in Secret news-agent-env; the
client is built at start-up, so a new key applies after a restart. Every call is still held against the one LLM
budget in the seo-agent first (app/seo_agent.py).

The models are the defaults when neither the job (Newsroom → Jobs), the crew configuration (CrewAI console) nor
NEWS_CREW_MODELS names one. Nothing here sets a value: empty means the deployment's value, then config.py's default.
"""

from __future__ import annotations

from ..registry import Field, Group

MODEL = r"claude-[a-z0-9][a-z0-9.-]{2,60}"

GROUP = Group(
    id="news-agent-llm",
    title="Claude",
    description="The news agent's Claude API key and the models its crews use when a job does not name one.",
    fields=(
        Field(
            "ANTHROPIC_API_KEY",
            "anthropic_api_key",
            "Anthropic API key",
            "The news agent's own key (use a workspace of its own in the Anthropic Console, with its spend limit). Stored sealed; it is never shown again. Applies after a restart.",
            "secret",
            apply="restart",
        ),
        Field("NEWS_DRAFT_MODEL", "draft_model", "Default writing model", "The crews' model when the job and the crew configuration name none.", "text", pattern=MODEL),
        Field("NEWS_CLASSIFY_MODEL", "classify_model", "Default classifying model", "For short classification calls when nothing else names a model.", "text", pattern=MODEL),
        Field(
            "NEWS_LLM_TIMEOUT_SECONDS",
            "llm_timeout_s",
            "Model request time limit",
            "How long one request to Claude may take. Applies after a restart.",
            "number",
            apply="restart",
            unit="seconds",
            min=30,
            max=900,
        ),
    ),
)
