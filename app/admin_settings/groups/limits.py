"""Settings group "news-agent-limits": how long and how much a news crew or flow may run.

Defaults are the ones in app/config.py; nothing here sets a value. The ranges only refuse values the code could not
work with (config.py's own clamps).
"""

from __future__ import annotations

from ..registry import Field, Group

GROUP = Group(
    id="news-agent-limits",
    title="Limits",
    description="How many turns, requests, tokens and seconds a news crew or flow may use.",
    fields=(
        Field("CREW_MAX_ITER", "crew_max_iter", "Turns per agent", "The most turns one crew agent takes on a task. An agent's own setting in the crew configuration wins.", "integer", unit="turns", min=1, max=20),
        Field("CREW_MAX_RPM", "crew_max_rpm", "Model requests per minute, per crew", "A crew waits rather than send more than this many model requests in a minute.", "integer", unit="requests per minute", min=1, max=600),
        Field("CREW_MAX_TOKENS_PER_RUN", "crew_max_tokens_per_run", "Tokens per crew run", "A crew run stops as incomplete once it has used this many tokens. A job's own token limit per draft comes on top.", "integer", unit="tokens", min=1000, max=10_000_000),
        Field("CREW_TIMEOUT_SECONDS", "crew_timeout_s", "Crew run time limit", "A crew run that takes longer stops as incomplete.", "integer", unit="seconds", min=60, max=7200),
        Field("NEWS_FLOW_TIMEOUT_SECONDS", "flow_timeout_s", "Flow run time limit", "The wall clock for one flow run. Applies after a restart.", "integer", apply="restart", unit="seconds", min=60, max=21_600),
        Field("SCHEDULE_CATCHUP_MINUTES", "schedule_catchup_min", "Missed schedule catch-up", "How late a missed schedule time still runs, e.g. after the agent was down.", "integer", unit="minutes", min=2, max=1440),
    ),
)
