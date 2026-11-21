"""crew_test_flow: one crew, an edited configuration, the inputs of an earlier run (the CrewAI console's Test run, D-98).

Like Jenkins Replay: an admin edits a crew's prompts in the console (/admin/crewai/crews/<crew>), picks a finished run
of a flow that uses that crew, and runs the crew again on that run's inputs with the edited configuration, without
saving it. The configuration under test travels in this run's inputs, so the run records exactly which prompts
produced its output; the run page offers to save them as a new version.

What a test run does and does not do:
- It reads the source run's inputs (seo_flow_runs.inputs: the news job and the story) and the story's items and the
  job's writing profile, as news_draft_flow and the profile playground read them.
- It runs the crew with the same limits, budget guard, run history and transcripts as any run.
- It writes nothing anywhere else: no draft and no news article; its output stays on this run (outputs.result) for the
  console to show.

The news agent has one crew, `news`. The SEO crews' test runs stay in the seo-agent (moks-seo-agent/app/flows/crew_test.py);
moks-cms sends each test run to the agent that owns the crew.

One test run per crew at a time (the lock target is the crew).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal

from crewai.flow import persist, start
from pydantic import Field, model_validator

from ..crews import config as crew_config
from ..llm import BudgetExceeded
from ..news import desk as nd
from ..runs import INSTANCE_ONLY
from ..state import UUID, RunState, _In
from .base import SeoFlow
from .news import NewsHelpers

log = logging.getLogger("news_agent.flows")

CrewId = Literal["news"]
OUTPUT_CHARS = 60_000  # a whole article body fits; anything longer is cut in the run's outputs


class CrewTestInput(_In):
    crew: CrewId
    source_run_id: str = Field(pattern=UUID)
    config: dict[str, Any] = Field(default_factory=dict)
    version: int | None = Field(default=None, ge=1, le=1_000_000)  # the saved version under test, if it is one
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _config_fits(self) -> CrewTestInput:
        crew_config.validate(crew_config.registry()[self.crew], self.config)  # OverlayError is a ValueError: 422
        return self


class CrewTestState(RunState):
    crew: str = ""
    source_run_id: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    version: int | None = None
    note: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)
    crew_config: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class ReplayError(Exception):
    """The source run cannot be replayed by this crew (wrong flow, or it stopped before the crew's inputs existed)."""


def _cut(v: Any) -> Any:
    if isinstance(v, str) and len(v) > OUTPUT_CHARS:
        return v[:OUTPUT_CHARS] + f"\n\n[cut at {OUTPUT_CHARS:,} characters]"
    if isinstance(v, dict):
        return {k: _cut(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_cut(x) for x in v]
    return v


@persist(INSTANCE_ONLY)
class CrewTestFlow(NewsHelpers, SeoFlow[CrewTestState]):
    FLOW: ClassVar[str] = "crew_test_flow"

    @start()
    async def replay(self) -> None:
        s = self.state
        async with self.step("load_source"):
            src = await self.deps.store.get(s.source_run_id) if self.deps.store is not None else None
            crew = crew_config.registry()[s.crew]
            if not src:
                s.errors.append("the source run was not found")
                s.outcome = "failed"
                return
            s.source = {"run_id": str(src.get("run_id")), "flow": src.get("flow_name"), "number": src.get("run_number"), "status": src.get("status")}
            if src.get("flow_name") not in crew.replay:
                s.errors.append(f"a {src.get('flow_name')} run cannot be replayed by the {s.crew} crew (it replays {', '.join(crew.replay)})")
                s.outcome = "failed"
                return
        token = crew_config.CREW_OVERRIDE.set({s.crew: {"config": s.config, "version": s.version}})
        try:
            async with self.step(f"crew_{s.crew}"):
                eff = await crew_config.load(s.crew, None)
                s.crew_config = eff.summary()
                try:
                    s.result = _cut(await self._run(s.crew, src))
                    s.outcome = "done" if s.result.get("status") == "completed" else "incomplete"
                except ReplayError as e:
                    s.errors.append(str(e))
                    s.outcome = "failed"
                except BudgetExceeded as e:
                    self.add_budget_cost(e)
                    s.errors.append(f"budget: {e.detail}")
                    s.outcome = "paused_budget"
        finally:
            crew_config.CREW_OVERRIDE.reset(token)

    async def _run(self, crew: str, src: dict[str, Any]) -> dict[str, Any]:
        if crew == "news":
            return await self._news(src.get("inputs") or {})
        raise ReplayError(f"unknown crew {crew}")

    async def _news(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """The story's items and the job's writing profile, as news_draft_flow and the profile playground read them."""
        job_id = inputs.get("job_id")
        cluster_id = inputs.get("cluster_id") or next(iter(inputs.get("cluster_ids") or []), None)
        if not job_id or not cluster_id:
            raise ReplayError("the source run names no news job and story")
        d = await self.desk()
        job = nd.as_job(await d.store.get_job(str(job_id)) or {"id": str(job_id), "name": "test run"})
        items = (await d.store.items_by_cluster([str(cluster_id)])).get(str(cluster_id), [])[:12]
        if not items:
            raise ReplayError("the story's items are gone (news items are kept for a limited time)")
        wp = await d.store.writing_profile(job.writing_profile_key, job.config.writing_profile_version) or {"key": job.writing_profile_key, "version": 0, "required_sections": ["Why it matters"], "banned_phrases": []}
        cluster = await d.store.get_cluster(str(cluster_id)) or {"title": "story"}
        mode = "digest" if inputs.get("mode") == "digest" else "single"
        crew = (d.crew_factory or (lambda lm: nd.NewsDeskCrew(d.settings, lm, d.web, d.retriever)))(self.deps.llm)
        brief = {"mode": mode, "title": cluster.get("title") or "story", "cluster_id": str(cluster_id), "length_words": [job.config.length_min_words, job.config.length_max_words]}
        r = await crew.run(brief, items=items, writing_profile=wp, image_profile={}, allow_hosts=nd._allowed_hosts(items, d), max_tokens=job.config.budget.max_tokens_per_draft, models=job.config.models)
        self.state.add_cost(r.cost_usd, r.tokens)
        dr = r.draft
        return {"status": r.status, "reason": dr.incomplete_reason, "tokens": r.tokens, "cost_usd": r.cost_usd, "draft": {"title": dr.title, "dek": dr.dek, "meta_description": dr.meta_description, "body_mdx": dr.body_mdx, "sources": [x.model_dump() for x in dr.sources], "flagged": [f.model_dump() for f in dr.fact_check_report.flagged]}}
