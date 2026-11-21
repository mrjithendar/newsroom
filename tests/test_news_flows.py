"""News desk flows end to end with a fake web app, deterministic model and fake crew (real CrewAI Flow engine)."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.news.images import Brief, clamp_alt, cover_variants, render_template
from app.runs import RunError
from tests.news_fixtures import JOB, NOW, SRC_BAD, FakeNewsCrew, NewsWeb, good_draft, news_container, png, seed

SAFE = {"safe": True, "faces": False, "logos_or_trademarks": False, "text_artifacts": False, "brief_match": 0.8, "alt_text": "Abstract blue grid of connected nodes with a highlighted container block expanding in the centre of the frame", "decorative": False}
UNSAFE = {**SAFE, "safe": False, "faces": True, "brief_match": 0.95}


def story_embed() -> Any:
    """Deterministic embeddings: feed items titled "Kubernetes …" (the two fixture articles about the same release) land
    close together; everything else (draft paragraphs in the similarity gate) uses the hashed bag-of-words model."""
    from app.embeddings import HashEmbeddings

    h = HashEmbeddings(32)

    async def embed(texts: list[str]) -> list[list[float]]:
        base = await h.embed(texts)
        out = []
        for t, b in zip(texts, base, strict=True):
            if t.startswith("Kubernetes") and "pod resize" in t.lower():
                v = np.random.default_rng(1).normal(size=32) + np.random.default_rng(len(t)).normal(scale=0.05, size=32)
                out.append((v / np.linalg.norm(v)).tolist())
            else:
                out.append(b)
        return out

    return embed


def setup(llm: Any, *, crew: FakeNewsCrew | None = None, enabled: bool = True, image_provider: str = "template", job_cfg: dict[str, Any] | None = None, web: NewsWeb | None = None) -> tuple[Any, NewsWeb, Any]:
    web = web or NewsWeb()
    c = news_container(llm, web, crew or FakeNewsCrew(), enabled=enabled)
    c.deps.extra["news_embed"] = story_embed()
    store = c.deps.extra["news_store"]
    seed(store, job_cfg=job_cfg, image_provider=image_provider)
    return c, web, store


async def run(c: Any, flow: str, inputs: dict[str, Any]) -> dict[str, Any]:
    r = await c.runner.kickoff(flow, inputs, triggered_by="manual")
    return await c.runner.wait(r["run_id"], 60)


def news_run(store: Any, flow_run: dict[str, Any]) -> dict[str, Any]:
    return store.runs[flow_run["state_json"]["news_run_id"]]


async def test_happy_path_drafts_one_story_from_two_sources(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    c, web, store = setup(llm)
    fr = await run(c, "news_job_flow", {"job_id": JOB})
    assert fr["status"] == "succeeded", fr
    nr = news_run(store, fr)
    assert nr["status"] == "succeeded" and nr["job_version"] == 3
    steps = [s["step"] for s in store.step_rows]
    for s in ("fetch", "dedupe", "classify", "score", "select", "draft", "quality", "image", "save"):
        assert s in steps, steps
    assert nr["counters"]["fetched"] == 2 and nr["counters"]["source_errors"] == 1  # the broken source did not fail the run
    assert store.srcs[SRC_BAD]["consecutive_errors"] == 1 and "error" in store.srcs[SRC_BAD]["last_status"]
    clusters = {i["cluster_id"] for i in store.items.values()}
    assert len(clusters) == 1  # same story, two sources → one cluster
    assert len(web.saved) == 1
    doc = web.saved[0]["document"]
    assert doc["news"]["job_id"] == JOB and doc["news"]["quality"]["passed"] and doc["news"]["image"]["provider"] == "template"
    assert doc["body_mdx"] and doc["sources"] and web.saved[0]["categories"] == ["kubernetes"]
    img = doc["news"]["image"]
    assert 80 <= len(img["alt"]) <= 125 and img["aiGenerated"] is False and set(web.media[0]["files"]) == {"og", "hero", "w800", "w400"}
    art = next(iter(store.articles.values()))
    assert art["status"] == "ready_for_review" and art["draft_id"] and art["generated_doc"]
    cands = store.cand_rows
    assert len(cands) == 1 and cands[0]["decision"] == "selected" and set(cands[0]["breakdown"]["features"]) >= {"credibility", "recency", "novelty"}
    assert any(row["level"] == "warning" and "Broken feed" in row["message"] for row in store.log_rows)
    # Nothing published, ever.
    assert web.published == [] and not any(k == "publish" for k, _ in web.calls)
    assert not any(r["flow_name"] == "publish_flow" for r in c.store.rows.values())


async def test_dry_run_creates_nothing_and_spends_no_image_budget(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    c, web, store = setup(llm, image_provider="gemini")
    web.gemini = [png()]
    fr = await run(c, "news_job_flow", {"job_id": JOB, "dry_run": True})
    nr = news_run(store, fr)
    assert nr["dry_run"] and nr["trigger"] == "dry_run" and nr["status"] == "succeeded"
    assert web.saved == [] and web.media == [] and web.gemini  # provider never called
    assert nr["image_cost_usd"] == 0
    a = next(iter(store.articles.values()))
    assert a["status"] == "dry_run" and a["generated_doc"]["news"]["image"]["stored"] is False


async def test_dry_run_without_llm_selects_only(llm: Any) -> None:
    crew = FakeNewsCrew()
    c, web, store = setup(llm, crew=crew)
    fr = await run(c, "news_job_flow", {"job_id": JOB, "dry_run": True, "llm": False})
    assert news_run(store, fr)["counters"]["selected"] == 1 and crew.runs == [] and web.saved == []


async def test_rerun_is_idempotent_and_a_slot_runs_once(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    c, web, store = setup(llm)
    await run(c, "news_job_flow", {"job_id": JOB, "slot": "20261110T0130Z", "trigger": "cron"})
    again = await run(c, "news_job_flow", {"job_id": JOB, "slot": "20261110T0130Z", "trigger": "cron"})
    assert again["outputs"]["outcome"] == "skipped"
    # Sources were fetched within their interval: a manual rerun fetches nothing new and drafts nothing new.
    rerun = await run(c, "news_job_flow", {"job_id": JOB})
    nr = news_run(store, rerun)
    assert len(web.saved) == 1 and nr["counters"].get("selected", 0) == 0 and nr["counters"].get("sources_skipped_interval") == 3


async def test_one_active_run_per_job(llm: Any) -> None:
    c, _web, _store = setup(llm)
    a = await c.runner.kickoff("news_job_flow", {"job_id": JOB}, triggered_by="manual")
    b = await c.runner.kickoff("news_job_flow", {"job_id": JOB}, triggered_by="manual")
    assert b["existing"] and b["run_id"] == a["run_id"]
    await c.runner.wait(a["run_id"], 60)


async def test_quality_failure_is_saved_with_reasons_not_queued(llm: Any) -> None:
    from tests.news_fixtures import K8S_TEXT

    bad = good_draft().model_copy(update={"body_mdx": K8S_TEXT + "\n\n## Why it matters\n\nThis matters because the project counts 99 enhancements now, and teams can resize running pods without any restarts at all in production clusters."})
    c, web, store = setup(llm, crew=FakeNewsCrew([bad]))
    fr = await run(c, "news_job_flow", {"job_id": JOB})
    a = next(iter(store.articles.values()))
    assert a["status"] == "failed_quality" and web.saved == []
    assert set(a["quality"]["failed"]) >= {"source_similarity", "grounding"} and a["similarity"]["detail"]["top_matches"]
    assert fr["state_json"]["children"][0]["outcome"] == "failed_quality"


async def test_crew_limit_saves_an_incomplete_draft_for_review(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    partial = good_draft().model_copy(update={"incomplete": True, "incomplete_reason": "crew token cap 120000 reached"})
    c, web, store = setup(llm, crew=FakeNewsCrew([partial]))
    await run(c, "news_job_flow", {"job_id": JOB})
    a = next(iter(store.articles.values()))
    assert a["status"] == "incomplete" and web.saved[0]["flags"][0]["code"] == "incomplete" and web.saved[0]["flags"][0]["severity"] == "error"


async def test_image_safety_rejection_falls_back_to_the_template(llm: Any) -> None:
    llm.queue("news.vision", UNSAFE, UNSAFE, SAFE)  # two generated variants (both show a face) → template → alt check
    c, web, store = setup(llm, image_provider="gemini")
    web.gemini = [png(), png(color=(200, 50, 50))]
    await run(c, "news_job_flow", {"job_id": JOB})
    imgs = list(store.image_rows.values())
    assert [i["status"] for i in imgs].count("rejected_safety") == 2
    chosen = [i for i in imgs if i["selected"]]
    assert len(chosen) == 1 and chosen[0]["provider"] == "template" and not chosen[0]["ai_generated"]
    nr = next(iter(store.runs.values()))
    assert nr["image_cost_usd"] == pytest.approx(0.08)  # both generated images were paid for


async def test_safe_ai_image_is_marked_ai_generated(llm: Any) -> None:
    llm.queue("news.vision", SAFE, {**SAFE, "brief_match": 0.9}, SAFE)
    c, web, _store = setup(llm, image_provider="gemini")
    web.gemini = [png(), png(color=(10, 120, 60))]
    await run(c, "news_job_flow", {"job_id": JOB})
    assert web.media[0]["aiGenerated"] is True
    og = web.media[0]["files"]["og"]["data"]
    import base64

    assert b"trainedAlgorithmicMedia" in base64.b64decode(og)


async def test_budget_cap_pauses_the_job_without_drafting(llm: Any) -> None:
    from app.llm_budgets import Budget

    # The one LLM budget is used up (there is no per-job news LLM cap any more; an old value in the job is ignored).
    c, web, store = setup(llm, job_cfg={"budget": {"llm_daily_usd": 100}})
    llm.guard.budget.value = Budget(daily_usd=0.01)
    llm.guard.store.preexisting_usd = 0.02
    fr = await run(c, "news_job_flow", {"job_id": JOB})
    assert fr["status"] == "paused_budget"
    nr = news_run(store, fr)
    assert nr["status"] == "paused_budget" and "budget" in nr["error_summary"] and web.saved == [] and store.step_rows == []


async def test_budget_reached_mid_draft_pauses_that_draft(llm: Any) -> None:
    from app.llm import BudgetExceeded

    c, web, store = setup(llm, crew=FakeNewsCrew([BudgetExceeded("news_job_llm_day", "this job's daily LLM budget $1.00 would be exceeded")]))
    fr = await run(c, "news_job_flow", {"job_id": JOB})
    assert fr["status"] == "paused_budget" and next(iter(store.articles.values()))["status"] == "paused_budget" and web.saved == []


async def test_kill_switch_off_runs_nothing(llm: Any) -> None:
    c, web, store = setup(llm, enabled=False)
    fr = await run(c, "news_job_flow", {"job_id": JOB})
    assert news_run(store, fr)["status"] == "skipped" and not any(k == "news_fetch" for k, _ in web.calls)
    d = await run(c, "news_dispatch_flow", {})
    assert d["outputs"]["outcome"] == "skipped"


async def test_web_validation_errors_end_as_failed_quality(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    web = NewsWeb(save_errors=["title: another published article already uses this title"])
    c, _w, store = setup(llm, web=web)
    await run(c, "news_job_flow", {"job_id": JOB})
    a = next(iter(store.articles.values()))
    assert a["status"] == "failed_quality" and "web_validation" in a["quality"]["failed"]


async def test_digest_mode_writes_one_article_for_several_stories(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    web = NewsWeb()
    from tests.news_fixtures import rss

    web.feeds["https://feeds.example.org/k8s.xml"] = rss([("Kubernetes 1.35 released with stable in-place pod resize", "https://blog.example.org/k8s-135", "pod resize story"), ("Terraform adds a new provider lock format", "https://blog.example.org/tf", "terraform kubernetes provider lock file format change for platform teams")])
    crew = FakeNewsCrew()
    c, _w, store = setup(llm, crew=crew, web=web, job_cfg={"min_score": 0.0})
    store.jobs[JOB]["mode"] = "daily_digest"
    await run(c, "news_job_flow", {"job_id": JOB})
    assert crew.runs and crew.runs[0]["brief"]["mode"] == "digest" and len(crew.runs[0]["brief"]["digest_stories"]) >= 2
    assert len([a for a in store.articles.values() if a["mode"] == "digest"]) == 1


async def test_retry_only_redrafts_the_failed_stories(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    c, web, store = setup(llm, crew=FakeNewsCrew([RuntimeError("model overloaded"), good_draft()]))
    first = await run(c, "news_job_flow", {"job_id": JOB})
    assert first["state_json"]["children"][0]["outcome"] == "failed" and web.saved == []
    rid = first["state_json"]["news_run_id"]
    retry = await run(c, "news_job_flow", {"job_id": JOB, "retry_of": rid, "trigger": "retry"})
    assert len(web.saved) == 1 and news_run(store, retry)["parent_run_id"] == rid


async def test_dispatcher_starts_due_jobs_once_per_slot(llm: Any) -> None:
    llm.queue("news.vision", SAFE)
    c, _web, store = setup(llm)
    store.jobs[JOB]["cron"] = "0 6 * * *"
    store.jobs[JOB]["timezone"] = "UTC"
    d1 = await run(c, "news_dispatch_flow", {})
    due = d1["state_json"]["result"]["due"]
    assert len(due) == 1 and due[0]["slot"] == "20261110T0600Z"
    await c.runner.wait(due[0]["run_id"], 60)
    d2 = await run(c, "news_dispatch_flow", {})
    child = await c.runner.wait(d2["state_json"]["result"]["due"][0]["run_id"], 60)
    assert child["outputs"]["outcome"] == "skipped"  # same slot: the second dispatch is a no-op
    assert store.jobs[JOB]["next_run_at"] is not None


async def test_regenerate_cover_from_review(llm: Any) -> None:
    llm.queue("news.vision", SAFE, SAFE)
    c, web, store = setup(llm)
    await run(c, "news_job_flow", {"job_id": JOB})
    aid = next(iter(store.articles))
    r = await run(c, "news_image_flow", {"article_id": aid, "prompt_tweak": "more green"})
    assert r["status"] == "succeeded" and web.image_updates and web.image_updates[0][1]["alt"]


async def test_train_flow_trains_the_classifier_and_skips_the_ranker_without_feedback(llm: Any) -> None:
    c, _web, store = setup(llm)
    rng = np.random.default_rng(0)
    for i in range(40):
        k8s = i % 2 == 0
        iid = f"item{i}"
        v = rng.normal(size=32) + (5 if k8s else -5)
        store.items[iid] = {"id": iid, "title": "kubernetes pod" if k8s else "cve exploit", "excerpt": "", "clean_text": "", "embedding": (v / np.linalg.norm(v)).tolist(), "entities": {}, "fetched_at": NOW, "source_id": "s"}
        store.labels.append({"item_id": iid, "topics": ["kubernetes" if k8s else "cybersecurity"], "source": "human"})
    fr = await run(c, "news_train_flow", {})
    res = fr["state_json"]["result"]
    assert res["classifier"]["trained"] and res["classifier"]["promoted"] and not res["ranker"]["trained"]
    assert store.models[-1]["kind"] == "classifier" and store.models[-1]["active"]


async def test_style_review_proposes_but_never_activates(llm: Any) -> None:
    llm.queue("news.style", {"explanation": "The owner shortens intros.", "system_style": "Lead with the change in one sentence.", "examples": [{"before": "x", "after": "y", "why": "z"}]})
    c, _web, store = setup(llm)
    store.articles["a1"] = {"id": "a1", "status": "published", "post_id": "p1", "generated_doc": {"body_mdx": "Long intro sentence one.\n\nBody."}, "created_at": NOW, "published_at": NOW}
    store.posts["p1"] = {"body": "Short intro.\n\nBody.", "title": "t"}
    fr = await run(c, "news_style_review_flow", {"profile_key": "news-default"})
    assert fr["state_json"]["result"]["news-default"]["proposed"]
    v2 = next(w for w in store.writing if w["key"] == "news-default" and w["version"] == 2)
    assert v2["status"] == "proposed" and next(w for w in store.writing if w["status"] == "active")["version"] == 1


async def test_playground_saves_nothing(llm: Any) -> None:
    c, web, store = setup(llm)
    await run(c, "news_job_flow", {"job_id": JOB, "dry_run": True, "llm": False})
    cid = next(iter(store.clusters))
    fr = await run(c, "news_playground_flow", {"job_id": JOB, "cluster_id": cid, "profile_key": "news-default", "version_a": 1})
    assert fr["state_json"]["result"]["a"]["title"] and web.saved == [] and web.media == []


# --- the no-auto-publish guard --------------------------------------------------------------------------------------

NEWS_CODE = [Path("app/flows/news.py"), *sorted(Path("app/news").glob("*.py"))]


def test_no_code_path_from_news_to_publishing() -> None:
    for p in NEWS_CODE:
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"publish", "verify", "submit", "decide"}, f"{p}: .{node.attr}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {"publish_flow", "approval"}, f"{p}: {node.value!r}"
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {"flows.publish", "services.review", "..services.review"} and not (node.module or "").endswith(("publish", "review")), f"{p}: import {node.module}"


async def test_news_flows_cannot_start_publish_flow(llm: Any) -> None:
    c, _web, _store = setup(llm)
    for trig in ("flow", "manual", "cron", "mcp"):
        with pytest.raises(RunError):
            await c.runner.kickoff("publish_flow", {"draft_id": "22222222-2222-4222-8222-222222222222", "approval_id": "x", "accepted_change_ids": []}, triggered_by=trig)


# --- images (unit) --------------------------------------------------------------------------------------------------


def test_template_and_optimised_variants_strip_metadata() -> None:
    import io

    from PIL import Image

    data = render_template(Brief(title="Kubernetes 1.35 makes in-place pod resize stable", category_label="Kubernetes", tone="sky", subject="x", seed_key="a"))
    assert render_template(Brief(title="t" * 20, category_label="K", tone="sky", subject="x", seed_key="a")) == render_template(Brief(title="t" * 20, category_label="K", tone="sky", subject="x", seed_key="a"))  # deterministic
    src = io.BytesIO()
    im = Image.open(io.BytesIO(data))
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"  # camera make
    im.convert("RGB").save(src, "JPEG", exif=exif)
    assert b"SecretCam" in src.getvalue()
    v = cover_variants(src.getvalue(), ai_generated=False)
    assert {k: (w, h) for k, (_b, w, h) in v.items()} == {"og": (1200, 630), "hero": (1600, 900), "w800": (800, 450), "w400": (400, 225)}
    out = Image.open(io.BytesIO(v["og"][0]))
    assert out.format == "WEBP" and not out.getexif() and b"SecretCam" not in v["og"][0]


def test_alt_text_clamp() -> None:
    t, ok = clamp_alt("word " * 40)
    assert len(t) <= 125 and ok is not None
    t2, ok2 = clamp_alt("Abstract blue grid of connected nodes with a highlighted container block expanding in the centre")
    assert ok2 and t2.endswith(".")


_ = asyncio


# --- MCP (news tools) -----------------------------------------------------------------------------------------------


async def test_news_mcp_tools_read_run_and_reject_but_never_approve(llm: Any) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from app.mcp import create_edge_app
    from tests.test_mcp import as_admin

    llm.queue("news.vision", SAFE)
    c, web, store = setup(llm)
    await run(c, "news_job_flow", {"job_id": JOB})
    art = next(iter(store.articles.values()))
    app = create_edge_app(c)
    async with as_admin(["seo:read", "seo:flows", "seo:review:reject"]), create_connected_server_and_client_session(app.state.admin) as s:
        names = {t.name for t in (await s.list_tools()).tools}
        assert {"list_news_jobs", "get_news_run", "kickoff_news_job", "reject_news_draft"} <= names
        assert not [n for n in names if "approve" in n or "publish" in n]
        drafts = (await s.call_tool("list_news_drafts", {})).structuredContent
        assert drafts["drafts"][0]["review_url"].endswith(art["draft_id"])
        runs = (await s.call_tool("list_news_runs", {"job_id": JOB})).structuredContent
        detail = (await s.call_tool("get_news_run", {"run_id": runs["runs"][0]["id"]})).structuredContent
        assert any(st["step"] == "select" for st in detail["steps"]) and "untrusted" in str(detail["logs"]).lower()
        r = await s.call_tool("kickoff_news_job", {"job_id": JOB})
        assert not r.isError
        await c.runner.wait(r.structuredContent["run_id"], 60)
        rej = await s.call_tool("reject_news_draft", {"draft_id": art["draft_id"], "reason": "not for us"})
        assert not rej.isError and store.articles[art["id"]]["status"] == "rejected"
        bad = await s.call_tool("kickoff_flow", {"flow_name": "news_draft_flow", "inputs": {}})
        assert bad.isError  # only news_job_flow, via its own tool
    assert web.published == []
    async with create_connected_server_and_client_session(app.state.public) as s:
        store.posts["p1"] = {"collection": "news", "status": "published", "title": "Kubernetes 1.35 resize", "path": "/news/k8s-135", "body": "pod resize", "description": "d", "published_at": NOW, "updated_at": None}
        res = (await s.call_tool("search_news", {"query": "resize", "since": "2026-11-01"})).structuredContent
        assert res["results"][0]["url"].endswith("/news/k8s-135")
