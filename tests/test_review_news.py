"""Fixes from the news desk ML review: capped ingest-time Claude calls, classifier train/serve parity and evaluation,
ranker thresholds and labels, the grounding gate, CVE clustering, near-dup, monitoring/rollback/drift and the budget
reservation."""

from __future__ import annotations

import asyncio
import dataclasses
import random
from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from app.llm import BudgetExceeded, CallSpec, Usage
from app.llm_budgets import Budget
from app.metrics import REGISTRY
from app.news import classifier as clf
from app.news import desk as nd
from app.news.budget import NewsBudget, NewsLlm
from app.news.cluster import Clusterer, ClusterState
from app.news.config import FEATURES, JobConfig, NewsSettings
from app.news.dedupe import Deduper, Sig, minhash_of
from app.news.monitoring import histogram, psi
from app.news.quality import SourceDoc, gate_grounding, snippet_in_sources
from app.news.schemas import Claim, FactSheet, TopicLabels
from app.news.scoring import RankerModel, TrainRow, train_ranker
from app.news.select import Candidate, Counts, select
from app.news.store import MemoryNewsStore
from tests.conftest import settings
from tests.news_fixtures import JOB, NOW, good_draft, seed
from tests.test_news_flows import run, setup

TOPICS = [clf.Topic("kubernetes", "Kubernetes", ["kubernetes", "pod"]), clf.Topic("cybersecurity", "Security", ["cve", "exploit"])]


def unit(v: Any) -> list[float]:
    a = np.asarray(v, dtype=np.float64)
    return (a / np.linalg.norm(a)).tolist()


def gauge(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


def deps(llm: Any, store: MemoryNewsStore, **news: Any) -> nd.DeskDeps:
    async def embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    return nd.DeskDeps(store=store, web=None, llm=llm, embed=embed, retriever=None, settings=settings(), news=NewsSettings(**news), clock=lambda: NOW)


def mem_store() -> MemoryNewsStore:
    s = MemoryNewsStore()
    s.clock = lambda: NOW
    seed(s)
    return s


def flat_classifier(dim: int = 32) -> clf.TopicModel:
    """p = 0.5 for every topic: always below the 0.55 confidence threshold."""
    n = dim + clf.HASH_BUCKETS + 2
    return clf.TopicModel("clf-flat", ["kubernetes", "cybersecurity"], np.zeros((2, n)), np.zeros(2), dim)


# --- 1. ingest-time Claude calls go through the news caps ----------------------------------------------------------


async def test_with_news_llm_caps_and_books_ingest_classification(llm: Any) -> None:
    store = mem_store()
    await store.create_run({"id": "r-ingest", "kind": "ingest", "job_id": None})
    d = nd.with_news_llm(deps(llm, store), ["r-ingest"])
    assert isinstance(d.llm, NewsLlm) and d.llm.budget.job is None  # global scope
    assert nd.with_news_llm(d, ["x"]) is d  # never wrapped twice
    llm.queue("news.classify", TopicLabels(items=[{"i": 0, "topics": ["kubernetes"]}]))
    assert await nd.claude_labels(d, ["kubernetes pod"], TOPICS) == [["kubernetes"]]
    assert store.runs["r-ingest"]["cost_usd"] > 0  # booked to news_runs
    # There is no news LLM cap any more: an old value in the row is ignored...
    old_cap = nd.with_news_llm(deps(llm, store, llm_daily_usd=0.0), ["r-ingest"])
    llm.queue("news.classify", TopicLabels(items=[{"i": 0, "topics": ["kubernetes"]}]))
    assert await nd.claude_labels(old_cap, ["kubernetes pod"], TOPICS) == [["kubernetes"]]
    # ...and the one LLM budget is what refuses a news call.
    llm.guard.budget.value = Budget(daily_usd=1e-9)
    with pytest.raises(BudgetExceeded) as e:
        await nd.claude_labels(nd.with_news_llm(deps(llm, store), ["r-ingest"]), ["kubernetes pod"], TOPICS)
    assert e.value.scope == "llm_daily"


async def test_job_ingest_fallback_respects_the_llm_budget_and_labels_low_confidence(llm: Any) -> None:
    from tests.test_news_flows import SAFE

    llm.queue("news.vision", SAFE)
    c, _web, store = setup(llm)
    store.settings = {"max_drafts_per_day": 10}
    llm.guard.budget.value = Budget(daily_usd=1e-9)  # the one LLM budget: nothing spent yet, but no call fits
    m = flat_classifier()
    store.models.append({"kind": "classifier", "version": m.version, "active": True, "artifact": m.to_bytes(), "metrics": {}, "trained_at": NOW})
    await run(c, "news_job_flow", {"job_id": JOB})
    # Without the cap the raw LLM would have been asked (and FakeLlm has no news.classify answer queued).
    assert not [s for s in llm.calls if s.purpose == "news.classify"]
    methods = {t["method"] for cl in store.clusters.values() for t in cl["topics"]}
    assert methods == {"model_low_conf"}
    assert gauge("news_topic_assignments_total", method="model_low_conf")


async def test_claude_topic_probability_is_configurable(llm: Any) -> None:
    store = mem_store()
    await store.create_run({"id": "r1", "kind": "ingest", "job_id": None})
    m = flat_classifier(2)
    store.models.append({"kind": "classifier", "version": m.version, "active": True, "artifact": m.to_bytes(), "metrics": {}, "trained_at": NOW})
    store.clusters["c1"] = {"id": "c1", "title": "Kubernetes pod", "centroid": [1.0, 0.0], "entities": {"set": []}, "status": "new", "last_seen": NOW, "topics": []}
    store.items["i1"] = {"id": "i1", "cluster_id": "c1", "source_id": "s", "title": "Kubernetes pod", "excerpt": "", "embedding": [1.0, 0.0], "entities": {}, "fetched_at": NOW}
    llm.queue("news.classify", TopicLabels(items=[{"i": 0, "topics": ["kubernetes"]}]))
    d = nd.with_news_llm(deps(llm, store, claude_topic_p=0.7), ["r1"])
    from app.news.runlog import RunRecorder

    await nd.classify_and_novelty(d, RunRecorder(store, "r1"), ["c1"], llm_ok=True)
    assert store.clusters["c1"]["topics"] == [{"slug": "kubernetes", "p": 0.7, "method": "claude"}]


# --- 2. train/serve parity ------------------------------------------------------------------------------------------


async def test_classifier_features_identical_at_training_and_serving() -> None:
    store = mem_store()
    rng = np.random.default_rng(4)
    item = {"id": "i1", "cluster_id": "c1", "source_id": "s", "title": "Kubernetes kubelet pod bug", "excerpt": "A CVE in the kubelet.", "clean_text": "", "embedding": unit(rng.normal(size=8)), "entities": {"product": ["Kubernetes"], "cve": ["CVE-2026-1111"], "meta": {"cvss": 9.0}}, "fetched_at": NOW}
    store.items["i1"] = item
    store.labels.append({"item_id": "i1", "topics": ["kubernetes"], "source": "human"})
    (train_row,) = await store.label_rows()
    served = clf.representative((await store.items_by_cluster(["c1"]))["c1"], item["embedding"])
    assert served is not None
    x_train = clf.featurize(*clf.item_inputs(train_row), TOPICS, 8)
    x_serve = clf.featurize(*clf.item_inputs(served), TOPICS, 8)
    assert np.array_equal(x_train, x_serve)
    # classify_cluster predicts from exactly those features.
    m = clf.TopicModel("clf-r", ["kubernetes", "cybersecurity"], rng.normal(size=(2, len(x_train))), np.zeros(2), 8)
    p = nd.classify_cluster(deps(None, store), m, TOPICS, [served], item["embedding"], "ignored joined text")
    expected = m.proba(x_train[None, :])[0]
    assert dict(p.labels)[p.labels[0][0]] == round(float(expected.max()), 4)


def test_representative_is_the_member_closest_to_the_centroid() -> None:
    a = {"id": "a", "embedding": unit([1, 0, 0])}
    b = {"id": "b", "embedding": unit([0.2, 1, 0])}
    dup = {"id": "c", "embedding": unit([0, 1, 0]), "duplicate_of": "b"}
    assert clf.representative([a, b, dup], unit([0, 1, 0]))["id"] == "b"  # type: ignore[index]
    assert clf.representative([a, b], None)["id"] == "a"  # type: ignore[index]


# --- 3. classifier evaluation ---------------------------------------------------------------------------------------


def _samples(n: int, seed: int = 1, *, extra_topic: bool = False) -> tuple[list[tuple[np.ndarray, set[str]]], list[str], list[str]]:
    rng = random.Random(seed)
    samples, groups, sources = [], [], []
    for i in range(n):
        k8s = i % 2 == 0
        emb = unit([1.0 if k8s else 0.1, 0.1 if k8s else 1.0, rng.random(), rng.random()])
        t = "kubernetes pod" if k8s else "cve exploit"
        samples.append((clf.featurize(emb, set(), t, TOPICS, 4), {"kubernetes"} if k8s else {"cybersecurity"}))
        groups.append(f"cl{i // 3}")
        sources.append("human" if i % 4 == 0 else "claude")
    return samples, groups, sources


def test_grouped_split_never_splits_a_cluster() -> None:
    groups = [f"g{i // 4}" for i in range(100)]
    tr, te = clf.grouped_split(groups, 0.2, 7)
    assert len(te) >= 18 and not {groups[i] for i in tr} & {groups[i] for i in te}
    assert sorted([*tr.tolist(), *te.tolist()]) == list(range(100))


def test_classifier_reports_excluded_topics_human_subset_and_refits_on_all_rows() -> None:
    samples, groups, sources = _samples(90)
    order = ["kubernetes", "cybersecurity", "mlops"]  # mlops: no positive label anywhere
    m = clf.train(samples, order, 4, "clf-t", groups=groups, sources=sources)
    assert m is not None
    assert m.metrics["excluded_topics"] == ["mlops"] and "note" in m.metrics["per_topic"]["mlops"]
    assert m.eval_topics == ["kubernetes", "cybersecurity"] and m.metrics["macro_f1"] >= 0.9
    assert m.metrics["human_test"] >= 5 and m.metrics["macro_f1_human"] is not None
    held = {groups[i] for i in m.eval_idx}
    assert held and not held & {groups[i] for i in range(90) if i not in set(m.eval_idx)}
    # The shipped model is the all-rows refit, not the evaluation model.
    x = np.stack([s[0] for s in samples])
    y = np.asarray([[1 if t in s[1] else 0 for t in order] for s in samples])
    coef, _b, _ca, _cb, _ex = clf._fit_all(x, y, order, groups, 7)
    assert np.allclose(m.coef, coef)
    assert len(m.metrics["confidence_hist"]) == 10


def test_calibration_is_stored_as_plain_arrays_and_old_artifacts_load() -> None:
    import io

    samples, groups, _ = _samples(60)
    m = clf.train(samples, ["kubernetes", "cybersecurity"], 4, "clf-c", groups=groups)
    assert m is not None and m.cal_a is not None and m.cal_b is not None and (m.cal_a > 0).all()
    again = clf.TopicModel.from_bytes(m.version, m.to_bytes())
    assert again.cal_a is not None and np.allclose(again.cal_a, m.cal_a) and np.allclose(again.proba(samples[0][0][None, :]), m.proba(samples[0][0][None, :]))
    z = np.load(io.BytesIO(m.to_bytes()), allow_pickle=False)  # nothing in it needs pickle
    assert set(z.files) == {"coef", "intercept", "topics", "dim", "cal_a", "cal_b"}
    buf = io.BytesIO()
    np.savez_compressed(buf, coef=m.coef, intercept=m.intercept, topics=np.asarray(m.topics), dim=np.asarray([4]))
    old = clf.TopicModel.from_bytes("old", buf.getvalue())
    assert old.cal_a is None and old.proba(samples[0][0][None, :]).shape == (1, 2)


def _seed_labels(store: MemoryNewsStore, n: int = 40) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        k8s = i % 2 == 0
        iid = f"item{i}"
        v = rng.normal(size=32) + (5 if k8s else -5)
        store.items[iid] = {"id": iid, "cluster_id": f"c{i // 2}", "title": "kubernetes pod" if k8s else "cve exploit", "excerpt": "", "clean_text": "", "embedding": (v / np.linalg.norm(v)).tolist(), "entities": {}, "fetched_at": NOW, "source_id": "s"}
        store.labels.append({"item_id": iid, "topics": ["kubernetes" if k8s else "cybersecurity"], "source": "human"})


async def test_promotion_compares_both_models_on_the_same_holdout_and_needs_a_strict_gain(llm: Any) -> None:
    from app.news.runlog import RunRecorder

    store = mem_store()
    _seed_labels(store)
    await store.create_run({"id": "t1", "kind": "train", "job_id": None})
    d = deps(llm, store)
    first = await nd.train_models(d, RunRecorder(store, "t1"))
    assert first["classifier"]["promoted"]
    served_f1 = gauge("news_model_macro_f1", model="classifier")
    assert served_f1 == first["classifier"]["metrics"]["macro_f1"] == gauge("news_model_auc", model="classifier")
    d.clock = lambda: NOW + timedelta(days=7)
    second = await nd.train_models(d, RunRecorder(store, "t1"))
    cls = second["classifier"]
    assert not cls["promoted"]  # same data → equal F1 → no churn
    assert cls["metrics"]["current"]["version"] == store.models[0]["version"] and cls["metrics"]["current"]["macro_f1"] is not None
    assert [m["active"] for m in store.models if m["kind"] == "classifier"] == [True, False]
    assert gauge("news_model_macro_f1", model="classifier") == served_f1  # not overwritten by the unpromoted model


# --- 4/5. ranker threshold, labels, comparison, exploration ---------------------------------------------------------


def _rows(n: int, signal: str, seed: int = 3, weights: dict[str, float] | None = None) -> list[TrainRow]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        f = {k: rng.random() for k in FEATURES}
        out.append(TrainRow(f"r{i // 6}", NOW + timedelta(hours=i), [f[k] for k in FEATURES], 1 if f[signal] > 0.55 else 0, weights))
    return out


def test_v2_threshold_matches_v1_selection_rate() -> None:
    w = JobConfig().effective_weights()
    ref = _rows(600, "keyword_overlap", seed=11)
    model, rep = train_ranker(_rows(300, "keyword_overlap"), w, "lgbm-t", reference=ref)
    assert model is not None and rep["score_map"]["v1"] and len(rep["score_map"]["v2"]) == 101 and len(rep["score_hist"]) == 10
    from app.news.scoring import _v1

    v2 = model.predict([r.features for r in ref])
    for min_score in (0.4, 0.5, 0.6):
        t = model.threshold_for(min_score)
        assert t is not None
        v1_rate = float(np.mean([_v1(r, w) >= min_score for r in ref]))
        v2_rate = float(np.mean(v2 >= t))
        # LightGBM scores tie (a tree ensemble is a step function): a threshold on a tie passes the whole tie, so the
        # v2 rate is never below the v1 rate and exceeds it by at most the tied mass at the threshold.
        tie = float(np.mean(np.isclose(v2, t)))
        assert v1_rate - 0.02 <= v2_rate <= v1_rate + tie + 0.02, (min_score, v1_rate, v2_rate, tie)
    assert model.threshold_for(0.5) > model.threshold_for(0.4)  # type: ignore[operator]
    assert model.threshold_for(5.0) > v2.max()  # type: ignore[operator]
    again = RankerModel.from_bytes("x", model.to_bytes(), rep)
    assert again.threshold_for(0.5) == model.threshold_for(0.5)
    assert RankerModel.from_bytes("x", model.to_bytes()).threshold_for(0.5) is None  # old model: falls back to min_score


def test_selection_uses_the_candidates_own_threshold() -> None:
    def cand(score: float, threshold: float | None) -> Candidate:
        return Candidate(cluster_id=f"c{score}", title="t", score=score, breakdown={}, features={}, model_version="lgbm-x", topics=["kubernetes"], source_ids=["s"], text="kubernetes", threshold=threshold)

    cfg = JobConfig(min_score=0.45)
    a, b = cand(0.30, 0.25), cand(0.31, None)
    select([a, b], cfg, Counts(), global_cap=10)
    assert a.decision == "selected" and b.decision == "rejected" and "0.45" in b.reason


def test_exploration_is_opt_in_and_marked() -> None:
    def cands() -> list[Candidate]:
        return [Candidate(cluster_id="lo", title="t", score=0.1, breakdown={}, features={}, model_version="v1", topics=["kubernetes"], source_ids=["s"], text="kubernetes")]

    off = cands()
    select(off, JobConfig(min_score=0.45), Counts(), global_cap=10)
    assert off[0].decision == "rejected" and not off[0].explored
    on = cands()
    select(on, JobConfig(min_score=0.45), Counts(), global_cap=10, explore_rate=1.0, rng=random.Random(1))
    assert on[0].decision == "selected" and on[0].explored and "exploration" in on[0].reason


def test_ranker_uses_each_jobs_weights_and_must_beat_the_serving_v2() -> None:
    w = JobConfig().effective_weights()
    rows = _rows(300, "keyword_overlap")
    _m, rep_default = train_ranker(rows, w, "a")
    only_kw = {k: (1.0 if k == "keyword_overlap" else 0.0) for k in FEATURES}
    _m, rep_job = train_ranker(_rows(300, "keyword_overlap", weights=only_kw), w, "b")
    assert rep_job["v1"]["auc"] > rep_default["v1"]["auc"]  # v1 baseline scored with the job's own weights
    model, _rep = train_ranker(rows, w, "c")
    assert model is not None
    again, rep2 = train_ranker(rows, w, "d", current=model)
    assert again is None and not rep2["promoted"] and rep2["current"]["version"] == "c" and "serving model" in rep2["reason"]


async def test_writing_rejections_are_not_ranker_labels(llm: Any) -> None:
    from app.news.runlog import RunRecorder

    store = mem_store()
    await store.create_run({"id": "t1", "kind": "train", "job_id": None})
    for i, cat in enumerate(["style", "inaccurate", "off_topic", None]):
        store.cand_rows.append({"run_id": f"r{i}", "job_id": JOB, "cluster_id": f"c{i}", "features": {k: 0.5 for k in FEATURES}, "score": 0.5, "breakdown": {}, "model_version": "v1-weighted", "decision": "selected", "reason": "", "rank": 1, "created_at": NOW})
        store.articles[f"a{i}"] = {"id": f"a{i}", "run_id": f"r{i}", "cluster_id": f"c{i}", "review_outcome": "rejected" if cat else "approved", "reject_category": cat, "status": "rejected"}
    out = await nd.train_models(deps(llm, store, ranker_min_feedback=100), RunRecorder(store, "t1"))
    assert out["ranker"]["dropped_writing_rejects"] == 2 and "2 labelled decisions" in out["ranker"]["reason"]


# --- 6. grounding ---------------------------------------------------------------------------------------------------

SRC = SourceDoc("https://blog.example.org/k8s", "Kubernetes 1.35 released", "Example", "The Kubernetes project released version 1.35. The project was founded in 2014 and joined the CNCF.")


def _draft(sentence: str) -> Any:
    return good_draft().model_copy(update={"body_mdx": f"## Why it matters\n\n{sentence}\n", "dek": "", "key_takeaways": []})


def test_a_fabricated_fact_sheet_claim_does_not_ground_itself() -> None:
    facts = FactSheet(claims=[Claim(id="f9", claim="The release fixes 17 bugs", source_url=SRC.url, snippet="fixes 17 bugs in the scheduler", kind="number")])
    g = gate_grounding(_draft("Kubernetes 1.35 fixes 17 bugs in the scheduler."), [SRC], facts)
    assert not g.passed and "17" in g.failures[0] and g.detail["claims_unverified"] == ["f9"] and g.warnings
    ok = FactSheet(claims=[Claim(id="f1", claim="1.35 is out", source_url=SRC.url, snippet="The  Kubernetes project released, version 1.35", kind="version")])
    assert gate_grounding(_draft("Kubernetes released version 1.35."), [SRC], ok).passed


def test_numbers_match_whole_tokens_only() -> None:
    facts = FactSheet()
    assert not gate_grounding(_draft("The project now has 10 maintainers."), [SourceDoc("u", "", "p", "Released in 2010 with 4 maintainers.")], facts).passed
    assert gate_grounding(_draft("The project now has 10 maintainers."), [SourceDoc("u", "", "p", "It has 10 maintainers.")], facts).passed
    assert not gate_grounding(_draft("Upgrade to 1.3 today."), [SourceDoc("u", "", "p", "Version 1.35 is out.")], facts).passed


def test_snippet_fuzzy_match() -> None:
    src = [SourceDoc("u", "", "p", "In-place pod resize has graduated to stable, so operators can change the CPU and memory of a running container.")]
    assert snippet_in_sources("in place pod resize has graduated to stable so operators can change the cpu", src)
    assert snippet_in_sources("pod resize graduated to stable, so operators can change the CPU and memory of running containers", src)
    assert not snippet_in_sources("pod resize was removed and operators must recreate every container", src)
    assert not snippet_in_sources("", src)


# --- 7. CVE clustering and entity caps ------------------------------------------------------------------------------


def test_cve_rule_picks_the_best_overlap_and_skips_roundups() -> None:
    c = Clusterer(max_cves=5)
    roundup_cves = {f"CVE-2026-{1000 + i}" for i in range(8)}
    big, _ = c.assign(item_id="r", title="Patch roundup", embedding=unit([0, 0, 1]), entities=set(), cves=roundup_cves, source_id="s1", at=NOW)
    pair, _ = c.assign(item_id="p", title="Two CVEs", embedding=unit([1, 0, 0]), entities=set(), cves={"CVE-2026-1000", "CVE-2026-9999"}, source_id="s2", at=NOW)
    assert pair != big  # the roundup cluster (8 CVEs) is no CVE magnet
    small, _ = c.assign(item_id="s", title="One of them", embedding=unit([1, 1, 0]), entities=set(), cves={"CVE-2026-1000"}, source_id="s5", at=NOW + timedelta(minutes=1))
    assert small == pair  # the only eligible CVE cluster
    single, _ = c.assign(item_id="o", title="Other advisory", embedding=unit([1, 1, 1]), entities=set(), cves={"CVE-2026-7777"}, source_id="s6", at=NOW)
    joined, created = c.assign(item_id="q", title="Same pair", embedding=unit([0, 1, 0]), entities=set(), cves={"CVE-2026-7777", "CVE-2026-1000", "CVE-2026-9999"}, source_id="s3", at=NOW)
    assert joined == pair and not created and single != pair  # Jaccard 2/3 beats 1/3: the best overlap, not the first
    other, created = c.assign(item_id="z", title="Another roundup", embedding=unit([0, -1, 1]), entities=set(), cves={f"CVE-2026-{1000 + i}" for i in range(6)}, source_id="s4", at=NOW)
    assert created and other not in (big, pair)


def test_entity_and_cve_caps_keep_the_most_frequent() -> None:
    c = ClusterState(id="c", title="t", centroid=None)
    for i in range(100):
        c.entities.add(f"org:a{i:03d}")
        c.entity_counts[f"org:a{i:03d}"] += 1
    c.entities.add("product:zzz")
    c.entity_counts["product:zzz"] += 9
    c.cve_ids |= {f"CVE-2026-{i:04d}" for i in range(40)} | {"CVE-2026-9999"}
    c.cve_counts.update({f"CVE-2026-{i:04d}": 1 for i in range(40)})
    c.cve_counts["CVE-2026-9999"] = 4
    row = nd._cluster_row(c, NOW, None)
    assert row["entities"]["set"][0] == "product:zzz" and len(row["entities"]["set"]) == 80  # alphabetically it would be dropped
    assert row["cve_ids"][0] == "CVE-2026-9999" and len(row["cve_ids"]) == 30
    assert row["entities"]["counts"]["product:zzz"] == 9 and row["entities"]["cve_counts"]["CVE-2026-9999"] == 4


# --- 8. near-dup ----------------------------------------------------------------------------------------------------


def test_short_texts_skip_minhash_but_not_exact_dedupe() -> None:
    a, b = "Kubernetes 1.35 released today", "Kubernetes 1.35 released today!"
    d = Deduper.from_existing([Sig("a", "u1", "h1", minhash_of(a), n_tokens=4)], min_tokens=30)
    assert d.check(Sig("b", "u2", "h2", minhash_of(b), n_tokens=4)).reason is None
    assert d.check(Sig("c", "u1", "h3", minhash_of(b), n_tokens=4)).reason == "exact"
    long_d = Deduper.from_existing([Sig("a", "u1", "h1", minhash_of(a))], min_tokens=30)  # unknown length: as before
    assert long_d.check(Sig("b", "u2", "h2", minhash_of(b))).reason == "near"


def test_embedding_matrix_is_cached_until_add() -> None:
    d = Deduper.from_existing([Sig("a", "u1", "h1", None, unit([1, 0, 0]))])
    m1 = d._matrix()
    assert d._matrix() is m1
    d.add(Sig("b", "u2", "h2", None, unit([0, 1, 0])))
    assert d._matrix().shape == (2, 3)
    assert d.check(Sig("c", "u3", "h3", None, unit([0, 1, 0.01]))).duplicate_of == "b"


# --- 9. monitoring, rollback, drift ---------------------------------------------------------------------------------


def test_psi() -> None:
    ref = histogram(np.linspace(0, 1, 1000))
    assert psi(ref, ref) == 0.0
    assert (psi(ref, histogram(np.full(100, 0.95))) or 0) > 0.25
    assert psi([], ref) is None


async def test_rollback_and_drift_gauges(llm: Any) -> None:
    store = mem_store()
    m1, m2 = flat_classifier(), flat_classifier()
    await store.save_model("classifier", "clf-1", metrics={"macro_f1": 0.8, "confidence_hist": histogram(np.full(50, 0.55))}, n_samples=50, artifact=m1.to_bytes(), active=True)
    await store.save_model("classifier", "clf-2", metrics={"macro_f1": 0.6}, n_samples=50, artifact=m2.to_bytes(), active=True)
    await store.save_model("ranker", "lgbm-nope", metrics={}, n_samples=50, artifact=None, active=False)
    d = deps(llm, store)
    r = await nd.activate_model_version(d, "classifier", "clf-1")
    assert r["version"] == "clf-1" and (await store.active_model("classifier") or {})["version"] == "clf-1"
    assert gauge("news_model_macro_f1", model="classifier") == 0.8 == gauge("news_model_auc", model="classifier")
    with pytest.raises(ValueError):
        await nd.activate_model_version(d, "ranker", "lgbm-nope")  # no artifact: cannot serve
    assert [m["version"] for m in await store.list_models("classifier")] == ["clf-2", "clf-1"]
    store.clusters["c1"] = {"id": "c1", "last_seen": NOW, "topics": [{"slug": "kubernetes", "p": 0.95, "method": "model"}]}
    out = await nd.drift(d)
    assert (out["classifier"] or 0) > 0.25 and gauge("news_model_score_psi", model="classifier") == out["classifier"]


async def test_news_mcp_lists_models_read_only(llm: Any) -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    from app.mcp import create_edge_app
    from tests.test_mcp import as_admin

    c, _web, store = setup(llm)
    m = flat_classifier()
    await store.save_model("classifier", "clf-1", metrics={"macro_f1": 0.8, "per_topic": {"x": 1}}, n_samples=50, artifact=m.to_bytes(), active=True)
    app = create_edge_app(c)
    async with as_admin(["seo:read"]), create_connected_server_and_client_session(app.state.admin) as s:
        tools = {t.name: t for t in (await s.list_tools()).tools}
        assert tools["list_news_models"].annotations.readOnlyHint is True  # type: ignore[union-attr]
        res = (await s.call_tool("list_news_models", {"kind": "classifier"})).structuredContent
        assert res["models"][0]["version"] == "clf-1" and res["models"][0]["active"] and res["models"][0]["metrics"] == {"macro_f1": 0.8}


# --- 10. budget reservation -----------------------------------------------------------------------------------------


class SlowLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def structured(self, spec: CallSpec, schema: type[Any]) -> tuple[Any, Usage]:
        self.calls += 1
        await asyncio.sleep(0.05)
        return schema.model_validate({"items": []}), Usage(spec.model, input_tokens=100, output_tokens=10)


async def test_the_news_desk_has_no_llm_cap_of_its_own_and_books_every_call() -> None:
    # Concurrent calls and overspending are the one LLM budget's job (tests/test_llm_budgets.py). Here: an old news
    # cap of 0 in the row refuses nothing, and each call's actual cost lands on the news run.
    store = mem_store()
    await store.create_run({"id": "r1", "kind": "ingest", "job_id": None})
    spec = CallSpec(model="claude-haiku-5-5", purpose="news.classify", prompt_version="t", cached_system=[], user="x" * 700, max_tokens=1500)
    inner = SlowLlm()
    budget = NewsBudget(store, NewsSettings(llm_daily_usd=0.0, llm_monthly_usd=0.0), None, "r1", lambda: NOW)
    llm = NewsLlm(inner, budget, ["r1"])  # type: ignore[arg-type]
    res = await asyncio.gather(*(llm.structured(dataclasses.replace(spec), TopicLabels) for _ in range(3)), return_exceptions=True)
    assert not [r for r in res if isinstance(r, BaseException)] and inner.calls == 3
    actual = Usage(spec.model, input_tokens=100, output_tokens=10).cost_usd()
    assert store.runs["r1"]["cost_usd"] == pytest.approx(3 * actual)
    st = await budget.status()
    assert st["global"]["caps"]["llm_day"] is None and st["global"]["caps"]["llm_month"] is None
    assert not [k for k in await budget.ratios() if "llm" in k] and "global_image_day" in await budget.ratios()  # image caps stay


async def test_reservation_is_released_when_the_call_fails() -> None:
    store = mem_store()
    await store.create_run({"id": "r1", "kind": "ingest", "job_id": None})

    class Boom:
        async def structured(self, spec: CallSpec, schema: type[Any]) -> tuple[Any, Usage]:
            raise RuntimeError("overloaded")

    llm = NewsLlm(Boom(), NewsBudget(store, NewsSettings(), None, "r1", lambda: NOW), ["r1"])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        await llm.structured(CallSpec(model="claude-haiku-5-5", purpose="news.classify", prompt_version="t", cached_system=[], user="x", max_tokens=100), TopicLabels)
    assert store.runs["r1"]["cost_usd"] == 0
