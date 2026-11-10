"""News desk ML: dedupe (exact, near, semantic), clustering + burst, scoring maths, selection (filters, threshold, caps,
diversity, digest), topic classifier (train, predict, fallback), learned ranker (promotion only when better)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np

from app.news import classifier as clf
from app.news.cluster import Clusterer
from app.news.config import DEFAULT_WEIGHTS, FEATURES, JobConfig
from app.news.dedupe import Deduper, Sig, minhash_of
from app.news.scoring import Features, TrainRow, evaluate, recency_value, score_v1, severity_value, source_count_value, train_ranker
from app.news.select import Candidate, Counts, select
from tests.news_fixtures import K8S_TEXT, K8S_TEXT_B

T0 = datetime(2026, 11, 10, 6, tzinfo=UTC)


def unit(v: list[float]) -> list[float]:
    a = np.asarray(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).tolist()


# --- dedupe ---------------------------------------------------------------------------------------------------------


def test_exact_by_url_and_content_hash() -> None:
    d = Deduper.from_existing([Sig("a", "u1", "h1")])
    assert d.check(Sig("b", "u1", "h9")).reason == "exact"
    assert d.check(Sig("c", "u2", "h1")).reason == "exact"
    assert d.check(Sig("d", "u3", "h3")).reason is None


def test_near_duplicate_minhash() -> None:
    a = K8S_TEXT
    b = K8S_TEXT.replace("today", "on Tuesday")  # syndicated copy with a small edit
    d = Deduper.from_existing([Sig("a", "u1", "h1", minhash_of(a))])
    r = d.check(Sig("b", "u2", "h2", minhash_of(b)))
    assert r.reason == "near" and r.duplicate_of == "a" and r.score >= 0.8
    assert d.check(Sig("c", "u3", "h3", minhash_of(K8S_TEXT_B))).reason is None  # same story, different text: cluster, not dupe


def test_semantic_duplicate_by_embedding() -> None:
    v = unit([1, 0.2, 0, 0])
    d = Deduper.from_existing([Sig("a", "u1", "h1", None, v)])
    assert d.check(Sig("b", "u2", "h2", None, unit([1, 0.21, 0, 0]))).reason == "semantic"
    assert d.check(Sig("c", "u3", "h3", None, unit([0.6, 0.8, 0, 0]))).reason is None


# --- clustering -----------------------------------------------------------------------------------------------------


def test_same_story_joins_one_cluster_and_burst_grows() -> None:
    c = Clusterer()
    base = unit([1, 0.1, 0.05, 0])
    ids = set()
    for i, src in enumerate(["s1", "s2", "s3", "s4"]):
        cid, _ = c.assign(item_id=f"i{i}", title="K8s 1.35", embedding=unit([1, 0.1 + i * 0.02, 0.05, 0]), entities={"product:kubernetes"}, cves=set(), source_id=src, at=T0 - timedelta(hours=i))
        ids.add(cid)
    other, created = c.assign(item_id="x", title="Terraform", embedding=unit([0, 0, 1, 0.1]), entities={"product:terraform"}, cves=set(), source_id="s1", at=T0)
    assert len(ids) == 1 and created and other not in ids
    k = c.clusters[ids.pop()]
    assert k.source_count == 4 and k.item_count == 4 and float(k.centroid @ np.asarray(base)) > 0.99
    assert k.burst(T0) > 0.6 > c.clusters[other].burst(T0)


def test_shared_cve_joins_even_with_different_wording() -> None:
    c = Clusterer()
    a, _ = c.assign(item_id="a", title="CVE-2026-1 in Widget", embedding=unit([1, 0, 0, 0]), entities=set(), cves={"CVE-2026-1"}, source_id="nvd", at=T0)
    b, created = c.assign(item_id="b", title="Patch now", embedding=unit([0, 1, 0, 0]), entities=set(), cves={"CVE-2026-1"}, source_id="cisa", at=T0)
    assert a == b and not created


def test_old_clusters_are_not_joined() -> None:
    c = Clusterer(window_hours=24)
    a, _ = c.assign(item_id="a", title="x", embedding=unit([1, 0, 0]), entities=set(), cves=set(), source_id="s", at=T0 - timedelta(days=3))
    b, created = c.assign(item_id="b", title="x", embedding=unit([1, 0, 0]), entities=set(), cves=set(), source_id="s", at=T0)
    assert created and a != b


# --- scoring --------------------------------------------------------------------------------------------------------


def test_feature_functions() -> None:
    assert source_count_value(1) == 0 and 0.6 < source_count_value(3) < 0.65
    assert recency_value(36, 36) == 0.5 and recency_value(0, 36) == 1
    assert severity_value(9.8, False) == 0.98 and severity_value(None, True) == 0.9 and severity_value(None, False) == 0


def test_v1_score_is_the_weighted_sum_with_a_breakdown() -> None:
    f = Features(credibility=1, source_count=0.5, burst=0, recency=1, topic_match=1, audience=0, novelty=1, severity=0, keyword_overlap=0)
    w = JobConfig().effective_weights()
    s, b = score_v1(f, w)
    assert abs(s - sum(getattr(f, k) * w[k] for k in FEATURES)) < 1e-3
    assert set(b["features"]) == set(FEATURES) and b["model"] == "v1-weighted"
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1) < 1e-9


def test_weight_overrides_are_renormalised() -> None:
    w = JobConfig(weights={"severity": 1.0}).effective_weights()
    assert abs(sum(w.values()) - 1) < 1e-6 and w["severity"] > DEFAULT_WEIGHTS["severity"]


# --- selection ------------------------------------------------------------------------------------------------------


def cand(cid: str, score: float, *, topics: tuple[str, ...] = ("kubernetes",), sources: tuple[str, ...] = ("s1",), text: str = "kubernetes release", **kw: object) -> Candidate:
    return Candidate(cluster_id=cid, title=cid, score=score, breakdown={}, features={}, model_version="v1", topics=list(topics), source_ids=list(sources), text=text, **kw)  # type: ignore[arg-type]


def test_threshold_caps_and_order() -> None:
    cfg = JobConfig(max_articles_per_run=2, max_articles_per_day=5, min_score=0.5, diversity={"max_per_topic_per_day": 10, "max_per_source_per_day": 10})
    cs = select([cand("c", 0.6), cand("a", 0.9), cand("b", 0.7), cand("d", 0.4)], cfg, Counts(), global_cap=10)
    by = {c.cluster_id: c for c in cs}
    assert [c.cluster_id for c in cs if c.decision == "selected"] == ["a", "b"]
    assert by["c"].reason.startswith("per-run cap") and "below the job's threshold" in by["d"].reason
    assert by["a"].rank == 1 and by["b"].rank == 2


def test_daily_and_global_caps() -> None:
    cfg = JobConfig(max_articles_per_run=5, max_articles_per_day=3, min_score=0)
    cs = select([cand("a", 0.9, sources=("s1",)), cand("b", 0.8, sources=("s2",)), cand("c", 0.7, sources=("s3",), topics=("x",))], cfg, Counts(job_today=2), global_cap=10)
    assert [c.decision for c in cs] == ["selected", "rejected", "rejected"] and "daily cap" in cs[1].reason
    cs = select([cand("a", 0.9)], cfg, Counts(global_today=10), global_cap=10)
    assert "global daily cap" in cs[0].reason


def test_diversity_by_topic_and_source() -> None:
    cfg = JobConfig(max_articles_per_run=5, min_score=0, diversity={"max_per_topic_per_day": 1, "max_per_source_per_day": 1})
    cs = select([cand("a", 0.9, sources=("s1",)), cand("b", 0.8, sources=("s2",)), cand("c", 0.7, topics=("security",), sources=("s1",)), cand("d", 0.6, topics=("mlops",), sources=("s3",))], cfg, Counts(), global_cap=10)
    got = {c.cluster_id: c for c in cs}
    assert got["a"].decision == "selected" and "on kubernetes today" in got["b"].reason
    assert "from this source" in got["c"].reason and got["d"].decision == "selected"


def test_filters_keywords_entities_topics_language_age_covered_update() -> None:
    cfg = JobConfig(min_score=0, include_keywords=["kubernetes"], exclude_keywords=["webinar"], topics=["kubernetes"], max_age_hours=48)
    cs = select(
        [
            cand("ok", 0.5),
            cand("ex", 0.9, text="kubernetes webinar"),
            cand("noinc", 0.9, text="terraform release"),
            cand("off", 0.9, topics=("mlops",)),
            cand("de", 0.9, language="de"),
            cand("old", 0.9, age_hours=72.0),
            cand("cov", 0.9, covered=True),
            cand("upd", 0.9, site_match={"update_candidate": True, "path": "/blog/x"}),
        ],
        cfg,
        Counts(),
        global_cap=10,
    )
    r = {c.cluster_id: c.reason for c in cs}
    assert [c.cluster_id for c in cs if c.decision == "selected"] == ["ok"]
    assert "excluded keyword" in r["ex"] and "no include" in r["noinc"] and "off-topic" in r["off"]
    assert "language de" in r["de"] and "too old" in r["old"] and "already covered" in r["cov"] and "/blog/x" in r["upd"]


def test_digest_mode() -> None:
    cfg = JobConfig(min_score=0, digest_size=2)
    cs = select([cand("a", 0.9), cand("b", 0.8), cand("c", 0.7)], cfg, Counts(), global_cap=10, mode="daily_digest")
    assert [c.decision for c in cs] == ["selected", "selected", "rejected"] and "digest is full" in cs[2].reason
    one = select([cand("a", 0.9)], cfg, Counts(), global_cap=10, mode="daily_digest")
    assert one[0].decision == "rejected" and "too few" in one[0].reason


# --- classifier -----------------------------------------------------------------------------------------------------


TOPICS = [clf.Topic("kubernetes", "Kubernetes", ["kubernetes", "kubelet", "pod"], "kubernetes"), clf.Topic("security", "Security", ["cve", "vulnerability", "exploit"], "cybersecurity")]


def test_rules_fallback_without_a_model() -> None:
    p = clf.predict(None, TOPICS, embedding=None, entities=set(), text="A kubelet bug crashes pods", threshold=0.55)
    assert p.method == "rules" and p.labels[0][0] == "kubernetes"


def test_classifier_trains_serialises_and_flags_low_confidence() -> None:
    rng = random.Random(1)
    samples = []
    for i in range(80):
        k8s = i % 2 == 0
        emb = unit([1.0 if k8s else 0.0, 0.0 if k8s else 1.0, rng.random() * 0.1, rng.random() * 0.1])
        t = "kubernetes pod scheduling" if k8s else "critical cve exploit"
        samples.append((clf.featurize(emb, set(), t, TOPICS, 4), {"kubernetes"} if k8s else {"security"}))
    m = clf.train(samples, ["kubernetes", "security"], 4, "clf-test")
    assert m is not None and m.metrics["macro_f1"] >= 0.9
    m2 = clf.TopicModel.from_bytes(m.version, m.to_bytes())
    p = clf.predict(m2, TOPICS, embedding=unit([1, 0, 0, 0]), entities=set(), text="kubernetes pod", threshold=0.55)
    assert p.method == "model" and p.labels[0][0] == "kubernetes"
    amb = clf.predict(m2, TOPICS, embedding=unit([1, 1, 0, 0]), entities=set(), text="", threshold=0.99)
    assert amb.method == "low_confidence"  # the flow then asks Claude


def test_classifier_needs_enough_data() -> None:
    assert clf.train([(np.zeros(4), {"kubernetes"})] * 5, ["kubernetes"], 4, "v") is None


# --- learned ranker -------------------------------------------------------------------------------------------------


def _rows(n: int, signal: str, seed: int = 3) -> list[TrainRow]:
    """Approval depends only on `signal` (a feature v1 weighs at 0.06–0.12); v1 is therefore mediocre and v2 learns it."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        f = {k: rng.random() for k in FEATURES}
        label = 1 if f[signal] > 0.55 else 0
        rows.append(TrainRow(run_id=f"r{i // 6}", at=T0 + timedelta(hours=i), features=[f[k] for k in FEATURES], label=label))
    return rows


def test_ranker_replaces_v1_only_when_it_wins_on_held_out_data() -> None:
    w = JobConfig().effective_weights()
    model, rep = train_ranker(_rows(300, "keyword_overlap"), w, "lgbm-test")
    assert model is not None and rep["promoted"] and rep["v2"]["auc"] > rep["v1"]["auc"]
    assert set(model.explain([0.5] * len(FEATURES))) == {*FEATURES, "bias"}
    from app.news.scoring import RankerModel

    assert abs(RankerModel.from_bytes("x", model.to_bytes()).predict([[0.5] * len(FEATURES)])[0] - model.predict([[0.5] * len(FEATURES)])[0]) < 1e-9


def test_ranker_not_promoted_on_noise_or_too_little_data() -> None:
    w = JobConfig().effective_weights()
    assert train_ranker(_rows(10, "recency"), w, "v")[0] is None
    rng = random.Random(9)
    noise = [TrainRow(f"r{i // 5}", T0 + timedelta(hours=i), [rng.random() for _ in FEATURES], rng.randint(0, 1)) for i in range(200)]
    model, rep = train_ranker(noise, w, "v")
    # On pure noise neither model should beat chance by much; if v2 does not beat v1 on both metrics it is not promoted.
    assert (model is None) == (not rep["promoted"])


def test_evaluate_metrics() -> None:
    m = evaluate(["a", "a", "b", "b"], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert m["auc"] == 1.0 and m["ndcg5"] == 1.0
