"""Multi-label topic classifier (docs/news/ML.md "Topics").

Input features: the item embedding ⊕ 64 hashed entity buckets ⊕ one keyword-hit feature per topic.
Model: one logistic regression per topic (scikit-learn, class_weight=balanced), Platt-calibrated. Serialised as plain
arrays (npz), not pickle, so loading a model from the database can never execute code.

Training and serving use the same inputs (`item_inputs`): one news item's embedding, entities and "title excerpt".
A cluster is classified on its representative item (the member closest to the centroid), never on the centroid or the
joined text of several items, so the features at serving time are built exactly as they were at training time.

Prediction: calibrated probabilities per topic; labels = topics with p ≥ 0.5 (at least the best one). When the best
probability is below the confidence threshold (default 0.55), the flow asks Claude (Haiku) to choose from the allowed
topics; when there is no model yet, or Claude is unavailable or over budget, it falls back to keyword rules. Bootstrap:
Claude labels up to 1,000 recent items once (news_train_flow bootstrap=true); the model then retrains weekly on those
labels plus the owner's category corrections (human labels win over Claude's).

Evaluation (`train`): a grouped hold-out (by cluster, so two reports of one story never sit on both sides), Platt
parameters from grouped out-of-fold decision values, metrics on the hold-out, then the shipped model is refit on all
rows. Topics with a single class in the training split are reported (`excluded_topics`) and left out of macro F1.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .monitoring import histogram
from .text import entity_set

HASH_BUCKETS = 64


@dataclass
class Topic:
    slug: str
    label: str
    keywords: list[str] = field(default_factory=list)
    category_slug: str | None = None
    tags: list[str] = field(default_factory=list)


def keyword_hits(text: str, topic: Topic) -> int:
    t = text.lower()
    return sum(1 for k in topic.keywords if k and re.search(rf"(?<![\w-]){re.escape(k.lower())}(?![\w-])", t))


def rule_topics(text: str, topics: list[Topic], max_labels: int = 3) -> list[tuple[str, float]]:
    """Keyword fallback: share of a topic's keywords found, capped; deterministic."""
    out = []
    for tp in topics:
        h = keyword_hits(text, tp)
        if h:
            out.append((tp.slug, round(min(1.0, 0.35 + 0.15 * h), 3)))
    out.sort(key=lambda x: -x[1])
    return out[:max_labels]


def item_inputs(row: dict[str, Any]) -> tuple[list[float] | None, set[str], str]:
    """The classifier's raw inputs for one news item (a news_items row or a label row): used for training AND serving."""
    ents = entity_set({k: v for k, v in (row.get("entities") or {}).items() if k != "meta"})
    return row.get("embedding"), ents, f"{row.get('title') or ''} {row.get('excerpt') or ''}"


def representative(items: list[dict[str, Any]], centroid: list[float] | None) -> dict[str, Any] | None:
    """The cluster member the classifier sees: the non-duplicate item closest to the centroid (else the first one)."""
    if not items:
        return None
    pool = [i for i in items if not i.get("duplicate_of")] or items
    if centroid:
        c = np.asarray(centroid, dtype=np.float32)
        best, best_s = None, -2.0
        for i in pool:
            e = i.get("embedding")
            if not e or len(e) != len(c):
                continue
            v = np.asarray(e, dtype=np.float32)
            n = float(np.linalg.norm(v) * np.linalg.norm(c))
            s = float(v @ c) / n if n > 0 else -1.0
            if s > best_s:
                best, best_s = i, s
        if best is not None:
            return best
    return pool[0]


def featurize(embedding: list[float] | None, entities: set[str], text: str, topics: list[Topic], dim: int) -> np.ndarray:
    e = np.zeros(dim, dtype=np.float32)
    if embedding:
        v = np.asarray(embedding[:dim], dtype=np.float32)
        e[: len(v)] = v
    h = np.zeros(HASH_BUCKETS, dtype=np.float32)
    for ent in entities:
        h[int(hashlib.md5(ent.encode(), usedforsecurity=False).hexdigest(), 16) % HASH_BUCKETS] = 1.0
    k = np.asarray([min(3, keyword_hits(text, t)) / 3 for t in topics], dtype=np.float32)
    return np.concatenate([e, h, k])


@dataclass
class TopicModel:
    version: str
    topics: list[str]  # model column order
    coef: np.ndarray  # (n_topics, n_features)
    intercept: np.ndarray  # (n_topics,)
    dim: int
    metrics: dict[str, Any] = field(default_factory=dict)
    cal_a: np.ndarray | None = None  # Platt slope per topic: p = σ(a·z + b); None = uncalibrated (older models)
    cal_b: np.ndarray | None = None
    eval_idx: list[int] = field(default_factory=list)  # hold-out sample indices of the train() call (not stored)
    eval_topics: list[str] = field(default_factory=list)  # topics macro F1 was computed over (not stored)

    def decision(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x @ self.coef.T + self.intercept)

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = self.decision(x)
        if self.cal_a is not None and self.cal_b is not None:
            z = z * self.cal_a + self.cal_b
        return np.asarray(1 / (1 + np.exp(-np.clip(z, -50, 50))))

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        n = len(self.topics)
        a = self.cal_a if self.cal_a is not None else np.ones(n)
        b = self.cal_b if self.cal_b is not None else np.zeros(n)
        # Plain arrays only (loaded with allow_pickle=False): calibration is two float vectors, not a fitted object.
        np.savez_compressed(buf, coef=self.coef, intercept=self.intercept, topics=np.asarray(self.topics), dim=np.asarray([self.dim]), cal_a=np.asarray(a, dtype=np.float64), cal_b=np.asarray(b, dtype=np.float64))
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, version: str, data: bytes, metrics: dict[str, Any] | None = None) -> TopicModel:
        z = np.load(io.BytesIO(data), allow_pickle=False)
        files = set(z.files)
        return cls(version, [str(t) for t in z["topics"]], z["coef"], z["intercept"], int(z["dim"][0]), metrics or {}, z["cal_a"] if "cal_a" in files else None, z["cal_b"] if "cal_b" in files else None)

    def predict_matrix(self, inputs: list[tuple[list[float] | None, set[str], str]], topics: list[Topic]) -> np.ndarray:
        """Calibrated probabilities (n, len(self.topics)) for raw item inputs, featurised the way this model was trained."""
        by_slug = {t.slug: t for t in topics}
        cols = [by_slug.get(s, Topic(s, s)) for s in self.topics]
        if not inputs:
            return np.zeros((0, len(self.topics)))
        x = np.stack([featurize(e, ents, text, cols, self.dim) for e, ents, text in inputs])
        return self.proba(x)


@dataclass
class Prediction:
    labels: list[tuple[str, float]]
    confidence: float
    method: str  # model | low_confidence | rules


def predict(model: TopicModel | None, topics: list[Topic], *, embedding: list[float] | None, entities: set[str], text: str, threshold: float) -> Prediction:
    allowed = [t for t in topics]
    # A vector of another size came from another embeddings model than the one this classifier was trained on:
    # featurize would cut or pad it into nonsense, so the rules answer until a model trained on these vectors serves.
    if model is not None and embedding and len(embedding) != model.dim:
        model = None
    if model is None or not allowed:
        r = rule_topics(text, allowed)
        return Prediction(r, r[0][1] if r else 0.0, "rules")
    by_slug = {t.slug: t for t in allowed}
    cols = [t for t in model.topics if t in by_slug]
    if not cols:
        r = rule_topics(text, allowed)
        return Prediction(r, r[0][1] if r else 0.0, "rules")
    x = featurize(embedding, entities, text, [by_slug.get(s, Topic(s, s)) for s in model.topics], model.dim)
    p = model.proba(x[None, :])[0]
    pairs = sorted(((s, round(float(p[i]), 4)) for i, s in enumerate(model.topics) if s in by_slug), key=lambda kv: -kv[1])
    labels = [kv for kv in pairs if kv[1] >= 0.5][:3] or pairs[:1]
    conf = pairs[0][1] if pairs else 0.0
    return Prediction(labels, conf, "model" if conf >= threshold else "low_confidence")


# --- training -------------------------------------------------------------------------------------------------------


def grouped_split(groups: list[str], frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole groups (clusters) go to the hold-out until it holds ≈ frac of the rows; returns (train_idx, test_idx)."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    target = max(1, round(len(groups) * frac))
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    test: list[int] = []
    for g in order:
        if len(test) >= target:
            break
        if len(test) + len(by_group[g]) > len(groups) - 1:
            continue
        test += by_group[g]
    test_set = set(test)
    train = [i for i in range(len(groups)) if i not in test_set]
    return np.asarray(train, dtype=int), np.asarray(sorted(test), dtype=int)


def _platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """1-D logistic fit of the labels on decision values (Platt scaling); identity when it cannot be fitted."""
    from sklearn.linear_model import LogisticRegression

    if len(set(y.tolist())) < 2:
        return 1.0, 0.0
    lr = LogisticRegression(C=1e4, max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return (a, b) if a > 0 else (1.0, 0.0)  # a non-positive slope would invert the ranking: keep the raw scores


def _fit_topic(x: np.ndarray, y: np.ndarray, groups: list[str], seed: int, folds: int = 3) -> tuple[np.ndarray, float, float, float]:
    """Logistic regression on all given rows + Platt parameters from grouped out-of-fold decision values."""
    from sklearn.linear_model import LogisticRegression

    def fit(xx: np.ndarray, yy: np.ndarray) -> LogisticRegression:
        m = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=seed)
        m.fit(xx, yy)
        return m

    full = fit(x, y)
    uniq = sorted(set(groups))
    k = min(folds, len(uniq))
    oof = np.full(len(y), np.nan)
    if k >= 2:
        perm = np.random.default_rng(seed + 1).permutation(len(uniq))
        fold_of = {uniq[g]: pos % k for pos, g in enumerate(perm)}
        fid = np.asarray([fold_of[g] for g in groups])
        for f in range(k):
            tr, te = fid != f, fid == f
            if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
                continue
            oof[te] = fit(x[tr], y[tr]).decision_function(x[te])
    ok = ~np.isnan(oof)
    a, b = _platt(oof[ok], y[ok]) if ok.sum() >= 10 else (1.0, 0.0)
    return full.coef_[0], float(full.intercept_[0]), a, b


def _fit_all(x: np.ndarray, y: np.ndarray, topic_order: list[str], groups: list[str], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Per-topic fits; a topic with one class only gets a constant and is returned in the excluded list."""
    coef = np.zeros((len(topic_order), x.shape[1]), dtype=np.float64)
    intercept = np.zeros(len(topic_order), dtype=np.float64)
    ca, cb = np.ones(len(topic_order)), np.zeros(len(topic_order))
    excluded = []
    for j, t in enumerate(topic_order):
        yj = y[:, j]
        if len(set(yj.tolist())) < 2:
            intercept[j] = -6.0 if yj.sum() == 0 else 6.0
            excluded.append(t)
            continue
        coef[j], intercept[j], ca[j], cb[j] = _fit_topic(x, yj, groups, seed)
    return coef, intercept, ca, cb, excluded


def topic_metrics(y_true: np.ndarray, proba: np.ndarray, topic_order: list[str], eval_topics: list[str]) -> dict[str, Any]:
    """Per-topic P/R/F1 at p ≥ 0.5, macro F1 over eval_topics, and the Brier score (calibration) over the same topics.
    Columns of y_true and proba follow topic_order."""
    from sklearn.metrics import precision_recall_fscore_support

    per: dict[str, Any] = {}
    briers = []
    for j, t in enumerate(topic_order):
        if t not in eval_topics:
            continue
        pred = (proba[:, j] >= 0.5).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true[:, j], pred, average="binary", zero_division=0)
        per[t] = {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3), "support": int(y_true[:, j].sum())}
        briers.append(float(np.mean((proba[:, j] - y_true[:, j]) ** 2)))
    f1s = [v["f1"] for v in per.values()]
    return {"per_topic": per, "macro_f1": round(float(np.mean(f1s)), 3) if f1s else None, "brier": round(float(np.mean(briers)), 4) if briers else None}


def macro_f1_on(model: TopicModel, inputs: list[tuple[list[float] | None, set[str], str]], labels: list[set[str]], topics: list[Topic], eval_topics: list[str]) -> float | None:
    """Macro F1 of any model (e.g. the serving one) on given raw rows over `eval_topics`, featurised the way that model
    was trained. A topic the model does not know is predicted negative throughout (F1 0)."""
    if not inputs or not eval_topics:
        return None
    proba = model.predict_matrix(inputs, topics)
    cols = [proba[:, model.topics.index(t)] if t in model.topics else np.zeros(len(inputs)) for t in eval_topics]
    y = np.asarray([[1 if t in ls else 0 for t in eval_topics] for ls in labels])
    return topic_metrics(y, np.stack(cols, axis=1), eval_topics, eval_topics)["macro_f1"]


def train(samples:list[tuple[np.ndarray, set[str]]], topic_order: list[str], dim: int, version: str, seed: int = 7, *, groups: list[str] | None = None, sources: list[str] | None = None) -> TopicModel | None:
    """samples: (feature vector, label set); groups: cluster id per sample (default: every sample its own group);
    sources: label source per sample ("human" | "claude"), for the human-only hold-out metric.

    1. grouped 80/20 split; 2. fit + Platt on the 80 %, metrics on the 20 % (`eval_idx` and `eval_topics` stay on the
    returned object so the caller can score the current model on the same rows); 3. the shipped model is refit and
    recalibrated on all rows."""
    if len(samples) < 30:
        return None
    g = [str(x) for x in groups] if groups is not None else [str(i) for i in range(len(samples))]
    tr, te = grouped_split(g, 0.2, seed)
    x = np.stack([s[0] for s in samples])
    y = np.asarray([[1 if t in s[1] else 0 for t in topic_order] for s in samples])
    coef, intercept, ca, cb, excluded = _fit_all(x[tr], y[tr], topic_order, [g[i] for i in tr], seed)
    held = TopicModel(version, topic_order, coef, intercept, dim, cal_a=ca, cal_b=cb)
    eval_topics = [t for t in topic_order if t not in excluded]
    m: dict[str, Any] = {"n": len(samples), "test": len(te), "groups": len(set(g)), "split": "grouped by cluster", "calibration": "platt (grouped out-of-fold)", "excluded_topics": excluded}
    if len(te):
        m.update(topic_metrics(y[te], held.proba(x[te]), topic_order, eval_topics))
        if sources is not None:
            hum = np.asarray([i for i in te if sources[i] == "human"], dtype=int)
            m["human_test"] = len(hum)
            m["macro_f1_human"] = topic_metrics(y[hum], held.proba(x[hum]), topic_order, eval_topics)["macro_f1"] if len(hum) >= 5 else None
    else:
        m.update({"per_topic": {}, "macro_f1": None, "brier": None})
    for t in excluded:
        m["per_topic"][t] = {"support": int(y[:, topic_order.index(t)].sum()), "note": "one class only in training data: excluded from macro F1"}
    coef, intercept, ca, cb, _ex = _fit_all(x, y, topic_order, g, seed)  # the shipped model: refit on all rows
    final = TopicModel(version, topic_order, coef, intercept, dim, m, ca, cb, eval_idx=[int(i) for i in te], eval_topics=eval_topics)
    # Confidence distribution of the shipped model on its training rows: the reference for the drift check (PSI).
    m["confidence_hist"] = histogram(np.max(final.proba(x), axis=1)) if len(topic_order) else []
    return final
