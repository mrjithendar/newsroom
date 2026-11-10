"""Multi-label topic classifier (docs/news/ML.md "Topics").

Input features: the item/cluster embedding ⊕ 64 hashed entity buckets ⊕ one keyword-hit feature per topic.
Model: one logistic regression per topic (scikit-learn, class_weight=balanced). Serialised as plain arrays (npz), not
pickle, so loading a model from the database can never execute code.

Prediction: probabilities per topic; labels = topics with p ≥ 0.5 (at least the best one). When the best probability is
below the confidence threshold (default 0.55), the flow asks Claude (Haiku) to choose from the allowed topics; when
there is no model yet, or Claude is unavailable or over budget, it falls back to keyword rules. Bootstrap: Claude labels
up to 1,000 recent items once (news_train_flow bootstrap=true); the model then retrains weekly on those labels plus
the owner's category corrections (human labels win over Claude's).
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.coef.T + self.intercept
        return 1 / (1 + np.exp(-z))

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, coef=self.coef, intercept=self.intercept, topics=np.asarray(self.topics), dim=np.asarray([self.dim]))
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, version: str, data: bytes, metrics: dict[str, Any] | None = None) -> TopicModel:
        z = np.load(io.BytesIO(data), allow_pickle=False)
        return cls(version, [str(t) for t in z["topics"]], z["coef"], z["intercept"], int(z["dim"][0]), metrics or {})


@dataclass
class Prediction:
    labels: list[tuple[str, float]]
    confidence: float
    method: str  # model | claude | rules


def predict(model: TopicModel | None, topics: list[Topic], *, embedding: list[float] | None, entities: set[str], text: str, threshold: float) -> Prediction:
    allowed = [t for t in topics]
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


def train(samples: list[tuple[np.ndarray, set[str]]], topic_order: list[str], dim: int, version: str, seed: int = 7) -> TopicModel | None:
    """samples: (feature vector, label set). Holds out 20 % for per-topic precision/recall/F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support

    if len(samples) < 30:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    cut = max(1, int(len(samples) * 0.8))
    tr, te = idx[:cut], idx[cut:]
    x = np.stack([s[0] for s in samples])
    y = np.asarray([[1 if t in s[1] else 0 for t in topic_order] for s in samples])
    coef = np.zeros((len(topic_order), x.shape[1]), dtype=np.float64)
    intercept = np.zeros(len(topic_order), dtype=np.float64)
    per_topic: dict[str, Any] = {}
    for j, t in enumerate(topic_order):
        yj = y[tr, j]
        if len(set(yj.tolist())) < 2:
            intercept[j] = -6.0 if yj.sum() == 0 else 6.0
            per_topic[t] = {"support": int(y[:, j].sum()), "note": "one class only in training data"}
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=seed)
        clf.fit(x[tr], yj)
        coef[j], intercept[j] = clf.coef_[0], clf.intercept_[0]
        if len(te):
            pred = clf.predict(x[te])
            p, r, f, _ = precision_recall_fscore_support(y[te, j], pred, average="binary", zero_division=0)
            per_topic[t] = {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3), "support": int(y[:, j].sum())}
    f1s = [v["f1"] for v in per_topic.values() if "f1" in v]
    return TopicModel(version, topic_order, coef, intercept, dim, {"per_topic": per_topic, "macro_f1": round(float(np.mean(f1s)), 3) if f1s else None, "n": len(samples), "test": len(te)})
