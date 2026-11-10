"""Ranking (docs/news/ML.md "Ranking").

v1: transparent weighted features in [0, 1]; score = Σ weight × value; every candidate stores the full breakdown.
v2: a LightGBM model trained on the owner's approve/reject decisions plus later performance (binary objective, so the
    output is a probability comparable to the job's score threshold). It replaces v1 only when it beats v1 on the same
    held-out data on **both** AUC and NDCG@5 (per run). Its explanation is LightGBM's per-feature contributions
    (`pred_contrib`), shown next to the v1 breakdown. The model version is recorded on every candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .config import FEATURES

V1 = "v1-weighted"


def source_count_value(n: int) -> float:
    return round(1 - math.exp(-max(0, n - 1) / 2), 4)


def recency_value(age_hours: float, half_life_hours: float) -> float:
    return round(0.5 ** (max(0.0, age_hours) / max(1.0, half_life_hours)), 4)


def severity_value(cvss: float | None, exploited: bool) -> float:
    v = (cvss or 0.0) / 10
    if exploited:
        v = max(v, 0.9)
    return round(max(0.0, min(1.0, v)), 4)


@dataclass
class Features:
    credibility: float = 0.0
    source_count: float = 0.0
    burst: float = 0.0
    recency: float = 0.0
    topic_match: float = 0.0
    audience: float = 0.0
    novelty: float = 1.0
    severity: float = 0.0
    keyword_overlap: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {f: round(float(getattr(self, f)), 4) for f in FEATURES}

    def vector(self) -> list[float]:
        return [float(getattr(self, f)) for f in FEATURES]


def score_v1(f: Features, weights: dict[str, float]) -> tuple[float, dict[str, Any]]:
    values = f.as_dict()
    parts = {k: {"value": values[k], "weight": round(weights.get(k, 0.0), 4), "contribution": round(values[k] * weights.get(k, 0.0), 4)} for k in FEATURES}
    return round(sum(p["contribution"] for p in parts.values()), 4), {"model": V1, "features": parts}


# --- v2 -------------------------------------------------------------------------------------------------------------


@dataclass
class TrainRow:
    run_id: str
    at: datetime
    features: list[float]
    label: int  # 1 approved (or performed well), 0 rejected


@dataclass
class RankerModel:
    version: str
    booster: Any

    def predict(self, rows: list[list[float]]) -> np.ndarray:
        return np.asarray(self.booster.predict(np.asarray(rows, dtype=np.float64)), dtype=np.float64)

    def explain(self, row: list[float]) -> dict[str, float]:
        contrib = self.booster.predict(np.asarray([row], dtype=np.float64), pred_contrib=True)[0]
        return {**{f: round(float(c), 4) for f, c in zip(FEATURES, contrib[:-1], strict=True)}, "bias": round(float(contrib[-1]), 4)}

    def to_bytes(self) -> bytes:
        return self.booster.model_to_string().encode()

    @classmethod
    def from_bytes(cls, version: str, data: bytes) -> RankerModel:
        import lightgbm as lgb

        return cls(version, lgb.Booster(model_str=data.decode()))


def ndcg_by_run(run_ids: list[str], labels: list[int], scores: list[float], k: int = 5) -> float:
    from sklearn.metrics import ndcg_score

    groups: dict[str, list[int]] = {}
    for i, r in enumerate(run_ids):
        groups.setdefault(r, []).append(i)
    vals = []
    for idx in groups.values():
        if len(idx) < 2 or len({labels[i] for i in idx}) < 2:
            continue
        vals.append(float(ndcg_score([[labels[i] for i in idx]], [[scores[i] for i in idx]], k=k)))
    return round(float(np.mean(vals)), 4) if vals else 0.0


def evaluate(run_ids: list[str], labels: list[int], scores: list[float]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else 0.5
    return {"auc": round(auc, 4), "ndcg5": ndcg_by_run(run_ids, labels, scores), "n": len(labels)}


def train_ranker(rows: list[TrainRow], weights: dict[str, float], version: str, seed: int = 7) -> tuple[RankerModel | None, dict[str, Any]]:
    """Time-ordered split (oldest 80 % train, newest 20 % test). Returns (model, report); model is None when it should
    not replace v1 (too little data, one class only, or not better than v1 on held-out AUC and NDCG@5)."""
    import lightgbm as lgb

    rows = sorted(rows, key=lambda r: r.at)
    if len(rows) < 20 or len({r.label for r in rows}) < 2:
        return None, {"promoted": False, "reason": f"not enough labelled decisions ({len(rows)})"}
    cut = max(1, int(len(rows) * 0.8))
    train, test = rows[:cut], rows[cut:]
    if len({r.label for r in train}) < 2 or len({r.label for r in test}) < 2:
        return None, {"promoted": False, "reason": "train or test split has one class only"}
    x = np.asarray([r.features for r in train], dtype=np.float64)
    y = np.asarray([r.label for r in train])
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 5, "feature_fraction": 1.0, "bagging_fraction": 1.0, "lambda_l2": 1.0, "verbose": -1, "seed": seed, "deterministic": True, "num_threads": 1}
    booster = lgb.train(params, lgb.Dataset(x, label=y, feature_name=list(FEATURES)), num_boost_round=120)
    model = RankerModel(version, booster)
    xt = [r.features for r in test]
    v2_scores = model.predict(xt).tolist()
    v1_scores = [score_v1(_features_from(r.features), weights)[0] for r in test]
    ids = [r.run_id for r in test]
    labels = [r.label for r in test]
    m2, m1 = evaluate(ids, labels, v2_scores), evaluate(ids, labels, v1_scores)
    better = m2["auc"] > m1["auc"] and m2["ndcg5"] >= m1["ndcg5"]
    report = {"promoted": better, "v2": m2, "v1": m1, "train": len(train), "test": len(test), "reason": "beats v1 on held-out AUC and NDCG@5" if better else "does not beat v1 on held-out data"}
    return (model if better else None), report


def _features_from(v: list[float]) -> Features:
    return Features(**dict(zip(FEATURES, v, strict=True)))
