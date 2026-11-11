"""Ranking (docs/news/ML.md "Ranking").

v1: transparent weighted features in [0, 1]; score = Σ weight × value; every candidate stores the full breakdown.
v2: a LightGBM model trained on the owner's approve/reject decisions (binary objective). It replaces v1 only when it
    beats v1 (each row scored with its own job's weights) and the current v2, if any, on the same held-out data on
    **both** AUC and NDCG@5 (per run). Its explanation is LightGBM's per-feature contributions (`pred_contrib`), shown
    next to the v1 breakdown. The model version is recorded on every candidate.
    Its probabilities are not on v1's scale, so a job's `min_score` (tuned for v1) is mapped through quantiles stored
    with the model: the v2 threshold keeps the share of candidates v1 would have passed (`RankerModel.threshold_for`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from .config import FEATURES
from .monitoring import histogram

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
    weights: dict[str, float] | None = None  # the v1 weights of the job that scored this candidate (None: defaults)


QUANTILES = 101


@dataclass
class RankerModel:
    version: str
    booster: Any
    # Score map (stored in the model's metrics): v1 and v2 score quantiles over the same reference candidates.
    v1_quantiles: list[float] | None = None
    v2_quantiles: list[float] | None = None

    def threshold_for(self, min_score: float) -> float | None:
        """The v2 score that passes the same share of candidates as `min_score` passes under v1 (quantile mapping).
        None without a score map (an older model): the caller then falls back to min_score."""
        if not self.v1_quantiles or not self.v2_quantiles:
            return None
        levels = np.linspace(0.0, 1.0, len(self.v1_quantiles))
        below = float(np.interp(min_score, self.v1_quantiles, levels, left=0.0, right=1.0))  # share v1 rejects
        if below >= 1.0:
            return float(self.v2_quantiles[-1]) + 1e-9  # v1 would pass nothing
        return float(np.interp(below, np.linspace(0.0, 1.0, len(self.v2_quantiles)), self.v2_quantiles))  # not rounded: ties

    def predict(self, rows: list[list[float]]) -> np.ndarray:
        return np.asarray(self.booster.predict(np.asarray(rows, dtype=np.float64)), dtype=np.float64)

    def explain(self, row: list[float]) -> dict[str, float]:
        contrib = self.booster.predict(np.asarray([row], dtype=np.float64), pred_contrib=True)[0]
        return {**{f: round(float(c), 4) for f, c in zip(FEATURES, contrib[:-1], strict=True)}, "bias": round(float(contrib[-1]), 4)}

    def to_bytes(self) -> bytes:
        return self.booster.model_to_string().encode()

    @classmethod
    def from_bytes(cls, version: str, data: bytes, metrics: dict[str, Any] | None = None) -> RankerModel:
        import lightgbm as lgb

        sm = (metrics or {}).get("score_map") or {}
        return cls(version, lgb.Booster(model_str=data.decode()), sm.get("v1") or None, sm.get("v2") or None)


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


def _v1(r: TrainRow, default: dict[str, float]) -> float:
    return score_v1(_features_from(r.features), r.weights or default)[0]


def score_map(model: RankerModel, reference: list[TrainRow], weights: dict[str, float]) -> dict[str, list[float]]:
    """v1 and v2 score quantiles over reference candidates (all decisions, not only reviewed ones)."""
    if not reference:
        return {}
    q = np.linspace(0.0, 1.0, QUANTILES)
    v1 = np.asarray([_v1(r, weights) for r in reference], dtype=np.float64)
    v2 = model.predict([r.features for r in reference])
    return {"v1": [float(x) for x in np.quantile(v1, q)], "v2": [float(x) for x in np.quantile(v2, q)]}  # exact: v2 scores tie a lot


def train_ranker(rows: list[TrainRow], weights: dict[str, float], version: str, seed: int = 7, *, current: RankerModel | None = None, reference: list[TrainRow] | None = None) -> tuple[RankerModel | None, dict[str, Any]]:
    """Time-ordered split (oldest 80 % train, newest 20 % test). Returns (model, report); model is None when it should
    not be promoted (too little data, one class only, or not better on held-out AUC and NDCG@5 than v1 and than the
    `current` v2). v1 scores each row with the row's own job weights (`TrainRow.weights`; `weights` otherwise).
    `reference`: candidates (any decision) for the v1→v2 score map; the training rows when not given."""
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
    v1_scores = [_v1(r, weights) for r in test]
    ids = [r.run_id for r in test]
    labels = [r.label for r in test]
    m2, m1 = evaluate(ids, labels, v2_scores), evaluate(ids, labels, v1_scores)
    better = m2["auc"] > m1["auc"] and m2["ndcg5"] >= m1["ndcg5"]
    reason = "beats v1 on held-out AUC and NDCG@5" if better else "does not beat v1 on held-out data"
    report: dict[str, Any] = {"v2": m2, "v1": m1, "train": len(train), "test": len(test)}
    if current is not None:
        # The serving v2 may have seen some of these rows in training, which favours it: a conservative comparison.
        mc = evaluate(ids, labels, current.predict(xt).tolist())
        report["current"] = {**mc, "version": current.version}
        if better and not (m2["auc"] > mc["auc"] and m2["ndcg5"] >= mc["ndcg5"]):
            better, reason = False, f"beats v1 but not the serving model {current.version} on held-out data"
        elif better:
            reason = f"beats v1 and the serving model {current.version} on held-out AUC and NDCG@5"
    ref = reference or rows
    report["score_map"] = score_map(model, ref, weights)
    model.v1_quantiles, model.v2_quantiles = report["score_map"].get("v1"), report["score_map"].get("v2")
    report["score_hist"] = histogram(model.predict([r.features for r in ref]).tolist())  # drift reference (PSI)
    report.update({"promoted": better, "reason": reason})
    return (model if better else None), report


def _features_from(v: list[float]) -> Features:
    return Features(**dict(zip(FEATURES, v, strict=True)))
