"""News desk model monitoring (docs/news/ML.md "Monitoring"): the gauges that app/metrics.py does not define, and the
drift check (PSI of the serving model's score distribution vs the one it had on its training rows).

The metrics live in the service's shared registry, so they are scraped with everything else.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from prometheus_client import Counter, Gauge

from ..metrics import REGISTRY

# `news_model_auc` (app/metrics.py) is kept for the existing dashboard: it holds the *serving* model's held-out AUC
# (ranker) or macro F1 (classifier), set only when a model is promoted (or read back from the active model). The
# correctly named gauges below carry the same numbers per metric.
NEWS_MODEL_MACRO_F1 = Gauge("news_model_macro_f1", "Held-out macro F1 of the serving topic classifier.", ["model"], registry=REGISTRY)
NEWS_MODEL_HOLDOUT_AUC = Gauge("news_model_holdout_auc", "Held-out AUC of the serving ranker (v2).", ["model"], registry=REGISTRY)
NEWS_MODEL_SCORE_PSI = Gauge("news_model_score_psi", "Population stability index of the serving model's scores in recent runs vs its training rows (> 0.25: drift).", ["model"], registry=REGISTRY)
NEWS_TOPIC_ASSIGNMENTS = Counter("news_topic_assignments_total", "Cluster topic assignments by method (model, model_low_conf = kept low-confidence guess, claude, rules).", ["method"], registry=REGISTRY)

TOPIC_METHODS = {"model", "model_low_conf", "claude", "rules"}
BINS = 10


def histogram(values: Iterable[float], bins: int = BINS) -> list[float]:
    """Share of values in each of `bins` equal-width bins over [0, 1] (scores and probabilities are in that range)."""
    v = np.clip(np.asarray(list(values), dtype=np.float64), 0.0, 1.0)
    if v.size == 0:
        return []
    counts, _ = np.histogram(v, bins=bins, range=(0.0, 1.0))
    return [round(float(c) / v.size, 6) for c in counts]


def psi(reference: list[float], current: list[float], eps: float = 1e-4) -> float | None:
    """Population stability index between two binned distributions (same bins). None when either is empty."""
    if not reference or not current or len(reference) != len(current):
        return None
    total = 0.0
    for r, c in zip(reference, current, strict=True):
        r, c = max(r, eps), max(c, eps)
        total += (c - r) * math.log(c / r)
    return round(total, 4)
