"""The news desk after an embeddings model change (bge-small ONNX → bge-m3 on Ollama): stored vectors of the previous
model are re-embedded, centroids rebuilt, the paraphrase cutoff and the classifier follow the model. Moved from
moks-seo-agent tests/test_embeddings_model.py with the news desk (2026-11-21)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pytest

from app import calibration as cal
from app.news import classifier as clf
from app.news import desk as nd
from app.news.config import NewsSettings, Quality
from app.news.runlog import RunRecorder
from app.news.store import MemoryNewsStore
from tests.conftest import settings
from tests.news_fixtures import NOW

BGE_M3 = "ollama:bge-m3:567m"


def test_the_news_paraphrase_default_is_bge_smalls_cutoff() -> None:
    default = Quality().max_paragraph_cosine
    assert default == cal.BGE_SMALL.paragraph_cosine
    assert cal.effective_paragraph_cosine(default, BGE_M3) == cal.CALIBRATIONS["bge-m3"].paragraph_cosine


def deps(store: MemoryNewsStore, dim: int, calls: list[list[str]], model: str = BGE_M3, **news: Any) -> nd.DeskDeps:
    async def embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[1.0] + [0.0] * (dim - 1) for _ in texts]

    return nd.DeskDeps(store=store, web=None, llm=None, embed=embed, retriever=None, settings=settings(), news=NewsSettings(**news), clock=lambda: NOW, extra={"news_embedding_model": model, "news_embedding_dim": dim})  # type: ignore[arg-type]


def item(store: MemoryNewsStore, n: int, embedding: list[float], cluster_id: str | None = None) -> str:
    iid = f"item-{n}"
    store.items[iid] = {"id": iid, "source_id": "s1", "url_hash": f"u{n}", "content_hash": f"c{n}", "minhash": None, "embedding": embedding, "cluster_id": cluster_id, "duplicate_of": None,
                        "title": f"Story {n}", "excerpt": f"Text of story {n}", "entities": {}, "published_at": NOW - timedelta(hours=10 - n), "fetched_at": NOW - timedelta(hours=10 - n)}
    return iid


async def test_window_vectors_of_the_old_model_are_re_embedded_newest_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nd, "REEMBED_PER_RUN", 2)
    store = MemoryNewsStore()
    store.clock = lambda: NOW
    for n in range(3):
        item(store, n, [0.6, 0.8])  # bge-small-era vectors (here: 2 dimensions)
    calls: list[list[str]] = []
    d = deps(store, 4, calls)
    await nd.dedupe_and_cluster(d, RunRecorder(store, "r1"), [], window_days=14)
    assert calls == [["Story 2. Text of story 2", "Story 1. Text of story 1"]]  # the same text as at ingest, capped
    assert [len(store.items[f"item-{n}"]["embedding"]) for n in range(3)] == [2, 4, 4]  # the oldest waits for the next run
    await nd.dedupe_and_cluster(d, RunRecorder(store, "r2"), [], window_days=14)
    assert [len(store.items[f"item-{n}"]["embedding"]) for n in range(3)] == [4, 4, 4]
    await nd.dedupe_and_cluster(d, RunRecorder(store, "r3"), [], window_days=14)
    assert len(calls) == 2  # nothing stale any more: no embedding call


def test_a_centroid_of_the_old_model_is_rebuilt_from_current_member_vectors() -> None:
    current = {"m1": {"id": "m1", "embedding": [1.0, 0.0, 0.0]}, "m2": {"id": "m2", "embedding": [0.0, 1.0, 0.0]}}
    members = [{"id": "m1", "embedding": [0.6, 0.8]}, {"id": "m2", "embedding": [0.8, 0.6]}, {"id": "gone", "embedding": [1.0, 0.0]}]
    c = nd._centroid({"centroid": [0.7, 0.7]}, members, current, 3)
    assert c is not None and np.allclose(c, [2**-0.5, 2**-0.5, 0.0])
    assert np.allclose(nd._centroid({"centroid": [0.0, 0.0, 1.0]}, members, current, 3), [0, 0, 1])  # current size: kept
    assert nd._centroid({"centroid": [0.7, 0.7]}, members[2:], {}, 3) is None  # no current member vector: none
    assert nd._centroid({"centroid": None}, members, current, 3) is None


def test_quality_gate_takes_the_models_paraphrase_cutoff() -> None:
    store = MemoryNewsStore()
    assert nd._quality(deps(store, 1024, [])).max_paragraph_cosine == cal.CALIBRATIONS["bge-m3"].paragraph_cosine
    assert nd._quality(deps(store, 384, [], model="onnx:bge-small-en-v1.5")).max_paragraph_cosine == 0.93
    custom = deps(store, 1024, [], quality={"max_paragraph_cosine": 0.9})
    assert nd._quality(custom).max_paragraph_cosine == 0.9


def test_classifier_of_the_old_model_answers_with_rules() -> None:
    dim = 32
    model = clf.TopicModel("clf-old", ["kubernetes"], np.ones((1, dim + clf.HASH_BUCKETS + 1)), np.zeros(1), dim)
    topics = [clf.Topic("kubernetes", "Kubernetes", ["kubernetes", "pod"])]
    old = clf.predict(model, topics, embedding=[0.1] * dim, entities=set(), text="a kubernetes pod", threshold=0.55)
    assert old.method != "rules"
    new = clf.predict(model, topics, embedding=[0.1] * 1024, entities=set(), text="a kubernetes pod", threshold=0.55)
    assert new.method == "rules" and new.labels and new.labels[0][0] == "kubernetes"
