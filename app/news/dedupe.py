"""Deduplication in three levels (docs/news/ML.md "Dedupe"):

1. exact: the same normalised URL (from any source) or the same content hash (title + text, normalised);
2. near-duplicate: MinHash LSH over word 5-gram shingles (datasketch), confirmed by the estimated Jaccard;
3. semantic: embedding cosine above a high threshold (a syndicated or lightly reworded copy of the same text).

A duplicate is stored (so source health and "seen" counts stay honest) with `duplicate_of` + `dedupe_reason`, and it
still joins its original's cluster, raising that story's source count. Different articles *about* the same story are
not duplicates: they are clustered (cluster.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .text import shingles

NUM_PERM = 64
# datasketch ≥ 2.0 needs the scheme named whenever a MinHash is rebuilt from stored values; changing it invalidates
# every stored signature (news_items.minhash), so it is fixed here.
SCHEME = "affine32"


def minhash_of(text: str) -> list[int]:
    from datasketch import MinHash

    m = MinHash(num_perm=NUM_PERM, seed=7, scheme=SCHEME)
    for s in shingles(text, 5):
        m.update(s.encode("utf-8"))
    # affine32 values are < 2^32, so they fit a Postgres bigint.
    return [int(v) for v in m.hashvalues]


def _minhash(values: list[int]) -> Any:
    from datasketch import MinHash

    return MinHash(num_perm=NUM_PERM, seed=7, hashvalues=np.array(values, dtype=np.uint64), scheme=SCHEME)


@dataclass
class Sig:
    id: str
    url_hash: str
    content_hash: str
    minhash: list[int] | None = None
    embedding: list[float] | None = None
    cluster_id: str | None = None
    source_id: str | None = None


@dataclass
class DedupeResult:
    reason: str | None  # exact | near | semantic | None
    duplicate_of: str | None
    score: float = 0.0


@dataclass
class Deduper:
    near_threshold: float = 0.8
    semantic_threshold: float = 0.95
    by_url: dict[str, str] = field(default_factory=dict)
    by_hash: dict[str, str] = field(default_factory=dict)
    sigs: dict[str, Sig] = field(default_factory=dict)
    _lsh: Any = None
    _emb_ids: list[str] = field(default_factory=list)
    _emb: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        from datasketch import MinHashLSH

        # Candidate generation uses a lower LSH threshold (banding is probabilistic and misses pairs just above it);
        # the decision is the estimated Jaccard ≥ near_threshold below.
        self._lsh = MinHashLSH(threshold=max(0.3, self.near_threshold - 0.3), num_perm=NUM_PERM)

    @classmethod
    def from_existing(cls, existing: list[Sig], near_threshold: float = 0.8, semantic_threshold: float = 0.95) -> Deduper:
        d = cls(near_threshold=near_threshold, semantic_threshold=semantic_threshold)
        for s in existing:
            d.add(s)
        return d

    def add(self, s: Sig) -> None:
        self.sigs[s.id] = s
        self.by_url.setdefault(s.url_hash, s.id)
        self.by_hash.setdefault(s.content_hash, s.id)
        if s.minhash and s.id not in self._lsh:
            self._lsh.insert(s.id, _minhash(s.minhash))
        if s.embedding:
            v = np.asarray(s.embedding, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                self._emb_ids.append(s.id)
                self._emb.append(v / n)

    def check(self, s: Sig) -> DedupeResult:
        if s.url_hash in self.by_url and self.by_url[s.url_hash] != s.id:
            return DedupeResult("exact", self.by_url[s.url_hash], 1.0)
        if s.content_hash in self.by_hash and self.by_hash[s.content_hash] != s.id:
            return DedupeResult("exact", self.by_hash[s.content_hash], 1.0)
        if s.minhash:
            mh = _minhash(s.minhash)
            best, best_j = None, 0.0
            for other in self._lsh.query(mh):
                o = self.sigs.get(other)
                if not o or not o.minhash or other == s.id:
                    continue
                j = float(mh.jaccard(_minhash(o.minhash)))
                if j > best_j:
                    best, best_j = other, j
            if best and best_j >= self.near_threshold:
                return DedupeResult("near", best, round(best_j, 4))
        if s.embedding and self._emb:
            v = np.asarray(s.embedding, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                sims = np.stack(self._emb) @ (v / n)
                i = int(np.argmax(sims))
                if float(sims[i]) >= self.semantic_threshold and self._emb_ids[i] != s.id:
                    return DedupeResult("semantic", self._emb_ids[i], round(float(sims[i]), 4))
        return DedupeResult(None, None)
