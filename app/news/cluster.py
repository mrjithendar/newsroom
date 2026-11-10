"""Incremental story clustering (docs/news/ML.md "Clustering").

Each new (non-duplicate) item joins the most similar open cluster when, within the cluster's time window,
  cosine(item, centroid) ≥ join_cosine, or
  cosine ≥ entity_cosine and entity Jaccard ≥ min_entity_overlap (shared products / orgs / CVE ids), or
  they share a CVE id (the same advisory reported by several sources);
otherwise it starts a new cluster. Centroids are running means (re-normalised). The burst score measures how fast
coverage is growing across *distinct sources*: sources that joined in the last 12 hours vs the 24 hours before.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .text import jaccard


@dataclass
class Member:
    source_id: str
    at: datetime
    item_id: str = ""


@dataclass
class ClusterState:
    id: str
    title: str
    centroid: np.ndarray | None
    entities: set[str] = field(default_factory=set)
    cve_ids: set[str] = field(default_factory=set)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    members: list[Member] = field(default_factory=list)
    severity: float | None = None
    exploited: bool = False
    status: str = "new"
    dirty: bool = False

    @property
    def item_count(self) -> int:
        return len(self.members)

    @property
    def source_count(self) -> int:
        return len({m.source_id for m in self.members})

    def burst(self, now: datetime) -> float:
        recent = {m.source_id for m in self.members if now - m.at <= timedelta(hours=12)}
        before = {m.source_id for m in self.members if timedelta(hours=12) < now - m.at <= timedelta(hours=36)} - recent
        growth = len(recent) - len(before) / 2
        return round(max(0.0, min(1.0, 1 - math.exp(-max(0.0, growth) / 2.5))), 4)


def _unit(v: list[float] | np.ndarray | None) -> np.ndarray | None:
    if v is None or len(v) == 0:
        return None
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else None


@dataclass
class Clusterer:
    join_cosine: float = 0.80
    entity_cosine: float = 0.66
    min_entity_overlap: float = 0.34
    window_hours: float = 96
    clusters: dict[str, ClusterState] = field(default_factory=dict)

    def best(self, emb: np.ndarray | None, entities: set[str], cves: set[str], at: datetime) -> tuple[ClusterState | None, float]:
        best, best_s = None, 0.0
        for c in self.clusters.values():
            if c.last_seen and at - c.last_seen > timedelta(hours=self.window_hours):
                continue
            if cves and cves & c.cve_ids:
                return c, 1.0
            cos = float(c.centroid @ emb) if (c.centroid is not None and emb is not None) else 0.0
            ent = jaccard(entities, c.entities)
            ok = cos >= self.join_cosine or (cos >= self.entity_cosine and ent >= self.min_entity_overlap)
            s = cos + 0.25 * ent
            if ok and s > best_s:
                best, best_s = c, s
        return best, best_s

    def assign(self, *, item_id: str, title: str, embedding: list[float] | None, entities: set[str], cves: set[str], source_id: str, at: datetime, severity: float | None = None, exploited: bool = False, force_cluster: str | None = None) -> tuple[str, bool]:
        """Returns (cluster_id, created)."""
        emb = _unit(embedding)
        c = self.clusters.get(force_cluster) if force_cluster else None
        if c is None:
            c, _ = self.best(emb, entities, cves, at)
        created = c is None
        if c is None:
            c = ClusterState(id=str(uuid.uuid4()), title=title[:300], centroid=emb, first_seen=at, last_seen=at)
            self.clusters[c.id] = c
        else:
            if emb is not None:
                n = len(c.members)
                c.centroid = _unit((c.centroid * n + emb) / (n + 1)) if c.centroid is not None else emb
            c.last_seen = max(c.last_seen or at, at)
            c.first_seen = min(c.first_seen or at, at)
        c.members.append(Member(source_id, at, item_id))
        c.entities |= entities
        c.cve_ids |= cves
        if severity is not None:
            c.severity = max(c.severity or 0.0, severity)
        c.exploited = c.exploited or exploited
        c.dirty = True
        return c.id, created
