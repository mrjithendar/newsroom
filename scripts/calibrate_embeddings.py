#!/usr/bin/env python3
"""Map the news desk's cosine cutoffs from bge-small to another embeddings model (app/calibration.py).

Cosines of two models are not on one scale, but on the same pairs of texts they rank alike. So a cutoff keeps its
meaning if it keeps the same share of pairs above it: the bge-small cutoff t passes a share s of the pairs, and the new
model's cutoff is the cosine that passes the same share s of the same pairs (quantile mapping). Agreement is the F1 of
the two decisions on those pairs: how often both models call a pair the same way.

The pairs are the kind the desk compares. Items: title + summary from the news desk's default feeds, embedded as at
ingest (desk._item_text); every pair of them (the cluster cutoffs, 0.66 / 0.80). Copies: each item lightly reworded as a
syndicated copy would be (sentence order changed, every 8th word dropped, another title prefix), paired with its
original (the high tail: dedupe 0.95, the paragraph gate 0.93).

  kubectl -n embeddings port-forward svc/bge-m3 11434:11434 &
  uv run python scripts/calibrate_embeddings.py --model-dir /models/bge-small-en-v1.5 \
      --ollama-url http://127.0.0.1:11434 --ollama-model bge-m3:567m

The Ask the Blog floor is not mapped: scripts/rag_eval.py measures it on the evaluation set ("calibrated floor").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.calibration import BGE_SMALL  # noqa: E402
from app.embeddings import OllamaEmbeddings, OnnxEmbeddings  # noqa: E402
from app.news.feeds import parse_feed  # noqa: E402

# The news desk's seeded sources (moks-cms src/db/migrations, news_sources).
FEEDS = [
    ("rss", "https://kubernetes.io/feed.xml"),
    ("rss", "https://www.cncf.io/feed/"),
    ("rss", "https://aws.amazon.com/blogs/aws/feed/"),
    ("rss", "https://aws.amazon.com/about-aws/whats-new/recent/feed/"),
    ("rss", "https://cloudblog.withgoogle.com/rss/"),
    ("rss", "https://www.hashicorp.com/blog/feed.xml"),
    ("rss", "https://openai.com/news/rss.xml"),
]
CUTOFFS = {
    "cluster_entity_cosine": BGE_SMALL.cluster_entity_cosine,
    "cluster_join_cosine": BGE_SMALL.cluster_join_cosine,
    "paragraph_cosine": BGE_SMALL.paragraph_cosine,
    "dedupe_semantic_cosine": BGE_SMALL.dedupe_semantic_cosine,
}


def item_text(title: str, summary: str) -> str:
    return f"{title}. {summary[:1200]}"  # app/news/desk.py _item_text


def reworded(title: str, summary: str) -> str:
    """A syndicated copy: other title prefix, sentences rotated by one, every 8th word dropped."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", summary) if s]
    rotated = sentences[1:] + sentences[:1] if len(sentences) > 1 else sentences
    words = " ".join(rotated).split()
    kept = " ".join(w for k, w in enumerate(words) if k % 8 != 7)
    return item_text(f"Report: {title}", kept)


async def fetch_items(per_feed: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers={"User-Agent": "calibrate-embeddings/1.0"}) as c:
        for kind, url in FEEDS:
            try:
                r = await c.get(url)
                r.raise_for_status()
            except httpx.HTTPError as e:
                print(f"skipped {url}: {type(e).__name__}", file=sys.stderr)
                continue
            items = [(i.title.strip(), re.sub(r"\s+", " ", i.summary).strip()) for i in parse_feed(kind, r.text, url)]
            items = [(t, s) for t, s in items if t and len(s.split()) >= 12][:per_feed]
            print(f"{len(items):3d} items from {url}", file=sys.stderr)
            out += items
    return out


def cosines(vecs: list[list[float]], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Item × item pairs (upper triangle) and item × its copy, from vectors [items..., copies...]."""
    v = np.asarray(vecs, dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    items, copies = v[:n], v[n:]
    iu = np.triu_indices(n, k=1)
    return (items @ items.T)[iu], np.einsum("ij,ij->i", items, copies)


def mapped(small: np.ndarray, other: np.ndarray, t: float) -> tuple[float, float, float]:
    share = float(np.mean(small >= t))
    cut = float(np.quantile(other, 1 - share)) if share > 0 else float(other.max())
    a, b = small >= t, other >= cut
    tp = int(np.sum(a & b))
    f1 = 2 * tp / (int(a.sum()) + int(b.sum())) if (a.sum() + b.sum()) else 1.0
    return share, cut, f1


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/models/bge-small-en-v1.5")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", default="bge-m3:567m")
    ap.add_argument("--per-feed", type=int, default=40)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    items = await fetch_items(a.per_feed)
    if len(items) < 30:
        print(f"only {len(items)} items: not enough pairs", file=sys.stderr)
        return 2
    texts = [item_text(t, s) for t, s in items] + [reworded(t, s) for t, s in items]
    small = await OnnxEmbeddings(a.model_dir).embed(texts)
    other_emb = OllamaEmbeddings(a.ollama_url, a.ollama_model)
    other = await other_emb.embed(texts)
    n = len(items)
    s_pairs, s_copies = cosines(small, n)
    o_pairs, o_copies = cosines(other, n)
    s_all, o_all = np.concatenate([s_pairs, s_copies]), np.concatenate([o_pairs, o_copies])
    rank = float(np.corrcoef(np.argsort(np.argsort(s_all)), np.argsort(np.argsort(o_all)))[0, 1])

    rows = {name: mapped(s_all, o_all, t) for name, t in CUTOFFS.items()}
    print(f"\n{n} items, {len(s_pairs)} item pairs + {n} item/copy pairs; Spearman rank correlation of the two models' cosines: {rank:.3f}")
    print(f"copies: bge-small cosine median {np.median(s_copies):.3f} (min {s_copies.min():.3f}); {other_emb.name} median {np.median(o_copies):.3f} (min {o_copies.min():.3f})")
    print(f"item pairs: bge-small median {np.median(s_pairs):.3f}; {other_emb.name} median {np.median(o_pairs):.3f}\n")
    print(f"| cutoff | bge-small | pairs above | {other_emb.name} | agreement (F1) |")
    print("|---|---|---|---|---|")
    for name, (share, cut, f1) in rows.items():
        print(f"| {name} | {CUTOFFS[name]:.2f} | {share:.2%} | {cut:.3f} | {f1:.3f} |")
    if a.json:
        print(json.dumps({"items": n, "spearman": rank, **{k: {"bge_small": CUTOFFS[k], "share": v[0], "mapped": v[1], "f1": v[2]} for k, v in rows.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
