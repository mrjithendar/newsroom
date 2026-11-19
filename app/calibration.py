"""Similarity cutoffs per embeddings model.

A cosine of 0.80 means something different for every model: bge-small packs related texts into a narrow band near 1,
bge-m3 spreads them out. Every cutoff that compares two vectors is therefore a property of the model that made them,
and changing EMBEDDINGS_PROVIDER / OLLAMA_EMBED_MODEL must change the cutoffs with it. This table is where they live.

- ask_min_score: Ask the Blog's floor (ASK_BLOG_MIN_SCORE overrides it): the highest cosine that loses no expected
  answer of the evaluation set (scripts/rag_eval.py "calibrated floor"; moks-cms docs/seo/RAG.md "Score calibration").
- cluster_join_cosine / cluster_entity_cosine: the news desk's clusterer (app/news/cluster.py): an item joins a
  cluster on cosine alone, or on the lower cosine when it also shares entities.
- dedupe_semantic_cosine: the news deduper's "semantic" duplicate (a syndicated or lightly reworded copy; dedupe.py).
- paragraph_cosine: the news quality gate's paraphrase check (Quality.max_paragraph_cosine; an admin value other than
  bge-small's default 0.93 wins, see effective_paragraph_cosine).

bge-small's numbers are the ones the code used before this table existed. A model that is not listed (hash, api,
nomic-embed-text) keeps them, as it did then. How the other rows were measured is in scripts/calibrate_embeddings.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Calibration:
    ask_min_score: float
    cluster_join_cosine: float
    cluster_entity_cosine: float
    dedupe_semantic_cosine: float
    paragraph_cosine: float


BGE_SMALL = Calibration(ask_min_score=0.66, cluster_join_cosine=0.80, cluster_entity_cosine=0.66, dedupe_semantic_cosine=0.95, paragraph_cosine=0.93)

# bge-m3:567m on mk4's Ollama, measured 2026-11-19 with scripts/calibrate_embeddings.py: 189 items from the desk's seven
# default feeds, 17,766 item pairs + 189 item/reworded-copy pairs, Spearman 0.770 between the two models' cosines.
# Each bge-small cutoff keeps its share of pairs (agreement F1 with bge-small's decision in brackets):
#   cluster_entity 0.66 → 0.523 (0.68) · cluster_join 0.80 → 0.699 (0.87) · paragraph 0.93 → 0.916 (0.98) ·
#   dedupe_semantic 0.95 → 0.933 (0.96). bge-m3 spreads unrelated items lower (pair median 0.47 vs 0.61) and keeps
#   reworded copies as close (copy median 0.97 vs 0.98), so the low cutoffs move most.
# ask_min_score is INTERIM: bge-small's 0.66 mapped the same way (0.523), not yet rag_eval.py's calibrated floor on the
# evaluation set, which is how the bge-small value was set. Replace it with that measurement before jithendar.com
# switches EMBEDDINGS_PROVIDER to ollama (or set ASK_BLOG_MIN_SCORE there).
BGE_M3 = Calibration(ask_min_score=0.52, cluster_join_cosine=0.70, cluster_entity_cosine=0.52, dedupe_semantic_cosine=0.93, paragraph_cosine=0.92)

CALIBRATIONS: dict[str, Calibration] = {
    "bge-small-en-v1.5": BGE_SMALL,
    "bge-m3": BGE_M3,
}


def model_key(name: str) -> str:
    """An Embeddings.name → its model: "onnx:bge-small-en-v1.5" → "bge-small-en-v1.5", "ollama:bge-m3:567m" → "bge-m3"."""
    rest = name.split(":", 1)[1] if ":" in name else name
    return rest.split(":", 1)[0]


def calibration_for(name: str) -> Calibration:
    return CALIBRATIONS.get(model_key(name), BGE_SMALL)


def effective_paragraph_cosine(setting: float, name: str) -> float:
    """The Settings page stores max_paragraph_cosine, and its default (0.93) was bge-small's. That default means "the
    model's own cutoff"; any other value is an admin's deliberate choice and is used as it is."""
    return calibration_for(name).paragraph_cosine if setting == BGE_SMALL.paragraph_cosine else setting
