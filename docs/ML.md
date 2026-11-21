# News desk ML

CPU-only, in the news agent's process, as plain Python steps inside the flows. Libraries: scikit-learn 1.9.1, LightGBM 4.7.0,
datasketch 2.0.0, numpy. No MLflow: models are rows in `news_models` with their metrics; one active per kind.
Measured on the development Mac: importing the ML libraries adds ≈ 70 MiB; training the classifier on 1,000 items
(384-dim embeddings + 72 extra features) or the ranker on 2,000 rows takes ≈ 0.1 s each.

## Normalisation (`app/news/text.py`)

- Canonical URL: lower-case host without `www.`, default ports dropped, tracking parameters removed (`utm_*`, `gclid`,
  `fbclid`, `mc_*`, `ref`, …), remaining query sorted, fragment and trailing slash dropped. `url_hash` = SHA-256 of it.
- Clean text: scripts/styles/nav/footers dropped, paragraphs kept, entities decoded, bounded (excerpt ≤ 4,000 chars,
  text ≤ 20,000).
- Language: stop-word vote (en/de/fr/es) over the first 400 words; `null` when there is too little text (N-09).
- Entities: CVE ids, version strings, and a gazetteer of products and organisations (N-08). KEV/NVD feeds add CVSS and
  "known exploited".

## Dedupe (`app/news/dedupe.py`)

| Level | Method | Threshold |
|---|---|---|
| exact | same normalised URL (any source) or same content hash (title + text, normalised) | equal |
| near-duplicate | MinHash (64 permutations, `affine32`) over word 5-gram shingles; LSH for candidates (threshold 0.5), decision on estimated Jaccard. Skipped for texts under `near_dup_min_tokens` (30) words: a title plus a one-line summary has too few shingles for a stable estimate (exact and semantic still apply) | ≥ 0.80 |
| semantic | cosine of embeddings (the service's embedding model; bge-small in production) | ≥ 0.95 |

A duplicate is stored with `duplicate_of` and joins its original's cluster, so source counts stay honest. Tested on
fixtures: a syndicated copy with a small edit (Jaccard 0.84) is a near-duplicate; another article about the same release
is not (it clusters instead).

The text is the feed's title + summary: full article text is fetched only for drafting, so it is not available at
dedupe time. The 64 permutations stay: every stored signature (`news_items.minhash`) has exactly 64 values, and a
MinHash cannot be compared or rebuilt with a different count, so changing it would need a re-hash of the window. The
stacked embedding matrix of the 14-day window is built once per ingest and rebuilt only after an item is added.

## Clustering (`app/news/cluster.py`)

Incremental, over clusters seen in the last 96 hours. An item joins the best cluster when cosine(item, centroid) ≥ 0.80,
or cosine ≥ 0.66 with entity Jaccard ≥ 0.34, or they share CVE ids; otherwise it starts a new one. The CVE rule picks
the cluster with the highest CVE Jaccard (not the first one sharing an id), and it is skipped for an item naming more
than `cve_roundup_max` (5) CVE ids and for clusters already holding more than that (advisory roundups would otherwise
pull unrelated stories together). Centroids are running means. Entities and CVE ids are capped on save (80 / 30) by how
many members named them, with the counts stored in `entities.counts` / `entities.cve_counts`, so a reload keeps the
frequent ones (the cap used to be alphabetical). **Burst** = 1 − exp(−g / 2.5), where g = distinct sources that joined in the last 12 h − half of those that joined
in the 24 h before (clipped to 0).

## Novelty vs the site

The cluster title is searched in the site's pgvector index (the existing retriever). Best hit ≥ 0.86 → the cluster is an
`update_candidate`: it is not drafted as new news, and the candidate's reason names the existing article (to update it
through the SEO optimisation flow). When pgvector is not available, novelty is 1 (no match).

## Topics (`app/news/classifier.py`)

- Features: embedding ⊕ 64 hashed entity buckets ⊕ one keyword-hit feature per topic, built by `item_inputs` from **one
  item** (its embedding, entities, "title excerpt"). Training rows are items; at serving time a cluster is classified on
  its representative item (the non-duplicate member closest to the centroid), so the features are constructed exactly
  as in training (a parity test asserts identical vectors). Keyword rules (no model yet) still read the cluster's text.
- Model: one balanced logistic regression per topic (multi-label), Platt-calibrated, stored as numpy arrays (no pickle;
  the calibration is two arrays `cal_a`, `cal_b`; older artefacts without them load uncalibrated).
- Prediction: calibrated p; labels with p ≥ 0.5 (at least the best); if the best p < `classifier_threshold` (0.55)
  Claude Haiku picks 1–3 topics from the allowed list; without a model, keyword rules. Topics record their `method`:
  `model`, `model_low_conf` (a low-confidence guess kept because Claude was off, failed or over budget), `claude`,
  `rules`. Claude returns labels, not probabilities: its topics get `claude_topic_p` (Settings, default 0.6), which feeds
  `topic_match`.
- The Claude fallback during ingest goes through the news budget (global caps, booked to the job or ingest run); before
  this it bypassed every cap (app/llm.py's BudgetGuard skips `news.*` purposes).
- Bootstrap: "Bootstrap labels with Claude" labels up to 1,000 recent items once (≈ 50 calls of 20 items).
- Training: weekly and on demand, from Claude labels plus the owner's category corrections (human labels win).
  1. Grouped hold-out: ≈ 20 % of the rows, whole clusters at a time (two reports of one story never sit on both sides).
  2. Fit on the rest; Platt parameters from grouped 3-fold out-of-fold decision values; per-topic precision/recall/F1
     at p ≥ 0.5, macro F1 and the Brier score on the hold-out. `macro_f1_human` is the same on the hold-out rows with a
     human label (≥ 5 needed). A topic with one class only in the training split is listed in `excluded_topics` and
     left out of macro F1 for **both** models.
  3. The serving model is scored on the same hold-out rows and topics (featurised its own way). The new model is
     promoted only if its macro F1 is higher by more than 0.005 (strict; the serving model may have trained on some of
     those rows, so this is conservative). With no serving model, it is promoted.
  4. The shipped model is refit (and recalibrated) on all rows.

## Ranking (`app/news/scoring.py`)

v1 (transparent): score = Σ wᵢ·fᵢ, weights renormalised to 1 (job overrides allowed). Every candidate stores each
feature's value, weight and contribution.

| Feature | Value |
|---|---|
| credibility | mean of the top 3 distinct sources' credibility |
| source_count | 1 − exp(−(n − 1)/2) |
| burst | cluster burst (above) |
| recency | 0.5^(age / half-life), half-life 36 h by default |
| topic_match | the cluster's probability for the job's topics (or its best topic) |
| audience | similarity to the closest site article × (0.5 + 0.5 × its share of top-article clicks, GSC 28 d); 0.3 when unknown |
| novelty | 1 − similarity to the closest site article |
| severity | CVSS / 10; known exploited ⇒ ≥ 0.9 |
| keyword_overlap | share of the story's top terms that appear in the site's GSC queries |

Default weights: credibility .14, source_count .12, burst .10, recency .16, topic_match .14, audience .10, novelty .12,
severity .06, keyword_overlap .06.

v2 (learned): LightGBM, binary objective (N-12), 120 rounds, 7 leaves, deterministic. Labels: approved/scheduled = 1,
rejected = 0, except rejections whose `reject_category` is `style` or `inaccurate`: those judge the writing, not the
story choice, and are dropped (`dropped_writing_rejects` in the report). Performance at +7/+28 days is collected in
`news_feedback` for later use as a stronger label. Trains once there are ≥ `ranker_min_feedback` (100) labelled
decisions. Time-ordered split (oldest 80 % / newest 20 %). Promoted only if it beats, on the same held-out rows, on
**both** AUC and NDCG@5 (per run): v1, each row scored with its own job's current weights, **and** the serving v2 if
there is one. Its explanation is LightGBM's per-feature contributions, shown next to the v1 breakdown. Tests: a
synthetic signal v1 underweights is learned and promoted; pure noise is not; a copy of the serving model is not.

**Threshold.** v2's probabilities are not on v1's scale, so a job's `min_score` (tuned for v1) is not compared with
them directly. Training stores a score map with the model (`metrics.score_map`: 101 quantiles of v1 and v2 scores over
the last 5,000 candidates of any decision); at serving time the job's `min_score` is mapped to the v2 score that passes
the same share of candidates (`RankerModel.threshold_for`). The mapped threshold is in each candidate's breakdown and
its rejection reason. LightGBM scores tie, so the v2 pass rate can exceed v1's by the tied mass at the threshold.
Models trained before the map existed fall back to `min_score`.

## Selection (`app/news/select.py`)

In this order, each candidate gets exactly one reason: language → max age → excluded keywords/entities → include
keywords/entities → job topics → already covered in the dedupe window → existing site article (update candidate) →
score threshold (v1: `min_score`; v2: the mapped threshold above) → per-run cap → per-day cap (job time zone) → global
daily cap → diversity (per topic, per source per day).

Exploration (opt-in, `explore_rate` in Settings, default 0, max 0.2): each candidate held back only by the threshold
becomes eligible with that probability, so the ranker also gets labels for stories it would not have picked. It still
goes through every cap and diversity rule, is marked `explored` in its breakdown and says so in its reason. Digest mode: the top `digest_size` eligible stories become one article (counts as one toward the caps).
"Both": singles first, then a digest of eligible stories held back only by the per-run cap or diversity.

## Monitoring (`app/news/monitoring.py`)

- `news_model_macro_f1{model="classifier"}`, `news_model_holdout_auc{model="ranker"}`: the **serving** model's held-out
  metric, set on promotion, on rollback and by the dispatcher every 15 minutes from `news_models` (restart-safe).
  `news_model_auc` keeps its old meaning for the dashboard (ranker AUC, classifier macro F1) and follows the same rule;
  before, it was overwritten by every trained classifier, promoted or not.
- `news_topic_assignments_total{method}`: model, model_low_conf, claude, rules. The share of `model_low_conf` is the
  rate at which a low-confidence guess was kept.
- `news_model_score_psi{model}`: drift check, population stability index (10 bins on [0, 1]) of the serving model's
  scores vs the same scores on its training rows. Ranker: candidate scores from the last 10 runs it scored. Classifier:
  the top topic probability of clusters classified by the model in the last 7 days. Above 0.25 the inputs have shifted;
  retrain, and compare the hold-out metrics.
- Rollback: `NewsStore.activate_model(kind, version)` (one transaction; only a version with a stored artefact) via
  `app.news.desk.activate_model_version`, which also resets the gauges. The read-only MCP tool `list_news_models` lists
  the stored versions and their metrics. A mutating MCP tool is deliberately not offered (RUNBOOK "Model rollback").
