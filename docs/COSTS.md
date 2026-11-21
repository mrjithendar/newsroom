# News desk costs

**Estimates.** The crews ran only against a deterministic test model in this work; no live Claude call was made, so the
token counts below are estimated from the prompt sizes, not measured. Replace them with real numbers from the run detail
(`news_runs.tokens`, `cost_usd`) after the first week. Prices: the platform's price table (`app/llm.py`, first-party list
prices as of 2026-10): Sonnet 5.5 $2 / $10 per million input/output tokens, Haiku 5.5 $0.10 / $0.50.

## Per draft (single story, default settings)

| Step | Model | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| Researcher (2–3 calls incl. tool results; up to 12 items × 3,000 chars + one primary source) | Sonnet | ~30,000 | ~2,000 | $0.080 |
| Analyst | Sonnet | ~8,000 | ~1,000 | $0.026 |
| Writer | Sonnet | ~10,000 | ~2,000 | $0.040 |
| Fact-checker | Sonnet | ~10,000 | ~2,000 | $0.040 |
| SEO editor | Sonnet | ~12,000 | ~2,500 | $0.049 |
| Art director | Sonnet | ~4,000 | ~300 | $0.011 |
| Vision: safety per variant + alt text on the final image | Haiku | ~4,000 each | ~200 | ~$0.001 |
| **Crew total** | | **~75,000** | **~10,000** | **≈ $0.25** |

Cached system blocks (the rules + each agent's prompt) bring the input part down somewhat after the first call in a
5-minute window. A per-draft token cap (`max_tokens_per_draft`, default 120,000) bounds the worst case at ≈ $0.35.

Covers: template $0. Gemini or OpenAI: 2 variants × the configured price (default estimate $0.04) = $0.08 + the vision checks.

Classification fallback (Haiku, 20 stories per call): < $0.001 per story. It runs during ingest and counts against the
**global** news caps (booked to the job or ingest run that ingested); over the cap the model's low-confidence guess is
kept (`model_low_conf`). Bootstrap (one-off, up to 1,000 items):
≈ $0.10–0.30. Style review (weekly, Sonnet, ≤ 12 diffs): ≈ $0.05–0.10. Training: CPU only, free.

## Per approved article

Not every draft is approved. Assuming 15 % fail the quality gates (they still cost a crew run) and 60 % of the rest are
approved:

| Covers | Cost per draft | Cost per approved article |
|---|---|---|
| template | ≈ $0.25 | ≈ $0.25 / (0.85 × 0.6) ≈ **$0.49** |
| Gemini/OpenAI (2 variants) | ≈ $0.33 | ≈ **$0.65** |

The analytics page shows the real figure ("cost per approved article", including rejected and failed drafts).

## Per day and month at the defaults

One job, daily, `max_articles_per_run` 3: ≈ 3 × $0.25 ≈ **$0.75/day**, ≈ $23/month if it found 3 stories every day.
The defaults then stop it earlier, by design:

| Cap | Default | Effect |
|---|---|---|
| job LLM per day | $1.00 | ≈ 4 drafts/day |
| job LLM per month | $15.00 | ≈ 60 drafts/month (≈ 20 days at 3/day) |
| job images per day / month | $0.50 / $5.00 | only matters with a paid provider |
| global LLM per day / month | $2.00 / $20.00 | all jobs together |
| global images per day / month | $1.00 / $10.00 | |
| global drafts per day | 10 | |

When a cap is reached, the run ends as **paused (budget)** with the cap named; nothing is sent first (every call is
checked against an upper-bound estimate). The estimate is *reserved* before the call: booked on the run's row under a
lock and then verified against the caps, so parallel drafts cannot all pass the same check and overspend together; the
reservation is released when the call ends and the actual cost booked. News spend never counts against the SEO admin
budget (N-07).

Levers: fewer articles per run/day, Haiku for the researcher/analyst (job → Writing → models per agent), fewer items per
story, a lower token cap, template covers.
