# Quality gates

Code, not the model's judgement (`app/news/quality.py`). A draft reaches the review queue only if every
gate passes; otherwise it is saved as `failed_quality` with the reasons and the closest-matching passages, visible in the
run detail. Failed drafts are not retried automatically (N-13); "Retry failed drafts" re-drafts on demand. After the
agent's gates, the web app runs v4's new-article validators (MDX compiles, internal links exist, title unique site-wide);
an error there is gate `web_validation`. Thresholds are global (Settings → Quality gates).

| Gate | Checks | Default thresholds |
|---|---|---|
| `source_similarity` | per paragraph (quotes removed) against every source: share of 8-grams found in that source; longest verbatim word run; embedding cosine against each source paragraph. Reports the top matches. | overlap ≤ 12 %; run < 25 words; cosine ≤ 0.93 |
| `grounding` | every sentence with a number, version, date or CVE: each such token must appear in the fetched **sources** (titles and text) as a whole token ("10" is not supported by "2010", "1.3" not by "1.35"; numbers compared as digits, dates by month + day). The fact sheet is the researcher's own output, so its claim text is not evidence: each claim's snippet is checked against the source text (normalised case/punctuation/whitespace, then a fuzzy token-window match ≥ 0.85) and unverified claims are listed as warnings and in `claims_unverified`. Versions inside code blocks too. Products/organisations not in any source: warning. | any unsupported critical token fails |
| `quotes` | quoted spans of ≥ 4 words: verbatim in a source, ≤ 25 words, at most one per source, attributed (said/according to/… or the publisher named in the sentence) | 25 words, 1 per source |
| `value` | a “Why it matters” section with ≥ 30 words; share of words in analysis sections (why it matters, impact, who is affected, what to do, migration, recommendations) ≥ 25 %; length within the job's range (single stories) | 25 %; job min/max words |
| `style` | banned phrases (profile), required sections (profile), no H1, balanced code fences, no raw script/iframe/style, Flesch reading ease, links: https and either a cited source or an allowlisted host | Flesch 25–75 |
| `seo` | title ≤ ~60 (warning above 60, fail above 70), meta 110–160, JSON-LD is a `NewsArticle` with a headline ≤ 110, ≥ 1 category, ≤ 3 internal links, ≥ 1 source | |
| `duplicate` | no open draft from another job and no published article for the same story; the database's one-active-draft index is the final guard | |

Tests (`tests/test_news_quality.py`): a well-grounded original draft passes every gate; **a draft that copies a source
paragraph fails** `source_similarity` and reports that source first; **an unsupported number fails** `grounding`, also when the fact sheet invents a claim and snippet for it, and a
number that only occurs inside a longer one fails (`tests/test_review_news.py`); invented,
over-long, unattributed and second quotes from one source each fail; a thin restatement fails `value`; banned phrases,
missing sections and off-allowlist links fail `style`; SEO problems fail `seo`.

The crew is also told the rules (CREW_RULES in `app/news/crew.py`), and the fact-checker removes unsupported claims,
but the gates do not trust either: they recheck everything in code.
