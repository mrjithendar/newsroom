"""Quality gates (docs/news/QUALITY_GATES.md). Code, not the model's own judgement: a draft reaches the review queue only
when every gate passes; otherwise it is saved as `failed_quality` with the reasons and the highest-matching passages.

Gates: source_similarity · grounding · quotes · value · style · seo · duplicate. The web app re-validates the parts
that need its data (MDX compiles, internal links exist, the title is unique site-wide) before the draft is stored.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import JobConfig, Quality
from .schemas import FactSheet, NewsDraft
from .text import CVE, GAZETTEER, VERSION, flesch_reading_ease, host_of, ngrams, normalize_url, sentences, words

Embed = Callable[[list[str]], Awaitable[list[list[float]]]]

ANALYSIS_HEADINGS = re.compile(r"^(why it matters|what it means|impact|who('s| is) affected|what to do|what you should do|action items?|migration|how to (upgrade|migrate|respond)|recommendations?|our take|analysis)\b", re.I)
QUOTE = re.compile(r"[“\"]([^”\"\n]{12,600})[”\"]")
ATTRIBUTION = re.compile(r"\b(said|says|wrote|writes|according to|noted|notes|stated|states|announced|explained|told|warned|in a (blog )?post|in (its|the) (advisory|release notes|announcement|changelog))\b", re.I)
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE = re.compile(rf"\b{MONTHS}\.? \d{{1,2}}(?:, \d{{4}})?\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,2}} {MONTHS} \d{{4}}\b")
NUMBER = re.compile(r"(?<![\w.#/-])(\d+(?:[.,]\d+)*\s?(?:%|percent|ms|s\b|x\b|GB|GiB|MB|TB|vCPU|CPU|nodes?|users?|requests?|hours?|days?|minutes?|seconds?)?)(?![\w.])", re.I)
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")
TRIVIAL_NUMBERS = {"0", "1", "2", "3", "one", "two"}


@dataclass
class Gate:
    name: str
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.failures.append(msg[:400])

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failures": self.failures[:20], "warnings": self.warnings[:20], **({"detail": self.detail} if self.detail else {})}


@dataclass
class Report:
    gates: dict[str, Gate]

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates.values())

    @property
    def failed_gates(self) -> list[str]:
        return [n for n, g in self.gates.items() if not g.passed]

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "failed": self.failed_gates, "gates": {n: g.as_dict() for n, g in self.gates.items()}}


@dataclass
class SourceDoc:
    url: str
    title: str
    publisher: str
    text: str


# --- helpers --------------------------------------------------------------------------------------------------------


def strip_code(md: str) -> str:
    return re.sub(r"(```|~~~)[\s\S]*?\1", " ", md)


def code_blocks(md: str) -> list[str]:
    return [m.group(2) for m in re.finditer(r"(```|~~~)[^\n]*\n([\s\S]*?)\1", md)]


def paragraphs(md: str) -> list[str]:
    out = []
    for block in re.split(r"\n\s*\n", strip_code(md)):
        b = block.strip()
        if not b or b.startswith("#") or b.startswith("|") or b.startswith("<"):
            continue
        out.append(re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", b, flags=re.M))
    return out


def headings(md: str) -> list[tuple[int, str]]:
    return [(len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"^(#{1,6})\s+(.+?)\s*#*\s*$", strip_code(md), re.M)]


def sections(md: str) -> list[tuple[str, str]]:
    """(heading, text) for every H2/H3 section; text before the first heading is ("", …)."""
    out: list[tuple[str, str]] = []
    cur = ""
    buf: list[str] = []
    for line in strip_code(md).splitlines():
        m = re.match(r"^#{2,3}\s+(.+?)\s*$", line)
        if m:
            out.append((cur, "\n".join(buf)))
            cur, buf = m.group(1), []
        else:
            buf.append(line)
    out.append((cur, "\n".join(buf)))
    return out


def without_quotes(text: str) -> str:
    return QUOTE.sub(" ", text)


def _norm(s: str) -> str:
    return " ".join(words(s))


def longest_shared_run(a: list[str], b_ngrams_by_len: Callable[[int], set[tuple[str, ...]]], cap: int) -> int:
    """Longest k (≤ cap) such that some k-word run of `a` also occurs in the source (binary search on k)."""
    lo, hi = 0, min(cap, len(a))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        grams = b_ngrams_by_len(mid)
        if any(g in grams for g in ngrams(a, mid)):
            lo = mid
        else:
            hi = mid - 1
    return lo


# --- gates ----------------------------------------------------------------------------------------------------------


async def gate_similarity(draft: NewsDraft, sources: list[SourceDoc], q: Quality, embed: Embed | None) -> Gate:
    g = Gate("source_similarity")
    paras = [p for p in paragraphs(draft.body_mdx) if len(words(without_quotes(p))) >= 12]
    src_tokens = [(s, words(s.text)) for s in sources if s.text]
    cache: dict[tuple[int, int], set[tuple[str, ...]]] = {}

    def grams(i: int, n: int) -> set[tuple[str, ...]]:
        key = (i, n)
        if key not in cache:
            cache[key] = set(ngrams(src_tokens[i][1], n))
        return cache[key]

    matches: list[dict[str, Any]] = []
    for pi, p in enumerate(paras):
        toks = words(without_quotes(p))
        mine = ngrams(toks, q.ngram)
        for si, (s, _toks) in enumerate(src_tokens):
            if not mine:
                continue
            shared = sum(1 for x in mine if x in grams(si, q.ngram))
            overlap = shared / len(mine)
            run = longest_shared_run(toks, functools.partial(grams, si), q.max_shared_run_words + 5) if shared else 0
            if overlap > 0 or run >= q.ngram:
                matches.append({"paragraph": pi, "source": s.url, "overlap": round(overlap, 3), "longest_run": run, "excerpt": p[:240]})
            if overlap > q.max_ngram_overlap:
                g.fail(f"paragraph {pi + 1} shares {overlap:.0%} of its {q.ngram}-word phrases with {host_of(s.url)} (max {q.max_ngram_overlap:.0%})")
            if run >= q.max_shared_run_words:
                g.fail(f"paragraph {pi + 1} copies a {run}-word run from {host_of(s.url)} outside a quote (max {q.max_shared_run_words - 1})")
    if embed and paras and src_tokens:
        src_paras = [(s.url, sp) for s, _ in src_tokens for sp in re.split(r"\n\s*\n", s.text) if len(words(sp)) >= 12][:300]
        if src_paras:
            vecs = await embed([without_quotes(p) for p in paras] + [sp for _, sp in src_paras])
            a = _unit(np.asarray(vecs[: len(paras)], dtype=np.float32))
            b = _unit(np.asarray(vecs[len(paras) :], dtype=np.float32))
            sims = a @ b.T
            for pi in range(len(paras)):
                j = int(np.argmax(sims[pi]))
                c = float(sims[pi, j])
                if c > q.max_paragraph_cosine:
                    g.fail(f"paragraph {pi + 1} is a near paraphrase of a passage on {host_of(src_paras[j][0])} (cosine {c:.2f} > {q.max_paragraph_cosine:.2f})")
                matches.append({"paragraph": pi, "source": src_paras[j][0], "cosine": round(c, 3), "excerpt": paras[pi][:240], "source_excerpt": src_paras[j][1][:240]})
    matches.sort(key=lambda m: (-(m.get("overlap") or 0), -(m.get("cosine") or 0), -(m.get("longest_run") or 0)))
    g.detail = {"top_matches": matches[:8], "paragraphs_checked": len(paras), "sources": len(src_tokens)}
    return g


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


def critical_tokens(sentence: str) -> set[str]:
    out: set[str] = set()
    for m in CVE.findall(sentence):
        out.add(m.upper())
    for m in DATE.findall(sentence):
        out.add(m)
    s = DATE.sub(" ", CVE.sub(" ", sentence))
    for m in VERSION.findall(s):
        out.add(m)
    s = VERSION.sub(" ", s)
    for m in NUMBER.findall(s):
        num = re.match(r"\d+(?:[.,]\d+)*", m)
        if num and num.group(0) not in TRIVIAL_NUMBERS:
            out.add(num.group(0))
    return out


def _supported(token: str, corpus: str, corpus_digits: str) -> bool:
    t = token.lower()
    if t in corpus:
        return True
    digits = re.sub(r"[^\d.]", "", t)
    if digits and re.search(rf"(?<![\d.]){re.escape(digits)}(?![\d])", corpus_digits):
        return True
    if DATE.fullmatch(token):  # "Nov 4" vs "November 4, 2026": compare month + day
        m = re.match(r"([A-Za-z]{3})[a-z]*\.? (\d{1,2})", token)
        if m and re.search(rf"{m.group(1).lower()}[a-z]*\.? 0?{m.group(2)}\b", corpus):
            return True
    return False


def gate_grounding(draft: NewsDraft, sources: list[SourceDoc], facts: FactSheet) -> Gate:
    g = Gate("grounding")
    corpus = "\n".join([*(c.claim + " " + c.snippet for c in facts.claims), *(s.title + " " + s.text for s in sources)]).lower()
    corpus_digits = re.sub(r"(?<=\d),(?=\d{3})", "", corpus)
    unmapped: list[dict[str, Any]] = []
    checked = 0
    text = strip_code(draft.body_mdx) + "\n\n" + draft.dek + "\n\n" + "\n".join(draft.key_takeaways)
    for sent in sentences(re.sub(LINK, r"\1", text)):
        if sent.startswith("#"):
            continue
        toks = critical_tokens(sent)
        if not toks:
            continue
        checked += 1
        missing = sorted(t for t in toks if not _supported(t, corpus, corpus_digits))
        if missing:
            unmapped.append({"sentence": sent[:300], "unsupported": missing})
            g.fail(f"unsupported {', '.join(missing[:4])} in: “{sent[:140]}”")
    for code in code_blocks(draft.body_mdx):
        for v in set(VERSION.findall(code)):
            if not _supported(v, corpus, corpus_digits):
                g.fail(f"code block uses version {v}, which no source mentions")
    # Named entities: a product or organisation the sources never mention is a warning (context is allowed).
    body_low = strip_code(draft.body_mdx).lower()
    for _kind, names in GAZETTEER.items():
        for name in names:
            if len(name) > 3 and re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", body_low) and name.lower() not in corpus:
                g.warnings.append(f"{name} is mentioned but not in any source (fine as context; check it is not a claim)")
    g.detail = {"sentences_checked": checked, "unmapped": unmapped[:20]}
    return g


def gate_quotes(draft: NewsDraft, sources: list[SourceDoc], q: Quality) -> Gate:
    g = Gate("quotes")
    per_source: dict[str, int] = {}
    body = strip_code(draft.body_mdx)
    for m in QUOTE.finditer(body):
        quote = m.group(1).strip()
        n = len(words(quote))
        if n < 4:
            continue  # a quoted term, not a quotation
        if n > q.max_quote_words:
            g.fail(f"quote of {n} words (max {q.max_quote_words}): “{quote[:80]}…”")
        nq = _norm(quote)
        src = next((s for s in sources if nq and nq in _norm(s.text)), None)
        if src is None:
            g.fail(f"quote not found in any source (quotes must be verbatim): “{quote[:80]}”")
            continue
        per_source[src.url] = per_source.get(src.url, 0) + 1
        if per_source[src.url] > 1:
            g.fail(f"more than one quote from {host_of(src.url)}")
        start = body.rfind("\n", 0, m.start()) + 1
        end = body.find("\n", m.end())
        sentence = body[start : end if end > 0 else len(body)]
        names = [src.publisher, src.title.split(" ")[0] if src.title else "", host_of(src.url).split(".")[0]]
        if not ATTRIBUTION.search(sentence) and not any(nm and nm.lower() in sentence.lower() for nm in names):
            g.fail(f"quote is not attributed: “{quote[:80]}”")
    g.detail = {"quotes_per_source": per_source}
    return g


def gate_value(draft: NewsDraft, cfg: JobConfig, q: Quality, required: list[str]) -> Gate:
    g = Gate("value")
    secs = sections(draft.body_mdx)
    total = len(words(strip_code(draft.body_mdx)))
    analysis = sum(len(words(t)) for h, t in secs if h and ANALYSIS_HEADINGS.match(h))
    has_why = any(h and re.match(r"why it matters", h, re.I) and len(words(t)) >= 30 for h, t in secs)
    if not has_why:
        g.fail("no “Why it matters” section with at least 30 words of analysis")
    share = analysis / total if total else 0.0
    if share < q.min_analysis_share:
        g.fail(f"original analysis is {share:.0%} of the article (min {q.min_analysis_share:.0%}); it reads as a restatement")
    if not draft.digest_items:
        if total < cfg.length_min_words:
            g.fail(f"{total} words, below the job's minimum of {cfg.length_min_words}")
        if total > cfg.length_max_words:
            g.fail(f"{total} words, above the job's maximum of {cfg.length_max_words}")
    g.detail = {"words": total, "analysis_words": analysis, "analysis_share": round(share, 3), "required_sections": required}
    return g


def gate_style(draft: NewsDraft, q: Quality, banned: list[str], required: list[str], allowed_hosts: set[str]) -> Gate:
    g = Gate("style")
    body = draft.body_mdx
    low = body.lower()
    for p in banned:
        if p and p.lower() in low:
            g.fail(f"banned phrase “{p}”")
    hs = [h.lower() for _, h in headings(body)]
    for r in required:
        if not any(h.startswith(r.lower()) for h in hs):
            g.fail(f"required section “{r}” is missing")
    if re.search(r"^#\s", strip_code(body), re.M):
        g.fail("body contains an H1 (the title is the H1)")
    if body.count("```") % 2:
        g.fail("unbalanced code fence")
    if re.search(r"<\s*(script|iframe|style)\b", body, re.I):
        g.fail("raw script/iframe/style in the body")
    fre = flesch_reading_ease(strip_code(body))
    if not q.readability_min <= fre <= q.readability_max:
        g.fail(f"readability (Flesch reading ease) {fre} outside {q.readability_min}–{q.readability_max}")
    src_urls = {normalize_url(s.url) for s in draft.sources}
    for _text, href in LINK.findall(body):
        if href.startswith("/") or href.startswith("#"):
            continue
        if not href.startswith("https://"):
            g.fail(f"link {href[:80]} is not https")
            continue
        if normalize_url(href) not in src_urls and host_of(href) not in allowed_hosts:
            g.fail(f"link {href[:100]} is neither a cited source nor on the allowlist")
    g.detail = {"readability": fre}
    return g


def gate_seo(draft: NewsDraft, q: Quality) -> Gate:
    g = Gate("seo")
    t = draft.title.strip()
    if not t:
        g.fail("title missing")
    elif len(t) > q.title_max + 10:
        g.fail(f"title is {len(t)} characters (max ~{q.title_max})")
    elif len(t) > q.title_max:
        g.warnings.append(f"title is {len(t)} characters; ≤ {q.title_max} shows fully in results")
    m = draft.meta_description.strip()
    if not q.meta_min <= len(m) <= q.meta_max:
        g.fail(f"meta description is {len(m)} characters ({q.meta_min}–{q.meta_max})")
    ld = draft.schema_jsonld
    if not isinstance(ld, dict) or ld.get("@type") != "NewsArticle" or not ld.get("headline"):
        g.fail("JSON-LD must be a NewsArticle with a headline")
    else:
        try:
            json.dumps(ld)
        except (TypeError, ValueError):
            g.fail("JSON-LD is not serialisable")
        if len(str(ld.get("headline"))) > 110:
            g.fail("JSON-LD headline over 110 characters")
    if not draft.categories:
        g.fail("no category")
    internal = [h for _t, h in LINK.findall(draft.body_mdx) if h.startswith("/")]
    if len(internal) > q.max_internal_links or len(draft.internal_links) > q.max_internal_links:
        g.fail(f"more than {q.max_internal_links} internal links")
    if not draft.sources:
        g.fail("no sources (news must cite its sources)")
    return g


def gate_duplicate(duplicate_reason: str | None) -> Gate:
    g = Gate("duplicate")
    if duplicate_reason:
        g.fail(duplicate_reason)
    return g


async def run_gates(draft: NewsDraft, *, sources: list[SourceDoc], facts: FactSheet, cfg: JobConfig, q: Quality, banned: list[str], required: list[str], allowed_hosts: set[str], embed: Embed | None, duplicate_reason: str | None) -> Report:
    gates = {
        "source_similarity": await gate_similarity(draft, sources, q, embed),
        "grounding": gate_grounding(draft, sources, facts),
        "quotes": gate_quotes(draft, sources, q),
        "value": gate_value(draft, cfg, q, required),
        "style": gate_style(draft, q, banned, required, allowed_hosts),
        "seo": gate_seo(draft, q),
        "duplicate": gate_duplicate(duplicate_reason),
    }
    return Report(gates)
