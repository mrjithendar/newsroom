"""Quality gates: a well-grounded original draft passes; a copied source paragraph, an unsupported number, an invented or
over-long quote, a thin restatement, banned phrases and SEO problems each fail with a reason."""

from __future__ import annotations

from app.embeddings import HashEmbeddings
from app.news.config import JobConfig, Quality
from app.news.quality import SourceDoc, run_gates
from tests.news_fixtures import GOOD_BODY, K8S_TEXT, K8S_TEXT_B, facts, good_draft

SOURCES = [SourceDoc("https://blog.example.org/k8s-135", "Kubernetes 1.35 released", "Example Blog", K8S_TEXT), SourceDoc("https://releases.example.net/kubernetes/v1.35", "v1.35", "Example releases", K8S_TEXT_B)]
CFG = JobConfig(length_min_words=150, length_max_words=900)
Q = Quality()


async def gates(draft, *, dup: str | None = None, banned: list[str] | None = None, required: list[str] | None = None):  # type: ignore[no-untyped-def]
    emb = HashEmbeddings()
    return await run_gates(draft, sources=SOURCES, facts=facts(), cfg=CFG, q=Q, banned=banned or ["game-changer"], required=required or ["Why it matters"], allowed_hosts={"kubernetes.io"}, embed=lambda t: emb.embed(t), duplicate_reason=dup)


async def test_good_draft_passes_every_gate() -> None:
    r = await gates(good_draft())
    assert r.passed, r.as_dict()
    assert r.gates["grounding"].detail["sentences_checked"] >= 2


async def test_copied_source_paragraph_fails_similarity() -> None:
    d = good_draft()
    d = d.model_copy(update={"body_mdx": K8S_TEXT + "\n\n" + GOOD_BODY.split("\n\n", 1)[1]})
    r = await gates(d)
    g = r.gates["source_similarity"]
    assert not g.passed and any("copies a" in f or "shares" in f for f in g.failures)
    assert g.detail["top_matches"][0]["source"] == "https://blog.example.org/k8s-135"


async def test_unsupported_number_fails_grounding() -> None:
    d = good_draft()
    d = d.model_copy(update={"body_mdx": GOOD_BODY.replace("The project counts 42 enhancements in this cycle.", "The project counts 57 enhancements and cuts restart time by 80% in this cycle.")})
    r = await gates(d)
    g = r.gates["grounding"]
    assert not g.passed and any("57" in f for f in g.failures) and any("80" in f for f in g.failures)


async def test_unsupported_version_in_code_fails() -> None:
    d = good_draft().model_copy(update={"body_mdx": GOOD_BODY + "\n\n```bash\nkubeadm upgrade apply v1.36.2\n```"})
    assert any("1.36.2" in f for f in (await gates(d)).gates["grounding"].failures)


async def test_quotes_must_be_verbatim_short_attributed_and_one_per_source() -> None:
    q_ok = "In-place pod resize has graduated to stable"
    body = GOOD_BODY + f'\n\nAccording to the Example Blog, "{q_ok}."'
    assert (await gates(good_draft().model_copy(update={"body_mdx": body}))).gates["quotes"].passed
    invented = GOOD_BODY + '\n\nThe maintainers said "this is the most important release we have ever shipped for operators".'
    assert any("not found" in f for f in (await gates(good_draft().model_copy(update={"body_mdx": invented}))).gates["quotes"].failures)
    long = GOOD_BODY + '\n\nThe blog said "' + " ".join(K8S_TEXT.split(". ")[1:3]) + '".'
    assert any("words (max" in f for f in (await gates(good_draft().model_copy(update={"body_mdx": long}))).gates["quotes"].failures)
    two = GOOD_BODY + f'\n\nThe blog said "{q_ok}". It also said "The release contains 42 enhancements in total".'
    assert any("more than one quote" in f for f in (await gates(good_draft().model_copy(update={"body_mdx": two}))).gates["quotes"].failures)
    unattributed = GOOD_BODY + f'\n\nIt is official: "{q_ok}".'
    assert any("not attributed" in f for f in (await gates(good_draft().model_copy(update={"body_mdx": unattributed}))).gates["quotes"].failures)


async def test_thin_restatement_fails_value() -> None:
    thin = "Kubernetes released version 1.35 with in-place pod resize promoted to stable and 42 enhancements in this cycle for operators."
    r = await gates(good_draft().model_copy(update={"body_mdx": thin}))
    f = " ".join(r.gates["value"].failures)
    assert "Why it matters" in f and "original analysis" in f and "below the job's minimum" in f


async def test_style_banned_phrase_missing_section_and_links() -> None:
    body = GOOD_BODY.replace("a welcome change", "a game-changer") + "\n\nSee [the vendor](https://evil.example.com/x) and [docs](https://kubernetes.io/docs/x) and [site](/blog/ingress-tls)."
    r = await gates(good_draft().model_copy(update={"body_mdx": body}), required=["Why it matters", "What to do", "Migration notes"])
    f = " ".join(r.gates["style"].failures)
    assert "game-changer" in f and "Migration notes" in f and "evil.example.com" in f and "kubernetes.io" not in f


async def test_seo_gate() -> None:
    d = good_draft().model_copy(update={"title": "x" * 80, "meta_description": "short", "schema_jsonld": {"@type": "BlogPosting"}, "categories": [], "sources": []})
    f = " ".join((await gates(d)).gates["seo"].failures)
    assert "title is 80" in f and "meta description" in f and "NewsArticle" in f and "no category" in f and "no sources" in f


async def test_duplicate_gate() -> None:
    r = await gates(good_draft(), dup="story already published")
    assert not r.passed and r.failed_gates == ["duplicate"]
