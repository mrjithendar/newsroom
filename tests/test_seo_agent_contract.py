"""Contract: the seo-agent still serves what app/seo_agent.py calls, to the news agent's token, in the shapes it reads.
Parsed from moks-seo-agent's source (tests/seo_agent.py finds it), so a renamed route, a dropped token or a changed
field fails here instead of in production."""

from __future__ import annotations

import ast
import re

from tests.seo_agent import seo


def routes(src: str) -> dict[tuple[str, str], str]:
    """(method, path) → the auth dependency of each route in moks-seo-agent app/main.py."""
    return {(m.upper(), p): dep for m, p, dep in re.findall(r'@app\.(get|post|put|delete)\("([^"]+)", dependencies=\[Depends\((\w+)\)\]\)', src)}


def test_the_news_agents_three_routes_take_its_token_and_nothing_else_does() -> None:
    src = seo("app/main.py").read_text()
    r = routes(src)
    for route in (("POST", "/search"), ("POST", "/llm-budget/reserve"), ("POST", "/llm-budget/settle")):
        assert r.get(route) == "auth_or_news", f"{route} must accept SEO_AGENT_NEWS_TOKEN (auth_or_news)"
    others = {k: v for k, v in r.items() if k not in {("POST", "/search"), ("POST", "/llm-budget/reserve"), ("POST", "/llm-budget/settle")}}
    assert others and all(v == "auth" for v in others.values()), {k: v for k, v in others.items() if v != "auth"}
    assert 'env.get("SEO_AGENT_NEWS_TOKEN"' in seo("app/config.py").read_text()


def test_search_body_and_answer_match_what_the_client_sends_and_reads() -> None:
    src = seo("app/main.py").read_text()
    tree = ast.parse(src)
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "SearchBody")
    fields = {t.target.id for t in body.body if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)}
    assert {"q", "k"} <= fields and "le=10" in ast.get_source_segment(src, body)  # type: ignore[operator]
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "search")
    text = ast.get_source_segment(src, fn) or ""
    for key in ('"hits"', '"article_id"', '"title"', '"path"', '"heading"', '"anchor"', '"content"', '"score"'):
        assert key in text, f"/search no longer answers {key} (app/seo_agent.py SeoAgentClient.search reads it)"


def test_reserve_and_settle_answer_as_the_client_expects() -> None:
    src = seo("app/main.py").read_text()
    reserve = src[src.index('@app.post("/llm-budget/reserve"') : src.index('@app.post("/llm-budget/settle"')]
    assert '"max_usd"' in reserve and '"purpose"' in reserve and '{"id":' in reserve
    assert 'status_code=402' in reserve and '"scope": e.scope' in reserve  # BudgetExceeded(scope, detail) here
    assert "0 <= max_usd <= 1000" in reserve  # the client clamps to this range
    settle = src[src.index('@app.post("/llm-budget/settle"') :]
    assert '"id"' in settle[:800] and "[0-9a-f]{32}" in settle[:800]
