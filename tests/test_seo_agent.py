"""The news agent's calls to the seo-agent (app/seo_agent.py): the one LLM budget held from outside, with the web app's
rules (moks-cms/src/lib/seo-intel/llm-gate.ts), and the site search that never fails a run."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from app.llm import BudgetExceeded, FakeLlm, MemorySpendStore
from app.seo_agent import UNCHECKED, RemoteBudgetGuard, RemoteRetriever, SeoAgentClient
from app.tools import ToolContext


def client(handler: Any, token: str = "n" * 40, url: str = "http://seo-agent:8080") -> tuple[SeoAgentClient, list[tuple[str, dict[str, Any]]]]:
    seen: list[tuple[str, dict[str, Any]]] = []

    def wrap(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        seen.append((req.url.path, body))
        assert req.headers["authorization"] == f"Bearer {token}"
        return handler(req, body)

    return SeoAgentClient(url, token, transport=httpx.MockTransport(wrap)), seen


def holds(req: httpx.Request, body: dict[str, Any]) -> httpx.Response:
    if req.url.path == "/llm-budget/reserve":
        return httpx.Response(200, json={"id": "a" * 32})
    if req.url.path == "/llm-budget/settle":
        return httpx.Response(200, json={"released": True})
    return httpx.Response(404)


async def test_not_set_up_allows_the_call_and_asks_nobody() -> None:
    c, seen = client(holds, token="")
    assert not c.configured
    assert await c.reserve(1.0, "news.crew") is None
    assert seen == []


async def test_an_older_seo_agent_without_the_budget_allows_the_call() -> None:
    c, _ = client(lambda r, b: httpx.Response(404))
    assert await c.reserve(1.0, "news.crew") is None


async def test_a_budget_that_does_not_fit_refuses_the_call_with_its_reason() -> None:
    c, _ = client(lambda r, b: httpx.Response(402, json={"error": "The monthly LLM budget is reached: $9.99 of $10.00 used this month.", "scope": "llm_monthly"}))
    with pytest.raises(BudgetExceeded) as e:
        await c.reserve(0.5, "news.crew")
    assert e.value.scope == "llm_monthly" and "monthly LLM budget" in e.value.detail


@pytest.mark.parametrize("answer", [httpx.Response(503), httpx.Response(500, json={}), httpx.Response(401), httpx.Response(403)])
async def test_a_budget_that_cannot_be_checked_refuses_the_call(answer: httpx.Response) -> None:
    c, _ = client(lambda r, b: answer)
    with pytest.raises(BudgetExceeded) as e:
        await c.reserve(0.5, "news.crew")
    assert e.value.scope == "llm_unchecked"


async def test_an_unreachable_seo_agent_refuses_the_call() -> None:
    def down(req: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    c, _ = client(down)
    with pytest.raises(BudgetExceeded) as e:
        await c.reserve(0.5, "news.crew")
    assert e.value.detail == UNCHECKED


async def test_each_call_is_held_before_and_settled_after_its_usage_row() -> None:
    order: list[str] = []
    spend = MemorySpendStore()

    def handler(req: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        order.append(f"{req.url.path}:{len(spend.rows)}")  # how many usage rows exist at that moment
        return holds(req, body)

    c, seen = client(handler)
    guard = RemoteBudgetGuard(spend, c)
    llm = FakeLlm(guard)
    llm.queue("news.crew", {"x": 1})
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    from app.llm import CallSpec

    await llm.structured(CallSpec(model="claude-sonnet-5-5", purpose="news.crew", prompt_version="t", cached_system=["sys"], user="hi", max_tokens=1000), Out)
    await asyncio.sleep(0)  # the settle runs as its own task
    await asyncio.gather(*guard._settling)
    assert order == ["/llm-budget/reserve:0", "/llm-budget/settle:1"]
    reserve = seen[0][1]
    assert reserve["purpose"] == "news.crew" and 0 < reserve["max_usd"] < 1
    assert spend.rows[0]["purpose"] == "news.crew"  # the usage row is still written here, so the budget counts it


async def test_a_refused_hold_sends_no_model_call() -> None:
    c, _ = client(lambda r, b: httpx.Response(402, json={"error": "no", "scope": "llm_daily"}))
    llm = FakeLlm(RemoteBudgetGuard(MemorySpendStore(), c))
    from app.llm import CallSpec

    with pytest.raises(BudgetExceeded):
        await llm.text(CallSpec(model="claude-haiku-5-5", purpose="news.classify", prompt_version="t", cached_system=[], user="x"))
    assert llm.calls == []


async def test_exhausted_is_a_zero_dollar_hold() -> None:
    c, seen = client(holds)
    assert await RemoteBudgetGuard(MemorySpendStore(), c).exhausted() is False
    assert [p for p, _ in seen] == ["/llm-budget/reserve", "/llm-budget/settle"] and seen[0][1]["max_usd"] == 0
    full, _ = client(lambda r, b: httpx.Response(402, json={"error": "used up", "scope": "llm_monthly"}))
    assert await RemoteBudgetGuard(MemorySpendStore(), full).exhausted() is True
    down, _ = client(lambda r, b: httpx.Response(503))
    assert await RemoteBudgetGuard(MemorySpendStore(), down).exhausted() is False  # said at the first real call instead


async def test_site_search_returns_hits_and_never_fails_a_run() -> None:
    def search(req: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        assert req.url.path == "/search" and body == {"q": "kubernetes resize", "k": 3}
        return httpx.Response(200, json={"hits": [{"article_id": "a1", "title": "Pod resize", "path": "/blog/resize", "heading": "Resize", "anchor": "resize", "content": "text", "score": 0.91}, {"bad": True}]})

    c, _ = client(search)
    hits = await RemoteRetriever(c).search("kubernetes resize", k=3)
    assert [(h.path, h.score, h.url) for h in hits] == [("/blog/resize", 0.91, "/blog/resize#resize")]

    now = [0.0]
    down, seen = client(lambda r, b: httpx.Response(503))
    r = RemoteRetriever(down, monotonic=lambda: now[0])
    assert await r.search("x y") == [] and not r.available  # one failure: skipped for a while
    assert await r.search("x y") == [] and len(seen) == 1
    now[0] = 61.0
    assert r.available
    assert not RemoteRetriever(SeoAgentClient("", "")).available


async def test_the_crews_search_tool_says_when_search_is_unavailable() -> None:
    ctx = ToolContext(web=None, retriever=RemoteRetriever(SeoAgentClient("", "")), docs_allowlist=())
    assert "search_unavailable" in await ctx.vector_search("kubernetes")
    assert "search_refused" in await ctx.vector_search("k")
