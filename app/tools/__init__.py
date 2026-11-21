"""Agent tools: read-only, allowlisted.

Forked from moks-seo-agent/app/tools/__init__.py@bb9e553 on 2026-11-21.

ALLOWED_TOOLS is the complete list an agent or crew member can ever hold. There is no shell, no arbitrary HTTP, no
code execution and no database write among them: the one write in the system ("create/update draft") is done by flow
steps through the web app's draft API, never by an agent. `build_tools` refuses any name outside the allowlist, and a test asserts the
crew's agents hold exactly these.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

# The News Desk Crew's tools (app/news/crew.py), all implemented by NewsToolContext (app/news/tools.py). The SEO crews'
# tools (read_article, style_guide, fetch_allowlisted_docs, …) are the seo-agent's and do not exist here.
ALLOWED_TOOLS = frozenset({"gsc_queries_for_topic"})
NEWS_TOOLS = frozenset({"read_news_item", "fetch_allowlisted", "vector_search_site", "fact_sheet", "writing_profile", "image_profile"})
ALL_TOOLS = ALLOWED_TOOLS | NEWS_TOOLS

EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
SECRETISH = re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{12,}|ya29\.[\w-]+|ghp_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def redact(text: str) -> str:
    return EMAIL.sub("[email]", SECRETISH.sub("[redacted]", text))


def safe_queries(queries: list[str]) -> list[str]:
    return [q[:120] for q in queries if not (EMAIL.search(q) or PHONE.search(q) or SECRETISH.search(q))]


def data_block(name: str, content: Any) -> str:
    """Same as dataBlock() in moks-cms/src/lib/seo-intel/llm.ts: untrusted content, closing tags neutralised."""
    import json

    text = redact(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=1))
    safe = re.sub(r"[^a-z0-9_-]", "", name, flags=re.I)
    return f'<data name="{safe}">\n{re.sub(r"</?data\b", "‹data", text, flags=re.I)}\n</data>'


class ToolError(Exception):
    pass


def check_docs_url(url: str, allowlist: tuple[str, ...]) -> str:
    """https only, default port, no credentials, host = an allowlisted domain or its subdomain, never an IP literal.

    The web app's fetcher re-checks the same allowlist and resolves DNS inside the socket (private ranges refused),
    so a hostname that resolves to an internal address is still blocked there.
    """
    if len(url) > 2000:
        raise ToolError("URL too long")
    parts = urlsplit(url.strip())
    if parts.scheme != "https":
        raise ToolError("only https URLs are allowed")
    if parts.username or parts.password:
        raise ToolError("credentials in URLs are not allowed")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise ToolError("missing host")
    try:
        ipaddress.ip_address(host)
        raise ToolError("IP addresses are not allowed")
    except ValueError:
        pass
    if parts.port not in (None, 443):
        raise ToolError("non-default ports are not allowed")
    if not any(host == d or host.endswith("." + d) for d in allowlist):
        raise ToolError(f"{host} is not on the documentation allowlist")
    return parts.geturl()


_TAGS = re.compile(r"<(script|style|noscript|svg|nav|footer|header)\b[\s\S]*?</\1>|<[^>]+>", re.I)
# The page's own content. Documentation sites put a language menu and a sidebar of every page before it: on a Kubernetes
# page that is the first 490 KB of HTML, so without this the first 12,000 characters of text were navigation only.
_CONTENT_TAGS = ("main", "article")
# A section heading with an id: the anchor a URL fragment (#range-vector-selectors) points to.
_HEADING_RE = re.compile(r"<h([2-6])\b[^>]*?\bid\s*=\s*[\"']?([^\"'\s>]+)[^>]*>([\s\S]*?)</h\1>", re.I)
_SECTION_MARK = "\ue000"  # a private-use character: marks a heading's anchor in the text, never left in the output
_LINK_RE = re.compile(r"<a\b[^>]*?\bhref\s*=\s*[\"']?(https://[^\"'\s>]+)[^>]*>([\s\S]*?)</a>", re.I)
SHORT_PAGE_CHARS = 2000  # a "moved" notice or an index page: its links are listed, they are where the content is


def _entities(text: str) -> str:
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#x27;|&#39;", "'", text)
    text = re.sub(r"&quot;", '"', text)
    return re.sub(r"&amp;", "&", text)


def main_content(html: str) -> str:
    """The first <main> element (else <article>) up to its last closing tag, or to the end when the page was cut
    before it (the web app sends at most 300,000 characters); the whole page when it has neither."""
    for tag in _CONTENT_TAGS:
        start = re.search(rf"<{tag}(?=[\s>])[^>]*>", html, re.I)
        if start:
            end = html.lower().rfind(f"</{tag}>")
            return html[start.end() : end if end > start.end() else len(html)]
    return html


def html_to_text(html: str, limit: int = 12_000, *, section: str = "", list_sections: bool = False) -> str:
    """Readable text of a page: the <main> (else <article>) element when the page has one, without scripts, styles
    and navigation, at most `limit` characters.

    `section` (a URL fragment, without '#') starts the text at the heading with that id. With `list_sections`
    (fetch_allowlisted_docs; the news tool has no sections), text that was cut ends with the anchors of the sections
    that did not fit, so a later fetch of `url#anchor` can read the one it needs, and a short page (a "moved" notice,
    an index) ends with its links, which are otherwise lost with the tags."""
    content = main_content(html)

    def mark(h: re.Match[str]) -> str:
        title = " ".join(_entities(_TAGS.sub(" ", h.group(3))).split())
        return f"\n\n{_SECTION_MARK}{h.group(2)}{_SECTION_MARK}{title}\n\n"

    text = _entities(_TAGS.sub(" ", _HEADING_RE.sub(mark, content)))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if section:
        at = text.find(f"{_SECTION_MARK}{section}{_SECTION_MARK}")
        if at >= 0:
            text = text[at:]
    anchor = re.compile(f"{_SECTION_MARK}([^{_SECTION_MARK}\\s]+){_SECTION_MARK}")

    def plain(t: str) -> str:
        return anchor.sub("", t).replace(_SECTION_MARK, "")  # a mark the cut split in two is dropped too

    if len(text) > limit:
        later = anchor.findall(text[limit:]) if list_sections else []
        out = plain(text[:limit])
        if later:
            out += f"\n\n[Cut at {limit:,} of {len(text):,} characters. Sections not shown (fetch the page URL with one of these anchors to read from it): {', '.join(f'#{a}' for a in later[:40])}]"
        return out
    out = plain(text)
    if list_sections and len(out) < SHORT_PAGE_CHARS:
        links = [(" ".join(_entities(_TAGS.sub(" ", t)).split()), _entities(u)) for u, t in _LINK_RE.findall(content)]
        listed = [f"- {t or u}: {u}" for t, u in dict.fromkeys(links)][:10]
        if listed:
            out += "\n\n[Links on this page]\n" + "\n".join(listed)
    return out


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[[str], Awaitable[str]]


def build_tools(names: list[str], ctx: ToolContext) -> list[ToolSpec]:
    unknown = [n for n in names if n not in ALL_TOOLS]
    if unknown:
        raise ToolError(f"tool(s) not on the allowlist: {', '.join(unknown)}")
    specs = getattr(ctx, "news_tool_specs", None)
    if specs is not None:
        reg = specs()
        return [reg[n] for n in names]
    news = [n for n in names if n in NEWS_TOOLS]
    if news:
        raise ToolError(f"news tool(s) need a news tool context: {', '.join(news)}")
    registry: dict[str, ToolSpec] = {
        "gsc_queries_for_topic": ToolSpec("gsc_queries_for_topic", "Search Console queries (28 days, aggregated) that contain the topic words. Input: a topic.", ctx.gsc_queries_for_topic),
    }
    return [registry[n] for n in names]


SEARCH_LIMIT = 6


class ToolContext:
    """Implements the shared tools on top of the web app callback API and the site search. All results are data blocks.

    `retriever` is the site search (app/seo_agent.py RemoteRetriever: the seo-agent's index of this site's articles,
    reached over POST /search); `docs_allowlist` the hosts a fetch may reach."""

    def __init__(self, web: Any, retriever: Any, docs_allowlist: tuple[str, ...], on_source: Callable[[str, str], None] | None = None):
        self.web = web
        self.retriever = retriever
        self.docs_allowlist = docs_allowlist
        self.on_source = on_source
        self.calls: list[tuple[str, str]] = []

    async def vector_search(self, query: str) -> str:
        self.calls.append(("vector_search", query))
        q = (query or "").strip()[:500]
        if len(q) < 2:
            return data_block("search_refused", {"reason": "query is too short"})
        if self.retriever is None or not getattr(self.retriever, "available", True):
            return data_block("search_unavailable", {"reason": "site search is unavailable (the seo-agent's index cannot be reached)"})
        try:
            hits = await self.retriever.search(q, k=SEARCH_LIMIT)
        except Exception as e:  # the seo-agent is down or slow: the crew goes on without related articles
            return data_block("search_unavailable", {"reason": f"site search failed: {type(e).__name__}"})
        return data_block("search_results", [{"title": h.title, "path": h.path, "section": h.heading_path, "anchor": h.anchor, "excerpt": h.content[:400], "score": round(float(h.score), 3)} for h in hits[:SEARCH_LIMIT]])

    async def gsc_queries_for_topic(self, topic: str) -> str:
        self.calls.append(("gsc_queries_for_topic", topic))
        rows = await self.web.gsc_queries(topic=topic.strip()[:120], limit=25)
        keep = set(safe_queries([r["query"] for r in rows]))
        return data_block("search_console_queries_28d", [r for r in rows if r["query"] in keep])
