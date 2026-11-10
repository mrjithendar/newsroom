"""Read-only tools for the News Desk Crew. Every result is an untrusted data block (closing tags neutralised).

- read_news_item: full cleaned text of one fetched item in the story's cluster (by id); nothing else.
- fetch_allowlisted: one https page on the story's allowlist (the cluster's source hosts + configured primary-source and
  documentation domains), fetched by the web app's SSRF-safe fetcher (robots.txt honoured, no paywalled hosts). Every
  page fetched becomes a citable source; anything else cited is flagged.
- vector_search_site: this site's own articles (pgvector), for "related" and internal links.
- fact_sheet: the Researcher's claims (after the research task finishes).
- writing_profile / image_profile: the job's versioned profiles.
- gsc_queries_for_topic: aggregated Search Console queries (personal-data-looking queries removed).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..tools import ToolContext, ToolError, ToolSpec, check_docs_url, data_block, html_to_text
from .text import host_of


class NewsToolContext(ToolContext):
    def __init__(self, web: Any, retriever: Any, *, items: list[dict[str, Any]], allow_hosts: tuple[str, ...], writing_profile: dict[str, Any], image_profile: dict[str, Any], on_source: Callable[[str, str, str], None] | None = None):
        super().__init__(web, retriever, allow_hosts)
        self.items = {str(i["id"]): i for i in items}
        self.allow_hosts = allow_hosts
        self.writing = writing_profile
        self.image = image_profile
        self.facts: dict[str, Any] = {}
        self.on_fetch = on_source
        self.fetched: dict[str, str] = {}  # url → cleaned text (citable)

    def news_tool_specs(self) -> dict[str, ToolSpec]:
        return {
            "read_news_item": ToolSpec("read_news_item", "Read the full text of one news item from this story's sources. Input: the item id.", self.read_news_item),
            "fetch_allowlisted": ToolSpec("fetch_allowlisted", "Fetch a primary source (release notes, advisory, docs): https and allowlisted domains only. Input: the URL.", self.fetch_allowlisted),
            "vector_search_site": ToolSpec("vector_search_site", "Search this site's own articles by meaning. Input: a short query.", self.vector_search),
            "fact_sheet": ToolSpec("fact_sheet", "The researcher's fact sheet: every allowed claim with its source URL and snippet. Input: anything.", self.fact_sheet),
            "writing_profile": ToolSpec("writing_profile", "The job's writing profile: style, structure, tone, audience, banned phrases, required sections. Input: anything.", self.writing_profile),
            "image_profile": ToolSpec("image_profile", "The job's image profile: style prompt, negative prompt, palette, safety rules. Input: anything.", self.image_profile),
            "gsc_queries_for_topic": ToolSpec("gsc_queries_for_topic", "Search Console queries (28 days, aggregated) containing the topic words. Input: a topic.", self.gsc_queries_for_topic),
        }

    async def read_news_item(self, item_id: str) -> str:
        self.calls.append(("read_news_item", item_id))
        i = self.items.get(item_id.strip().strip('"'))
        if not i:
            return data_block("not_found", {"id": item_id[:60], "known_ids": list(self.items)[:20]})
        return data_block("news_item", {"id": str(i["id"]), "url": i["url"], "title": i["title"], "publisher": i.get("source_name"), "published_at": str(i.get("published_at") or ""), "text": str(i.get("clean_text") or i.get("excerpt") or "")[:12_000]})

    async def fetch_allowlisted(self, url: str) -> str:
        self.calls.append(("fetch_allowlisted", url))
        try:
            clean = check_docs_url(url, self.allow_hosts)
        except ToolError as e:
            return data_block("fetch_refused", {"url": url[:300], "reason": str(e)})
        r = await self.web.news_fetch({"url": clean, "purpose": "primary_source"})
        if r.get("error") or int(r.get("status", 0)) != 200:
            return data_block("fetch_failed", {"url": clean, "status": r.get("status"), "reason": r.get("error")})
        text = html_to_text(str(r.get("body", "")))
        self.fetched[clean] = text
        if self.on_fetch:
            self.on_fetch(clean, str(r.get("title", "")), host_of(clean))
        return data_block("primary_source", {"url": r.get("finalUrl", clean), "title": r.get("title", ""), "text": text})

    async def fact_sheet(self, _input: str) -> str:
        self.calls.append(("fact_sheet", ""))
        return data_block("fact_sheet", self.facts or {"claims": [], "note": "the research task has not finished yet"})

    async def writing_profile(self, _input: str) -> str:
        self.calls.append(("writing_profile", ""))
        keep = ("name", "version", "system_style", "structure_template", "tone", "audience", "banned_phrases", "required_sections", "examples")
        return data_block("writing_profile", json.loads(json.dumps({k: self.writing.get(k) for k in keep}, default=str)))

    async def image_profile(self, _input: str) -> str:
        self.calls.append(("image_profile", ""))
        keep = ("name", "version", "style_prompt", "negative_prompt", "palette", "safety_rules")
        return data_block("image_profile", json.loads(json.dumps({k: self.image.get(k) for k in keep}, default=str)))
