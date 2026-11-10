"""Feed parsing for every source type: RSS 2.0, Atom (incl. GitHub release feeds), JSON (JSON Feed 1.1, CISA KEV, NVD
CVE API 2.0), sitemaps (Google News sitemap titles) and one allowlisted HTML page.

Fetching happens elsewhere (the web app's SSRF-safe fetcher, docs/news/DECISIONS.md N-05); this module only parses the
bytes it is given. XML with a DOCTYPE or ENTITY declaration is refused outright, so entity-expansion attacks
("billion laughs", external entities) never reach the parser.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from .text import clean_text

MAX_ITEMS_PER_FETCH = 200


class FeedError(Exception):
    pass


@dataclass
class RawItem:
    url: str
    title: str
    published_at: datetime | None
    summary: str
    extra: dict[str, Any] = field(default_factory=dict)


def parse_date(v: str | None) -> datetime | None:
    if not v:
        return None
    v = v.strip()
    try:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=UTC)
    except ValueError:
        pass
    try:
        d = parsedate_to_datetime(v)
        return d if d.tzinfo else d.replace(tzinfo=UTC)
    except (TypeError, ValueError, IndexError):
        return None


def _xml(body: str) -> ET.Element:
    head = body[:4000].lower()
    if "<!doctype" in head or "<!entity" in body.lower():
        raise FeedError("XML with DOCTYPE/ENTITY declarations is refused")
    try:
        return ET.fromstring(body.encode("utf-8", "replace"))  # noqa: S314 - see above
    except ET.ParseError as e:
        raise FeedError(f"invalid XML: {str(e)[:120]}") from e


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child(el: ET.Element, *names: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) in names:
            return c
    return None


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def parse_rss_or_atom(body: str) -> list[RawItem]:
    root = _xml(body)
    out: list[RawItem] = []
    tag = _local(root.tag)
    if tag == "rss" or tag == "rdf":
        channel = _child(root, "channel") if tag == "rss" else root
        nodes = [n for n in (channel if channel is not None else []) if _local(n.tag) == "item"] if tag == "rss" else [n for n in root if _local(n.tag) == "item"]
        for it in nodes:
            link = _text(_child(it, "link")) or _text(_child(it, "guid"))
            title = _text(_child(it, "title"))
            desc = _text(_child(it, "encoded")) or _text(_child(it, "description"))
            date = parse_date(_text(_child(it, "pubdate")) or _text(_child(it, "date")))
            if link and title:
                out.append(RawItem(link, title, date, clean_text(desc, 4000)))
    elif tag == "feed":
        for e in root:
            if _local(e.tag) != "entry":
                continue
            link = ""
            for ln in e:
                if _local(ln.tag) == "link" and ln.get("rel", "alternate") == "alternate" and ln.get("href"):
                    link = ln.get("href", "")
                    break
            title = _text(_child(e, "title"))
            summary = _text(_child(e, "content")) or _text(_child(e, "summary"))
            date = parse_date(_text(_child(e, "published")) or _text(_child(e, "updated")))
            if link and title:
                out.append(RawItem(link, title, date, clean_text(summary, 4000)))
    else:
        raise FeedError(f"not an RSS or Atom feed (root <{tag}>)")
    return out[:MAX_ITEMS_PER_FETCH]


def parse_json(body: str) -> list[RawItem]:
    try:
        data = json.loads(body)
    except ValueError as e:
        raise FeedError("invalid JSON") from e
    out: list[RawItem] = []
    if isinstance(data, dict) and isinstance(data.get("items"), list) and str(data.get("version", "")).startswith("https://jsonfeed.org"):
        for it in data["items"]:
            url, title = it.get("url") or it.get("external_url"), it.get("title")
            if url and title:
                out.append(RawItem(str(url), str(title), parse_date(it.get("date_published")), clean_text(str(it.get("content_text") or it.get("summary") or it.get("content_html") or ""), 4000)))
    elif isinstance(data, dict) and isinstance(data.get("vulnerabilities"), list) and data.get("catalogVersion"):
        # CISA Known Exploited Vulnerabilities catalogue: every entry is exploited in the wild (no CVSS in the feed).
        for v in data["vulnerabilities"]:
            cve = str(v.get("cveID", ""))
            if not cve:
                continue
            title = f"{cve}: {v.get('vulnerabilityName') or (str(v.get('vendorProject', '')) + ' ' + str(v.get('product', ''))).strip()}"
            summary = f"{v.get('shortDescription', '')} Required action: {v.get('requiredAction', '')} Due: {v.get('dueDate', '')}."
            out.append(RawItem(f"https://nvd.nist.gov/vuln/detail/{cve}", title[:300], parse_date(v.get("dateAdded")), clean_text(summary, 4000), {"cve": cve, "kev": True, "vendor": v.get("vendorProject"), "product": v.get("product")}))
    elif isinstance(data, dict) and isinstance(data.get("vulnerabilities"), list):
        # NVD CVE API 2.0
        for w in data["vulnerabilities"]:
            c = w.get("cve") or {}
            cve = str(c.get("id", ""))
            if not cve:
                continue
            desc = next((d.get("value", "") for d in c.get("descriptions", []) if d.get("lang") == "en"), "")
            score = None
            for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30"):
                m = (c.get("metrics") or {}).get(key) or []
                if m:
                    score = (m[0].get("cvssData") or {}).get("baseScore")
                    break
            out.append(RawItem(f"https://nvd.nist.gov/vuln/detail/{cve}", f"{cve}: {desc[:140]}", parse_date(c.get("published")), clean_text(desc, 4000), {"cve": cve, "cvss": float(score) if score is not None else None}))
    else:
        raise FeedError("unsupported JSON shape (JSON Feed 1.1, CISA KEV or NVD CVE API 2.0)")
    return out[:MAX_ITEMS_PER_FETCH]


def parse_sitemap(body: str) -> list[RawItem]:
    """Only entries with a title are usable as news (Google News sitemaps carry <news:title>); others are skipped."""
    root = _xml(body)
    out: list[RawItem] = []
    for u in root:
        if _local(u.tag) != "url":
            continue
        loc = _text(_child(u, "loc"))
        news = _child(u, "news")
        title = _text(_child(news, "title")) if news is not None else ""
        date = parse_date(_text(_child(news, "publication_date")) if news is not None else _text(_child(u, "lastmod")))
        if loc and title:
            out.append(RawItem(loc, title, date, ""))
    return out[:MAX_ITEMS_PER_FETCH]


def parse_html_page(body: str, url: str) -> list[RawItem]:
    """An explicitly allowlisted page (e.g. a "what's new" page without a feed) becomes one item; its content hash
    decides whether it changed."""
    m = re.search(r"<title[^>]*>([\s\S]*?)</title>", body, re.I)
    title = clean_text(m.group(1), 300) if m else url
    main = re.search(r"<(main|article)\b[\s\S]*?</\1>", body, re.I)
    return [RawItem(url, title, None, clean_text(main.group(0) if main else body, 4000))]


def parse_github_releases(body: str) -> list[RawItem]:
    """GitHub REST API `GET /repos/{owner}/{repo}/releases` (github.com's own robots.txt disallows the .atom feeds for
    every crawler, so the official API is the allowed way in). Drafts and pre-releases are skipped."""
    try:
        data = json.loads(body)
    except ValueError as e:
        raise FeedError("invalid JSON from the GitHub releases API") from e
    if not isinstance(data, list):
        raise FeedError(str((data or {}).get("message", "unexpected GitHub API response"))[:200] if isinstance(data, dict) else "unexpected GitHub API response")
    out = []
    for r in data:
        if not isinstance(r, dict) or r.get("draft") or r.get("prerelease") or not r.get("html_url"):
            continue
        repo = "/".join(str(r["html_url"]).split("/")[3:5])
        name = str(r.get("name") or r.get("tag_name") or "release")
        title = name if repo.split("/")[-1].lower() in name.lower() else f"{repo} {name}"
        out.append(RawItem(str(r["html_url"]), title[:300], parse_date(r.get("published_at")), clean_text(str(r.get("body") or ""), 4000), {"tag": r.get("tag_name")}))
    return out[:MAX_ITEMS_PER_FETCH]


def parse_feed(source_type: str, body: str, url: str) -> list[RawItem]:
    if source_type == "github_releases" and body.lstrip().startswith(("[", "{")):
        return parse_github_releases(body)
    if source_type in ("rss", "atom", "github_releases"):
        return parse_rss_or_atom(body)
    if source_type == "json_api":
        return parse_json(body)
    if source_type == "sitemap":
        return parse_sitemap(body)
    if source_type == "html_allowlisted":
        return parse_html_page(body, url)
    raise FeedError(f"unknown source type {source_type}")
