"""News desk: URL normalisation, cleaning, language, entities, SimHash, feeds (incl. hostile XML), cron, job config."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.news import feeds, text
from app.news.config import JobConfig, load_news_settings
from app.news.cron import Cron, CronError
from tests.moks_cms import www
from tests.news_fixtures import FEEDS, K8S_TEXT


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("https://www.Example.com/a/b/?utm_source=rss&utm_medium=x&id=3#frag", "https://example.com/a/b?id=3"),
        ("https://example.com:443//x//y/", "https://example.com/x/y"),
        ("http://example.com/?b=2&a=1&fbclid=zz", "http://example.com/?a=1&b=2"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com:8443/p", "https://example.com:8443/p"),
    ],
)
def test_normalize_url(raw: str, want: str) -> None:
    assert text.normalize_url(raw) == want


def test_url_hash_ignores_tracking_and_www() -> None:
    assert text.url_hash("https://www.example.com/post?utm_campaign=a") == text.url_hash("https://example.com/post")


def test_clean_text_drops_scripts_and_keeps_paragraphs() -> None:
    t = text.clean_text("<p>One &amp; two</p><script>alert(1)</script><nav>menu</nav><p>Three</p>")
    assert t == "One & two\n\nThree" or t == "One & two\nThree"
    assert "alert" not in t and "menu" not in t


def test_language() -> None:
    assert text.detect_language(K8S_TEXT) == "en"
    assert text.detect_language("Die neue Version ist nicht mit der alten kompatibel und das ist ein Problem für die Betreiber der Cluster") == "de"
    assert text.detect_language("short") is None


def test_entities() -> None:
    e = text.extract_entities("Kubernetes 1.35 fixes CVE-2026-12345", "Google and AWS ship Terraform v1.14.2; see cve-2026-9999.")
    assert "Kubernetes" in e["product"] and "Terraform" in e["product"]
    assert {"Google", "AWS"} <= set(e["org"])
    assert e["cve"] == ["CVE-2026-12345", "CVE-2026-9999"]
    assert "1.35" in e["version"] and "1.14.2" in e["version"]


def test_simhash_is_close_for_near_duplicates() -> None:
    a = text.simhash(K8S_TEXT)
    b = text.simhash(K8S_TEXT.replace("today", "this week"))
    c = text.simhash("Terraform state locking with DynamoDB tables is deprecated in favour of S3 native locking.")
    assert text.hamming(a, b) < text.hamming(a, c)
    assert -(2**63) <= a < 2**63


def test_flesch_and_sentences() -> None:
    assert len(text.sentences("One fact. Two facts! Three?")) == 3
    assert 0 < text.flesch_reading_ease(K8S_TEXT) < 100


# --- feeds ----------------------------------------------------------------------------------------------------------


def test_rss_and_atom() -> None:
    r = feeds.parse_feed("rss", FEEDS["https://feeds.example.org/k8s.xml"], "u")
    assert len(r) == 1 and r[0].title.startswith("Kubernetes 1.35") and r[0].published_at == datetime(2026, 11, 10, 4, tzinfo=UTC)
    a = feeds.parse_feed("github_releases", FEEDS["https://releases.example.net/k8s.atom"], "u")
    assert a[0].url == "https://releases.example.net/kubernetes/v1.35"


def test_doctype_and_entities_are_refused() -> None:
    bomb = '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;">]><rss><channel><item><title>&lol2;</title><link>x</link></item></channel></rss>'
    with pytest.raises(feeds.FeedError):
        feeds.parse_feed("rss", bomb, "u")
    with pytest.raises(feeds.FeedError):
        feeds.parse_feed("rss", "<html><body>not a feed</body></html>", "u")


def test_json_kev_nvd_and_jsonfeed() -> None:
    kev = json.dumps({"catalogVersion": "2026.11.10", "vulnerabilities": [{"cveID": "CVE-2026-0001", "vendorProject": "Example", "product": "Widget", "vulnerabilityName": "Example Widget RCE", "dateAdded": "2026-11-09", "shortDescription": "Remote code execution.", "requiredAction": "Patch.", "dueDate": "2026-11-30"}]})
    r = feeds.parse_feed("json_api", kev, "u")
    assert r[0].extra["kev"] and r[0].url.endswith("CVE-2026-0001")
    nvd = json.dumps({"vulnerabilities": [{"cve": {"id": "CVE-2026-0002", "published": "2026-11-09T10:00:00.000", "descriptions": [{"lang": "en", "value": "A flaw."}], "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]}}}]})
    assert feeds.parse_feed("json_api", nvd, "u")[0].extra["cvss"] == 9.8
    jf = json.dumps({"version": "https://jsonfeed.org/version/1.1", "items": [{"url": "https://e.org/a", "title": "A", "content_text": "x"}]})
    assert feeds.parse_feed("json_api", jf, "u")[0].title == "A"
    with pytest.raises(feeds.FeedError):
        feeds.parse_feed("json_api", "{}", "u")


def test_sitemap_needs_titles() -> None:
    xml = '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9"><url><loc>https://e.org/a</loc><news:news><news:publication_date>2026-11-10</news:publication_date><news:title>Story A</news:title></news:news></url><url><loc>https://e.org/b</loc></url></urlset>'
    r = feeds.parse_feed("sitemap", xml, "u")
    assert [x.url for x in r] == ["https://e.org/a"]


# --- cron -----------------------------------------------------------------------------------------------------------


def test_cron_in_a_time_zone() -> None:
    c = Cron("0 7 * * 1-5", "Asia/Kolkata")  # 07:00 IST = 01:30 UTC
    nxt = c.next_after(datetime(2026, 11, 10, 2, 0, tzinfo=UTC))  # Tuesday 07:30 IST
    assert nxt == datetime(2026, 11, 11, 1, 30, tzinfo=UTC)
    assert c.next_after(datetime(2026, 11, 13, 2, 0, tzinfo=UTC)) == datetime(2026, 11, 16, 1, 30, tzinfo=UTC)  # Fri → Mon


def test_cron_steps_lists_and_due_window() -> None:
    c = Cron("*/15 6,18 * * *", "UTC")
    assert c.matches(datetime(2026, 11, 10, 18, 45, tzinfo=UTC)) and not c.matches(datetime(2026, 11, 10, 17, 45, tzinfo=UTC))
    now = datetime(2026, 11, 10, 6, 7, tzinfo=UTC)
    assert c.last_due(now, timedelta(minutes=16)) == datetime(2026, 11, 10, 6, 0, tzinfo=UTC)
    assert c.last_due(datetime(2026, 11, 10, 12, 0, tzinfo=UTC), timedelta(minutes=16)) is None
    assert Cron("0 9 * * 7", "UTC").matches(datetime(2026, 11, 15, 9, 0, tzinfo=UTC))  # 7 = Sunday


@pytest.mark.parametrize("bad", ["* * *", "61 * * * *", "0 25 * * *", "*/0 * * * *", "x * * * *"])
def test_cron_rejects(bad: str) -> None:
    with pytest.raises(CronError):
        Cron(bad, "UTC")
    with pytest.raises(CronError):
        Cron("0 7 * * *", "Mars/Olympus")


# --- config ---------------------------------------------------------------------------------------------------------


FIXTURE = "docs/news/fixtures/job-config.json"


def test_shared_job_config_fixture_parses() -> None:
    cfg = JobConfig.model_validate(json.loads(www(FIXTURE).read_text()))
    assert abs(sum(cfg.effective_weights().values()) - 1) < 1e-6


def test_job_config_refuses_unknown_and_bad_values() -> None:
    with pytest.raises(ValueError):
        JobConfig.model_validate({"max_articels_per_run": 3})
    with pytest.raises(ValueError):
        JobConfig.model_validate({"weights": {"vibes": 0.5}})
    with pytest.raises(ValueError):
        JobConfig.model_validate({"publishing_slots": ["25:00"]})
    with pytest.raises(ValueError):
        JobConfig.model_validate({"models": {"writer": "gpt-5"}})


def test_kill_switch_defaults_off_and_settings_can_only_pause() -> None:
    assert not load_news_settings(None, {}).enabled
    assert load_news_settings({"paused": False}, {"NEWS_DESK_ENABLED": "true"}).enabled
    assert not load_news_settings({"paused": True}, {"NEWS_DESK_ENABLED": "true"}).enabled
    assert not load_news_settings({"paused": False, "env_enabled": True}, {}).enabled  # the row cannot fake the deployment's value
    # P5: the admin's switch (row key "enabled") decides once saved; NEWS_DESK_ENABLED=false is the emergency override.
    assert load_news_settings({"enabled": True}, {}).enabled
    assert not load_news_settings({"enabled": False}, {"NEWS_DESK_ENABLED": "true"}).enabled
    assert not load_news_settings({"enabled": True, "paused": True}, {}).enabled  # an old pause is still honoured
    for off in ("false", "0", "off", "no"):
        assert not load_news_settings({"enabled": True}, {"NEWS_DESK_ENABLED": off}).enabled
    assert not load_news_settings({"enabled": "yes"}, {}).enabled  # not a boolean: treated as never saved
    assert not load_news_settings({"desk_enabled": True, "env_forced_off": False}, {"NEWS_DESK_ENABLED": "false"}).enabled
    assert load_news_settings({"enabled": True, "max_drafts_per_day": 3}, {}).max_drafts_per_day == 3  # the rest of the row still loads
    assert load_news_settings({}, {"NEWS_IMAGE_PROVIDER": "gemini"}).image_provider == "gemini"


def test_github_releases_api_skips_drafts_and_prereleases() -> None:
    body = json.dumps([
        {"html_url": "https://github.com/kubernetes/kubernetes/releases/tag/v1.35.0", "name": "Kubernetes v1.35.0", "tag_name": "v1.35.0", "published_at": "2026-11-10T04:00:00Z", "body": "Notes", "draft": False, "prerelease": False},
        {"html_url": "https://github.com/kubernetes/kubernetes/releases/tag/v1.36.0-alpha.1", "name": "v1.36.0-alpha.1", "tag_name": "v1.36.0-alpha.1", "prerelease": True},
        {"html_url": "https://github.com/hashicorp/terraform/releases/tag/v1.14.2", "name": "v1.14.2", "tag_name": "v1.14.2", "published_at": "2026-11-09T04:00:00Z", "body": "", "draft": False, "prerelease": False},
    ])
    r = feeds.parse_feed("github_releases", body, "u")
    assert [x.title for x in r] == ["Kubernetes v1.35.0", "hashicorp/terraform v1.14.2"]
    with pytest.raises(feeds.FeedError):
        feeds.parse_feed("github_releases", json.dumps({"message": "API rate limit exceeded"}), "u")
