"""The seo-agent (mrjithendar/moks-seo-agent) that the contract tests compare this service with: the routes the news
agent calls there (POST /search, POST /llm-budget/reserve|settle) and what they accept and answer. Found as

- $SEO_AGENT_DIR, a moks-seo-agent checkout (a wrong path is an error, never a skip), or
- ../moks-seo-agent, a checkout next to this repository.

With neither, the tests that need it are skipped. CI has no checkout (no cross-repository credentials, by choice).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _checkout() -> Path | None:
    env = os.environ.get("SEO_AGENT_DIR")
    if env:
        if not (Path(env) / "app" / "main.py").is_file():
            raise RuntimeError(f"SEO_AGENT_DIR={env} is not a moks-seo-agent checkout (no app/main.py)")
        return Path(env)
    sibling = ROOT.parent / "moks-seo-agent"
    return sibling if (sibling / "app" / "main.py").is_file() else None


CHECKOUT = _checkout()


def seo(rel: str) -> Path:
    if CHECKOUT is None:
        pytest.skip("moks-seo-agent not found: set SEO_AGENT_DIR to a checkout, or check it out next to this repository")
    return CHECKOUT / rel
