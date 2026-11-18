"""The web app (moks-cms) that the contract tests compare the agent with.

The agent reads and writes the web app's database and calls its callback API, and some tests check that both sides
still agree by reading moks-cms source: its migrations, the callback server's route table and a few shared constants.
moks-cms is a separate repository (moks-tech/moks-cms), found here as

- $MOKS_CMS_DIR, a moks-cms checkout (a wrong path is an error, never a skip), or
- ../moks-cms, a checkout next to this repository (local development).

With neither, the tests that need it are skipped. CI has no moks-cms checkout, so they run only locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _checkout() -> Path | None:
    env = os.environ.get("MOKS_CMS_DIR")
    if env:
        if not (Path(env) / "src" / "db" / "migrations").is_dir():
            raise RuntimeError(f"MOKS_CMS_DIR={env} is not a moks-cms checkout (no src/db/migrations)")
        return Path(env)
    sibling = ROOT.parent / "moks-cms"
    return sibling if (sibling / "src" / "db" / "migrations").is_dir() else None


CHECKOUT = _checkout()


def www(rel: str) -> Path:
    """`<rel>` in the moks-cms checkout; skips the calling test when there is none."""
    if CHECKOUT is None:
        pytest.skip("moks-cms not found: set MOKS_CMS_DIR to a moks-cms checkout, or check it out next to this repository")
    return CHECKOUT / rel
