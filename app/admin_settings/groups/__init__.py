"""One module per settings group (append only)."""

from __future__ import annotations

from ..registry import Group, check_registry
from . import embeddings, limits, llm

GROUPS: list[Group] = [
    llm.GROUP,
    limits.GROUP,
    embeddings.GROUP,
]

check_registry(GROUPS)
