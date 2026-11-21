"""Process logging: one JSON object per line on stderr.

Forked from moks-seo-agent/app/logs.py@bb9e553 on 2026-11-21.

The format string this replaces only looked like JSON: a quote or a newline in a message broke the line, tracebacks
were printed as extra plain-text lines, and the fields passed with `extra=` (event, tool, ref, …) were dropped, so the
reference an MCP client is shown for a failed tool call could not be found in the logs.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

# Attributes every LogRecord carries. Anything else on a record was passed with `extra=` and is logged as a field.
_RECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime", "taskName"}


def _context() -> dict[str, str]:
    """trace_id/span_id of the current span (Grafana links a Loki line to its Jaeger trace) and the flow run id."""
    out: dict[str, str] = {}
    try:
        from .tracing import current_ids

        ids = current_ids()
        if ids:
            out["trace_id"], out["span_id"] = ids
        from .runs import CURRENT_RUN_ID

        run_id = CURRENT_RUN_ID.get()
        if run_id:
            out["run_id"] = run_id
    except Exception:  # noqa: S110 - a log line must never fail because of its context
        pass
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {"level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for key, value in record.__dict__.items():
            if key not in _RECORD_ATTRS and key not in out:
                out[key] = value
        for key, value in _context().items():
            out.setdefault(key, value)  # a field passed with extra= (e.g. a crew thread's run_id) wins
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            out["stack"] = self.formatStack(record.stack_info)
        return json.dumps(out, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Root logger → stderr as JSON lines, at LOG_LEVEL (default INFO). Does nothing if logging is already set up."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), handlers=[handler])
