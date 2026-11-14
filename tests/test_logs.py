"""Log lines are JSON whatever the message holds, and carry the fields passed with `extra=`."""

from __future__ import annotations

import json
import logging
import sys

from app.logs import JsonFormatter


def record(msg: str, *args: object, exc: bool = False, **extra: object) -> logging.LogRecord:
    r = logging.LogRecord("seo_agent", logging.WARNING, __file__, 1, msg, args, sys.exc_info() if exc else None)
    for key, value in extra.items():
        setattr(r, key, value)
    return r


def test_a_message_with_quotes_and_newlines_is_one_valid_json_line() -> None:
    line = JsonFormatter().format(record('article "%s" failed\nsecond line', "ingress-tls"))
    assert "\n" not in line
    assert json.loads(line) == {"level": "WARNING", "logger": "seo_agent", "msg": 'article "ingress-tls" failed\nsecond line'}


def test_extra_fields_are_logged() -> None:
    out = json.loads(JsonFormatter().format(record("mcp tool failed", event="mcp_tool_failed", tool="get_overview", ref="a1b2c3", when=object)))
    assert out["event"] == "mcp_tool_failed" and out["tool"] == "get_overview" and out["ref"] == "a1b2c3"
    assert isinstance(out["when"], str)  # anything JSON cannot hold is written as text, never an error
    # A field may not replace the line's own keys.
    assert json.loads(JsonFormatter().format(record("real", level="FAKE")))["level"] == "WARNING"


def test_a_traceback_stays_inside_the_line() -> None:
    try:
        raise ValueError('boom "quoted"')
    except ValueError:
        line = JsonFormatter().format(record("flow run failed", exc=True, event="flow_failed"))
    assert "\n" not in line
    out = json.loads(line)
    assert out["event"] == "flow_failed" and 'ValueError: boom "quoted"' in out["exc"] and "Traceback" in out["exc"]
