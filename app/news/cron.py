"""A small 5-field cron evaluator (minute hour day-of-month month day-of-week) with IANA time zones.

Supports `*`, lists `1,5`, ranges `1-5`, steps `*/15` and `1-30/5`, names for months/weekdays, and the Vixie rule that
when both day fields are restricted a day matches either. No @reboot/@yearly aliases except @hourly/@daily/@weekly.
Kept in-house (≈80 lines, unit-tested) instead of a new dependency for the dispatcher's "which jobs are due" question.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

ALIASES = {"@hourly": "0 * * * *", "@daily": "0 0 * * *", "@weekly": "0 0 * * 0"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DAYS = {d: i for i, d in enumerate(["sun", "mon", "tue", "wed", "thu", "fri", "sat"])}


class CronError(ValueError):
    pass


def _field(spec: str, lo: int, hi: int, names: dict[str, int] | None = None) -> tuple[set[int], bool]:
    out: set[int] = set()
    restricted = spec != "*"
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/", 1)
            if not s.isdigit() or int(s) < 1:
                raise CronError(f"bad step in {spec!r}")
            step = int(s)
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            x, y = part.split("-", 1)
            a, b = _num(x, names), _num(y, names)
        else:
            a = b = _num(part, names)
            if step > 1:
                b = hi
        if not (lo <= a <= hi and lo <= b <= hi and a <= b):
            raise CronError(f"{spec!r} is outside {lo}-{hi}")
        out.update(range(a, b + 1, step))
    return out, restricted


def _num(v: str, names: dict[str, int] | None) -> int:
    v = v.strip().lower()
    if names and v in names:
        return names[v]
    if not v.isdigit():
        raise CronError(f"{v!r} is not a number")
    return int(v)


class Cron:
    def __init__(self, expr: str, tz: str = "UTC"):
        e = ALIASES.get(expr.strip(), expr.strip())
        parts = e.split()
        if len(parts) != 5:
            raise CronError("a cron expression has 5 fields: minute hour day month weekday")
        try:
            self.tz = ZoneInfo(tz)
        except Exception as ex:
            raise CronError(f"unknown time zone {tz!r}") from ex
        self.minutes, _ = _field(parts[0], 0, 59)
        self.hours, _ = _field(parts[1], 0, 23)
        self.dom, self.dom_r = _field(parts[2], 1, 31)
        self.months, _ = _field(parts[3], 1, 12, MONTHS)
        dow, self.dow_r = _field(parts[4], 0, 7, DAYS)  # 0 and 7 are both Sunday
        self.dow = {d % 7 for d in dow}

    def _day_ok(self, d: datetime) -> bool:
        dom_ok = d.day in self.dom
        dow_ok = (d.isoweekday() % 7) in self.dow
        if self.dom_r and self.dow_r:
            return dom_ok or dow_ok
        return dom_ok and dow_ok

    def matches(self, when: datetime) -> bool:
        local = when.astimezone(self.tz)
        return local.minute in self.minutes and local.hour in self.hours and local.month in self.months and self._day_ok(local)

    def next_after(self, after: datetime, limit_days: int = 400) -> datetime:
        """The first fire time strictly after `after` (UTC result)."""
        t = after.astimezone(self.tz).replace(second=0, microsecond=0) + timedelta(minutes=1)
        end = t + timedelta(days=limit_days)
        while t < end:
            if t.month not in self.months:
                t = (t.replace(day=1, hour=0, minute=0) + timedelta(days=32)).replace(day=1)
                continue
            if not self._day_ok(t):
                t = (t + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if t.hour not in self.hours:
                t = (t + timedelta(hours=1)).replace(minute=0)
                continue
            if t.minute not in self.minutes:
                t += timedelta(minutes=1)
                continue
            # Wall-clock times skipped by a DST jump do not exist; zoneinfo normalises them forward, which is fine.
            return t.astimezone(UTC)
        raise CronError("no fire time within a year")

    def last_due(self, now: datetime, window: timedelta) -> datetime | None:
        """The latest fire time in (now - window, now], or None. The dispatcher runs every 15 minutes with a window
        slightly larger than that, and the run's slot (job id + this time) makes a double fire a no-op."""
        t = self.next_after(now - window - timedelta(minutes=1))
        last = None
        while t <= now:
            last = t
            t = self.next_after(t)
        return last


def validate(expr: str, tz: str) -> str | None:
    try:
        Cron(expr, tz)
        return None
    except CronError as e:
        return str(e)
