"""Check the components listed in catalog.toml: present, how old, what to do if not."""
from __future__ import annotations

import datetime as dt
import os
import re
import tomllib
from pathlib import Path

from lantern.config import LANTERN_DIR

CATALOG = LANTERN_DIR / "catalog.toml"


def load(path: Path = CATALOG) -> list[dict]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)["component"]


def _date(c: dict, present: list[str]) -> dt.date | None:
    df = c.get("date_file")
    if df and Path(df).exists():
        m = re.search(c.get("date_regex", r"(\d{4}-\d{2}-\d{2})"), Path(df).read_text(errors="ignore"))
        if m:
            try:
                return dt.datetime.strptime(m.group(1), c.get("date_format", "%Y-%m-%d")).date()
            except ValueError:
                pass
    times = []
    for p in present:
        try:
            times.append(os.stat(p).st_mtime)
        except OSError:
            pass
    return dt.date.fromtimestamp(max(times)) if times else None


def check(c: dict, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    need, alts = c.get("paths", []), c.get("any_of", [])
    missing = [p for p in need if not Path(p).exists()]
    present = [p for p in need + alts if Path(p).exists()]
    if missing or (alts and not any(Path(p).exists() for p in alts)):
        status, note = "missing", ", ".join(missing) or "none of: " + ", ".join(alts)
    else:
        status, note = "ok", ""
    when = _date(c, present) if present else None
    age = (today - when).days if when else None
    if status == "ok" and c.get("max_age_days", 0) and age is not None and age > c["max_age_days"]:
        status, note = "stale", f"{age} days old (limit {c['max_age_days']})"
    return {"name": c["name"], "kind": c.get("kind", ""), "used_for": c.get("used_for", ""), "status": status,
            "date": when.isoformat() if when else None, "note": note, "fix": c.get("fix", "") if status != "ok" else ""}


def check_all(path: Path = CATALOG) -> list[dict]:
    return [check(c) for c in load(path)]
