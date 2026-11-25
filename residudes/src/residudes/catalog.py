"""Check the components listed in catalog.toml: present, how old, what to do if not.

A path may be written as ``{key}``, which is filled from ``[databases]`` in residudes.toml.
That keeps a machine's own layout in the config a site writes, instead of in the catalog
everyone shares; a component whose key is unset reports ``unconfigured`` and says which key.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import tomllib
from pathlib import Path

from residudes.config import RESIDUDES_DIR

CATALOG = RESIDUDES_DIR / "catalog.toml"


_KEY = re.compile(r"\{([a-z0-9_]+)\}")


def _fill(value: str, paths: dict) -> tuple[str, list[str]]:
    """Replace every {key} from `paths`; return the value and the keys that had none."""
    missing = [k for k in _KEY.findall(value) if k not in paths]
    return _KEY.sub(lambda m: str(paths.get(m.group(1), m.group(0))), value), missing


def _resolve(c: dict, paths: dict) -> dict:
    c = dict(c)
    unset: list[str] = []
    for field in ("paths", "any_of"):
        if c.get(field):
            filled = [_fill(v, paths) for v in c[field]]
            c[field] = [v for v, _ in filled]
            unset += [k for _, ks in filled for k in ks]
    if c.get("date_file"):
        c["date_file"], ks = _fill(c["date_file"], paths)
        unset += ks
    c["unset"] = sorted(set(unset))
    return c


def load(path: Path = CATALOG, paths: dict | None = None) -> list[dict]:
    with open(path, "rb") as fh:
        return [_resolve(c, paths or {}) for c in tomllib.load(fh)["component"]]


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
    if c.get("unset"):
        keys = ", ".join(c["unset"])
        return {"name": c["name"], "kind": c.get("kind", ""), "used_for": c.get("used_for", ""),
                "status": "unconfigured", "date": None,
                "note": f"set [databases] {keys} in residudes.toml", "fix": c.get("fix", "")}
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


def check_all(path: Path = CATALOG, paths: dict | None = None) -> list[dict]:
    return [check(c) for c in load(path, paths)]
