"""`lantern-qa data fetch|verify|list`: put the data files of data/manifest.toml under the data root.

Sources are tried per file in this order unless restricted:
  local  the reference copy on the DeepFold cluster (copied, or symlinked with --link)
  url    a public upstream serving identical bytes (ProteinMPNN weights)
A fetched file is accepted only if its sha256 matches the manifest.
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import sys
import tomllib
import urllib.request
from pathlib import Path

from lantern_qa.paths import MANIFEST, data_root


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)["file"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _select(files: list[dict], only: list[str] | None) -> list[dict]:
    if not only:
        return files
    return [f for f in files if any(fnmatch.fnmatch(f["path"], pat) for pat in only)]


def status(entry: dict, root: Path, full: bool) -> str:
    p = root / entry["path"]
    if not p.exists():
        return "missing"
    if p.stat().st_size != entry["size"]:
        return "size-mismatch"
    if full and sha256(p) != entry["sha256"]:
        return "sha-mismatch"
    return "ok"


def _fetch_one(entry: dict, root: Path, link: bool, sources: list[str]) -> str:
    dest = root / entry["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for src in sources:
        if src == "local" and entry.get("local") and Path(entry["local"]).exists():
            if link:
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                dest.symlink_to(entry["local"])
                return "linked"
            shutil.copyfile(entry["local"], tmp)
        elif src == "url" and entry.get("url"):
            with urllib.request.urlopen(entry["url"], timeout=120) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh)
        else:
            continue
        if sha256(tmp) != entry["sha256"]:
            tmp.unlink()
            print(f"  sha256 mismatch from {src}: {entry['path']}", file=sys.stderr)
            continue
        os.replace(tmp, dest)
        return f"copied from {src}"
    return "unavailable"


def cmd_list(args) -> int:
    root = data_root()
    print(f"data root: {root}")
    for e in _select(load_manifest(), args.only):
        src = "local+url" if e.get("url") else "local"
        print(f"{status(e, root, full=False):14s} {e['size'] / 1e6:8.1f} MB  {src:9s}  {e['path']}")
    return 0


def cmd_fetch(args) -> int:
    root = data_root()
    sources = [args.source] if args.source else ["local", "url"]
    files = _select(load_manifest(), args.only)
    need = [e for e in files if args.force or status(e, root, full=False) != "ok"]
    print(f"data root: {root}  ({len(files)} files in scope, {len(need)} to fetch, "
          f"{sum(e['size'] for e in need) / 1e6:.0f} MB; mode={'link' if args.link else 'copy'})")
    if need and not args.link:
        free = shutil.disk_usage(root if root.exists() else root.parent if root.parent.exists() else Path.home()).free
        if free < sum(e["size"] for e in need) * 1.1:
            print(f"not enough free space under {root} ({free / 1e9:.1f} GB free)", file=sys.stderr)
            return 2
    bad = 0
    for e in need:
        how = _fetch_one(e, root, args.link, sources)
        bad += how == "unavailable"
        print(f"  {how:18s} {e['path']}")
    print("done" if not bad else f"{bad} file(s) could not be fetched (no reachable source)")
    return 1 if bad else 0


def cmd_verify(args) -> int:
    root = data_root()
    files = _select(load_manifest(), args.only)
    bad = 0
    for e in files:
        st = status(e, root, full=not args.quick)
        if st != "ok":
            bad += 1
            print(f"{st:14s} {e['path']}")
    print(f"data root: {root}  {len(files) - bad}/{len(files)} ok ({'size' if args.quick else 'sha256'} check)")
    return 1 if bad else 0


def add_parser(sub) -> None:
    p = sub.add_parser("data", help="fetch / verify / list the data files (models, potentials, weights)")
    s = p.add_subparsers(dest="data_cmd", required=True)
    for name, fn, hlp in (("list", cmd_list, "show every file and whether it is present"),
                          ("fetch", cmd_fetch, "copy (or --link) missing files into the data root"),
                          ("verify", cmd_verify, "check sha256 (or --quick: size only)")):
        q = s.add_parser(name, help=hlp)
        q.add_argument("--only", nargs="*", metavar="GLOB", help="restrict to manifest paths matching these globs")
        q.set_defaults(func=fn)
        if name == "fetch":
            q.add_argument("--link", action="store_true", help="symlink to the local reference copy instead of copying")
            q.add_argument("--source", choices=["local", "url"], help="use only this source")
            q.add_argument("--force", action="store_true", help="re-fetch files that already verify")
        if name == "verify":
            q.add_argument("--quick", action="store_true", help="size check only")
