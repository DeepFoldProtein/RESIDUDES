#!/usr/bin/env python3
"""Submit the benchmark matrix (model × backend × target) as Slurm jobs via `lantern-engine submit`.

    bench/run.py [--only pt2,of3] [--backends trt,torch] [--dry-run]

Outputs land in runs/bench/<target>_<sto>/<model>-<backend>/seed_<N>/ and the
submitted job ids in runs/bench/jobs.json. Re-running skips combinations whose
seed dir already holds a .lantern-engine/run.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lantern_engine.runners._common import meta_of  # noqa: E402
CFG = json.loads((ROOT / "bench" / "targets.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of model tags")
    ap.add_argument("--backends", default=None, help="comma list of backends")
    ap.add_argument("--targets", default=None, help="comma list of target names")
    ap.add_argument("--time", default="02:00:00")
    ap.add_argument("--suffix", default="", help="output dir suffix, e.g. `warm` for a second JAX pass with a hot compile cache")
    ap.add_argument("--site", default=None, help="site name or sites/*.toml path passed through to lantern-engine submit")
    ap.add_argument("--after", default=None, metavar="JOBID", help="Slurm dependency: start after this job succeeds")
    ap.add_argument("--max-queued", type=int, default=None, metavar="N",
                    help="submit only while fewer than N of this matrix's jobs are in the queue (call again later for the rest)")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()
    suffix = f"-{ns.suffix}" if ns.suffix else ""
    models = ns.only.split(",") if ns.only else CFG["models"]
    backends = ns.backends.split(",") if ns.backends else CFG["backends"]
    targets = [t for t in CFG["targets"] if not ns.targets or t["name"] in ns.targets.split(",")]
    jobs_path = ROOT / "runs" / "bench" / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    seed = CFG["seed"]
    in_queue = set(subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i"],
                                  capture_output=True, text=True).stdout.split())
    queued = sum(1 for k, v in jobs.items() if str(v.get("job")) in in_queue)
    remaining = 0
    for t in targets:
        inp = Path(CFG["root"]) / t["name"] / "inputs" / t["sto"] / "af3_msa" / "input.json"
        if not inp.is_file():
            print(f"skip {t['name']} {t['sto']}: no {inp}", file=sys.stderr)
            continue
        for m in models:
            for b in backends:
                out = ROOT / "runs" / "bench" / f"{t['name']}_{t['sto']}" / f"{m}-{b}{suffix}"
                key = f"{t['name']}_{t['sto']}/{m}-{b}{suffix}"
                if (meta_of(out / f"seed_{seed}") / "run.json").exists():
                    print(f"done  {key}")
                    continue
                prev = (jobs.get(key) or {}).get("job")
                if prev and str(prev) in in_queue:
                    print(f"queued {key} (job {prev} still in the queue)")
                    continue
                if ns.max_queued is not None and queued >= ns.max_queued:
                    remaining += 1
                    continue
                cmd = [str(ROOT / "bin" / "lantern-engine"), "submit", "--model", m, "--backend", b,
                       "--input", str(inp), "--out", str(out), "--seeds", str(seed),
                       "--gpu", CFG["gpu"], "--time", ns.time] + (["--site", ns.site] if ns.site else []) \
                    + (["--after", ns.after] if ns.after else [])
                if ns.dry_run:
                    print("dry  ", " ".join(cmd))
                    continue
                res = subprocess.run(cmd, capture_output=True, text=True)
                jid = next((ln.split()[-1] for ln in res.stdout.splitlines() if ln.startswith("Submitted")), None)
                if res.returncode != 0 or not jid:
                    print(f"FAIL  {key}: {res.stderr.strip()[-200:]}", file=sys.stderr)
                    continue
                jobs[key] = {"job": jid, "tokens": t["tokens"], "model": m, "backend": b,
                             "target": t["name"], "sto": t["sto"], "out": str(out / f"seed_{seed}")}
                queued += 1
                print(f"queued {key} → job {jid}")
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"{len(jobs)} job records in {jobs_path}; {queued} in the queue; {remaining} held back by --max-queued")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
