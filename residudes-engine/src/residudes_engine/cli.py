"""residudes-engine command line.

  residudes-engine setup    all | venv | checkpoints | site      one-time machine setup
  residudes-engine doctor                                        check paths, GPU, venv, TRT plugin
  residudes-engine predict  --model pt2 --input input.json [--backend auto|trt|jax|torch] [--out DIR] [--seed N]
  residudes-engine submit   --model pt2 --input input.json --seeds 1,2,3 [--gpu rtx4090]     (Slurm, one job per seed)
  residudes-engine verify   <seed_dir> [--ref <seed_dir>]
  residudes-engine engines  list | build --model pt2 [--max-seqlen 1024] [--slurm]
  residudes-engine models | sites | env
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import __version__, registry, runners
from . import setup as setup_mod
from .config import (RESIDUDES_ENGINE_ROOT, SiteError, available_sites, detect_gpu, find_libstdcxx,
                     load_site, runtime_env, venv_python)
from .runners._common import META_DIR, Request, load_af3_dialect, run_plan, write_manifest
from .verify import check_seed_dir, format_report


# ── argument plumbing ──────────────────────────────────────────────────────

def _add_predict_options(p: argparse.ArgumentParser, *, seeds: bool) -> None:
    p.add_argument("--model", "-m", required=True, choices=sorted(registry.MODELS), help="model tag")
    p.add_argument("--backend", "-b", default="auto", choices=("auto", *registry.BACKENDS),
                   help="auto (default) = trt when engines for this GPU exist, else torch")
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="AF3-dialect input.json (all models) or a native input (Boltz YAML / OF3 query)")
    p.add_argument("--out", "-o", type=Path, default=None,
                   help="output directory (default: engine_out/<input name>_<model>[/seed_<N>])")
    if seeds:
        p.add_argument("--seeds", default=None,
                       help="comma list / ranges, one job per seed → <out>/seed_<N>/ (default: seeds from input, one job)")
    else:
        p.add_argument("--seed", type=int, default=None, help="override modelSeeds with this single seed")
    p.add_argument("--num-recycles", type=int, default=None, help="trunk recycles (runner default 10)")
    p.add_argument("--diffusion-samples", type=int, default=None, help="samples per seed (default 5)")
    p.add_argument("--template-weight", type=float, default=None,
                   help="scale template-embedder output (PT2/OF3/BT2_TPL_WEIGHT)")
    p.add_argument("--pairing-mode", choices=("precomputed", "online"), default=None,
                   help="OF3 MSA pairing mode (default: pairing_mode.txt sidecar, else precomputed)")
    p.add_argument("--engines-dir", type=Path, default=None, help="override TRT engine directory")
    p.add_argument("--site", default=None, help="site name or sites/*.toml path (default: auto)")
    p.add_argument("--dry-run", action="store_true", help="print the plan without running")
    p.add_argument("--keep-native", action="store_true", help="do not flatten the runner's native layout")
    p.add_argument("extra", nargs="*", help="extra args forwarded to the runner (after --)")


def _expand_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            a_i, b_i = int(a), int(b)
            if a_i > b_i:
                raise SystemExit(f"bad seed range {tok!r}")
            out += list(range(a_i, b_i + 1))
        else:
            out.append(int(tok))
    return out


def _passthrough(ns: argparse.Namespace) -> list[str]:
    """Re-serialise predict options for `residudes-engine submit` → sbatch → `residudes-engine predict`."""
    args: list[str] = []
    for key, flag in (("num_recycles", "--num-recycles"), ("diffusion_samples", "--diffusion-samples"),
                      ("template_weight", "--template-weight"), ("pairing_mode", "--pairing-mode"),
                      ("engines_dir", "--engines-dir")):
        v = getattr(ns, key, None)
        if v is not None:
            args += [flag, str(v)]
    if ns.keep_native:
        args.append("--keep-native")
    return args


def _input_name(path: Path) -> str:
    d = load_af3_dialect(path)
    name = (d or {}).get("name") if d else None
    return str(name or path.stem)


def _default_out(ns: argparse.Namespace, model_tag: str) -> Path:
    if ns.out is not None:
        return ns.out
    base = Path("engine_out") / f"{_input_name(ns.input)}_{model_tag}"
    seed = getattr(ns, "seed", None)
    return base / f"seed_{seed}" if seed is not None else base


def _resolve_backend(requested: str, site, model, gpu) -> tuple[str, str]:
    """auto → trt when engines for this GPU exist, else torch (or the model's only
    backend, for AlphaFold 3). Returns (backend, reason)."""
    if requested != "auto":
        if requested not in model.backends:
            raise SystemExit(f"{model.name} runs on {', '.join(model.backends)} only; "
                             f"--backend {requested} is not available for {model.tag}")
        return requested, "requested"
    if "trt" not in model.backends:
        return model.backends[0], f"{model.tag} runs on {model.backends[0]}"
    fb = model.fallback
    if gpu is None or gpu.tag is None:
        return fb, f"no GPU class detected → {fb} (no engines needed)"
    d = registry.find_engine_dir(site.engines, gpu.tag, model)
    if d is not None:
        return "trt", f"engines found: {d}"
    return fb, f"no TRT engines for {model.tag} on {gpu.tag} → {fb} (build some with `residudes-engine engines build`)"


# ── commands ───────────────────────────────────────────────────────────────

def cmd_predict(ns: argparse.Namespace) -> int:
    site = load_site(ns.site)
    model = registry.get_model(ns.model)
    gpu = detect_gpu()
    backend, why = _resolve_backend(ns.backend, site, model, gpu)
    ns.out = _default_out(ns, model.tag)
    req = Request(
        model=model, backend=backend, input=ns.input, out=ns.out, site=site, gpu=gpu,
        seed=ns.seed, num_recycles=ns.num_recycles, diffusion_samples=ns.diffusion_samples,
        template_weight=ns.template_weight, pairing_mode=ns.pairing_mode,
        engines_dir=ns.engines_dir, extra_args=list(ns.extra), dry_run=ns.dry_run,
        keep_native=ns.keep_native,
    )
    if not req.input.exists():
        raise SystemExit(f"input not found: {req.input}")
    runner = runners.for_family(model.family)
    req.out.mkdir(parents=True, exist_ok=True)
    plan = runner.plan(req)

    print(f"▶ residudes-engine {__version__}  model={model.tag} ({model.name})  backend={req.backend} ({why})  site={site.name}")
    print(f"▶ gpu : {gpu.name + ' cc=' + gpu.compute_cap + ' tag=' + str(gpu.tag) if gpu else 'none detected'}")
    print(f"▶ out : {req.out}")
    print(f"▶ venv: {plan.venv}")
    for k, v in plan.notes.items():
        print(f"▶ {k}: {v}")
    if ns.dry_run:
        print("▶ cwd :", plan.cwd)
        print("▶ cmd :", " ".join(shlex.quote(c) for c in plan.cmd))
        keys = ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "LAYERNORM_TYPE", "PYTORCH_CUDA_ALLOC_CONF",
                "DS_BUILD_OPS", "OF3_PAIRING_MODE", "OF3_PER_SAMPLE_TOKEN_CUTOFF", "BOLTZ_CACHE", "HF_HOME",
                "TMPDIR")
        for k in keys + tuple(k for k in sorted(plan.env) if k.startswith(("RESIDUDES_ENGINE_JAX_", "XLA_", "JAX_", "PT2_", "OF3_TPL", "BT2_"))):
            if k in plan.env:
                print(f"  env {k}={plan.env[k]}")
        from .runners._common import remove_files
        remove_files(plan.cleanup)
        return 0

    t0 = time.time()
    rc = run_plan(req, plan)
    t1 = time.time()
    moved: list[Path] = []
    if not req.keep_native:
        try:
            moved = runner.post(req, plan)
        except Exception as e:  # noqa: BLE001 — never lose a finished prediction over cleanup
            print(f"▶ warning: post-processing failed: {e}", file=sys.stderr)
    reports = []
    if rc == 0:
        reports = [check_seed_dir(d) for d in (moved or [req.out])]
        if any(r.problems for r in reports):
            # A runner can exit 0 and still write nothing (e.g. OF3 when JAX silently
            # falls back to CPU). Never report success without sample_<i>/ directories.
            rc = 3
    man = write_manifest(req, plan, rc, t0, t1)
    print(f"▶ exit {rc} after {t1 - t0:.0f}s; manifest {man}")
    for r in reports:
        print(format_report(r))
    if rc == 3:
        print("▶ error: runner exited 0 but produced no sample_<i>/ directories — see inference.log",
              file=sys.stderr)
    return rc


def cmd_submit(ns: argparse.Namespace) -> int:
    site = load_site(ns.site)
    if site.scheduler != "slurm":
        raise SystemExit(f"site {site.name!r} scheduler is {site.scheduler!r}; `submit` needs slurm")
    model = registry.get_model(ns.model)
    if not ns.input.exists():
        raise SystemExit(f"input not found: {ns.input}")
    sl = site.slurm
    seeds = _expand_seeds(ns.seeds) if ns.seeds else [None]
    base = _default_out(ns, model.tag).resolve()
    constraint = None
    if ns.gpu:
        constraint = (sl.get("constraint") or {}).get(ns.gpu)
        if constraint is None:
            raise SystemExit(f"--gpu {ns.gpu!r} has no [slurm.constraint] entry in {site.path.name}")
    script = RESIDUDES_ENGINE_ROOT / "scripts" / "slurm" / "residudes-engine.sbatch"
    job_ids: list[str] = []
    for seed in seeds:
        out = base if seed is None else base / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        name = f"fmx-{model.tag}-{ns.backend}" + (f"-s{seed}" if seed is not None else "")
        cmd = ["sbatch", "-J", name, "-p", str(sl.get("partition", "normal")),
               "--gres", str(sl.get("gres", "gpu:1")), "-c", str(sl.get("cpus", 4)),
               "-t", str(ns.time or sl.get("time", "02:00:00")),
               "-o", str(out / "slurm.out"), "-e", str(out / "slurm.err"),
               f"--export=ALL,RESIDUDES_ENGINE_ROOT={RESIDUDES_ENGINE_ROOT},RESIDUDES_ENGINE_SITE_FILE={site.path}"]
        if constraint:
            cmd += ["-C", constraint]
        if ns.after:
            cmd += [f"--dependency=afterok:{ns.after}"]
        cmd += [str(script), "predict", "--model", model.tag, "--backend", ns.backend,
                "--input", str(ns.input.resolve()), "--out", str(out), "--site", str(site.path)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        cmd += _passthrough(ns)
        if ns.extra:
            cmd += ["--"] + list(ns.extra)
        print("▶", " ".join(shlex.quote(c) for c in cmd))
        if ns.dry_run:
            continue
        res = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        if res.returncode != 0:
            return res.returncode
        jid = res.stdout.strip().split()[-1] if res.stdout.strip() else "?"
        job_ids.append(jid)
        (out / META_DIR).mkdir(exist_ok=True)
        (out / META_DIR / "slurm_job_id").write_text(jid + "\n")
    if job_ids:
        print("▶ submitted:", " ".join(job_ids))
        print(f"▶ watch   : squeue -j {','.join(job_ids)}   |   verify: residudes-engine verify {base}{'/seed_<N>' if ns.seeds else ''}")
    return 0


def cmd_verify(ns: argparse.Namespace) -> int:
    ref = check_seed_dir(ns.ref) if ns.ref else None
    rc = 0
    for d in ns.seed_dirs:
        rep = check_seed_dir(d)
        print(format_report(rep, ref))
        print()
        if not rep.ok:
            rc = 1
    return rc



def _msa_section(site, deep: bool = False) -> bool:
    """What an MSA search needs: the binaries, and the databases at the names each tool expects.

    A missing database is cheap to find here and expensive to find inside a queued job, so this
    checks names and sizes; --deep then runs one real jackhmmer iteration to prove the stack works.
    """
    from . import msa as msa_mod
    cfg = dict(site.raw.get("msa", {}) if site else {})
    # the same overrides `residudes-engine msa` honours, so the doctor reports what a run would use
    for key, env in (("af3_db_dir", "RESIDUDES_ENGINE_AF3_DB_DIR"), ("mmseqs_db", "RESIDUDES_ENGINE_MMSEQS_DB"),
                     ("mmseqs_bin", "RESIDUDES_ENGINE_MMSEQS")):
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    ok = True

    tools = {b: shutil.which(b) for b in msa_mod.AF3_BINARIES}
    tools["mmseqs"] = shutil.which(cfg.get("mmseqs_bin") or "mmseqs")
    missing = [b for b, p in tools.items() if not p]
    print(f"  {'tools':12} {'✓' if not missing else '✗'} "
          + ", ".join(f"{b}" for b, p in tools.items() if p) + (f"  MISSING: {', '.join(missing)}" if missing else ""))

    db_dir = cfg.get("af3_db_dir")
    if not db_dir:
        print(f"  {'af3 dbs':12} (unset: [msa] af3_db_dir — --source af3 needs it)")
    else:
        root = Path(db_dir)
        rows = []
        for name, want, what in msa_mod.AF3_DATABASES:
            p = root / name
            if not p.exists():
                rows.append(f"    ✗ {name}  missing ({what})")
                ok = False
            elif p.is_dir():
                rows.append(f"    ✓ {name}/  ({what})")
            else:
                size = p.stat().st_size
                mark = "✓" if size >= want else "!"
                rows.append(f"    {mark} {name}  {size / 1e9:.1f} GB" + ("" if size >= want else f"  (expected >= {want / 1e9:.0f} GB: truncated?)"))
                ok &= size >= want
        print(f"  {'af3 dbs':12} {root}")
        for r in rows:
            print(r)

    mdb = cfg.get("mmseqs_db")
    if mdb:
        root = Path(mdb)
        found = sorted({f.name[: -len(".dbtype")] for f in root.glob("*.dbtype")}) if root.is_dir() else []
        print(f"  {'mmseqs db':12} {'✓' if found else '✗'} {root}" + (f"  ({', '.join(found[:6])})" if found else ""))
        ok &= bool(found)

    if deep and db_dir and tools.get("jackhmmer"):
        q = Path(tempfile.mkdtemp()) / "q.fasta"
        q.write_text(">probe\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n")
        target = Path(db_dir) / "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"   # the smallest one
        if target.is_file():
            cmd = [tools["jackhmmer"], "-N", "1", "--cpu", "4", "-o", "/dev/null", str(q), str(target)]
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(f"  {'deep check':12} {'✓' if r.returncode == 0 else '✗'} jackhmmer -N 1 over "
                  f"{target.name} in {time.time() - t0:.0f}s" + ("" if r.returncode == 0 else f": {r.stderr.strip()[:120]}"))
            ok &= r.returncode == 0
    return ok


def cmd_env(ns: argparse.Namespace) -> int:
    try:
        site = load_site(ns.site)
    except SiteError as e:
        print(f"✗ {e}")
        return 2
    gpu = detect_gpu()
    print(f"residudes-engine {__version__}  root={RESIDUDES_ENGINE_ROOT}")
    print(f"site     : {site.name}  ({site.path})")
    print(f"host     : {os.uname().nodename}   gpu: "
          + (f"{gpu.name} cc={gpu.compute_cap} {gpu.mem_mib} MiB tag={gpu.tag}" if gpu else "none"))
    try:
        import ssl  # noqa: F401
        ssl_note = "ssl ok"
    except ImportError:
        ssl_note = "no ssl module → downloads use curl"
    print(f"cli py   : {sys.executable} ({ssl_note}; curl {'found' if shutil.which('curl') else 'MISSING'})")

    def row(label: str, p: Path | None, kind: str = "dir") -> bool:
        if p is None:
            print(f"  {label:12} (unset)")
            return True
        ok = p.is_dir() if kind == "dir" else p.is_file()
        print(f"  {label:12} {'✓' if ok else '✗'} {p}")
        return ok

    ok = True
    ok &= row("core", site.core)
    ok &= row("venv", site.venv)
    row("venv_jax", site.venv_jax)
    ok &= row("checkpoints", site.checkpoints)
    ok &= row("engines", site.engines)
    row("kalign", site.kalign, "file")
    site.cache_dir.mkdir(parents=True, exist_ok=True)
    row("cache_dir", site.cache_dir)
    lib = find_libstdcxx(site.libstdcxx)
    print(f"  {'libstdc++':12} {'✓' if lib else '✗'} {lib or 'no libstdc++ with GLIBCXX_3.4.32 found (apt install g++-13, or set libstdcxx in the site file)'}")
    ok &= lib is not None
    print("MSA databases and tools:")
    ok &= _msa_section(site, deep=getattr(ns, "deep", False))
    print("models:")
    for tag, m in registry.MODELS.items():
        ck = site.checkpoints / m.checkpoint
        if m.family == "alphafold3":                       # licensed weights, placed by hand (site key af3_weights)
            d = site.af3_weights or (site.checkpoints / "af3")
            ck = next((d / f for f in ("af3.bin", "af3.bin.zst") if (d / f).is_file()), d / "af3.bin")
        line = f"  {tag:4} ckpt {'✓' if ck.exists() else '✗'} {ck}"
        if gpu and gpu.tag:
            e = m.engine_dir(site.engines, gpu.tag)
            has = bool(e and (e / 'pairformer_engine' / 'trt' / 'rank0.engine').is_file())
            mx = registry.engine_max_seq_len(e) if (e and has) else None
            line += f"\n       trt  {'✓' if has else '✗'} {e}" + (f"  (max_seq_len {mx})" if mx else "")
        else:
            line += "\n       trt  (no GPU here; engines resolved at run time by compute capability)"
        print(line)
    if ns.check:
        print("import check (site venv, protenix PYTHONPATH):")
        root = site.core
        env = runtime_env(site, site.venv, pythonpath=[root / "models" / "protenix", root / "thirdparty", root])
        code = (
            "import sys, torch, tensorrt as trt; print('  python', sys.version.split()[0], "
            "'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'tensorrt', trt.__version__)\n"
            "import tensorrt_bionemo; print('  tensorrt_bionemo ok from', tensorrt_bionemo.__file__)\n"
            "import protenix; print('  protenix from', protenix.__file__)\n"
            "try:\n"
            "    import jax, residudes_engine_jax, importlib.util as u\n"
            "    plug = 'cuda plugin OK' if (u.find_spec('jax_cuda12_plugin') or u.find_spec('jax_cuda13_plugin')) else 'NO CUDA PLUGIN → JAX would run on CPU; run `residudes-engine setup venv`'\n"
            "    print('  jax', jax.__version__, '|', plug, '| JAX modules from', residudes_engine_jax.__file__)\n"
            "except ImportError as e: print('  jax not installed → --backend jax unavailable in this venv (', e, ')')\n"
            "try:\n"
            "    names=[c.name for c in trt.get_plugin_registry().plugin_creator_list if 'Identity' in c.name]\n"
            "    print('  IdentitySZ plugin', 'loaded' if names else 'NOT FOUND', names)\n"
            "except Exception as e: print('  plugin registry check skipped:', e)\n"
        )
        # cwd = upstream Protenix tree, as protenix/inference.py chdir's before importing;
        # from the repo root the adapter stub `protenix/__init__.py` would shadow it.
        res = subprocess.run([str(venv_python(site.venv)), "-c", code], env=env,
                             cwd=str(root / "models" / "protenix"),
                             capture_output=True, text=True, timeout=600)
        sys.stdout.write(res.stdout)
        if res.returncode != 0:
            ok = False
            sys.stdout.write("  ✗ " + res.stderr.strip().splitlines()[-1] + "\n" if res.stderr.strip() else "")
    print("status   :", "OK" if ok else "PROBLEMS (see ✗ above)")
    return 0 if ok else 1


def cmd_run(ns: argparse.Namespace) -> int:
    """Run a Python script (or module) inside the runtime venv with the same
    environment a prediction gets — LD paths, caches, JAX knobs — for
    profiling and micro-benchmarks. `--jax` selects the JAX venv/env for
    `--model` (default pt2) exactly as `predict --backend jax` would."""
    try:
        site = load_site(ns.site)
    except SiteError as e:
        print(f"✗ {e}")
        return 2
    model = registry.get_model(ns.model)
    venv = site.jax_venv_for(model.tag) if ns.jax else site.venv
    env = runtime_env(site, venv, pythonpath=[site.core / "thirdparty"], jax=ns.jax,
                      model_tag=model.tag, family=model.family)
    env["RESIDUDES_ENGINE_CORE"] = str(site.core)
    args = list(ns.args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print("usage: residudes-engine run [--jax] [--model TAG] -- script.py [args…]   (or: -- -m module …)")
        return 2
    cmd = [str(venv_python(venv))] + args
    print("▶", shlex.join(cmd), file=sys.stderr)
    os.execve(cmd[0], cmd, env)


def cmd_models(ns: argparse.Namespace) -> int:
    print(f"{'tag':4} {'family':10} {'checkpoint':28} {'backends':16} name")
    for tag, m in registry.MODELS.items():
        print(f"{tag:4} {m.family:10} {m.checkpoint:28} {'/'.join(m.backends):16} {m.name}")
    for tag, desc in registry.EXTERNAL_MODELS.items():
        print(f"{tag:4} {'-':10} {'-':28} {desc} [not run by residudes-engine]")
    print("backends:", ", ".join(registry.BACKENDS), " gpu tags:", ", ".join(sorted(set(registry.GPUS.values()))))
    return 0


def cmd_sites(ns: argparse.Namespace) -> int:
    for name in available_sites():
        try:
            s = load_site(name)
            print(f"{name:10} scheduler={s.scheduler:6} core={s.core}")
        except SiteError as e:
            print(f"{name:10} ✗ {e}")
    try:
        print("auto-selected:", load_site().name)
    except SiteError as e:
        print("auto-selected: none —", e)
    return 0


def _ensure_jax_cuda_plugin(venv: Path, env: dict[str, str]) -> None:
    """The lock's `jax-cu13` extra pins jax 0.5.x, which has no `cuda13` extra, so uv drops
    the marker silently and the venv gets jax without a CUDA PJRT plugin → JAX falls back
    to CPU (bt2 errors out, OF3 exits 0 with no output). Install the cuda12 plugin at the
    locked jax version."""
    py = venv / "bin" / "python"
    probe = subprocess.run(
        [str(py), "-c",
         "import jax, importlib.util as u; "
         "print(jax.__version__, bool(u.find_spec('jax_cuda12_plugin') or u.find_spec('jax_cuda13_plugin')))"],
        capture_output=True, text=True, env=env,
    )
    if probe.returncode != 0:
        return  # jax not installed (cu12 flavour): nothing to do
    version, has_plugin = probe.stdout.split()
    if has_plugin == "True":
        return
    cmd = ["uv", "pip", "install", "--python", str(py), f"jax[cuda12]=={version}"]
    print("▶", " ".join(cmd), " (CUDA PJRT plugin missing from the lock's jax extra)")
    subprocess.run(cmd, env=env, check=False)


def cmd_setup(ns: argparse.Namespace) -> int:
    """`residudes-engine setup {all|venv|checkpoints|site}`."""
    if ns.what == "site":
        if not (ns.checkpoints and ns.engines):
            raise SystemExit("setup site needs --checkpoints DIR and --engines DIR")
        out = setup_mod.write_site(ns.name, checkpoints=Path(ns.checkpoints).resolve(),
                                   engines=Path(ns.engines).resolve(), scheduler=ns.scheduler,
                                   partition=ns.partition, hosts=None, force=ns.force)
        print(f"▶ wrote {out}  → next: residudes-engine setup venv ; residudes-engine setup checkpoints ; residudes-engine doctor")
        return 0
    if ns.what == "checkpoints":
        site = load_site(ns.site)
        tags = ns.models.split(",") if ns.models else sorted(registry.MODELS)
        return setup_mod.setup_checkpoints(site, tags, dry_run=ns.dry_run, force=ns.force,
                                           af3_from=getattr(ns, "af3_from", None))
    if ns.what == "all":
        if ns.slurm:
            raise SystemExit("setup all runs in the foreground (needs network); use `setup venv --slurm` separately")
        rc = _setup_venv(ns)
        if rc:
            return rc
        site = load_site(ns.site)
        rc = setup_mod.setup_checkpoints(site, sorted(registry.MODELS), dry_run=ns.dry_run, force=ns.force,
                                         af3_from=getattr(ns, "af3_from", None))
        if rc:
            return rc
        ns.check = True
        return cmd_env(ns)
    return _setup_venv(ns)


def _setup_venv(ns: argparse.Namespace) -> int:
    """Build the runtime venv at core/.venv with uv (locally or as a Slurm job)."""
    try:
        site = load_site(ns.site)
        root = site.core
        partition = str((site.slurm or {}).get("partition", "normal"))
    except SiteError:
        root = RESIDUDES_ENGINE_ROOT / "core"
        partition = "normal"
    if not (root / "pyproject.toml").is_file():
        raise SystemExit(f"no pyproject.toml under {root} — incomplete checkout?")
    cache = Path(os.environ.get("RESIDUDES_ENGINE_UV_CACHE_DIR") or RESIDUDES_ENGINE_ROOT / "runs" / "_setup" / "uv-cache")
    logs = RESIDUDES_ENGINE_ROOT / "runs" / "_setup"
    logs.mkdir(parents=True, exist_ok=True)
    if ns.slurm:
        cmd = ["sbatch", "-p", partition,
               f"--export=ALL,RESIDUDES_ENGINE_ROOT={RESIDUDES_ENGINE_ROOT},RESIDUDES_ENGINE_UV_CACHE_DIR={cache}",
               "-o", str(logs / "venv.%j.out"), "-e", str(logs / "venv.%j.err"),
               str(RESIDUDES_ENGINE_ROOT / "scripts" / "slurm" / "setup_venv.sbatch"), ns.cuda]
        print("▶", " ".join(shlex.quote(c) for c in cmd))
        if ns.dry_run:
            return 0
        res = subprocess.run(cmd, text=True)
        print(f"▶ logs in {logs}/venv.<jobid>.out ; afterwards: residudes-engine doctor")
        return res.returncode
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(cache)
    env["DS_BUILD_OPS"] = "0"
    env.pop("VIRTUAL_ENV", None)
    cache.mkdir(parents=True, exist_ok=True)
    # cu13 also pulls the in-venv JAX stack (extra `jax-cu13`): the JAX backend modules
    # import jax at module import, and --backend jax runs inside this venv unless venv_jax is set.
    extras = ["--extra", ns.cuda, "--extra", "boltz", "--extra", "openfold3"]
    if ns.cuda == "cu13":
        extras += ["--extra", "jax-cu13"]
    for attempt in (["uv", "sync", "--frozen", *extras], ["uv", "sync", *extras]):
        print("▶", " ".join(attempt), f"(cwd {root}, UV_CACHE_DIR {cache})")
        if ns.dry_run:
            return 0
        rc = subprocess.run(attempt, cwd=str(root), env=env).returncode
        if rc == 0:
            _ensure_jax_cuda_plugin(root / ".venv", env)
            print(f"▶ venv ready: {root / '.venv'}  → next: `residudes-engine doctor`")
            return 0
        print("▶ attempt failed; retrying without --frozen" if "--frozen" in attempt else "▶ uv sync failed")
    return 1


def cmd_engines(ns: argparse.Namespace) -> int:
    site = load_site(ns.site)
    gpu = detect_gpu()
    if ns.action == "list":
        print(f"engines root: {site.engines}   gpu here: {gpu.tag if gpu else 'none'}")
        print(f"  {'gpu':11} {'model':5} {'max_seq_len':>11}  dir")
        for g, tag, d, mx in setup_mod.list_engines(site, gpu):
            print(f"  {g:11} {tag:5} {str(mx or '-'):>11}  {d or '— (build: residudes-engine engines build --model ' + tag + ')'}")
        return 0
    if ns.action == "build":
        spec = registry.get_model(ns.model)
        if ns.slurm:
            args = ["engines", "build", "--model", spec.tag, "--kernel", ns.kernel,
                    "--max-seqlen", str(ns.max_seqlen), "--min-seqlen", str(ns.min_seqlen), *ns.extra]
            constraint = (site.slurm.get("constraint") or {}).get(ns.gpu) if ns.gpu else None
            return setup_mod.slurm_wrap(site, args, name=f"fmx-build-{spec.tag}-{ns.kernel}-{ns.max_seqlen}",
                                        log_dir=RESIDUDES_ENGINE_ROOT / "runs" / "_setup", time_limit=ns.time,
                                        gpu_constraint=constraint, dry_run=ns.dry_run)
        if gpu is None:
            raise SystemExit("no GPU on this machine; use --slurm or run on a GPU node")
        cmd, env, out = setup_mod.engine_build_plan(site, spec, gpu, kernel=ns.kernel, max_seqlen=ns.max_seqlen,
                                                    min_seqlen=ns.min_seqlen, extra=list(ns.extra))
        print(f"▶ building {spec.tag} {ns.kernel} max_seq_len={ns.max_seqlen} on {gpu.name} → {out}")
        print("▶", " ".join(shlex.quote(c) for c in cmd))
        if ns.dry_run:
            return 0
        out.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(cmd, env=env, cwd=str(site.core)).returncode
        if rc == 0:
            mx = registry.engine_max_seq_len(out)
            print(f"▶ done: {out} (max_seq_len {mx}); `residudes-engine predict --backend auto` will pick it up")
        return rc
    if ns.action == "pull":
        print("Pre-built engines are published per GPU class and TensorRT version under the\n"
              "DeepFoldProtein organisation on Hugging Face (repos *-trt / *-cuequiv-trt).\n"
              "Download the directory matching your GPU and TensorRT version and place it at\n"
              f"  {site.engines}/<gpu>/<model>_<kernel>_<maxN>/   (see docs/sites.md)\n"
              "Automatic pull is not implemented yet; `residudes-engine engines build` builds locally instead.")
        return 0
    return 1


# ── entry ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="residudes-engine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"residudes-engine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("predict", help="run one prediction on this machine")
    _add_predict_options(sp, seeds=False)
    sp.set_defaults(fn=cmd_predict)

    sp = sub.add_parser("submit", help="submit prediction job(s) to Slurm (one per seed)")
    _add_predict_options(sp, seeds=True)
    sp.add_argument("--gpu", default=None, help="GPU tag → Slurm constraint from the site file (e.g. rtx4090)")
    sp.add_argument("--time", default=None, help="Slurm time limit (default from site file)")
    sp.add_argument("--after", default=None, metavar="JOBID",
                    help="start only after this Slurm job has succeeded (e.g. an engine build)")
    sp.set_defaults(fn=cmd_submit)

    sp = sub.add_parser("verify", help="check seed dir(s) against the unified sample layout")
    sp.add_argument("seed_dirs", nargs="+", type=Path)
    sp.add_argument("--ref", type=Path, default=None, help="reference seed dir to compare against")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("env", help="show the resolved site and check the runtime")
    sp.add_argument("--site", default=None)
    sp.add_argument("--check", action="store_true", help="import torch/tensorrt/plugins inside the site venv")
    sp.set_defaults(fn=cmd_env)

    sp = sub.add_parser("doctor", help="full environment check (= env --check)")
    sp.add_argument("--deep", action="store_true",
                    help="also run one jackhmmer iteration against the smallest database (minutes)")
    sp.add_argument("--site", default=None)
    sp.set_defaults(fn=cmd_env, check=True)

    sp = sub.add_parser("setup", help="one-time machine setup: venv, model weights, site file")
    sp.add_argument("what", choices=("all", "venv", "checkpoints", "site"),
                    help="all = venv + checkpoints + doctor")
    sp.add_argument("--cuda", default="cu13", choices=("cu13", "cu12"), help="CUDA flavour of the venv (default cu13)")
    sp.add_argument("--slurm", action="store_true", help="venv: build as a Slurm job (scripts/slurm/setup_venv.sbatch)")
    sp.add_argument("--models", default=None, help="checkpoints: comma list of model tags (default all)")
    sp.add_argument("--af3-from", default=None, metavar="FILE|DIR|URL",
                    help="checkpoints: install AlphaFold 3 parameters you have been granted "
                         "(they are per-user and may not be redistributed, so nothing can fetch them for you)")
    sp.add_argument("--name", default="local", help="site: file name sites/<name>.toml (default local)")
    sp.add_argument("--checkpoints", default=None, help="site: directory for model weights")
    sp.add_argument("--engines", default=None, help="site: directory for TRT engines")
    sp.add_argument("--scheduler", default="none", choices=("none", "slurm", "pueue"), help="site: scheduler")
    sp.add_argument("--partition", default=None, help="site: Slurm partition")
    sp.add_argument("--force", action="store_true", help="overwrite existing site file / re-download weights")
    sp.add_argument("--site", default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=cmd_setup)

    sp = sub.add_parser("engines", help="TensorRT engines: list what exists, build for this GPU")
    sp.add_argument("action", choices=("list", "build", "pull"))
    sp.add_argument("--model", "-m", default=None, choices=sorted(registry.MODELS))
    sp.add_argument("--kernel", default="vanilla", choices=registry.KERNELS,
                    help="triangle-attention kernel baked into the engine (default vanilla)")
    sp.add_argument("--max-seqlen", type=int, default=1024, help="build-time token ceiling (default 1024)")
    sp.add_argument("--min-seqlen", type=int, default=4)
    sp.add_argument("--gpu", default=None, help="with --slurm: GPU tag → constraint from the site file")
    sp.add_argument("--slurm", action="store_true", help="build as a Slurm GPU job")
    sp.add_argument("--time", default=None, help="Slurm time limit for --slurm")
    sp.add_argument("--site", default=None)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("extra", nargs="*", help="extra flags for the build script (after --), e.g. --tri-chunk 16")
    sp.set_defaults(fn=cmd_engines)

    from . import cli_inputs
    cli_inputs.add_parsers(sub)          # msa · inputs · prepare

    sp = sub.add_parser("run", help="run a script inside the runtime venv with the prediction environment")
    sp.add_argument("--site", default=None)
    sp.add_argument("--model", "-m", default="pt2", choices=sorted(registry.MODELS),
                    help="model whose venv/JAX settings to use (default pt2)")
    sp.add_argument("--jax", action="store_true", help="use the JAX venv and JAX environment knobs")
    sp.add_argument("args", nargs=argparse.REMAINDER, help="-- script.py [args…]")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("models", help="list model tags")
    sp.set_defaults(fn=cmd_models)
    sp = sub.add_parser("sites", help="list site files")
    sp.set_defaults(fn=cmd_sites)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        return ns.fn(ns)
    except SiteError as e:
        print(f"residudes-engine: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
