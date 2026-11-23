"""GPU-free checks: every family × backend plans the expected command line,
native layouts flatten correctly, and the verifier reads the unified layout.

    cd lantern-engine && python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lantern_engine import registry  # noqa: E402
from lantern_engine.config import GPU, load_site  # noqa: E402
from lantern_engine.runners import for_family  # noqa: E402
from lantern_engine.runners._common import Request, flatten_seed_dirs  # noqa: E402
from lantern_engine.verify import check_seed_dir  # noqa: E402

AF3_INPUT = {
    "name": "T0000_A1", "dialect": "alphafold3", "version": 1, "modelSeeds": [7],
    "sequences": [{"protein": {"id": "A", "sequence": "MKVLAAGIV"}}],
}


class FakeSite:
    """A throwaway site: empty venv skeleton, empty checkpoints, fake engines."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.root = root
        (root / "venv" / "bin").mkdir(parents=True)
        (root / "venv" / "bin" / "python").touch()
        (root / "venv" / "lib" / "python3.11" / "site-packages" / "tensorrt_libs").mkdir(parents=True)
        (root / "ckpt" / "boltz2").mkdir(parents=True)
        for m in registry.MODELS.values():
            (root / "ckpt" / m.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            (root / "ckpt" / m.checkpoint).touch()
            if "rtx4090" not in m.engines:        # AlphaFold 3 has no TensorRT engines
                continue
            e = root / "engines" / m.engines["rtx4090"] / "pairformer_engine" / "trt"
            e.mkdir(parents=True)
            (e / "rank0.engine").touch()
            (e / "config.json").write_text(json.dumps({"pretrained_config": {"max_seq_len": 1024}}))
        core = ROOT / "core"
        (root / "site.toml").write_text(
            "[site]\n"
            'name = "fake"\nscheduler = "slurm"\n'
            f'core = "{core}"\nvenv = "{root / "venv"}"\nvenv_jax = "{root / "venv"}"\n'
            f'checkpoints = "{root / "ckpt"}"\nengines = "{root / "engines"}"\ncache_dir = "{root / "cache"}"\n'
            '[slurm]\npartition = "p"\n'
            '[jax]\nvenv = { pt2 = "venv_jax" }\n[jax.env]\nLANTERN_ENGINE_JAX_CUEQ = "0"\n'
        )
        self.input = root / "input.json"
        self.input.write_text(json.dumps(AF3_INPUT))
        self.site = load_site(str(root / "site.toml"))
        self.gpu = GPU(compute_cap="8.9", name="fake 4090", mem_mib=24000, tag="rtx4090")

    def request(self, tag: str, backend: str, **kw) -> Request:
        out = self.root / "out" / f"{tag}-{backend}"
        out.mkdir(parents=True, exist_ok=True)
        return Request(model=registry.get_model(tag), backend=backend, input=self.input, out=out,
                       site=self.site, gpu=self.gpu, **kw)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fs = FakeSite()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fs.tmp.cleanup()

    def test_every_family_and_backend_plans(self) -> None:
        for tag, spec in registry.MODELS.items():
            for backend in registry.BACKENDS:
                if backend not in spec.backends or (backend == "trt" and "rtx4090" not in spec.engines):
                    continue
                with self.subTest(model=tag, backend=backend):
                    req = self.fs.request(tag, backend, seed=3)
                    plan = for_family(spec.family).plan(req)
                    self.assertTrue(plan.cmd[0].endswith("venv/bin/python"))
                    self.assertTrue(Path(plan.cmd[1]).is_file(), f"runner script missing: {plan.cmd[1]}")
                    staged = json.loads(plan.staged_input.read_text())
                    self.assertEqual(staged["modelSeeds"], [3])
                    joined = " ".join(plan.cmd)
                    if backend == "trt":
                        self.assertIn("engines", joined)
                        self.assertIn(spec.engines["rtx4090"], joined)
                    else:
                        self.assertNotIn("engines", joined)
                    if backend == "torch" and spec.family != "alphafold3":
                        self.assertTrue("--no_trt" in plan.cmd or "--no-trt" in plan.cmd)
                    if backend == "jax":
                        if spec.family != "alphafold3":
                            self.assertIn("--jax_dtype", plan.cmd)
                        self.assertIn("XLA_PYTHON_CLIENT_PREALLOCATE", plan.env)
                    self.assertIn("tensorrt_libs", plan.env["LD_LIBRARY_PATH"])
                    self.assertEqual(plan.env["LAYERNORM_TYPE"], "torch_layernorm")
                    self.assertNotIn(str(ROOT / "src"), plan.env["PYTHONPATH"].split(":"))
                    for p in plan.cleanup:
                        if p.is_dir():
                            p.rmdir()

    def test_family_specifics(self) -> None:
        pt2 = for_family("protenix").plan(self.fs.request("pt2", "trt", seed=1, num_recycles=4, template_weight=2.0))
        self.assertIn("--model.N_cycle", pt2.cmd)
        self.assertEqual(pt2.cmd[pt2.cmd.index("--model.N_cycle") + 1], "4")
        self.assertEqual(pt2.env["PT2_TPL_WEIGHT"], "2.0")
        self.assertTrue(pt2.env["PYTHONPATH"].startswith(str(self.fs.site.core / "models" / "protenix")))

        of3 = for_family("openfold3").plan(self.fs.request("of3", "trt", seed=1, pairing_mode="online"))
        self.assertEqual(of3.env["OF3_PAIRING_MODE"], "online")
        self.assertEqual(of3.env["DS_BUILD_OPS"], "0")
        self.assertEqual(of3.env["OF3_PER_SAMPLE_TOKEN_CUTOFF"], "0")  # rtx4090 rule
        self.assertEqual(json.loads(of3.staged_input.read_text())["seeds"], [1])
        for p in of3.cleanup:
            p.rmdir()

        bt2 = for_family("boltz").plan(self.fs.request("bt2", "trt", seed=5, diffusion_samples=2))
        self.assertEqual(bt2.cmd[bt2.cmd.index("--seed") + 1], "5")
        self.assertEqual(bt2.cmd[bt2.cmd.index("--diffusion_samples") + 1], "2")
        self.assertTrue(bt2.env["BOLTZ_CACHE"].endswith("boltz2"))
        self.assertEqual(bt2.env["BT2_SEED"], "5")   # writer reads the seed from env, not --seed
        # seed falls back to the input's single modelSeeds entry
        bt2b = for_family("boltz").plan(self.fs.request("bt2", "torch"))
        self.assertEqual(bt2b.cmd[bt2b.cmd.index("--seed") + 1], "7")

    def test_alphafold3_specifics(self) -> None:
        from lantern_engine.cli import _resolve_backend
        af3 = for_family("alphafold3").plan(self.fs.request("af3", "jax", seed=2, num_recycles=3, template_weight=1.5))
        self.assertIn("--run_data_pipeline=false", af3.cmd)
        self.assertEqual(af3.cmd[af3.cmd.index("--num_recycles") + 1], "3")
        self.assertTrue(af3.cmd[af3.cmd.index("--model_dir") + 1].endswith("ckpt/af3"))
        self.assertEqual(af3.env["XLA_PYTHON_CLIENT_PREALLOCATE"], "true")
        self.assertEqual(af3.env["AF3_TPL_WEIGHT"], "1.5")
        self.assertIn("--xla_gpu_enable_triton_gemm=false", af3.env["XLA_FLAGS"])
        self.assertEqual(json.loads(af3.staged_input.read_text())["modelSeeds"], [2])
        tor = for_family("alphafold3").plan(self.fs.request("af3", "torch", seed=2))
        self.assertTrue(tor.cmd[1].endswith("core/alphafold3/inference.py"))
        self.assertEqual(tor.env["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
        self.assertEqual(tor.env["XLA_CLIENT_MEM_FRACTION"], "0.02")
        self.assertNotIn("XLA_PYTHON_CLIENT_MEM_FRACTION", tor.env)      # JAX 0.11 refuses the deprecated name beside the new one
        self.assertNotIn("CUDA_ROOT", tor.env)                            # no pip CUDA toolkit in the fake venv
        ptxas = self.fs.root / "venv" / "lib" / "python3.11" / "site-packages" / "nvidia" / "cu13" / "bin" / "ptxas"
        ptxas.parent.mkdir(parents=True, exist_ok=True); ptxas.touch()
        jx = for_family("alphafold3").plan(self.fs.request("af3", "jax", seed=1))
        self.assertEqual(jx.env["CUDA_ROOT"], str(ptxas.parents[1]))      # XLA's ptxas comes from the venv, not the node
        self.assertEqual(jx.env["PATH"].split(":")[0], str(ptxas.parent))  # and so does the one jax-triton finds on PATH
        self.assertIn(str(self.fs.site.core / "thirdparty"), tor.env["PYTHONPATH"].split(":"))
        spec = registry.get_model("af3")
        eng = self.fs.root / "engines" / "rtx4090" / "alphafold3_vanilla_1024"
        (eng / "pairformer_engine" / "trt").mkdir(parents=True, exist_ok=True)
        (eng / "pairformer_engine" / "trt" / "rank0.engine").touch()
        trt = for_family("alphafold3").plan(self.fs.request("af3", "trt", seed=2, engines_dir=eng))
        self.assertEqual(trt.cmd[trt.cmd.index("--engines_dir") + 1], str(eng))
        self.assertEqual(_resolve_backend("auto", self.fs.site, spec, self.fs.gpu)[0], "trt")   # the fake engine set is found
        with self.assertRaises(Exception):
            for_family("alphafold3").plan(self.fs.request("af3", "trt", engines_dir=self.fs.root / "nowhere"))
        self.assertEqual(_resolve_backend("auto", self.fs.site, spec, self.fs.gpu)[0], "trt")   # engines present → TensorRT
        with self.assertRaises(SystemExit):
            _resolve_backend("nope", self.fs.site, spec, self.fs.gpu)

    def test_trt_without_gpu_is_a_clear_error(self) -> None:
        req = self.fs.request("pt2", "trt")
        req.gpu = None
        with self.assertRaises(Exception) as ctx:
            for_family("protenix").plan(req)
        self.assertIn("--engines-dir", str(ctx.exception))


class FlattenTests(unittest.TestCase):
    def _mk(self, root: Path, rel: str) -> Path:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        return p

    def test_single_seed_lands_in_out(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            self._mk(out, "T0000_A1/seed_1/sample_0/model.cif")
            self._mk(out, "T0000_A1/seed_1/timing.json")
            moved = flatten_seed_dirs(out, [out / "T0000_A1" / "seed_1"])
            self.assertEqual(moved, [out])
            self.assertTrue((out / "sample_0" / "model.cif").is_file())
            self.assertTrue((out / "timing.json").is_file())
            self.assertFalse((out / "T0000_A1").exists())

    def test_multi_seed_keeps_seed_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            for s in (1, 2):
                self._mk(out, f"q/seed_{s}/sample_0/model.cif")
            moved = flatten_seed_dirs(out, sorted(out.glob("q/seed_*")))
            self.assertEqual(sorted(p.name for p in moved), ["seed_1", "seed_2"])
            self.assertTrue((out / "seed_2" / "sample_0" / "model.cif").is_file())
            self.assertFalse((out / "q").exists())


class VerifyTests(unittest.TestCase):
    def test_reads_unified_layout(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            seed = Path(t)
            (seed / "timing.json").write_text("{}")
            (seed / "distogram_logits.npz").write_bytes(b"0")
            for i, score in enumerate((0.3, 0.9)):
                s = seed / f"sample_{i}"
                (s / "arrays").mkdir(parents=True)
                (s / "model.cif").write_text("data_x\nATOM 1 N N . MET A 1 1 ? 0 0 0 1 90 1 A 1\nATOM 2 C CA . MET A 1 1 ? 1 0 0 1 90 1 A 1\n")
                (s / "summary.json").write_text(json.dumps({
                    "model_name": "pt2", "seed": 1, "sample_idx": i, "n_tokens": 1, "n_atoms": 2,
                    "plddt": 80 + i, "ptm": 0.5, "ranking_score": score}))
                (s / "meta.json").write_text("{}")
                for a in ("plddt_atom.npy", "plddt_token.npy", "pae_logits.npz", "pde_logits.npz", "contact_probs.npy"):
                    (s / "arrays" / a).write_bytes(b"0")
            rep = check_seed_dir(seed)
            self.assertTrue(rep.ok, [s.missing for s in rep.samples])
            self.assertEqual(rep.top().name, "sample_1")
            self.assertEqual(rep.samples[0].atoms, 2)
            self.assertEqual(rep.samples[0].pae_form, "logits")
            (seed / "sample_0" / "arrays" / "pae_logits.npz").unlink()
            self.assertFalse(check_seed_dir(seed).ok)
            # bt2_jax / ef2 writers store reduced Å arrays instead of logits — also accepted
            (seed / "sample_0" / "arrays" / "pae.npy").write_bytes(b"0")
            rep2 = check_seed_dir(seed)
            self.assertTrue(rep2.ok, [s.missing for s in rep2.samples])
            self.assertEqual(rep2.samples[0].pae_form, "Å")


class CliTests(unittest.TestCase):
    def test_parser_covers_every_subcommand(self) -> None:
        from lantern_engine.cli import build_parser
        p = build_parser()
        for argv in (
            ["predict", "--model", "pt2", "--input", "x.json", "--out", "o", "--seed", "1"],
            ["submit", "--model", "of3", "--input", "x.json", "--out", "o", "--seeds", "1,3-4", "--gpu", "rtx4090"],
            ["verify", "a", "b", "--ref", "r"],
            ["env", "--check"],
            ["setup", "venv", "--cuda", "cu12", "--slurm", "--dry-run"],
            ["models"], ["sites"],
        ):
            ns = p.parse_args(argv)
            self.assertTrue(callable(ns.fn), argv)

    def test_seed_spec_expansion(self) -> None:
        from lantern_engine.cli import _expand_seeds
        self.assertEqual(_expand_seeds("1,5-7,10"), [1, 5, 6, 7, 10])
        with self.assertRaises(SystemExit):
            _expand_seeds("7-5")


class DistributionTests(unittest.TestCase):
    """auto backend, default output dir, engine discovery, site file writer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fs = FakeSite()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fs.tmp.cleanup()

    def test_backend_auto_picks_trt_when_engines_exist(self) -> None:
        from lantern_engine.cli import _resolve_backend
        pt2 = registry.get_model("pt2")
        self.assertEqual(_resolve_backend("auto", self.fs.site, pt2, self.fs.gpu)[0], "trt")
        self.assertEqual(_resolve_backend("auto", self.fs.site, pt2, None)[0], "torch")
        no_engines = GPU(compute_cap="10.0", name="fake b200", mem_mib=1, tag="b200")
        self.assertEqual(_resolve_backend("auto", self.fs.site, pt2, no_engines)[0], "torch")
        self.assertEqual(_resolve_backend("jax", self.fs.site, pt2, self.fs.gpu)[0], "jax")

    def test_find_engine_dir_falls_back_to_locally_built_dirs(self) -> None:
        pt2 = registry.get_model("pt2")
        built = pt2.engine_dir_for_build(self.fs.site.engines, "b200", "vanilla", 2048)
        (built / "pairformer_engine" / "trt").mkdir(parents=True)
        (built / "pairformer_engine" / "trt" / "rank0.engine").touch()
        (built / "pairformer_engine" / "trt" / "config.json").write_text('{"pretrained_config": {"max_seq_len": 2048}}')
        self.assertEqual(registry.find_engine_dir(self.fs.site.engines, "b200", pt2), built)
        self.assertIsNone(registry.find_engine_dir(self.fs.site.engines, "h100", pt2))
        # best fit: the smallest engine that covers the input; largest when none does or N unknown
        big = pt2.engine_dir_for_build(self.fs.site.engines, "b200", "vanilla", 4096)
        (big / "pairformer_engine" / "trt").mkdir(parents=True)
        (big / "pairformer_engine" / "trt" / "rank0.engine").touch()
        (big / "pairformer_engine" / "trt" / "config.json").write_text('{"pretrained_config": {"max_seq_len": 4096}}')
        find = lambda n: registry.find_engine_dir(self.fs.site.engines, "b200", pt2, n_tokens=n)  # noqa: E731
        self.assertEqual(find(1000), built)
        self.assertEqual(find(3000), big)
        self.assertEqual(find(5000), big)
        self.assertEqual(find(None), big)

    def test_default_out_uses_input_name_model_and_seed(self) -> None:
        from lantern_engine.cli import _default_out
        ns = argparse.Namespace(out=None, input=self.fs.input, seed=3)
        self.assertEqual(_default_out(ns, "pt2"), Path("engine_out") / "T0000_A1_pt2" / "seed_3")
        ns = argparse.Namespace(out=Path("x"), input=self.fs.input, seed=None)
        self.assertEqual(_default_out(ns, "pt2"), Path("x"))

    def test_write_site_fills_paths(self) -> None:
        from lantern_engine import setup as setup_mod
        import lantern_engine.config as cfg
        tmp = Path(self.fs.tmp.name) / "sites"
        tmp.mkdir()
        (tmp / "local.example.toml").write_text((cfg.SITES_DIR / "local.example.toml").read_text())
        old = setup_mod.SITES_DIR
        setup_mod.SITES_DIR = tmp
        try:
            out = setup_mod.write_site("t", checkpoints=Path("/c"), engines=Path("/e"), scheduler="slurm",
                                       partition="gpu", hosts=["h1"], force=False)
            txt = out.read_text()
            self.assertIn('checkpoints = "/c"', txt)
            self.assertIn('engines = "/e"', txt)
            self.assertIn('scheduler = "slurm"', txt)
            self.assertIn('hosts = ["h1"]', txt)
            self.assertIn('partition = "gpu"', txt)
            with self.assertRaises(SystemExit):
                setup_mod.write_site("t", checkpoints=Path("/c"), engines=Path("/e"), scheduler="none",
                                     partition=None, hosts=None, force=False)
        finally:
            setup_mod.SITES_DIR = old


if __name__ == "__main__":
    unittest.main()
