"""The two renderers: molterm for every deck while the run goes on, PyMOL ray tracing for the final one.

The case behind both halves is H1388's peptide figure: five predictors' peptides merged into one file, every atom
as ball-and-stick, viewed from behind the receptor, rasterised without anti-aliasing. These hold the fixes."""
import os
import subprocess as sp
import time

from residudes import ops, slides
from residudes.config import Config
from residudes.dossier import Dossier


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def test_face_rows_put_the_camera_outside_the_partner():
    """A peptide lying along x, 6 A above a receptor patch at z = 0: the camera must look down on it from +z (the
    side away from the receptor), with the peptide running across the picture."""
    on = [[float(i), 0.3 * (i % 2), 6.0] for i in range(11)]
    near = [[float(x), float(y), 0.0] for x in range(-2, 13, 3) for y in (-3, 0, 3)]
    x, y, z = slides.face_rows(on, near)
    assert z[2] > 0.95                                         # towards the viewer = away from the receptor (the patch is not centred under it)
    assert abs(x[0]) > 0.95                                    # the peptide's long axis is the picture's width
    for a, b in ((x, y), (y, z), (x, z)):
        assert abs(sum(i * j for i, j in zip(a, b))) < 1e-6    # orthonormal ...
    assert all(abs(c - d) < 1e-6 for c, d in zip(_cross(x, y), z))   # ... and right-handed, or molterm mirrors it
    assert slides.face_rows(on, []) is None                    # nothing to face


def _pdb(path, chains):
    lines, n = [], 0
    for ch, pts in chains.items():
        for i, (px, py, pz) in enumerate(pts, 1):
            n += 1
            lines.append(f"ATOM  {n:5d}  CA  ALA {ch}{i:4d}    {px:8.3f}{py:8.3f}{pz:8.3f}  1.00 90.00           C")
    path.write_text("\n".join(lines) + "\nEND\n")


def test_the_molterm_deck_faces_a_small_chain(tmp_path):
    m = tmp_path / "m.pdb"
    _pdb(m, {"A": [(float(i * 3.8), 0.0, 6.0) for i in range(8)],
             "B": [(float(x), float(y), 0.0) for x in range(-4, 30, 4) for y in (-4, 0, 4)]})
    cam = slides.face_camera(str(m), ("A",))
    assert cam.startswith("# molterm camera v1") and "rot = " in cam
    z = [float(v) for v in cam.split("rot = ")[1].split()][6:9]
    assert z[2] > 0.95
    assert slides.face_camera(str(m), ()) is None and slides.face_camera(str(tmp_path / "none.cif"), ("A",)) is None
    mt = slides.scenes_mt([{"label": "M1", "path": str(m), "sto": "A1B1"}], ("A",), "view.cam")
    assert "camera load view.cam\nzoom M1" in mt and "orient" not in mt


def test_the_final_scenes_are_ray_traced_and_face_the_peptide():
    models = [{"label": f"M{i}", "path": f"/x/m{i}.cif", "sto": "A1B1"} for i in (1, 2)]
    pml = slides.scenes_pml(models, ("A",))
    for s in ("set ray_shadows, 1", "set antialias, 2", "set ambient_occlusion_mode, 1", "python end",
              "face M1 and chain A, M1 and not chain A, M1, 0",
              "png figs/M1_chains.png, width=1800, height=1350, dpi=300, ray=1",
              "png figs/overlay.png, width=2400, height=1700, dpi=300, ray=1", "set cartoon_transparency, 0, (M2)"):
        assert s in pml, s
    assert "orient" not in pml.split("python end")[1]          # the view is face's, not the peptide's own axes
    assert "orient M1" in slides.scenes_pml(models)            # nothing small: the old view


def test_a_stale_jpeg_does_not_stand_in_for_a_newer_picture(tmp_path):
    jpg, png = tmp_path / "M1_chains.jpg", tmp_path / "M1_chains.png"
    jpg.write_bytes(b"old")
    png.write_bytes(b"new")
    t = time.time()
    os.utime(jpg, (t - 100, t - 100))
    assert slides._pic(tmp_path, "M1_chains") == "figs/M1_chains.png"
    os.utime(jpg, (t + 1, t + 1))
    assert slides._pic(tmp_path, "M1_chains") == "figs/M1_chains.jpg"


def _target(tmp_path):
    import csv
    cfg = Config(work_root=str(tmp_path))
    d = Dossier(target="T")
    d.save(cfg.targets_dir)
    ddir = cfg.targets_dir / "T"
    with open(ddir / "pool.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm",
                                           "plddt", "has_clash", "path", "rs_z_runner"])
        w.writeheader()
        w.writerow({"sto": "A1B1", "variant": "v", "runner": "af3", "seed": 1, "sample": 0, "ranking_score": 0.8,
                    "iptm": 0.8, "ptm": 0.8, "plddt": 90, "has_clash": False, "path": "/x/m.cif", "rs_z_runner": 0})
    (ddir / "report" / "pml").mkdir(parents=True)
    (ddir / "report" / "pml" / "pep_split.pml").write_text("load /x/m.cif\npng figs/pep_split.png, width=1800, height=1350, dpi=300, ray=1\n")
    return cfg, d, ddir / "report"


def test_final_ray_traces_the_deck_and_the_agents_scenes_and_waits(tmp_path, monkeypatch):
    cfg, d, rep = _target(tmp_path)
    tools = {"pymol": "/opt/pymol/bin/pymol", "molterm": "/usr/bin/molterm"}
    monkeypatch.setattr(slides, "tool", lambda cfg, name: tools.get(name))
    monkeypatch.setattr(slides.shutil, "which", lambda n: "/usr/bin/sbatch" if n == "sbatch" else None)
    monkeypatch.setattr(slides, "render_mt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("molterm in a final build")))
    seen = {}

    def fake_sbatch(cmd, name, log, gpu, hours, after=None, cpus=8, wait=False):
        seen.update(cmd=cmd, wait=wait)
        for f in ("A1B1_chains", "A1B1_plddt", "pep_split"):
            (rep / "figs" / f"{f}.png").write_bytes(b"png")
        return "4242"
    monkeypatch.setattr(ops, "_sbatch", fake_sbatch)
    out = slides.slides_cmd(cfg, d, "t", final=True)
    assert seen["wait"] is True
    assert "-cq scenes.pml" in seen["cmd"] and "-cq pml/pep_split.pml" in seen["cmd"]
    assert "3 ray-traced figures" in out and (rep / "scenes.pml").exists()
    assert d.runs[-1]["job"] == "4242"

    (rep / "figs" / "pep_split.png").unlink()                  # a scene that wrote nothing is reported, not passed
    monkeypatch.setattr(ops, "_sbatch", lambda *a, **k: "4243")
    assert "were not written" in slides.slides_cmd(cfg, d, "t", final=True)


def test_final_without_pymol_says_the_deck_is_not_ray_traced(tmp_path, monkeypatch):
    cfg, d, rep = _target(tmp_path)
    monkeypatch.setattr(slides, "tool", lambda cfg, name: "/usr/bin/molterm" if name == "molterm" else None)
    monkeypatch.setattr(slides, "render_mt", lambda *a, **k: "molterm drew it")
    out = slides.slides_cmd(cfg, d, "t", final=True)
    assert "not ray-traced" in out and "molterm drew it" in out


def test_sbatch_wait_blocks_and_reports_a_failed_job(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ops.shutil, "which", lambda n: "/usr/bin/sbatch")
    monkeypatch.setattr(sp, "run", lambda args, **k: calls.append((args, k)) or sp.CompletedProcess(args, 1, "77\n", "exited 1"))
    try:
        ops._sbatch("true", "n", tmp_path / "r.%j.log", gpu=False, hours=1, wait=True)
    except ValueError as exc:
        assert "job 77" in str(exc) and "r.77.log" in str(exc)
    else:
        raise AssertionError("a failed job passed")
    assert "--wait" in calls[0][0] and calls[0][1]["timeout"]


def test_residudes_jobs_take_their_slurm_flags_from_the_config():
    """No site's partition name is baked in; a site names its own, its GPU type and anything else."""
    from residudes import ops
    assert ops.slurm_flags(False, {}) == [] and ops.slurm_flags(True, {}) == ["--gres=gpu:1"]
    flags = ops.slurm_flags(True, {"partition": "normal", "gres": "gpu:RTX4090:1", "extra": ["--exclude=ada"]})
    assert flags == ["-p", "normal", "--gres=gpu:RTX4090:1", "--exclude=ada"]


def test_a_model_shown_twice_is_copied_not_loaded_again():
    """molterm loads a path once; a cluster medoid that is also a slate model rendered blank."""
    from residudes import slides
    lines = slides.load_lines([{"path": "/r/a/model.cif", "label": "M4"}, {"path": "/r/b/model.cif", "label": "M5"},
                               {"path": "/r/a/model.cif", "label": "C24"}])
    assert lines == ["load /r/a/model.cif", "rename model M4", "load /r/b/model.cif", "rename model M5", "copy M4 as C24"]


def test_a_job_whose_dependency_failed_is_blocked_not_active(monkeypatch):
    """Slurm keeps a job queued after its afterok dependency failed; counted as pending, the target waited for good.
    It is left queued (the agent may re-point it) and asked about again, so a re-pointed run is active once more."""
    import subprocess as sp
    from residudes import ops
    calls, queue = [], {"out": "350909 PENDING DependencyNeverSatisfied\n"}
    def fake(args, **k):
        calls.append(args)
        return sp.CompletedProcess(args, 0, queue["out"] if args[0] == "squeue" else "", "")
    monkeypatch.setattr(ops.subprocess, "run", fake)
    assert ops._slurm_state("350909") == "BLOCKED" and not ops.active({"state": "BLOCKED"})
    assert not any(c[0] == "scancel" for c in calls)
    queue["out"] = "350909 PENDING Dependency\n"                      # re-pointed at a live job
    assert ops._slurm_state("350909") == "PENDING"


def test_a_run_of_several_jobs_is_tracked_as_a_whole(monkeypatch):
    """An agent recorded 20 seeds as one run, the ids one per line: every id is asked about."""
    import subprocess as sp
    from residudes import ops
    calls, queue = [], {"out": "1 PENDING DependencyNeverSatisfied\n2 PENDING DependencyNeverSatisfied\n"}
    def fake(args, **k):
        calls.append(args)
        out = queue["out"] if args[0] == "squeue" else ("COMPLETED\nFAILED\n" if args[0] == "sacct" else "")
        return sp.CompletedProcess(args, 0, out, "")
    monkeypatch.setattr(ops.subprocess, "run", fake)
    assert ops._slurm_state("1\n2\n3") == "BLOCKED"
    assert calls[0][:4] == ["squeue", "-h", "-j", "1,2,3"]
    queue["out"] = "3 RUNNING None\n"
    assert ops._slurm_state("1,2,3") == "RUNNING"                 # one still running: the run is active
    queue["out"] = ""
    assert ops._slurm_state("1 2") == "FAILED"                    # all finished: the first that did not complete


def test_relax_without_a_pipeline_says_so_instead_of_submitting(monkeypatch):
    """Path("") exists (it is the current directory): an unset [tools] minimization submitted the model path as a command."""
    import pytest
    from residudes import relax
    cfg = type("C", (), {"extra": {"tools": {"minimization": ""}}})()
    monkeypatch.setattr(relax, "MINIMIZATION", "")
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="no relaxation pipeline configured"):
        relax.script(cfg)
