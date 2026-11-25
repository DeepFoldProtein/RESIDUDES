"""`residudes relax`: the campaign's GROMACS + Rosetta relaxation, wrapped.

No cluster here — the pipeline is stubbed and the outputs it would write are made by hand, so what
these tests check is the command line, the staging, and the arithmetic of before and after.
"""
import csv
import json

import pytest

from residudes import ops, relax
from residudes.config import Config
from residudes.dossier import Dossier

# two residues, one clash-free backbone; the "relaxed" copy moves one CA by 0.3 A
PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.332   1.550   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       3.970   2.850   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.480   2.700   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.000   1.590   0.000  1.00  0.00           O
TER
END
"""


def _target(tmp_path, key="A1__af3_msa__af3__s1_0"):
    cfg = Config(work_root=str(tmp_path / "work"))
    d = Dossier(target="T1")
    d.sequence = {"text": ">A\nAA\n"}
    d.stoichiometry = {"given": "A1"}
    ddir = Dossier.path(cfg.targets_dir, "T1").parent
    ddir.mkdir(parents=True)
    src = tmp_path / "model.pdb"
    src.write_text(PDB)
    with open(ddir / "pool.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["sto", "variant", "runner", "seed", "sample", "path"])
        w.writeheader()
        w.writerow({"sto": "A1", "variant": "af3_msa", "runner": "af3", "seed": "1", "sample": "0", "path": str(src)})
    d.evidence["slate_proposal"] = [{"time": "2026-11-25T00:00:00+00:00", "by": "agent",
                                     "value": {"version": 1, "models": [key]}}]
    return cfg, d, ddir, src


def test_relaxation_runs_the_campaign_pipeline_under_restraints(tmp_path, monkeypatch):
    sent = {}
    monkeypatch.setattr(ops, "_sbatch", lambda cmd, name, log, gpu, hours, after=None, cpus=8: sent.update(
        cmd=cmd, name=name, gpu=gpu, hours=hours) or "900")
    monkeypatch.setattr(relax, "script", lambda cfg: "/tools/minimization.sh")
    cfg, d, ddir, _src = _target(tmp_path)

    out = relax.submit_cmd(cfg, d, "agent", sc_fc=1000)
    assert sent["cmd"].startswith("/tools/minimization.sh ")
    assert "--ssbond auto" in sent["cmd"]              # disulfides found by distance unless a file is given
    assert "--sc-fc 1000" in sent["cmd"] and "--bb-fc" not in sent["cmd"]   # only what was asked for
    assert sent["gpu"] is False                        # minimisation, not MD
    assert (ddir / "relax" / "1_A1__af3_msa__af3__s1_0" / "model.pdb").is_file()   # staged, original untouched
    assert d.runs[-1]["job"] == "900" and d.runs[-1]["what"].startswith("relax ")
    assert "--read" in out


def test_it_refuses_before_there_is_a_slate(tmp_path, monkeypatch):
    monkeypatch.setattr(relax, "script", lambda cfg: "/tools/minimization.sh")
    cfg, d, _ddir, _src = _target(tmp_path)
    d.evidence.pop("slate_proposal")
    with pytest.raises(ValueError, match="no slate proposal"):
        relax.submit_cmd(cfg, d, "agent")


def test_reading_measures_before_and_after_and_how_far_it_moved(tmp_path, monkeypatch):
    monkeypatch.setattr(relax, "script", lambda cfg: "/tools/minimization.sh")
    monkeypatch.setattr(ops, "_sbatch", lambda *a, **k: "901")
    cfg, d, ddir, _src = _target(tmp_path)
    relax.submit_cmd(cfg, d, "agent")

    wd = ddir / "relax" / "1_A1__af3_msa__af3__s1_0"
    for v in ("gromacs", "gromacs-rosetta"):
        vd = wd / "model_relaxed" / v
        vd.mkdir(parents=True)
        (vd / "relaxed.pdb").write_text(PDB.replace("3.970   2.850", "4.270   2.850"))   # one CA moved 0.3 A
    (wd / "model_relaxed" / "gromacs" / "relaxed.json").write_text(json.dumps({"mpscore": 1.23}))

    out = relax.read_cmd(cfg, d, "agent")
    got = d.evidence["relax"][-1]["value"]["A1__af3_msa__af3__s1_0"]
    assert set(got["after"]) == {"gromacs", "gromacs-rosetta"}          # the two that finished, not the four asked for
    assert got["after"]["gromacs"]["pipeline"] == {"mpscore": 1.23}     # whatever the pipeline recorded is kept
    moved = got["after"]["gromacs"]["ca_rmsd"]
    assert moved == pytest.approx(0.212, abs=0.01)                      # 0.3 A on one of two CAs
    assert "gromacs-rosetta" in out and "0.212 A" in out
    assert "never ranking" in out or "ranking" in out


def test_an_unfinished_run_says_so_instead_of_guessing(tmp_path, monkeypatch):
    monkeypatch.setattr(relax, "script", lambda cfg: "/tools/minimization.sh")
    monkeypatch.setattr(ops, "_sbatch", lambda *a, **k: "902")
    cfg, d, _ddir, _src = _target(tmp_path)
    relax.submit_cmd(cfg, d, "agent")
    out = relax.read_cmd(cfg, d, "agent")
    assert "(not finished)" in out
    assert d.evidence["relax"][-1]["value"]["A1__af3_msa__af3__s1_0"]["after"] == {}
