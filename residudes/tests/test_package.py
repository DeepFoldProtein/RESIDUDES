"""residudes package: geometry checks, the polymer-only copy, and the files (casp17-utils' prepare stood in for)."""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from residudes import package  # noqa: E402
from residudes.dossier import Dossier  # noqa: E402
from test_gate import _chain, _pdb  # noqa: E402


def test_checks_count_chains_lengths_breaks_and_clashes(tmp_path):
    good = _pdb(tmp_path / "good.pdb", {"A": _chain(0, 0), "B": _chain(0, 5.0)}, ligand=(7.6, -4.0, 0))
    r = package.check_model(str(good), "A2", ["AAAAAA"])
    assert r["chains"] == 2 and r["expected_chains"] == 2 and r["length_mismatch"] == [] and r["ok"]
    line = [(3.8 * i, 0.0, 0.0) for i in range(6)]           # straight: every C-N 1.4 A
    gap = [(x + (12.0 if i >= 3 else 0.0), y, z) for i, (x, y, z) in enumerate(line)]
    clean = package.check_model(str(_pdb(tmp_path / "line.pdb", {"A": line, "B": _chain(0, 5.0)})), "A2", ["AAAAAA"])
    broken = package.check_model(str(_pdb(tmp_path / "gap.pdb", {"A": gap, "B": _chain(0, 5.0)})), "A2", ["AAAAAA"])
    assert broken["breaks"] == clean["breaks"] + 1               # a 12 A gap between residues 3 and 4
    short = package.check_model(str(good), "A2", ["AAAAAAAA"])
    assert not short["ok"] and "A: 6 residues, sequence 8" in short["length_mismatch"]
    lost = _pdb(tmp_path / "lost.pdb", {"A": _chain(0, 0)})
    assert not package.check_model(str(lost), "A2", ["AAAAAA"])["ok"]
    clash = _pdb(tmp_path / "clash.pdb", {"A": _chain(0, 0), "B": _chain(0.3, 0.2)})
    assert package.check_model(str(clash), "A2", ["AAAAAA"])["clash_pairs"] > 0


def test_polymer_copy_drops_ligands(tmp_path):
    good = _pdb(tmp_path / "good.pdb", {"A": _chain(0, 0), "B": _chain(0, 5.0)}, ligand=(7.6, -4.0, 0))
    out, note = package.polymer_copy(str(good), tmp_path / "poly.cif")
    assert note == "pLDDT"
    st = gemmi.read_structure(str(out))
    assert [c.name for c in st[0]] == ["A", "B"] and all(r.name == "ALA" for c in st[0] for r in c)


def test_package_needs_approval_and_writes_files(tmp_path, monkeypatch):
    class Cfg:
        targets_dir = tmp_path / "targets"
        extra = {"tools": {"casp17_utils": str(tmp_path / "utils")}}
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "pyproject.toml").write_text("[project]\nname='x'\n")
    d = Dossier(target="T5")
    d.sequence["text"] = ">a\nAAAAAA\n"
    ddir = Dossier.path(Cfg.targets_dir, "T5").parent
    ddir.mkdir(parents=True)
    m = _pdb(tmp_path / "m.pdb", {"A": _chain(0, 0), "B": _chain(0, 5.0)})
    with open(ddir / "pool.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path", "rs_z_runner"])
        w.writerow(["A2", "msa", "af3", 1, 0, 0.9, 0.9, 0.9, 80, False, str(m), 1.0])
    d.evidence["slate_proposal"] = [{"time": "2026-11-22T01:00:00+09:00", "by": "agent", "value": {"version": 1, "models": ["A2__msa__af3__s1_0"]}}]
    d.stage = "SLATE_PROPOSED"
    with pytest.raises(ValueError):
        package.package_cmd(Cfg, d, "server", "DeepFold", None, False, False, "agent")      # not approved
    out = package.package_cmd(Cfg, d, None, "DeepFold", None, False, True, "agent")        # checks at any stage
    assert "| 1 | A2__msa__af3__s1_0 | yes | 2/2 |" in out and (ddir / "package" / "checks.md").exists()
    d.approve("a-person", "slate v1")
    d.move("APPROVED", "a-person")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("PFRMAT TS\n")
        return subprocess.CompletedProcess(cmd, 0, "Wrote", "")
    monkeypatch.setattr(package.subprocess, "run", fake_run)
    monkeypatch.setattr(package.shutil, "which", lambda x: "/usr/bin/uv")
    out = package.package_cmd(Cfg, d, "server", "DeepFold", "T1234", False, False, "agent")
    assert d.stage == "PACKAGED" and (ddir / "package" / "T1234_1.pdb").exists()
    c = calls[0]
    assert c[:5] == ["uv", "run", "--project", str(tmp_path / "utils"), "casp17"] and "prepare" in c and "--server" in c
    assert "submit" not in c and c[c.index("-i") + 1].endswith("polymer/model_1.cif")   # polymer-only input, never submit


def _setup(tmp_path, models_by_slot, sto="A2", seqs=">a\nAAAAAA\n", what="slate v1"):
    class Cfg:
        targets_dir = tmp_path / "targets"
        extra = {"tools": {"casp17_utils": str(tmp_path / "utils")}}
    (tmp_path / "utils").mkdir(parents=True, exist_ok=True)
    (tmp_path / "utils" / "pyproject.toml").write_text("[project]\nname='x'\n")
    d = Dossier(target="T6")
    d.sequence["text"] = seqs
    ddir = Dossier.path(Cfg.targets_dir, "T6").parent
    ddir.mkdir(parents=True, exist_ok=True)
    keys = []
    with open(ddir / "pool.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path", "rs_z_runner"])
        for i, path in enumerate(models_by_slot, start=1):
            w.writerow([sto, "msa", "af3", i, 0, 0.9, 0.9, 0.9, 80, False, str(path), 1.0])
            keys.append(f"{sto}__msa__af3__s{i}_0")
    d.evidence["slate_proposal"] = [{"time": "2026-11-22T01:00:00+09:00", "by": "agent", "value": {"version": 1, "models": keys}}]
    d.stage = "SLATE_PROPOSED"
    d.approve("a-person", what)
    if what == "slate v1":
        d.move("APPROVED", "a-person")
    return Cfg, d, ddir


def _fake_prepare(monkeypatch):
    def fake_run(cmd, **kw):
        Path(cmd[cmd.index("-o") + 1]).write_text("x")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    monkeypatch.setattr(package.subprocess, "run", fake_run)
    monkeypatch.setattr(package.shutil, "which", lambda x: "/usr/bin/uv")


def test_package_review_findings(tmp_path, monkeypatch):
    """Regressions for the review of residudes package (2026-11-22)."""
    _fake_prepare(monkeypatch)
    good = _pdb(tmp_path / "good.pdb", {"A": _chain(0, 0), "B": _chain(0, 5.0)})
    lost = _pdb(tmp_path / "lost.pdb", {"A": _chain(0, 0)})
    # an approval naming a slate that does not exist is refused, not replaced by the latest
    Cfg, d, ddir = _setup(tmp_path / "a", [good], what="slate v3")
    d.stage = "APPROVED"
    with pytest.raises(ValueError, match="names slate v3"):
        package.package_cmd(Cfg, d, "server", "M", None, False, False, "agent")
    # a missing model file or a flagged model: INCOMPLETE, no submit line, the stage stays
    Cfg, d, ddir = _setup(tmp_path / "b", [good, tmp_path / "gone.pdb", lost])
    out = package.package_cmd(Cfg, d, "server", "M", None, False, False, "agent")
    assert out.startswith("INCOMPLETE") and "submits" not in out.split("INCOMPLETE")[1].split("Nothing")[0]
    assert d.stage == "APPROVED" and d.log[-1]["note"].startswith("NEEDS HUMAN") and "slot 2" in out and "slot 3" in out
    # a rerun keeps the earlier package beside, never mixed in
    package.package_cmd(Cfg, d, "server", "M", None, False, False, "agent")
    assert any(p.name.startswith("package.replaced-") for p in ddir.iterdir())
    # more than 26 chains are written as mmCIF
    many = {c: [(x, y + 6.0 * k, z) for x, y, z in _chain(0, 0)] for k, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    big = _pdb(tmp_path / "big.pdb", many)
    Cfg, d, ddir = _setup(tmp_path / "c", [big], sto="A26")
    chk = package.check_model(str(big), "A26", ["AAAAAA"])
    assert chk["chains"] == 26 and chk["ok"]
    # a per-chain run is checked against its own entity's sequence
    one = _pdb(tmp_path / "one.pdb", {"A": _chain(0, 0)})
    assert package.check_model(str(one), "A1@B", ["AAAAAAAAAA", "AAAAAA"])["ok"]
    assert not package.check_model(str(one), "A1@B", ["AAAAAA", "AAAAAAAAAA"])["ok"]


def test_package_writes_mmcif_past_26_chains(tmp_path, monkeypatch):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("x")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    monkeypatch.setattr(package.subprocess, "run", fake_run)
    monkeypatch.setattr(package.shutil, "which", lambda x: "/usr/bin/uv")
    monkeypatch.setattr(package, "check_model", lambda *a, **k: {"chains": 36, "expected_chains": 36, "length_mismatch": [],
                                                                  "breaks": 0, "cis_nonpro": 0, "clash_pairs": 0, "atoms": 9, "ok": True})
    monkeypatch.setattr(package, "polymer_copy", lambda src, out, prov=None: (out.parent.mkdir(parents=True, exist_ok=True), out.write_text("x"), (out, "pLDDT"))[2])
    good = _pdb(tmp_path / "good.pdb", {"A": _chain(0, 0), "B": _chain(0, 5.0)})
    Cfg, d, ddir = _setup(tmp_path, [good])
    package.package_cmd(Cfg, d, "server", "M", "T9", False, False, "agent")
    assert "--mmcif" in seen[0] and seen[0][seen[0].index("-o") + 1].endswith("T9_1.cif") and d.stage == "PACKAGED"


def test_template_derived_model_gets_provenance_bfactors_and_parent(tmp_path, monkeypatch):
    """F38: a model copied from a deposited complex carried the deposit's B-factors (27-290) into the CASP file as
    if they were confidences, with PARENT N/A. Without a pLDDT it gets provenance B-factors and its template as PARENT."""
    import json
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_text("x")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")
    monkeypatch.setattr(package.subprocess, "run", fake_run)
    monkeypatch.setattr(package.shutil, "which", lambda x: "/usr/bin/uv")
    src = tmp_path / "results" / "A2" / "V3" / "tplderived" / "seed_1" / "sample_0"
    src.mkdir(parents=True)
    st = gemmi.read_structure(str(_pdb(tmp_path / "m.pdb", {"A": _chain(0, 0), "B": _chain(0, 5.0)})))
    for ch in st[0]:
        for r in ch:
            for a in r:
                a.b_iso = 150.0                                   # a deposit's B, not a confidence
    st.setup_entities()
    st.make_mmcif_document().write_file(str(src / "model.cif"))
    (src / "summary.json").write_text(json.dumps({"plddt": None, "template": "/x/refs/9U5Q.cif",
                                                  "chains": {"B": {"mutated": ["A3S"]}}}))
    Cfg, d, ddir = _setup(tmp_path, [src / "model.cif"])
    with open(ddir / "pool.csv") as fh:                         # the pool row has no pLDDT either
        rows = list(csv.DictReader(fh))
    rows[0]["plddt"] = ""
    with open(ddir / "pool.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    package.package_cmd(Cfg, d, "server", "M", "T9", False, False, "agent")
    assert seen[0][seen[0].index("--parent") + 1] == "9U5Q"
    poly = gemmi.read_structure(str(ddir / "package" / "polymer" / "model_1.cif"))
    b = {(ch.name, r.seqid.num): r[0].b_iso for ch in poly[0] for r in ch}
    assert max(b.values()) == 90 and b[("A", 1)] == 60 and b[("B", 3)] == 70 and len(set(b.values())) == 3
    assert "provenance" in (ddir / "package" / "checks.md").read_text() and "9U5Q" in (ddir / "package" / "checks.md").read_text()
