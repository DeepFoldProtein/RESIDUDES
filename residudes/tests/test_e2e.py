"""The command chain a campaign runs, end to end on a tiny synthetic pool in the residudes-engine layout:
target add -> pool -> gate --init -> gate --local -> gate --show -> slate -> review -> report -> slides -> approve -> export.
Each command has its own tests; this one catches breaks between them."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from residudes import cli  # noqa: E402
from residudes.dossier import Dossier  # noqa: E402

PEP, PROT = "GSKLV", "MKTAYIAKQRQISFVKSHFSRQ"


def _model(path: Path, bound: bool) -> None:
    """Chain A (peptide) and chain B (protein) as mmCIF; bound = the peptide lies along B's residues 10-14."""
    st = gemmi.Structure()
    st.cell = gemmi.UnitCell()
    m = gemmi.Model("1")
    for name, seq, y in (("A", PEP, 5.0 if bound else 40.0), ("B", PROT, 0.0)):
        ch = gemmi.Chain(name)
        for i, aa in enumerate(seq, start=1):
            r = gemmi.Residue()
            r.name, r.seqid = gemmi.expand_one_letter(aa, gemmi.ResidueKind.AA), gemmi.SeqId(i, " ")
            x = 3.8 * (i + (9 if name == "A" else 0))
            for an, dx, dy, el in (("N", -1.2, 0.5, "N"), ("CA", 0, 0, "C"), ("C", 1.2, 0.5, "C"), ("O", 1.5, 1.6, "O")):
                a = gemmi.Atom()
                a.name, a.element, a.pos = an, gemmi.Element(el), gemmi.Position(x + dx, y + dy + (0.8 if i % 2 else 0), 0.6 * (i % 3))
                r.add_atom(a)
            ch.add_residue(r)
        m.add_chain(ch)
    st.add_model(m)
    st.setup_entities()
    path.parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(str(path))


def test_the_command_chain(tmp_path, monkeypatch, capsys):
    root = tmp_path / "engine" / "T9"
    n = 0
    for runner, bound_seeds in (("af3", {1, 2}), ("of3", {1})):
        for seed in (1, 2, 3):
            for sample in (0, 1):
                d = root / "results" / "A1B1" / "msa" / runner / f"seed_{seed}" / f"sample_{sample}"
                bound = seed in bound_seeds
                _model(d / "model.cif", bound)
                (d / "summary.json").write_text(json.dumps({"ranking_score": 0.5 + 0.1 * bound + 0.01 * seed + 0.001 * sample,
                                                            "iptm": 0.4 + 0.3 * bound, "ptm": 0.6, "plddt": 80.0}))
                n += 1
    fasta = tmp_path / "T9.fasta"
    fasta.write_text(f">s1\n{PEP}\n>s2\n{PROT}\n")
    cfgf = tmp_path / "residudes.toml"
    cfgf.write_text(f'[paths]\nwork_root = "{tmp_path / "work"}"\n')
    monkeypatch.setenv("RESIDUDES_CONFIG", str(cfgf))
    monkeypatch.setenv("RESIDUDES_ACTOR", "agent")
    run = lambda *a: (cli.main(list(a)), capsys.readouterr().out)[1]
    ddir = tmp_path / "work" / "targets" / "T9"

    run("target", "add", "T9", "--fasta", str(fasta), "--stoich", "A1B1")
    out = run("pool", "T9", "--dir", str(root), "--threads", "2")
    assert f"{n}" in out or (ddir / "pool.csv").read_text().count("\n") == n + 1
    run("evidence", "T9", "binding_site", '"groove residues R10, Q11 and I12 of the protein"')
    out = run("gate", "T9", "--init", "--sto", "A1B1")
    draft = json.loads((ddir / "gates" / "draft_A1B1.json").read_text())
    site = next(g for g in draft["gates"] if g["name"] == "binding_site")
    assert site["site"] == "B:10,11,12" and site["partner"] == "A"          # chain found by sequence identity
    out = run("gate", "T9", "--spec", str(ddir / "gates" / "draft_A1B1.json"), "--sto", "A1B1", "--name", "site", "--local")
    assert "pass every gate" in out
    passing = [r for r in (ddir / "gates" / "site_A1B1.csv").read_text().splitlines()[1:] if r.split(",")[-2] == "1"]
    assert 0 < len(passing) < n                                             # the unbound models fail the site gate
    out = run("slate", "T9", "--no-interface")                              # --sto: the given A1B1
    assert "pass gates site" in out and "Rule: model 1 is af3's most confident" in out
    out = run("review", "T9")
    assert "results review for T9" in out
    out = run("report", "T9")
    assert "## Structural gates `site` at A1B1" in out
    if shutil.which("typst"):
        monkeypatch.setattr("residudes.slides.render_cmd", lambda *a, **k: "render skipped in the test")
        out = run("slides", "T9", "--no-render")
        assert (ddir / "report" / "report.typ").exists()
    d = Dossier.load(tmp_path / "work" / "targets", "T9")
    d.stage = "SLATE_PROPOSED"                                              # where the agent leaves it
    d.save(tmp_path / "work" / "targets")
    monkeypatch.setenv("RESIDUDES_ACTOR", "a-person")
    out = run("approve", "T9", "--what", "slate v1")
    assert "approved: slate v1" in out
    assert Dossier.load(tmp_path / "work" / "targets", "T9").stage == "APPROVED"
    out = run("package", "T9", "--check-only")
    assert "package checks" in out and (ddir / "package" / "checks.md").exists()
    out = run("export", "T9", str(tmp_path / "out"))
    assert (tmp_path / "out" / "analysis.md").exists() and (tmp_path / "out" / "gates" / "site_A1B1.summary.json").exists()
    d = Dossier.load(tmp_path / "work" / "targets", "T9")
    assert d.approvals and d.evidence["gates"] and d.evidence["slate_proposal"]
