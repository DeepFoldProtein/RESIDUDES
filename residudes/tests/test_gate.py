"""Structural gates (residudes gate), the slate filter they feed and the results-review triggers built on them.

Synthetic models: two 6-residue chains A and B side by side, and a one-atom ligand. Positions are chosen so each
gate's answer is known by construction."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

gemmi = pytest.importorskip("gemmi")

from residudes import consult, gate, slate  # noqa: E402
from residudes.dossier import Dossier  # noqa: E402


def _pdb(path: Path, chains: dict, ligand=None) -> Path:
    """chains: {name: [(x, y, z) of each residue's CA]}; every residue is ALA with N/CA/C/O/CB around its CA."""
    lines, n = [], 1
    for ch, cas in chains.items():
        for i, (x, y, z) in enumerate(cas, start=1):
            for name, dx, dy, el in (("N", -1.2, 0.5, "N"), ("CA", 0, 0, "C"), ("C", 1.2, 0.5, "C"), ("O", 1.5, 1.6, "O"),
                                     ("CB", 0, -1.5, "C")):
                lines.append(f"ATOM  {n:5d}  {name:<3s} ALA {ch}{i:4d}    {x + dx:8.3f}{y + dy:8.3f}{z:8.3f}  1.00 80.00           {el}")
                n += 1
        lines.append("TER")
    if ligand:
        x, y, z = ligand
        lines.append(f"HETATM{n:5d}  C1  LIG L   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 80.00           C")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _chain(x0: float, y: float, n: int = 6, z: float = 0.0) -> list:
    """A zig-zag, not a line: superposition is undefined for collinear points."""
    return [(x0 + 3.8 * i, y + (0.8 if i % 2 else 0.0), z + 0.6 * (i % 3)) for i in range(n)]


@pytest.fixture
def models(tmp_path):
    """m_good: B 5 A from A, ligand at A:3.  m_far: B 30 A away, ligand at A:6.  m_clash: B on top of A.
    m_lost: chain B missing."""
    a = _chain(0, 0)
    return {
        "good": _pdb(tmp_path / "good.pdb", {"A": a, "B": _chain(0, 5.0)}, ligand=(7.6, -4.0, 0)),
        "far": _pdb(tmp_path / "far.pdb", {"A": a, "B": _chain(0, 30.0)}, ligand=(19.0, -4.0, 0)),
        "clash": _pdb(tmp_path / "clash.pdb", {"A": a, "B": _chain(0.3, 0.2)}),
        "lost": _pdb(tmp_path / "lost.pdb", {"A": a}),
    }


SPEC = {"gates": [
    {"name": "iface", "type": "interface", "a": "A", "b": "B", "min_contacts": 3},
    {"name": "site", "type": "contact", "site": "A:2-3", "partner": "B", "within": 6.0},
    {"name": "clash", "type": "clash", "cutoff": 2.0, "max": 0}],
    "census": {"probe": "ligand", "within": 5.0, "sites": {"n_end": "A:1-3", "c_end": "A:4-6"}}}


def test_parsers():
    assert gate.parse_select("A:1-3,7,B:2") == {"A": {1, 2, 3, 7}, "B": {2}}
    assert gate.parse_select("A,B") == {"A": None, "B": None}
    assert gate.parse_partner("H,L") == ({"H", "L"}, False, None, None)
    assert gate.parse_partner("ligand:lig") == (set(), True, "LIG", None)
    assert gate.parse_partner("A:7") == ({"A"}, False, None, {"A": {7}})
    assert gate.expected_chains("A2B2C2") == 6 and gate.expected_chains("A1@B") == 1 and gate.expected_chains("weird") is None
    with pytest.raises(ValueError):
        gate.parse_select("12")                                  # a residue needs a chain
    with pytest.raises(ValueError):
        gate.check_spec({"gates": [{"name": "x", "type": "contact", "site": "A:1"}]})   # partner, within missing
    with pytest.raises(ValueError):
        gate.check_spec({"gates": [{"name": "x", "type": "interface", "a": "A", "b": "B"}]})  # no threshold


def test_evaluate_by_construction(models):
    good = gate.evaluate(str(models["good"]), SPEC, 2)
    assert good["complete_pass"] == 1 and good["iface_pass"] == 1 and good["site_pass"] == 1 and good["clash_pass"] == 1
    assert good["census"] == "n_end" and good["pass"] == 1
    far = gate.evaluate(str(models["far"]), SPEC, 2)
    assert far["iface_value"] == 0 and far["site_pass"] == 0 and far["census"] == "c_end" and far["pass"] == 0
    cl = gate.evaluate(str(models["clash"]), SPEC, 2)
    assert cl["clash_value"] > 0 and cl["clash_pass"] == 0 and cl["census"] == "other"
    lost = gate.evaluate(str(models["lost"]), SPEC, 2)
    assert lost["n_chains"] == 1 and lost["complete_pass"] == 0 and lost["pass"] == 0


def test_rmsd_and_reference_interface(models, tmp_path):
    shifted = _pdb(tmp_path / "shifted.pdb", {"A": [(x + 10, y + 3, z) for x, y, z in _chain(0, 0)], "B": _chain(10, 8.0)})
    bent = _pdb(tmp_path / "bent.pdb", {"A": [(x, y + (4.0 if i > 2 else 0), z) for i, (x, y, z) in enumerate(_chain(0, 0))],
                                        "B": _chain(0, 5.0)})
    spec = {"gates": [{"name": "core", "type": "rmsd", "select": "A", "ref": str(models["good"]), "max": 0.5, "min_pairs": 3},
                      {"name": "dimer", "type": "interface", "a": "A", "b": "B", "ref": str(models["good"]),
                       "ref_chains": {"A": "A", "B": "B"}, "min_fraction": 0.8}]}
    s = gate.evaluate(str(shifted), spec, 2)
    assert s["core_value"] < 0.01 and s["core_pass"] == 1           # a rigid move is not a deviation
    assert s["dimer_fraction"] == 1.0 and s["dimer_pass"] == 1
    b = gate.evaluate(str(bent), spec, 2)
    assert b["core_value"] > 0.5 and b["core_pass"] == 0
    f = gate.evaluate(str(models["far"]), spec, 2)
    assert f["dimer_fraction"] == 0.0 and f["dimer_pass"] == 0


def _pool(path: Path, models: dict, sto="A2") -> Path:
    rows = [("good", "af3", 0.70, 0.5), ("far", "af3", 0.95, 1.5), ("clash", "pt2", 0.60, 0.2), ("lost", "of3", 0.90, 1.0)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path", "rs_z_runner"])
        for i, (m, run, rs, z) in enumerate(rows):
            w.writerow([sto, "msa", run, i + 1, 0, rs, rs, rs, 80, False, str(models[m]), z])
    return path


def test_run_feeds_the_slate(models, tmp_path):
    pc = _pool(tmp_path / "pool.csv", models)
    out = tmp_path / "gates" / "default_A2.csv"
    s = gate.run(SPEC, pc, "A2", out, workers=1)
    assert s["n"] == 4 and s["n_pass"] == 1 and s["gates"]["complete"] == 3
    assert s["incomplete"] == {"msa/of3": 1} and s["census"]["af3"] == {"n_end": 1, "c_end": 1}
    keep, note = gate.for_slate(tmp_path, "A2", None, None)
    assert keep == {"A2__msa__af3__s1_0"} and note["pass"] == 1 and note["of"] == 4
    # the highest-confidence model (far) would lead without gates; with them only the passing model is a candidate
    ungated = slate.propose(pc, "A2", n=2)
    assert ungated["models"][0]["model"] == "A2__msa__af3__s2_0"
    gated = slate.propose(pc, "A2", n=2, allowed={"A2": keep})      # one passing model cannot fill two slots
    assert [m["model"] for m in gated["models"]] == ["A2__msa__af3__s1_0"] and gated["short"] == 1
    md = slate.write(gated | {"gated": {"A2": {"gates": "default", "pass": 1, "of": 4}}}, tmp_path / "sl", 1)[1].read_text()
    assert "1 of 4 models pass gates default" in md and "only 1 model" in md
    assert gate.for_slate(tmp_path, "A2", "none", None) == (None, None)
    with pytest.raises(ValueError):
        gate.for_slate(tmp_path, "A2", None, "c_end")               # c_end holds no passing model
    assert gate.failures(out)["A2__msa__af3__s2_0"] == ["iface", "site"]


def test_review_flags_lost_chains_and_unused_site_priors(models, tmp_path):
    pc = _pool(tmp_path / "pool.csv", models)
    num, trig = consult.checks(pc, None)
    lost = [t for t in trig if t.startswith("missing_chains")]
    assert lost and "A2/msa/of3 has 1 of 2 chains" in lost[0]
    assert num["chains_per_run"]["A2/msa/af3"] == 2

    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="T9")
    d.evidence["epitope"] = [{"time": "t", "by": "agent", "value": "A:2-3 (escape mutants)"}]
    ddir = Dossier.path(Cfg.targets_dir, "T9").parent
    ddir.mkdir(parents=True)
    (ddir / "pool.csv").write_text(pc.read_text())
    out = consult.review_cmd(Cfg, d, [], "agent")
    assert "site_prior_without_gate" in out and "missing_chains" in out


def test_census_split_is_a_trigger(tmp_path):
    """The ten most confident models split between two sites: a conversation, not more seeds (M1439, M2415)."""
    gdir = tmp_path / "gates"
    gdir.mkdir()
    pc = tmp_path / "pool.csv"
    with open(pc, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path", "rs_z_runner"])
        for i in range(6):
            w.writerow(["A1B1", "msa", "af3", i, 0, 0.5, 0.5, 0.5, 80, False, "/nonexistent.cif", 0.1 * i])
    with open(gdir / "default_A1B1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "rs_z_runner", "census", "pass"])
        for i in range(6):
            w.writerow([f"A1B1__msa__af3__s{i}_0", 0.1 * i, "bulge" if i % 2 else "sm_site", 1])
    num, trig = consult.checks(pc, None, gdir)
    assert num["census_top10_A1B1"] == {"bulge": 3, "sm_site": 3}
    assert any(t.startswith("site_split") for t in trig)


def test_gate_cmd_local_records_evidence(models, tmp_path):
    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="T8")
    ddir = Dossier.path(Cfg.targets_dir, "T8").parent
    ddir.mkdir(parents=True)
    _pool(ddir / "pool.csv", models)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    out = gate.gate_cmd(Cfg, d, str(spec), "A2", "epi", local=True, workers=1, actor="agent")
    assert "1 of 4 models pass" in out and "census (probe ligand)" in out and "missing chains" in out
    ev = d.evidence["gates"][-1]["value"]
    assert ev["name"] == "epi" and ev["n_pass"] == 1 and ev["spec"]["gates"][0]["name"] == "iface"
    assert (ddir / "gates" / "epi_A2.json").exists()
    with pytest.raises(ValueError):
        gate.gate_cmd(Cfg, d, str(spec), "A2", "bad-name", local=True, workers=1, actor="agent")


def test_gates_reach_the_report_and_the_slides(models, tmp_path):
    from residudes import report, slides

    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="T7")
    ddir = Dossier.path(Cfg.targets_dir, "T7").parent
    ddir.mkdir(parents=True)
    _pool(ddir / "pool.csv", models)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    gate.gate_cmd(Cfg, d, str(spec), "A2", "epi", local=True, workers=1, actor="agent")
    md = report.build(d)
    assert "## Structural gates `epi` at A2" in md and "| iface | interface A : B within 5.0 Å, ≥ 3 residue contacts | 2 |" in md
    assert "Site census (probe ligand)" in md and "msa/of3 (1)" in md
    assert "### gates" not in md                                  # not dumped as a prior
    typ = slides.build_typ(d, ddir)
    assert "== Structural gates #\"epi\" at #\"A2\"" in typ and "Site census" in typ


def test_draft_spec_reads_sites_from_the_evidence(models, tmp_path):
    assert gate.site_from_text("paratope within 6 A of A:611,615 (2A2 escape mutants)") == ("A:611,615", False)
    assert gate.site_from_text("escape mutants S611/R615") == ("A:611,615", True)       # chain guessed
    assert gate.site_from_text("LARP1 Ser548 in the 14-3-3 pocket") == ("A:548", True)
    assert gate.site_from_text("the recognition core of the intron") == (None, False)
    assert gate.chain_ids("A2B1") == ["A", "B", "C"] and len(gate.chain_ids("A28")) == 28 and gate.chain_ids("A28")[26] == "AA"

    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="T6")
    Dossier.path(Cfg.targets_dir, "T6").parent.mkdir(parents=True)
    d.evidence["epitope"] = [{"time": "t", "by": "agent", "value": {"site": "A:611,615", "source": "escape mutants"}}]
    d.evidence["recognition_site"] = [{"time": "t", "by": "agent", "value": "UGAGGAGGUUU at nt 20-30"}]
    d.evidence["public_structures"] = [{"time": "t", "by": "agent", "value": "8UFO apo spike"}]
    out = gate.init_cmd(Cfg, d, "A2B1", "agent")
    spec = json.loads((Dossier.path(Cfg.targets_dir, "T6").parent / "gates" / "draft_A2B1.json").read_text())
    names = [g["name"] for g in spec["gates"]]
    assert names == ["epitope", "assembled"] and "draft_A2B1.json" in out
    assert any("no clash gate" in n for n in spec["notes"])
    ep = spec["gates"][0]
    assert ep["site"] == "A:611,615" and ep["partner"] == "C" and "from evidence epitope" in ep["note"]   # not B: a copy of A
    assert any("recognition_site" in n for n in spec["notes"]) and any("public_structures" in n for n in spec["notes"])
    gate.check_spec(spec)                                         # the draft is a valid spec as written


def test_draft_spec_review_findings(tmp_path):
    """Regressions for the review of `residudes gate --init` (2026-11-22)."""
    # ordinary notation is not a residue: target ids, CDRs, stoichiometries, pH, a trailing count
    assert gate.site_from_text("CDR H3 and L1 contact HE-3 (H1392)") == (None, False)
    assert gate.site_from_text("the A24 cage, C2 symmetric, subsite S1") == (None, False)
    assert gate.site_from_text("stable at pH:7") == (None, False)
    assert gate.site_from_text("A:611, 2 residues") == ("A:611", False)
    assert gate.site_from_text("Lys49 and Arg56 bind the phosphate") == ("A:49,56", True)
    assert gate.entity_chains("A2B1C1") == {"A": ["A", "B"], "B": ["C"], "C": ["D"]}

    class Cfg:
        targets_dir = tmp_path / "targets"

    def dossier(name, **ev):
        d = Dossier(target=name)
        Dossier.path(Cfg.targets_dir, name).parent.mkdir(parents=True, exist_ok=True)
        for k, v in ev.items():
            d.evidence[k.replace("__", "-")] = [{"time": "t", "by": "agent", "value": v}] if v is not None else []
        return d

    # homo-oligomer: the other copies are not a partner; without a ligand the site is left to a person
    spec = gate.draft_spec(dossier("H1", epitope="A:10,12"), "A3")
    assert [g["name"] for g in spec["gates"]] == ["assembled"] and any("no other entity" in n for n in spec["notes"])
    assert any('"type": "symmetry"' in n for n in spec["notes"])        # suggested for a ring, not drafted
    spec = gate.draft_spec(dossier("H2", pocket="A:10,12", ligand_ccd="ATP"), "A3")
    assert spec["gates"][0]["partner"] == "ligand"
    # letters are model chain ids; a letter that is also another entity's name is kept, and the note says so
    spec = gate.draft_spec(dossier("H3", paratope="B:30-35"), "A2B1C1")
    g = spec["gates"][0]
    assert g["site"].startswith("B:30,31") and g["partner"] == "C,D" and "kept as model chain B" in g["note"]
    # colliding and empty keys
    spec = gate.draft_spec(dossier("H4", binding_site="A:5,6", binding__site="A:7,8", epitope=None), "A1B1")
    assert [g["name"] for g in spec["gates"]][:2] == ["binding_site", "binding_site_2"]
    # a second --init never overwrites the first (a person may have edited it)
    d = dossier("H5", epitope="A:5,6")
    gate.init_cmd(Cfg, d, "A1B1", "agent")
    out = gate.init_cmd(Cfg, d, "A1B1", "agent")
    assert "draft_A1B1_2.json" in out and (Dossier.path(Cfg.targets_dir, "H5").parent / "gates" / "draft_A1B1.json").exists()
    # numbers must be numbers (a string min_fraction used to crash the report later)
    with pytest.raises(ValueError):
        gate.check_spec({"gates": [{"name": "x", "type": "interface", "a": "A", "b": "B", "ref": "r.cif", "min_fraction": "0.6"}]})
    assert "≥ 60%" in gate.describe({"type": "interface", "a": "A", "b": "B", "ref": "r", "min_fraction": "0.6"})
    assert "(≥ 4 site residues)" in gate.describe({"type": "contact", "partner": "H", "within": 6, "site": "A:1", "min_residues": 4})
    # the latest results come in the order they were last shown
    d = Dossier(target="H6")
    for n in ("X", "Y", "Z", "X"):
        d.evidence.setdefault("gates", []).append({"value": {"name": n, "sto": "A3"}})
    assert [v["name"] for v in gate.latest_results(d)] == ["Y", "Z", "X"]


def test_chosen_slate_groups_parts_by_stoichiometry(models, tmp_path):
    pc = _pool(tmp_path / "pool.csv", models)
    with open(pc, "a", newline="") as fh:
        csv.writer(fh).writerow(["A1", "msa", "af3", 9, 0, 0.5, "", 0.5, 80, False, str(models["good"]), 0.0])
    s = slate.from_models(pc, ["A2__msa__af3__s1_0", "A1__msa__af3__s9_0", "A2__msa__af3__s2_0"])
    assert s["parts"] == [{"sto": "A2", "n": 2}, {"sto": "A1", "n": 1}] and s["sto"] == "A2:2,A1:1"


def test_approve_names_slate_models_that_fail_gates(models, tmp_path, monkeypatch, capsys):
    from residudes import cli
    cfgf = tmp_path / "residudes.toml"
    cfgf.write_text(f'[paths]\nwork_root = "{tmp_path / "work"}"\n')
    monkeypatch.setenv("RESIDUDES_CONFIG", str(cfgf))
    monkeypatch.setenv("RESIDUDES_ACTOR", "person")
    targets = tmp_path / "work" / "targets"
    d = Dossier(target="T5")
    ddir = Dossier.path(targets, "T5").parent
    ddir.mkdir(parents=True)
    pc = _pool(ddir / "pool.csv", models)
    gate.run(SPEC, pc, "A2", ddir / "gates" / "default_A2.csv", workers=1)
    d.evidence["slate_proposal"] = [{"time": "t", "by": "agent", "value": {"version": 1, "models": ["A2__msa__af3__s1_0", "A2__msa__af3__s2_0"]}}]
    d.save(targets)
    assert cli.main(["approve", "T5", "--what", "slate v1"]) == 0
    out = capsys.readouterr().out
    assert "approved: slate v1" in out and "slot 2 A2__msa__af3__s2_0: iface, site" in out and "s1_0" not in out
    d = Dossier.load(targets, "T5")
    assert d.approvals and "gate failures" in d.log[-1]["note"]


def test_qa_antibody_mode_builds_the_ab_command(tmp_path, monkeypatch):
    from residudes import ops
    seen = {}
    monkeypatch.setattr(ops, "_sbatch", lambda cmd, *a, **k: seen.setdefault("cmd", cmd) and "123")
    monkeypatch.setattr(ops, "target_dir", lambda cfg, d, explicit=None: tmp_path / "T1")

    class Cfg:
        targets_dir = tmp_path / "targets"
        residudes_qa = str(tmp_path / "lqa")
    d = Dossier(target="T1")
    Dossier.path(Cfg.targets_dir, "T1").parent.mkdir(parents=True)
    ops.qa_cmd(Cfg, d, "A1B1C1", "interface", None, "agent", "ab", "BC", "A")
    assert "interface run-all --mode ab" in seen["cmd"] and "--ab-chains BC --ag-chains A" in seen["cmd"]
    assert "interface_A1B1C1_ab" in seen["cmd"] and d.runs[-1]["what"] == "residudes-qa interface A1B1C1"
    with pytest.raises(ValueError):
        ops.qa_cmd(Cfg, d, "A1B1C1", "interface", None, "agent", "ab", None, None)


def test_live_check_findings(models, tmp_path):
    """From the live Claude Code check on H1388 (2026-11-22): a partner can be a residue, and --init finds the chain
    of residues written without one by their identity in the sequence."""
    spec = {"gates": [{"name": "near", "type": "contact", "site": "A:2-3", "partner": "B:2", "within": 6.0},
                      {"name": "far", "type": "contact", "site": "A:2-3", "partner": "B:6", "within": 6.0}]}
    r = gate.evaluate(str(models["good"]), spec, 2)
    assert r["near_pass"] == 1 and r["far_pass"] == 0              # B:6 is 15 A along the chain

    assert gate.residue_tokens("K49, R56, R131 and Y132 of 14-3-3; Ser548") == [("S", 548), ("K", 49), ("R", 56), ("R", 131), ("Y", 132)]
    pep, prot = "PKGLSASLPDL", "SDREQLVQKA" + "X" * 38 + "K" + "XXXXXX" + "R" + "X" * 74 + "RY" + "X" * 20
    assert prot[48] == "K" and prot[55] == "R" and prot[130:132] == "RY"
    assert gate.entity_by_sequence([("K", 49), ("R", 56), ("R", 131), ("Y", 132)], [pep, prot]) == "B"
    assert gate.entity_by_sequence([("K", 49)], [prot, prot]) is None                # ambiguous: two entities fit

    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="H7")
    d.sequence["text"] = f">s1\n{pep}\n>s2\n{prot}\n"
    d.evidence["binding_site"] = [{"time": "t", "by": "agent", "value": "phosphate pocket K49, R56, R131 and Y132 of 14-3-3"}]
    g = gate.draft_spec(d, "A1B1")["gates"][0]
    assert g["site"] == "B:49,56,131,132" and g["partner"] == "A" and "its sequence has these residues" in g["note"]
    d.evidence["binding_site"] = [{"time": "t", "by": "agent", "value": "pocket residues K19 and Q22 on chain B"}]
    g = gate.draft_spec(d, "A1B1")["gates"][0]
    assert g["site"] == "B:19,22" and "as the text says" in g["note"]


def test_export_carries_the_gate_specs_and_summaries(models, tmp_path):
    from residudes import export

    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="T4")
    ddir = Dossier.path(Cfg.targets_dir, "T4").parent
    ddir.mkdir(parents=True)
    _pool(ddir / "pool.csv", models)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    gate.gate_cmd(Cfg, d, str(spec), "A2", "epi", local=True, workers=1, actor="agent")
    export.run(d, ddir, tmp_path / "out")
    got = sorted(p.name for p in (tmp_path / "out" / "gates").iterdir())
    assert got == ["epi_A2.json", "epi_A2.summary.json"]                # the per-model CSV stays behind
    assert "## Structural gates `epi` at A2" in (tmp_path / "out" / "analysis.md").read_text()


def test_second_review_findings(models, tmp_path):
    """Regressions for the review of the threaded pool, partner selections, sequence placement and parking (2026-11-22)."""
    from residudes import daemon, pool
    # a step that ran out of budget or turns is not an API error
    assert daemon.api_error({"subtype": "error_max_budget_usd", "is_error": True, "result": ""}) is None
    assert daemon.api_error({"subtype": "error_max_turns", "is_error": True}) is None
    assert daemon.api_error({"subtype": "error_during_execution", "is_error": True, "result": ""}) == "error_during_execution"
    # a seed with sample_best next to sample_0..1 imports (int and str sample names sort together)
    seed = tmp_path / "T" / "results" / "A1" / "msa" / "af3" / "seed_1"
    for smp in ("sample_1", "sample_best", "sample_0"):
        (seed / smp).mkdir(parents=True)
        (seed / smp / "summary.json").write_text('{"ranking_score": 0.5, "ptm": 0.5}')
        (seed / smp / "model.cif").write_text("data_x\n")
    assert [r["sample"] for r in pool.scan(tmp_path / "T", threads=2)] == ["0", "1", "best"]
    (seed / "sample_1" / "model.cif").write_text("")                          # a 0-byte model is left out
    empty: list = []
    assert [r["sample"] for r in pool.scan(tmp_path / "T", threads=2, empty=empty)] == ["0", "best"]
    assert empty == ["A1/msa/af3/seed_1/sample_1"]
    # a partner on the site's own chain does not match the site residue or its chain neighbours
    near_self = {"gates": [{"name": "self", "type": "contact", "site": "A:2-3", "partner": "A:3", "within": 6.0}]}
    assert gate.evaluate(str(models["good"]), near_self, 2)["self_pass"] == 0
    far_part = {"gates": [{"name": "loop", "type": "contact", "site": "A:1", "partner": "A:5", "within": 20.0}]}
    assert gate.evaluate(str(models["good"]), far_part, 2)["loop_pass"] == 1
    # interface a / b residue selections count
    iface = {"gates": [{"name": "i1", "type": "interface", "a": "A:1", "b": "B:1-2", "min_contacts": 1},
                       {"name": "i2", "type": "interface", "a": "A:1", "b": "B:6", "min_contacts": 1}]}
    r = gate.evaluate(str(models["good"]), iface, 2)
    assert r["i1_pass"] == 1 and r["i2_pass"] == 0 and r["i2_value"] == 0
    # residues 1000+ are residues; accessions, His tags and the target's own ids are not
    assert gate.site_from_text("epitope: Q9UNE7 residues K49 R56, His6-tagged") == ("A:49,56", True)
    assert gate.site_from_text("contacts D1146 and E1144") == ("A:1144,1146", True)
    assert gate.site_from_text("H1388 pocket K49 R56", gate.target_ids("H2388")) == ("A:49,56", True)
    assert gate.target_ids("H2388") == {"H2388", "H1388"}

    def dossier(name, **ev):
        d = Dossier(target=name)
        for k, v in ev.items():
            d.evidence[k] = [{"time": "t", "by": "agent", "value": v}]
        return d
    # 'region' / 'description' keys are not ligand evidence
    spec = gate.draft_spec(dossier("X1", binding_site="A:10,12", domain_description="two domains", region="N-term"), "A2")
    assert all(g.get("partner") != "ligand" for g in spec["gates"]) and any("no other entity" in n for n in spec["notes"])
    # a chain the model does not have is left out, with a note
    spec = gate.draft_spec(dossier("X2", paratope="H:95-102"), "A2B1C1")
    assert [g["name"] for g in spec["gates"]] == ["assembled"] and any("names chain(s) H" in n for n in spec["notes"])


def test_approve_judges_a_slate_by_its_own_gates(models, tmp_path):
    pc = _pool(tmp_path / "pool.csv", models)
    gate.run(SPEC, pc, "A2", tmp_path / "gates" / "strict_A2.csv", workers=1)
    gate.run({"gates": [{"name": "clash", "type": "clash", "cutoff": 2.0, "max": 1000}]}, pc, "A2",
             tmp_path / "gates" / "loose_A2.csv", workers=1)                 # newer, passes everything readable
    prop = {"version": 1, "models": ["A2__msa__af3__s1_0", "A2__msa__af3__s2_0"], "gated": {"A2": {"gates": "strict"}}}
    assert gate.proposal_failures(tmp_path, prop) == {"A2__msa__af3__s2_0": ["iface", "site"]}
    assert gate.slate_failures(tmp_path, prop["models"]) == {}                # the latest set would have missed it
    assert gate.proposal_failures(tmp_path, prop | {"gate_fail": {}}) != {}     # an empty gate_fail on a rule proposal is not trusted
    chosen = {"version": 2, "chosen": True, "models": prop["models"], "gate_fail": {"A2__msa__af3__s2_0": ["site"]}}
    assert gate.proposal_failures(tmp_path, chosen) == {"A2__msa__af3__s2_0": ["site"]}


def test_symmetry_gate_on_a_c3_ring(tmp_path):
    import math

    def rot(xyz, k):
        a = 2 * math.pi * k / 3
        return [(x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), z) for x, y, z in xyz]
    base = [(10 + 1.5 * (i % 2), 3.8 * i, 0.6 * (i % 3)) for i in range(12)]          # one copy, off the axis
    sym = _pdb(tmp_path / "c3.pdb", {"A": base, "B": rot(base, 1), "C": rot(base, 2)})
    bent = _pdb(tmp_path / "c3_bent.pdb", {"A": base, "B": rot(base, 1), "C": [(x + 6, y, z) for x, y, z in rot(base, 2)]})
    spec = {"gates": [{"name": "c3", "type": "symmetry", "chains": "A,B,C", "max": 1.0}]}
    r = gate.evaluate(str(sym), spec, 3)
    assert r["c3_value"] < 0.01 and r["c3_pass"] == 1 and r["c3_pairs"] == 12
    r = gate.evaluate(str(bent), spec, 3)
    assert r["c3_value"] > 1.0 and r["c3_pass"] == 0
    assert "cyclic symmetry of A,B,C" in gate.describe(spec["gates"][0])


def test_stacking_gate_tells_a_beta_stack_from_a_loose_bundle(tmp_path):
    strand = [(3.4 * i, 0.4 * (i % 2), 0.0) for i in range(8)]
    stack = _pdb(tmp_path / "stack.pdb", {c: [(x, y, z + 4.8 * k) for x, y, z in strand] for k, c in enumerate("ABCD")})
    loose = _pdb(tmp_path / "loose.pdb", {c: [(x, y + 10.0 * k, z) for x, y, z in strand] for k, c in enumerate("ABCD")})
    spec = {"gates": [{"name": "xbeta", "type": "stacking"}]}
    r = gate.evaluate(str(stack), spec, 4)
    assert abs(r["xbeta_value"] - 4.8) < 0.01 and r["xbeta_pass"] == 1 and r["xbeta_fraction"] == 1.0
    r = gate.evaluate(str(loose), spec, 4)
    assert abs(r["xbeta_value"] - 10.0) < 0.01 and r["xbeta_pass"] == 0


def test_retro_audits_the_gates_against_the_native(tmp_path):
    """With native scores, the retrospective says whether the gates kept the good models (synthetic numbers)."""
    from residudes import retro
    gdir = tmp_path / "gates"
    gdir.mkdir()
    sc = {}
    with open(gdir / "epi_A2.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "rs_z_runner", "pass"])
        for i in range(20):                       # the good models (high DockQ) pass; the most confident one fails
            k = f"A2__msa__af3__s{i}_0"
            dq = 0.9 - 0.04 * i
            w.writerow([k, 2.0 if i == 15 else 0.1 * (20 - i) / 20, 1 if i < 10 else 0])
            sc[k] = {"dockq_wave": dq}
    g = retro._gate_audit(tmp_path, sc)[0]
    assert g["gates"] == "epi_A2" and g["metric"] == "dockq_wave" and g["n_pass"] == 10 and g["top10_pass"] == 1.0
    assert g["best_passes"] and abs(g["pick_all"] - 0.30) < 1e-9 and abs(g["pick_pass"] - 0.90) < 1e-9   # one runner: by z
    assert g["median_pass"] > g["median_fail"]


def test_robust_z_resists_narrow_runners_and_outliers():
    """The per-runner z the slate ranks by: a runner whose scores barely spread must not win on a 0.015 difference,
    and one extreme score must not squash the rest of its runner."""
    from residudes import pool
    rows = []
    for i in range(20):                                    # bt2: all 0.915 +- 0.007, best 0.93
        rows.append({"sto": "A1B1", "runner": "bt2", "ranking_score": 0.915 + (0.015 if i == 0 else 0.007 * ((i % 3) - 1))})
    for i in range(20):                                    # af3: two modes; the best 0.96 is a real outlier upwards
        rows.append({"sto": "A1B1", "runner": "af3", "ranking_score": 0.96 if i < 3 else 0.70 + 0.01 * (i % 5)})
    for i in range(20):                                    # pt2: sane scores plus two failed models scored 100
        rows.append({"sto": "A1B1", "runner": "pt2", "ranking_score": 100.0 if i < 2 else 0.60 + 0.01 * (i % 7)})
    pool._zscore(rows)
    best = lambda run: max(r["rs_z_runner"] for r in rows if r["runner"] == run and r["ranking_score"] < 50)
    assert best("af3") > best("bt2")                       # the plain z put bt2 first
    assert best("pt2") > 1.0                               # the plain z squashed pt2 to ~0


def test_slate_leads_with_af3_and_alternates_runners(tmp_path):
    """Model 1 is af3's most confident model; the other slots alternate runners (af3 first, then by best raw score)."""
    pc = tmp_path / "pool.csv"
    with open(pc, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path", "rs_z_runner"])
        rows = [("bt2", 0.95, 3.0), ("bt2", 0.94, 2.0), ("af3", 0.80, 1.5), ("af3", 0.78, 1.0), ("of3", 0.70, 2.5), ("of3", 0.60, 0.5)]
        for i, (run, rs, z) in enumerate(rows):
            w.writerow(["A2", "msa", run, i + 1, 0, rs, rs, rs, 80, False, f"/m{i}.cif", z])
    s = slate.propose(pc, "A2", n=5, use_interface=False)
    assert [m["runner"] for m in s["models"]] == ["af3", "bt2", "of3", "af3", "bt2"]
    assert s["models"][0]["model"] == "A2__msa__af3__s3_0"           # af3's best, though bt2's z is higher
    order = [r["runner"] for r in slate.interleave([{"runner": r, "ranking_score": rs, "score": z} for r, rs, z in rows if r != "af3"])]
    assert order[:2] == ["bt2", "of3"]                                # no af3: by each runner's best raw score


def test_third_review_findings(models, tmp_path):
    """Regressions for the review of the slate order, robust z, symmetry/stacking and the retro audit (2026-11-22)."""
    # non-numeric samples are kept as written
    assert slate.key("A3", "msa", "af3", "1", "best") == "A3__msa__af3__s1_best"
    assert slate.key("A3", "msa", "af3", "1.0", "2") == "A3__msa__af3__s1_2"

    def pool_csv(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path", "rs_z_runner"])
            for i, (run, rs) in enumerate(rows):
                w.writerow(["A2", "msa", run, i + 1, 0, rs, rs, rs, 80, False, f"/m{i}.cif", rs])
        return path
    # one runner with many candidates fills the slate once the others run out (the cap gives way last)
    pc = pool_csv(tmp_path / "p1.csv", [("af3", 0.9 - 0.01 * i) for i in range(20)] + [("of3", 0.5)])
    s = slate.propose(pc, "A2", n=5, use_interface=False)
    assert len(s["models"]) == 5 and s["short"] == 0 and [m["runner"] for m in s["models"]][:2] == ["af3", "of3"]
    # no af3: the fixed priority leads (bt2 before pt2), not the raw score (pt2 0.99 > bt2 0.60)
    pc = pool_csv(tmp_path / "p2.csv", [("pt2", 0.99), ("bt2", 0.60), ("of3", 0.95)])
    assert [m["runner"] for m in slate.propose(pc, "A2", n=3, use_interface=False)["models"]] == ["bt2", "pt2", "of3"]
    # a single-model slate is not overwritten by the second-runner fallback
    pc = pool_csv(tmp_path / "p3.csv", [("af3", 0.9)])
    assert [m["runner"] for m in slate.propose(pc, "A2", n=5, use_interface=False)["models"]] == ["af3"]
    # an interface gate with a reference but no ref_chains maps chains by id
    spec = {"gates": [{"name": "d", "type": "interface", "a": "A", "b": "B", "ref": str(models["good"]), "min_fraction": 0.8}]}
    r = gate.evaluate(str(models["good"]), spec, 2)
    assert r["d_fraction"] == 1.0 and r["d_pass"] == 1
    # a glycan bonded to the protein (its own chain, 1.45 A) is not a clash; two proteins 1.0 A apart are
    glyco = _pdb(tmp_path / "glyco.pdb", {"A": _chain(0, 0), "B": _chain(0, 8.0)}, ligand=(0.0, -1.5 - 1.45, 0.0))
    assert gate.evaluate(str(glyco), {"gates": [{"name": "c", "type": "clash", "max": 0}]}, 2)["c_value"] == 0
    assert gate.evaluate(str(models["clash"]), {"gates": [{"name": "c", "type": "clash", "max": 0}]}, 2)["c_value"] > 0


def test_fourth_review_findings(models, tmp_path):
    """Regressions for the review of setup, the drafts and the review triggers (2026-11-22)."""
    from residudes import setup
    from residudes.dossier import tkey
    # a monomer with no site evidence still gets a draft (complete only), and it runs
    d = Dossier(target="M1")
    spec = gate.draft_spec(d, "A1")
    assert spec["gates"] == [] and gate.check_spec(spec) is spec
    # fractional thresholds are not truncated
    r = gate.evaluate(str(models["clash"]), {"gates": [{"name": "c", "type": "clash", "max": 0.5}]}, 2)
    assert r["c_value"] > 0 and r["c_pass"] == 0
    # times from hosts in different zones compare as instants
    assert tkey("2026-11-22T10:00:00+09:00") < tkey("2026-11-22T01:30:00+00:00")
    assert tkey("") < tkey("2026-01-01T00:00:00+00:00")
    # setup repairs a dangling link to a residudes skill (a moved checkout) and refreshes its own copies
    proj = tmp_path / "p"
    dd = proj / ".agents" / "skills"
    dd.mkdir(parents=True)
    (dd / "residudes-selection").symlink_to(tmp_path / "old-checkout" / "skills" / "residudes-selection")
    setup.link_skills(proj)
    assert (dd / "residudes-selection" / "SKILL.md").exists()
    # doctor: a README among the skills is not a broken skill
    (dd / "README.md").write_text("notes")
    row = next(x for x in setup.skill_check([proj]) if x["folder"] == str(dd))
    assert row["broken"] == [] and row["missing"] == []


def test_site_split_ignores_failing_models_and_resubmission_sets_old_results_aside(models, tmp_path, monkeypatch):
    from residudes import ops
    gdir = tmp_path / "gates"
    gdir.mkdir()
    pc = tmp_path / "pool.csv"
    with open(pc, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt", "has_clash", "path", "rs_z_runner"])
        for i in range(6):
            w.writerow(["A1B1", "msa", "af3", i, 0, 0.5, 0.5, 0.5, 80, False, "/x.cif", 0.1 * i])
    with open(gdir / "default_A1B1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "rs_z_runner", "census", "pass"])
        for i in range(6):                         # the failing half sits elsewhere: not the slate's candidates
            w.writerow([f"A1B1__msa__af3__s{i}_0", 0.1 * i, "bulge" if i % 2 else "sm_site", i % 2])
    _, trig = consult.checks(pc, None, gdir)
    assert not any(t.startswith("site_split") for t in trig)

    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="T3")
    ddir = Dossier.path(Cfg.targets_dir, "T3").parent
    (ddir / "gates").mkdir(parents=True)
    _pool(ddir / "pool.csv", models)
    (ddir / "gates" / "epi_A2.csv").write_text("old")
    (ddir / "gates" / "epi_A2.summary.json").write_text("{}")
    monkeypatch.setattr(ops, "_sbatch", lambda *a, **k: "42")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC))
    gate.gate_cmd(Cfg, d, str(spec), "A2", "epi", local=False, workers=1, actor="agent")
    names = sorted(p.name for p in (ddir / "gates").iterdir())
    assert "epi_A2.csv" not in names and any(n.startswith("epi_A2.csv.replaced-") for n in names)
    assert "no result for gates epi" in gate.show_cmd(Cfg, d, "epi", "A2", "agent")


def test_cold_read_findings(models, tmp_path, monkeypatch, capsys):
    """Regressions for a newcomer's read of the docs (2026-11-22)."""
    from residudes import cli, ops
    cfgf = tmp_path / "residudes.toml"
    cfgf.write_text(f'[paths]\nwork_root = "{tmp_path / "work"}"\n')
    monkeypatch.setenv("RESIDUDES_CONFIG", str(cfgf))
    targets = tmp_path / "work" / "targets"
    # the daemon needs its targets named (or --all)
    assert cli.main(["daemon", "--once"]) == 2 and "name the targets" in capsys.readouterr().err
    # approve moves a proposed slate to APPROVED
    d = Dossier(target="T2")
    d.stoichiometry["given"] = "A2"
    d.stage = "SLATE_PROPOSED"
    d.evidence["slate_proposal"] = [{"time": "2026-11-22T01:00:00+09:00", "by": "agent", "value": {"version": 1, "models": []}}]
    d.save(targets)
    monkeypatch.setenv("RESIDUDES_ACTOR", "a-person")
    assert cli.main(["approve", "T2", "--what", "slate v1"]) == 0
    assert Dossier.load(targets, "T2").stage == "APPROVED"
    capsys.readouterr()
    # --sto falls back to the given stoichiometry; a spec naming a chain the models lack is refused
    ddir = Dossier.path(targets, "T2").parent
    _pool(ddir / "pool.csv", models)
    monkeypatch.setenv("RESIDUDES_ACTOR", "agent")
    assert cli.main(["gate", "T2", "--init"]) == 0 and (ddir / "gates" / "draft_A2.json").exists()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"gates": [{"name": "e", "type": "contact", "site": "A:1", "partner": "H,L", "within": 6}]}))
    assert cli.main(["gate", "T2", "--spec", str(bad), "--local"]) == 1
    assert "names chain(s) H, L" in capsys.readouterr().err
    # a failed sbatch leaves the earlier result where it was
    (ddir / "gates" / "epi_A2.csv").write_text("old")
    def boom(*a, **k):
        raise ValueError("sbatch failed: partition down")
    monkeypatch.setattr(ops, "_sbatch", boom)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(SPEC))
    assert cli.main(["gate", "T2", "--spec", str(good), "--name", "epi"]) == 1
    assert (ddir / "gates" / "epi_A2.csv").read_text() == "old"


def test_slate_leaves_antibody_interface_scores_out(tmp_path, monkeypatch):
    qa = tmp_path / "qa" / "interface_A2_ab"
    qa.mkdir(parents=True)
    (qa / "xgb_scores.csv").write_text("model_id,xgb_pred\naf3_seed1_sample0__msa.pdb,0.9\n")
    monkeypatch.delenv("RESIDUDES_USE_AB_INTERFACE", raising=False)
    assert slate.load_interface(tmp_path / "qa", "A2") == {}
    monkeypatch.setenv("RESIDUDES_USE_AB_INTERFACE", "1")
    assert slate.load_interface(tmp_path / "qa", "A2") == {"A2__msa__af3__s1_0": 0.9}


def test_final_review_findings(tmp_path, monkeypatch, capsys):
    """Regressions for the last review of the polish loop (2026-11-22)."""
    from residudes import cli, consult
    cfgf = tmp_path / "residudes.toml"
    cfgf.write_text(f'[paths]\nwork_root = "{tmp_path / "work"}"\n')
    monkeypatch.setenv("RESIDUDES_CONFIG", str(cfgf))
    monkeypatch.setenv("RESIDUDES_ACTOR", "a-person")
    targets = tmp_path / "work" / "targets"
    d = Dossier(target="T3", stage="SLATE_PROPOSED")
    d.stoichiometry["given"] = "A3"
    d.evidence["slate_proposal"] = [{"time": "2026-11-22T01:00:00+09:00", "by": "agent", "value": {"version": 1, "models": []}}]
    d.save(targets)
    # a slate version that does not exist is refused, and nothing is recorded
    assert cli.main(["approve", "T3", "--what", "slate v3"]) == 1 and "no slate v3" in capsys.readouterr().err
    assert not Dossier.load(targets, "T3").approvals
    # an approval that is not about a slate keeps the stage
    assert cli.main(["approve", "T3", "--what", "stoichiometry A3"]) == 0
    assert Dossier.load(targets, "T3").stage == "SLATE_PROPOSED"
    # --sto: a stoichiometry decision that differs from the given one has to be named
    from residudes.cli import _sto
    d = Dossier.load(targets, "T3")
    assert _sto(d, None) == "A3"
    d.decide("agent", "stoichiometry", "A2:3,A1:2 hedge", ["e"])
    with pytest.raises(ValueError, match="decision D1 chose"):
        _sto(d, None)
    assert _sto(d, "A2") == "A2"
    # architecture_split compares runners within a variant they share, not tops from different variants
    rows = []
    for var, run, seed, rs in (("af3_msa", "af3", 1, 0.9), ("af3_msa", "bt2", 1, 0.5), ("colab", "bt2", 2, 0.8)):
        rows.append({"sto": "A3", "variant": var, "runner": run, "seed": seed, "sample": 0, "ranking_score": rs,
                     "iptm": rs, "ptm": rs, "plddt": 80, "has_clash": False, "path": "x", "rs_z_runner": 0})
    with open(tmp_path / "pool.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for var, names in (("af3_msa", [("af3_seed1_sample0", 0), ("bt2_seed1_sample0", 1)]), ("colab", [("bt2_seed2_sample0", 0)])):
        p = tmp_path / "qa" / "cluster_A3" / "A3" / var / "m" / "results"
        p.mkdir(parents=True)
        (p / "x_clustering.csv").write_text("structure,cluster\n" + "".join(f"{n},{c}\n" for n, c in names))
    num, trig = consult.checks(tmp_path / "pool.csv", tmp_path / "qa")
    assert any(t.startswith("architecture_split") and "af3_msa" in t for t in trig)
    assert num["top_cluster_A3"]["af3_msa"] == {"af3": "af3_msa:0", "bt2": "af3_msa:1"}


def test_slate_says_when_it_relaxed_its_rules(tmp_path):
    rows = [{"key": f"A1__v__af3__s{i}_0", "sto": "A1", "runner": "af3", "variant": "v", "seed": str(i), "cluster": None}
            for i in range(3)]
    picked: list = []
    slate._pick(rows, 3, picked, cap=2)
    assert [r.get("relaxed") for r in picked] == [None, None, "runner cap"]


def test_slate_rule_line_states_the_real_cap(models, tmp_path):
    pc = _pool(tmp_path / "pool.csv", models)
    s = slate.propose(pc, "A2")
    md = slate.write(s, tmp_path / "out", 1)[1]
    assert f"at most {s['cap']} per runner" in md.read_text()


def test_relaxed_slots_reach_the_analysis():
    from residudes import report
    d = Dossier(target="T6")
    d.evidence["slate_proposal"] = [{"time": "2026-11-22T01:00:00+09:00", "by": "agent", "value": {
        "version": 1, "models": ["A1__v__af3__s1_0", "A1__v__af3__s2_0"], "relaxed": {"A1__v__af3__s2_0": "runner cap"}}}]
    assert "relaxed for want of candidates: A1__v__af3__s2_0 (runner cap)" in report.build(d)


def test_sites_carry_the_source_they_came_from(models, tmp_path):
    """A site prior is a reading of the literature, and only checkable if it says which reading (docs/literature.md)."""
    assert gate.source_of("epitope on Q542/P543 (escape selection, J Virol 2024, PMC11323900)") == "PMC11323900"
    assert gate.source_of({"site": "A:542,543", "source": "escape mutants, J Virol 2024"}) == "escape mutants, J Virol 2024"
    assert gate.source_of({"site": "A:10-20", "pmid": "38916399"}) == "pmid: 38916399"
    assert gate.source_of("the spike core, 8UFO apo crystal") == "8UFO"
    assert gate.source_of("paratope within 6 A of A:611,615") is None            # residue lists are not citations
    assert gate.source_of({"site": "A:10", "note": "seen in 2024, 0.59 of the models"}) is None   # nor years or scores

    class Cfg:
        targets_dir = tmp_path / "targets"
    d = Dossier(target="T11")
    ddir = Dossier.path(Cfg.targets_dir, "T11").parent
    ddir.mkdir(parents=True)
    _pool(ddir / "pool.csv", models)
    d.evidence["epitope"] = [{"time": "t", "by": "agent", "value": "A:611,615 escape mutants"}]
    assert "no source recorded" in gate.draft_spec(d, "A2B1")["gates"][0]["note"]
    assert "site_without_source" in consult.review_cmd(Cfg, d, [], "agent")
    d.evidence["epitope"] = [{"time": "t", "by": "agent",
                              "value": {"site": "A:611,615", "source": "escape mutants, J Virol 2024"}}]
    assert "source escape mutants, J Virol 2024" in gate.draft_spec(d, "A2B1")["gates"][0]["note"]
    assert "site_without_source" not in consult.review_cmd(Cfg, d, [], "agent")


def test_report_renders_structure_and_a_bibliography(tmp_path):
    """The analysis is the document a person reads: nested evidence keeps its shape, nothing is cut mid-claim,
    and every source is collected with the priors that rest on it (docs/literature.md)."""
    from residudes import report
    d = Dossier(target="T12")
    d.evidence["construct"] = [{"time": "t", "by": "agent", "value": {
        "chains": [{"id": "A", "len": 228, "what": "Fab heavy chain Fd"}, {"id": "B", "len": 215, "what": "kappa LC"}],
        "source": ["UniProt P03951 feature table", "alignment done here"]}}]
    d.evidence["epitope"] = [{"time": "t", "by": "agent", "value": {
        "select": "C:92-175", "source": "escape data, Blood 2026, doi 10.1182/blood.2025032276"}}]
    d.evidence["literature"] = [{"time": "t", "by": "agent", "value": {
        "claims": [{"claim": "the antibody targets the A2 domain " + "x" * 400,
                    "source": "PMID 42275249; doi 10.1182/blood.2025032276"}]}}]
    d.evidence["identity"] = [{"time": "t", "by": "agent", "value": {"family": "peptidase S1"}}]
    out = report.build(d)
    assert "| id | len | what |" in out                       # like-shaped records read as a table
    assert "x" * 400 in out                                   # a long claim is not truncated
    srcs, unsourced = report.sources(d)
    assert unsourced == ["identity"]                          # no source recorded, and the report says so
    assert "**No source recorded**: `identity`" in out
    bib = {t: ks for t, ks in srcs}
    doi = [t for t in bib if "42275249" in t]                 # the DOI and the PMID are one paper, not two entries
    assert len(doi) == 1 and sorted(bib[doi[0]]) == ["epitope", "literature"]
    assert "UniProt P03951 feature table" in bib and "alignment done here" in bib   # a list of sources, both kept


def test_report_embeds_the_written_note(tmp_path):
    from residudes import report
    d = Dossier(target="T13")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "target.md").write_text("# T13 — the target\n\n## Chains\n\nChain A is the Fd.\n")
    out = report.build(d, False, tmp_path)
    assert "## The target, written up" in out and "### Chains" in out and "Chain A is the Fd." in out
    assert "## The target, written up" not in report.build(d)


def test_site_from_value_prefers_the_numbering_that_was_recorded():
    """H1388 recorded the pocket as residues [49, 56, 131, 132] in input numbering and mentioned the native numbers
    in the same sentence. Reading the text first took the native ones, so the drafted gate was a residue off."""
    from residudes.gate import site_from_value
    val = {"chain": "B", "residues": [49, 56, 131, 132],
           "what": "phosphate pocket of 14-3-3 gamma (input numbering; native K50 R57 R132 Y133)"}
    assert site_from_value(val) == ("B:49,56,131,132", False, "residues as recorded")
    assert site_from_value({"select": "A:542,543", "note": "Gln542 Pro543"})[0] == "A:542,543"
    assert site_from_value({"residues": [7], "what": "pS-X-P"}) == ("A:7", True, "residues as recorded")
    sel, guessed, how = site_from_value("Lys49 and Arg56 line the pocket")
    assert (sel, guessed, how) == ("A:49,56", True, "read from the text")


def test_chain_map_draws_the_recorded_sites_and_names_what_it_cannot_place(tmp_path):
    """The map is drawn from the dossier alone: chains from the FASTA, marks from the site evidence, read the way
    gate --init reads it. A residue past the end of its chain is native numbering, so it is named, not drawn."""
    from residudes import figures
    from residudes.dossier import Dossier
    d = Dossier(target="T9", sequence={"text": ">T9,peptide|\nPKGLSASLPDL\n>T9,protein|\n" + "GA" * 60 + "\n"})
    d.evidence["motif"] = [{"time": "2026-01-01T09:00:00+09:00", "by": "agent",
                            "value": {"chain": "A", "residues": [7], "what": "pS-X-P; LARP1 S548"}}]
    d.evidence["binding_site"] = [{"time": "2026-01-01T09:01:00+09:00", "by": "agent",
                                   "value": {"chain": "B", "residues": [49, 56], "what": "pocket (native K50 R57)"}}]
    d.evidence["epitope"] = [{"time": "2026-01-01T09:02:00+09:00", "by": "agent",
                              "value": {"chain": "B", "residues": [542]}}]           # past the end: native numbering
    letters = {"A": 11, "B": 120}
    marks, skipped = figures.sites(d, letters)
    assert ("A", 7, "motif") in marks and ("B", 49, "binding_site") in marks         # recorded numbering, not K50/R57
    assert ("B", 56, "binding_site") in marks and not any(r in (50, 57) for _, r, _ in marks)
    assert skipped == ["`epitope` has B:542, past residue 120 - native numbering?"]

    svg = figures.chain_map_svg(d, tmp_path / "chain_map.svg").read_text()
    assert ">S7<" in svg and ">G49<" in svg and ">A56<" in svg                       # labelled with the real residue
    assert "peptide" in svg and "11 aa" in svg and "120 aa" in svg
    assert "not placed:" in svg and "epitope" in svg


def test_report_and_map_survive_the_shapes_they_will_actually_meet(tmp_path):
    """Dossiers in flight are incomplete: no sequence yet, evidence recorded as a bare string or a list, a
    dossier directory that does not exist. None of it should raise - a report that crashes at NEW is a report
    nobody runs."""
    from residudes import figures, report
    from residudes.dossier import Dossier
    assert "# X: target analysis" in report.build(Dossier(target="X"))
    assert report.build(Dossier(target="X", sequence={"text": "MKT"}), False, "/nonexistent/dir")
    d = Dossier(target="X", sequence={"text": ">a|\nMKT\n>b|\nMM\n"})
    d.evidence["identity"] = [{"time": "2026-01-01T00:00:00+09:00", "by": "agent", "value": "a string, not a dict"}]
    d.evidence["msa_depth"] = [{"time": "2026-01-01T00:00:00+09:00", "by": "agent", "value": [1, 2, 3]}]
    assert "## What the target is" in report.build(d)
    assert figures.chain_map_svg(Dossier(target="X"), tmp_path / "m.svg") is None       # nothing to draw
    d2 = Dossier(target="X", sequence={"text": ">a|\nMKT\n"})
    d2.evidence["epitope"] = [{"time": "2026-01-01T00:00:00+09:00", "by": "agent", "value": {"chain": "Z", "residues": [2]}}]
    assert figures.sites(d2, {"A": 3})[1] == ["`epitope` is on chain Z, which this FASTA does not have"]


# ---------------------------------------------------------------------------------- the per-evidence figures
def _h1388(tmp_path):
    """A dossier shaped like H1388's: a peptide and a protein, a literature record with sources, a pocket, a
    motif, and the chemistry the FASTA does not carry."""
    from residudes.dossier import Dossier
    t = "2026-11-23T08:00:00+09:00"
    d = Dossier(target="H1388", sequence={"text": ">H1388,peptide|\nPKGLSASLPDL\n>H1388,protein|\n"
                                                  + "MKGDYYRYLAEVAAGDDKKGIVDQSQQAYQEAFEISKKEMQPTHPIRLGLALNFSVFYYEIL" * 2 + "\n"})
    d.evidence["literature"] = [{"time": t, "by": "agent", "value": {"claims": [
        {"claim": "14-3-3 binds phosphoserine motifs; mode I consensus R-S-X-pS-X-P",
         "source": "Yaffe et al. 1997 Cell, PMID 9428519"},
        {"claim": "the pocket is K49, R56 in this construct", "source": "PMID 9428519; 6A5S"},
        {"claim": "a claim nobody wrote a source for"}],
        "not_known": "no structure of this complex exists", "searched": "Europe PMC 2026-11-23"}}]
    d.evidence["binding_site"] = [{"time": t, "by": "agent", "value": {
        "chain": "B", "residues": [49, 56], "what": "phosphate pocket (input numbering; native K50 R57)",
        "source": "Yaffe et al. 1997 PMID 9428519; 6A5S"}}]
    d.evidence["motif"] = [{"time": t, "by": "agent", "value": {
        "chain": "A", "residues": [7], "what": "peptide S7 in pS-X-P (S7-L8-P9); S5 rejected (S-A-S, no Pro at +2)",
        "source": "UniProt Q6PKG0"}}]
    d.evidence["chemistry"] = [{"time": t, "by": "agent", "value": {
        "modification": "phosphoserine (CCD SEP) on chain A residue 7",
        "candidates": {"S7": "chosen: pS-X-P motif", "S5": "rejected: no Pro at +2"}}}]
    d.decide("agent", "variant", "SEP at A7", ["motif", "chemistry: PMID 9428519"])
    return d


def test_each_kind_of_evidence_gets_its_own_figure(tmp_path):
    """The point of splitting them: a claim, a site and a motif are different questions, and one picture of all
    three answers none of them. Each figure is drawn only from the evidence it is about."""
    from residudes import figures
    d = _h1388(tmp_path)
    drawn, problems = figures.draw_all(d, tmp_path)
    assert problems == []
    assert set(drawn) == {"chain_map", "literature", "sites", "motif"}       # no pool here, so no clustering
    lit = drawn["literature"].read_text()
    assert "PMID 9428519" in lit and "no source recorded" in lit             # the claim without one is marked
    assert "binding_site" in lit and "D1" in lit                 # a shared PMID links the claim to the decision
    assert "no structure of this complex exists" in lit                      # what the reading does not settle
    sites = drawn["sites"].read_text()
    assert ">K<" in sites and ">49<" in sites and ">56<" in sites            # the residues, with their letters
    assert "native K50 R57" in sites and "phosphoserine (CCD SEP)" in sites


def test_the_motif_figure_does_the_arithmetic_the_sentence_claims(tmp_path):
    """H1388's lever: S7 sits in pS-X-P and S5 does not. The figure scans both patterns against the peptide
    rather than trusting the sentence, and fills only the match the record is actually about."""
    from residudes import figures
    d = _h1388(tmp_path)
    ms = figures.motifs(d)
    # the paper's full consensus is a pattern too, and it is read from the claim that quotes it
    assert {m["token"] for m in ms} == {"pS-X-P", "S-A-S", "R-S-X-pS-X-P"}
    pat = {m["token"]: m["pattern"] for m in ms}
    assert figures.hits(pat["pS-X-P"], "PKGLSASLPDL") == [7]                 # only S7 has a proline at +2
    assert figures.hits(pat["S-A-S"], "PKGLSASLPDL") == [5]
    assert figures.candidates(d) == {"S7": "chosen: pS-X-P motif", "S5": "rejected: no Pro at +2"}
    svg = figures.motif_svg(d, tmp_path / "motif.svg").read_text()
    assert "pS-X-P" in svg and "S-A-S" in svg and "✓ S7" in svg and "✗ S5" in svg
    # S-A-S covers residue 7 but is not about it: its anchor is S5, so nothing in that row is filled
    assert "nothing is filled" in svg
    assert "Fits nowhere in any chain" in svg and "R-S-X-pS-X-P" in svg


def test_the_clustering_figure_reads_both_shapes_residudes_qa_writes(tmp_path):
    """One variant at a time, residudes-qa names the directory after the variant; several at once, it pools them
    and writes the variant into each model's name. Reading the directory in that case silently invents a variant
    called pool-<hash>, and then no model matches the pool or the slate."""
    from residudes import figures
    qa = tmp_path / "qa" / "cluster_A1B1" / "A1B1"
    one = qa / "af3_msa" / "tm" / "results"
    one.mkdir(parents=True)
    (one / "tm_clustering.csv").write_text("cluster,structure,umap1,umap2\n0,af3_seed1_sample0.pdb,1.0,2.0\n")
    many = qa / "pool-befc310a" / "tm" / "results"
    many.mkdir(parents=True)
    (many / "tm_clustering.csv").write_text("cluster,structure,umap1,umap2\n"
                                            "1,af3_seed2_sample1__af3_msa_pS548.pdb,3.0,4.0\n")
    (tmp_path / "pool.csv").write_text(
        "sto,variant,runner,seed,sample,ranking_score,iptm,ptm,plddt,has_clash,path\n"
        "A1B1,af3_msa,af3,1,0,0.77,0.70,0.8,90,0,/x\n"
        "A1B1,af3_msa_pS548,af3,2,1,0.96,0.94,0.9,95,0,/y\n")
    pts = figures.cluster_points(tmp_path)
    assert {p["variant"] for p in pts} == {"af3_msa", "af3_msa_pS548"}
    assert {p["key"] for p in pts} == {"A1B1__af3_msa__af3__s1_0", "A1B1__af3_msa_pS548__af3__s2_1"}
    scores = figures.pool_scores(tmp_path)
    assert scores["A1B1__af3_msa_pS548__af3__s2_1"]["rs"] == 0.96         # the point can be looked up in the pool
    svg = figures.clusters_svg(figures.Dossier(target="H1388"), tmp_path / "c.svg", ddir=tmp_path).read_text()
    assert "af3_msa_pS548" in svg and "(0.960)" in svg      # the variant and its model, in the ranked list
