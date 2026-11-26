"""The MSA and template pictures the deck shows before the models (views.py)."""
import json

import numpy as np
import pytest

from residudes import slides, views
from residudes.config import Config
from residudes.dossier import Dossier


def test_neff_counts_each_cluster_of_near_identical_sequences_once():
    codes = np.array([[0, 1, 2, 3, 4]] * 3 + [[9, 9, 9, 9, 9]], np.uint8)
    nf, est = views.neff(codes)
    assert nf == pytest.approx(2.0) and est is False          # three copies weigh 1/3 each, the odd one 1


def test_neff_above_the_exact_limit_is_an_unbiased_sample(monkeypatch):
    rng = np.random.default_rng(1)
    base = rng.integers(0, 20, (40, 30))                       # 40 families of near-copies, 3,200 rows
    codes = np.repeat(base, 80, 0)
    noise = rng.random(codes.shape) < 0.05
    codes = np.where(noise, rng.integers(0, 20, codes.shape), codes).astype(np.uint8)
    est, flagged = views.neff(codes)
    monkeypatch.setattr(views, "NEFF_EXACT_MAX", 10_000)
    exact, _ = views.neff(codes)
    assert flagged and abs(est - exact) / exact < 0.1


def _a3m(path, query, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">q\n{query}\n" + "".join(f">{h}\n{s}\n" for h, s in rows))


def test_the_msa_picture_splits_rows_by_source_and_names_a_lone_query(tmp_path):
    ddir = tmp_path / "T"
    q = "ACDEFGHIKLMNPQRSTVWY"
    msa = ddir / "msas" / "colabfold" / "msa"
    _a3m(msa / "unpaired_B.a3m", q, [("UniRef100_A1", q), ("UniRef100_A2", q[:10] + "-" * 10),
                                     ("MGYP0001", "Y" * 20), ("ERR123", q[:15] + "aa" + "WWWWW")])
    _a3m(msa / "unpaired_A.a3m", "PKGL", [])
    _a3m(msa / "_colabfold" / "uniref.a3m", q, [("UniRef100_X", q)])   # ColabFold's raw files: not a second MSA
    (msa / "templates_B.a3m").write_text(">q\nAAAA\n")
    d = Dossier(target="T")
    assert set(views.msa_files(ddir, d)) == {("colabfold", "A"), ("colabfold", "B")}
    out = ddir / "report" / "figs" / "msa_depth.svg"
    st = views.msa_figure(ddir, d, out)
    (panel,) = st["panels"]
    assert panel["chain"] == "B" and panel["n"] == 4
    assert [(g["name"], g["n"]) for g in panel["groups"]] == [("UniRef", 2), ("environmental", 2)]
    assert st["query_only"] == ["A (colabfold)"]
    assert "data:image/png;base64," in out.read_text() and "Neff" in out.read_text()
    out.write_text("drawn")                                    # unchanged files: the numbers are reused, not redrawn
    assert views.msa_figure(ddir, d, out)["panels"] == st["panels"] and out.read_text() == "drawn"


def _entry(path):
    """A 1ABC entry: template chain A (30 Ala), a peptide P lying on it, and C, a packed copy of A."""
    import gemmi
    lines, n = [], 0
    for ch, count, dy in (("A", 30, 0.0), ("P", 8, 4.0), ("C", 30, -4.0)):
        for i in range(count):
            n += 1
            lines.append(f"ATOM  {n:5d}  CA  ALA {ch}{i + 1:4d}    {i * 3.8:8.3f}{dy:8.3f}{0.0:8.3f}  1.00 30.00           C")
    pdb = path.with_suffix(".pdb")
    pdb.write_text("\n".join(lines) + "\nEND\n")
    st = gemmi.read_structure(str(pdb))
    st.setup_entities()
    st.make_mmcif_document().write_file(str(path))


def test_templates_are_the_inputs_then_close_hits_trimmed_to_what_they_bound(tmp_path):
    cfg = Config(work_root=str(tmp_path))
    d = Dossier(target="T")
    d.sequence["text"] = ">pep\nPKGLSASL\n>rec\n" + "A" * 30 + "\n"
    d.save(cfg.targets_dir)
    ddir = cfg.targets_dir / "T"
    (ddir / "inputs" / "sep").mkdir(parents=True)
    (ddir / "inputs" / "sep" / "input.json").write_text(json.dumps({"sequences": [
        {"protein": {"id": "B", "sequence": "A" * 30, "templates": [{"mmcifPath": "/x/templates/B/pin_1abc_A.cif"}]}}]}))
    (ddir / "msas" / "cf" / "msa").mkdir(parents=True)
    (ddir / "msas" / "cf" / "msa" / "templates_B.m8").write_text(
        "102\t9zzz_B\t0.30\t30\t0\t0\t1\t30\t1\t30\t1e-5\t50\t30M\n")        # below 50 %: not drawn
    cache = ddir / "dl" / "pdb_entries"
    cache.mkdir(parents=True)
    _entry(cache / "1ABC.cif")
    (t,) = views.template_entries(cfg, d, ddir)
    assert (t["pdb"], t["chain"], t["for"], t["used"]) == ("1ABC", "A", "B", ["sep"])
    assert t["identity"] == 1.0 and [p["chain"] for p in t["partners"]] == ["P"] and t["small"] == ["P"]
    assert "C" not in {c.name for c in __import__("gemmi").read_structure(t["path"])[0]}   # the packed copy is gone
    assert "1ABC A → B" in views.template_caption(t) and "+ P (8 aa)" in views.template_caption(t)

    mt = slides.scenes_mt([], (), None, [t | {"camera": "tpl_1ABC_A.cam"}])
    assert "rename 1ABC_A tpl_1ABC_A" in mt and "camera load tpl_1ABC_A.cam" in mt and "screenshot figs/tpl_1ABC_A.png" in mt
    pml = slides.scenes_pml([], (), [t])
    assert "def face" in pml and "face tpl_1ABC_A and chain P, tpl_1ABC_A and chain A, tpl_1ABC_A, 0" in pml


def test_the_deck_shows_the_msa_picture_and_the_templates(tmp_path):
    cfg = Config(work_root=str(tmp_path))
    d = Dossier(target="T")
    d.save(cfg.targets_dir)
    ddir = cfg.targets_dir / "T"
    figs = ddir / "report" / "figs"
    figs.mkdir(parents=True)
    (figs / "msa_depth.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="660"></svg>')
    (figs / "msa_depth.json").write_text(json.dumps({"query_only": ["A (cf)"], "neff_identity": 0.8, "panels": [
        {"chain": "B", "n": 10377, "neff": 5965.8, "neff_estimated": True,
         "groups": [{"name": "UniRef", "n": 3775, "neff": 1686.9, "neff_estimated": True}]}]}))
    (figs / "templates.json").write_text(json.dumps({"templates": [
        {"label": "tpl_6S9K_A", "pdb": "6S9K", "chain": "A", "for": "B", "used": ["sep"], "identity": 1.0,
         "resolution": 1.6, "partners": [{"chain": "B", "residues": 22, "ligands": []}]}]}))
    (figs / "tpl_6S9K_A.png").write_bytes(b"png")
    typ = slides.build_typ(d, ddir)
    assert "== MSA depth by source" in typ and "Neff ≈5,966" in typ and "UniRef 3,775 (Neff ≈1,687)" in typ
    assert "== Templates" in typ and "figs/tpl_6S9K_A.png" in typ and "6S9K A → B · 100% identity · 1.60 Å" in typ
