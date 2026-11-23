"""The report deck is generated Typst: compile one with every kind of section and awkward text in it.
Skipped where typst is not installed (CI installs it)."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from lantern import slides
from lantern.dossier import Dossier

AWKWARD = 'a #hash, $dollar$, [brackets], "quotes", back\\slash, *star*, _under_, <angle>, @at, ≥ 60% · Å, 한글, `tick`'


@pytest.mark.skipif(not shutil.which("typst"), reason="typst not installed")
def test_the_deck_compiles_with_every_section(tmp_path):
    d = Dossier(target="T9999")
    for k in ("construct", "identity", "public_structures", "epitope", "stoichiometry_evidence"):
        d.evidence[k] = [{"time": "2026-11-22T01:00:00", "by": "agent", "value": {"note": AWKWARD, "n": 3}}]
    d.evidence["results_review"] = [{"time": "2026-11-22T02:00:00", "by": "agent",
                                     "value": {"checks": {}, "triggers": [f"site_split: {AWKWARD}"]}}]
    d.evidence["gates"] = [{"time": "2026-11-22T03:00:00", "by": "agent", "value": {
        "name": "epi", "sto": "A2B1", "n": 10, "n_pass": 4, "gates": {"complete": 10, "epitope": 4},
        "per_runner": {"af3": {"n": 10, "pass": 4, "best_pass": ["A2B1__msa__af3__s1_0", 0.8]}},
        "census": {"af3": {"bulge": 6, "other": 4}}, "incomplete": {},
        "spec": {"gates": [{"name": "epitope", "type": "contact", "site": "A:611,615", "partner": "C", "within": 6.0}],
                 "census": {"probe": "ligand", "sites": {"bulge": "B:1-12"}}}}}]
    d.decide("agent", "stoichiometry", AWKWARD, ["epitope", "construct"], alternatives=[AWKWARD], confidence="medium")
    c = d.consult("agent", "plan", [AWKWARD], "brief.md")
    d.reply("person", c["id"], AWKWARD)
    ddir = tmp_path / "T9999"
    (ddir / "report").mkdir(parents=True)
    (ddir / "report" / "report.typ").write_text(slides.build_typ(d, ddir))
    r = subprocess.run(["typst", "compile", "report.typ", "report.pdf"], cwd=ddir / "report", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (ddir / "report" / "report.pdf").stat().st_size > 10_000
    json.dumps(d.__dict__, default=str)
