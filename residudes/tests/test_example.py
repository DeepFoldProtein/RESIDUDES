"""The shipped H1388 example has to reproduce itself.

`residudes report` on the example's own dossier must write the analysis.md that ships beside it.
That is the example's whole claim — this is the document the pipeline produces from this record —
and it is also the broadest regression test there is: the stage machine, the evidence renderers,
the pool, the clustering and every figure, over a real campaign dossier rather than a fixture.
"""
import shutil
from pathlib import Path

import pytest

from residudes import report
from residudes.dossier import Dossier

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "H1388"


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    work = tmp_path_factory.mktemp("example")
    shutil.copytree(EXAMPLE, work / "targets" / "H1388")
    d = Dossier.load(work / "targets", "H1388")
    ddir = work / "targets" / "H1388"
    return report.run(d, ddir / "analysis_regenerated.md", ddir=ddir), ddir


def test_the_example_regenerates_its_own_analysis(regenerated):
    _out, ddir = regenerated
    got = (ddir / "analysis_regenerated.md").read_text()
    assert got == (EXAMPLE / "analysis.md").read_text(), (
        "residudes report no longer writes the analysis.md that ships with the example: "
        "regenerate it (residudes report H1388 -o analysis.md) if the change is intended")


def test_the_example_draws_every_figure_it_links(regenerated):
    out, ddir = regenerated
    assert "not drawn" not in out
    shipped = (EXAMPLE / "analysis.md").read_text()
    for name in ("chain_map", "literature", "sites", "motif", "clusters"):
        assert f"figs/{name}.svg" in shipped, f"the analysis no longer links {name}.svg"
        assert (ddir / "report" / "figs" / f"{name}.svg").is_file(), f"{name}.svg was not drawn"
    # the pool and its clustering come from the two tables the example ships
    assert "300 A1B1 models" in (ddir / "report" / "figs" / "clusters.svg").read_text()


def test_setup_reports_the_picture_tools(monkeypatch):
    """A missing USalign is the quiet kind of missing: molterm carries on and draws an unaligned overlay.
    Setup should say so before a deck is built, not after someone reads the figure."""
    from residudes import config, setup

    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/" + name if name == "typst" else None)
    rows = {c["name"]: c for c in setup._picture_tools(config.Config())}
    assert rows["typst"]["status"] == "ok"
    assert rows["molterm"]["status"] == "missing" and rows["USalign"]["status"] == "missing"
    assert "USalign" in rows["USalign"]["fix"]
    assert rows["pymol"]["status"] == "note"                  # the fallback: absent is not a problem

    cfg = config.Config()
    cfg.extra = {"tools": {"molterm": "/opt/molterm"}}        # [tools] in residudes.toml wins over PATH
    assert {c["name"]: c["path"] for c in setup._picture_tools(cfg)}["molterm"] == "/opt/molterm"
