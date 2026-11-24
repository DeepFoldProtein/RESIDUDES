"""Tests for per-group confidence (residudes_qa.cluster.analysis.group_confidence)."""
import tempfile
from pathlib import Path

import numpy as np

from residudes_qa.cluster.analysis.group_confidence import (
    read_mean_plddt, interface_pdockq, summarize_by_group,
    format_group_table, confidence_from_paths, plot_confidence_by_group,
)


def _write_struct(path, chain_coords, bfac=80.0):
    import gemmi
    st = gemmi.Structure(); m = gemmi.Model("1")
    for cname, coords in zip("ABCDEFGH", chain_coords):
        ch = gemmi.Chain(cname)
        for i, (x, y, z) in enumerate(coords, 1):
            r = gemmi.Residue(); r.name = "ALA"; r.seqid = gemmi.SeqId(i, " ")
            a = gemmi.Atom(); a.name = "CA"; a.element = gemmi.Element("C")
            a.pos = gemmi.Position(float(x), float(y), float(z))
            a.b_iso = float(bfac); a.occ = 1.0
            r.add_atom(a); ch.add_residue(r)
        m.add_chain(ch)
    st.add_model(m); st.write_pdb(str(path))


def test_read_mean_plddt():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.pdb"
        _write_struct(p, [rng.normal(0, 10, (20, 3))], bfac=73.5)
        assert abs(read_mean_plddt(p) - 73.5) < 1e-3


def test_interface_pdockq_range():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 5, (20, 3))
    with tempfile.TemporaryDirectory() as d:
        # two chains in contact -> pDockQ in (0, 0.742]
        close = Path(d) / "close.pdb"
        _write_struct(close, [a, a + np.array([3., 0, 0])])
        v = interface_pdockq(close)
        assert 0.0 < v <= 0.7425
        # two chains far apart -> no contacts -> 0.0
        far = Path(d) / "far.pdb"
        _write_struct(far, [a, a + np.array([500., 0, 0])])
        assert interface_pdockq(far) == 0.0


def test_summarize_by_group():
    labels = np.array([0, 0, 1, 1, 1, -1])
    vals = {"plddt": np.array([80., 82., 60., 62., 64., 99.])}
    names = ["m0", "OURS_relaxed_a", "x", "OURS_relaxed_b", "y", "z"]
    df = summarize_by_group(labels, vals, group_name="cluster",
                            names=names, highlight_prefix="OURS_")
    assert set(df["cluster"]) == {0, 1}                       # -1 dropped
    g0 = df[df.cluster == 0].iloc[0]
    g1 = df[df.cluster == 1].iloc[0]
    assert g0["n"] == 2 and abs(g0["plddt_mean"] - 81.0) < 1e-6
    assert g1["n"] == 3 and abs(g1["plddt_mean"] - 62.0) < 1e-6
    assert g0["highlight"] == "relaxed_a"
    assert g1["highlight"] == "relaxed_b"


def test_format_group_table():
    labels = np.array([0, 0, 1, 1])
    vals = {"plddt": np.array([80., 82., 60., 62.])}
    df = summarize_by_group(labels, vals, group_name="community")
    txt = format_group_table(df, group_name="community", footer="modularity 0.65")
    assert "commu" in txt and "plddt" in txt and "modularity 0.65" in txt
    assert "0" in txt and "1" in txt


def test_confidence_from_paths_and_plot():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        paths = []
        for k in range(4):
            p = d / f"m{k}.pdb"
            _write_struct(p, [rng.normal(0, 5, (15, 3)),
                              rng.normal(0, 5, (15, 3)) + 3.0], bfac=70 + k)
            paths.append(p)
        vals = confidence_from_paths(paths, pdockq=True)
        assert "plddt" in vals and "pdockq" in vals and len(vals["plddt"]) == 4
        fig = plot_confidence_by_group(np.array([0, 0, 1, 1]), vals["plddt"],
                                       value_name="pLDDT", group_name="community",
                                       output_path=d / "v.png")
        assert (d / "v.png").exists()
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
