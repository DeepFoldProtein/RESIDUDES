"""Tests for homo-oligomer chain canonicalization (residudes_qa.cluster.analysis.chain_canon)."""
import tempfile
from pathlib import Path

import numpy as np

from residudes_qa.cluster.analysis.chain_canon import (
    kabsch_tm, permutation_parity, best_chain_permutation,
    chain_ca_by_resnum, canonicalize_directory,
)


def _monomer(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 10, (n, 3))


def _write_trimer(path, chain_coords, bfac=80.0):
    """Write a 3-chain CA-only PDB with the given per-chain coords + B-factor."""
    import gemmi
    st = gemmi.Structure()
    m = gemmi.Model("1")
    for cname, coords in zip("ABC", chain_coords):
        ch = gemmi.Chain(cname)
        for i, (x, y, z) in enumerate(coords, 1):
            r = gemmi.Residue()
            r.name = "ALA"
            r.seqid = gemmi.SeqId(i, " ")
            a = gemmi.Atom()
            a.name = "CA"
            a.element = gemmi.Element("C")
            a.pos = gemmi.Position(float(x), float(y), float(z))
            a.b_iso = float(bfac)
            a.occ = 1.0
            r.add_atom(a)
            ch.add_residue(r)
        m.add_chain(ch)
    st.add_model(m)
    st.write_pdb(str(path))


def test_kabsch_tm_identity():
    P = _monomer(40)
    assert kabsch_tm(P, P.copy(), len(P)) > 0.999


def test_permutation_parity():
    base = ["A", "B", "C"]
    assert permutation_parity(("A", "B", "C"), base) == 0          # identity
    assert permutation_parity(("B", "C", "A"), base) == 0          # cyclic
    assert permutation_parity(("C", "A", "B"), base) == 0          # cyclic
    assert permutation_parity(("B", "A", "C"), base) == 1          # transposition
    assert permutation_parity(("A", "C", "B"), base) == 1          # transposition


def test_best_permutation_recovers_relabel():
    """A cyclically-relabelled copy must map back to the reference frame (TM~1)."""
    mono = _monomer(30)
    t1, t2 = np.array([40., 0, 0]), np.array([0, 40., 0])
    A, B, C = mono, mono + t1, mono + t2                # asymmetric assembly
    ref = {"A": dict(enumerate(A.tolist(), 1)),
           "B": dict(enumerate(B.tolist(), 1)),
           "C": dict(enumerate(C.tolist(), 1))}
    # model: chains relabelled (named A,B,C but holding B,C,A coords)
    model = {"A": dict(enumerate(B.tolist(), 1)),
             "B": dict(enumerate(C.tolist(), 1)),
             "C": dict(enumerate(A.tolist(), 1))}
    perm, tm = best_chain_permutation(model, ref)
    assert tm > 0.95                                    # recovered alignment
    assert perm == ("C", "A", "B")                      # maps model-C(=A coords)->slot A
    assert permutation_parity(perm, sorted(model)) == 0  # cyclic, not a transposition


def test_best_permutation_chain_count_mismatch():
    ref = {"A": {1: (0, 0, 0)}, "B": {1: (1, 0, 0)}, "C": {1: (0, 1, 0)}}
    model = {"A": {1: (0, 0, 0)}, "B": {1: (1, 0, 0)}}   # only 2 chains
    perm, tm = best_chain_permutation(model, ref)
    assert perm is None and tm is None


def test_canonicalize_directory_end_to_end():
    mono = _monomer(30)
    t1, t2 = np.array([40., 0, 0]), np.array([0, 40., 0])
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        inp = d / "in"
        out = d / "out"
        inp.mkdir()
        # ref-frame model + a chain-permuted copy of the SAME structure
        _write_trimer(inp / "m1.pdb", [mono, mono + t1, mono + t2])
        _write_trimer(inp / "m2.pdb", [mono + t1, mono + t2, mono])   # relabelled
        _write_trimer(inp / "m3.pdb", [mono + t2, mono, mono + t1])   # relabelled
        mdf = canonicalize_directory(inp, out, ref=str(inp / "m1.pdb"), workers=1)
        assert (mdf.status == "ok").all() and len(mdf) == 3
        assert len(list(out.glob("*.pdb"))) == 3
        # after canonicalization all three must be byte-equivalent in CA frame:
        import gemmi
        frames = []
        for f in sorted(out.glob("*.pdb")):
            ca = chain_ca_by_resnum(gemmi.read_pdb(str(f))[0])
            frames.append(np.array([ca["A"][i] for i in sorted(ca["A"])]))
        # chain A of every canonicalized model should be the same physical subunit
        assert np.allclose(frames[0], frames[1], atol=1e-2)
        assert np.allclose(frames[0], frames[2], atol=1e-2)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
