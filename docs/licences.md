# Licences

**[MIT](../LICENSE)** for this repository's own code and documentation — residudes, residudes-engine, residudes-qa, the
mascot scripts and the docs.

Two things it does **not** cover, and both matter before you reuse anything:

- **Vendored upstream code**, which keeps the licence its authors gave it:
  `residudes-engine/core/models/alphafold3/` (CC BY-NC-SA 4.0 — **non-commercial**),
  `residudes-engine/core/models/protenix/` and `…/openfold-3/` (Apache-2.0), ProteinMPNN and MassiveClustering under
  `residudes-qa/licenses/` (MIT). Model weights and databases are not in this repository and carry their own terms;
  the Dunbrack rotamer libraries need an academic licence and are not included. The three harness marks in the
  overview figures belong to their owners and say which harnesses read the same instructions, nothing more; their
  paths come from [simple-icons](https://simpleicons.org) (CC0-1.0) and Wikimedia Commons (public domain).
- **The [residudes-decoy](residudes-decoy.md) dataset, which is licensed separately from this repository.** The
  MIT licence here says nothing about it. Each structure keeps the terms of the predictor that made it — the
  194,183 AlphaFold 3 structures are non-commercial and **may not be used to train structure-prediction, scoring
  or ranking models**, the other 531,914 are Apache-2.0 or MIT — and the release's own annotations (index,
  metadata, input specifications, manifests) are covered by the `README.md` and `TERMS.md` distributed with the
  data, not by this file.

An archived release with a DOI would still be worth having, so that work building on this can point at something
stable beyond the abstract.

The data behind the reported numbers is listed in
[`residudes/docs/casp17_cases.md`](../residudes/docs/casp17_cases.md).
