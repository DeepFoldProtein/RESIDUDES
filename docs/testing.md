# Testing and continuous integration

GitHub Actions ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)) lints and tests residudes and
residudes-engine on every pull request, runs residudes-qa's clustering suite, checks every documentation link, and
measures the text in every figure; none of that needs a GPU. What CI cannot run is residudes-qa's scorers and the
parity checks against the CASP17 tools, which need the CUDA stack and the team's data: those run on the cluster
(`sbatch residudes-qa/tests/cluster/run_tests.sbatch`).

The diagrams are drawn by [`docs/figures/make_figures.py`](figures/make_figures.py) and the ones under each
tool's `docs/figures/`. They are hand-laid SVG, so [`docs/svg_text_overlap.py`](svg_text_overlap.py) measures
every label with the font the figure asks for and fails on text that collides, spills out of its box or runs off
the page — run it after regenerating a figure:

```sh
python3 docs/svg_text_overlap.py $(git ls-files '*.svg')
```
