# Provenance: post-run edits, what the run could see, and one fix mid-run

## What was edited after the run

Paths, everywhere they appear (logs, `dossier.json`, `analysis.md`, notes, inputs, QA tables), are written as
placeholders so that no one site's layout ships with the example:

| placeholder | what it stood for |
|---|---|
| `<WORK>` | the run's work root (dossiers under `<WORK>/targets/`) |
| `<RESIDUDES>` | the checkout the run used |
| `<AF3_DB>` · `<COLABFOLD_DB>` · `<FOLDSEEK_DB>` · `<FOLDSEEK_PDB_2026>` | the sequence and structure databases |
| `<CHECKPOINTS>` · `<ENGINES>` | model weights and the TensorRT engines |
| `<CONTAINERS>` · `<CASP17_UTILS>` · `<RELAX>` · `<PYMOL>` · `<KALIGN>` · `<CONDA>` | tools |
| `<SABDAB>` · `<CASP17>` · `<SITE>` · `<STORE>` · `~` | other site paths |

`analysis.md` and the figures were then regenerated from the edited dossier, so the example reproduces itself.
Not shipped: the 300 model files and the MSAs (1.2 GB; `pool.csv` lists every model with its scores), the
templates' coordinate files, and duplicate tables (parquet beside csv, the TM matrix).

## What the run was given, and what it could not be shielded from

The configuration named the databases and tools and nothing else: no existing predictions (`targets_root` empty,
so every model was predicted for this run) and not the team's CASP17 tree, which holds the old analysis and the
submission. Nothing told the agent to avoid anything; it was simply not pointed at it.

What no configuration can remove: the `residudes-intake` skill was written after CASP17 and quotes this target's
lesson as a worked case. The agent said so in its first step, and twice later that it had not used the line as
evidence. How much that shaped the plan is in [`retrospective.md`](retrospective.md).

## One fix, mid-run

The run stopped at its first checkpoint. The agent answered C1 under the mandate, ran `residudes resolve`, and was
refused by the command guard and the project settings, both of which blocked approve and resolve without looking
at the mandate. It wrote NEEDS HUMAN, as it should, and the daemon stepped it twice more before stopping - a second
bug, since fixed: an open checkpoint used to override the note.

Both were fixed in the code, a person's note on the dossier said so (02:07, by `deepfold`), and the daemon was
restarted. The agent resolved C1 itself on its next step; its answer from before the refusal stood. Steps 5 to 7
in [`steps.json`](../steps.json), $0.74, are the cost of that.

## The final deck, drawn after the run

The run built every deck with `--no-render`, as AGENTS.md then told it to, so the deck it left had no pictures
of the models. The two figures the agent drew itself were molterm's: rasterised without anti-aliasing, five
predictors' peptides merged into one file (so secondary structure paired the overlapping copies into β-slabs that
do not exist), every atom as ball-and-stick, viewed from behind the receptor.

After the run, `residudes slides H1388 --final` drew the deck the way a run now ends: PyMOL, ray-traced, in one
sbatch job (29 figures, 8 min 43 s with the queue). The agent's two scenes were rewritten as PyMOL scenes in
[`report/pml/`](../report/pml/) by a person, following the guidance that now ships in residudes-structure-view;
its molterm scenes stay in [`report/mt/`](../report/mt/). Its caption on the peptide slide said the receptor was
hidden, and now says grey. The deck's pictures ship at 1200 px (rendered at 1800 × 1350); `report.pdf` embeds those.
