---
name: residudes-structure-view
description: Use when a conversation turns on a shape — show the model, interface, gate site or ligand with molterm instead of describing it
---

# Showing a structure

A target is a shape, and a sentence about a shape is a poor substitute for the shape. When the
conversation is about a model, an interface, a site or a ligand pose, render it. The tools are
molterm's, through its MCP server (`fetch`, `render_pdb`, `screenshot`, `select`, `count`,
`measure`, `align`, `dssp`, `run_script`), with the `molterm` command line as the fallback.

The session's `.mcp.json` is written by `residudes chat`; if the tools are not there, say so once
and use the command line.

## When to show one

- **a model you are discussing** — the slate's first pick, an outlier, the one that fails a gate
- **an interface** — which chains touch, and where
- **a site that became a gate** — the residues the gate names, in place
- **a ligand or a modified residue** — where it sits, and what it contacts
- **a comparison** — two models aligned, with the RMSD that follows

Not for a routine progress line. A picture per message is noise; a picture at the moment a shape
decides something is the message.

## The command line

```sh
molterm --no-tui --script view.mt model.cif          # batch: no alt-screen, exits 1 if a command failed
```

`view.mt` is one command per line, without the leading colon:

```
show cartoon
color chain
orient
screenshot out.png
```

Useful lines, with their real syntax (the errors are explicit when it is wrong):

| what | line |
|---|---|
| secondary-structure colouring | `color secondary` |
| by chain, by rainbow, by SASA | `color chain` · `color rainbow` · `color sasa` |
| inter-chain contacts | `interface on` (then `interface legend`) |
| H-bonds, salt bridges, disulfides | `hbonds` · `saltbridge` · `disulfide` |
| solvent accessibility | `sasa` (total, per chain, mean relative) |
| secondary structure from coordinates | `dssp` |
| a view along the molecule's own axes | `orient` · `orient view 1 0 0` |
| the picture | `screenshot out.png` |

**Check the exit code.** A failed command leaves the previous state and the script carries on, so
a screenshot taken after a failed `color` is written with the old colouring. `molterm --no-tui
--script f.mt m.cif || echo "the figure is not trustworthy"` is the difference between a figure
and a figure-shaped file.

## For the report

Two renderers, for two moments. **While the run goes on** - the decks for the checkpoints and the slate - pictures
are molterm's, drawn in seconds. **The final deck**, after approval, is ray-traced by PyMOL: `residudes slides T
--final`. molterm rasterises (flat shading, no anti-aliasing, no shadows); it is the quick look, not the report.

### What makes a picture readable

These come from a deck figure that failed all of them (H1388, five predictors' peptides on 14-3-3):

- **One object per model.** Superpose models as separate files, never merged into one: secondary structure is
  computed per object, and copies that overlap in one file pair up into β-bridges that do not exist, drawn as
  flat slabs.
- **Sticks for the residue the slide is about,** not every atom of every model. Five overlapping peptides in full
  ball-and-stick are a tangle; their backbones as tubes, with the phosphoserine and the pocket residues as sticks,
  answer the question.
- **Look at a small chain from outside its partner.** `orient` looks along the selection's shortest axis, which
  for a peptide in a groove usually puts the receptor between the camera and the peptide. In PyMOL use `face`
  (below); in molterm, `turn y 180` until it is in front, or `camera load view.cam` (scenes run from `report/`),
  which `residudes slides` writes for the deck's own small chains.
- **Context, quietly.** The partner in light grey cartoon, not hidden: without it, "the peptide sits 7 Å off" has
  nothing to sit on.
- **Read the PNG before it goes in.**

### While the run goes on: molterm

The deck's own figures are written by `residudes slides T` into `report/scenes.mt`. When you write a scene of your
own, copy that file's `set` block rather than the defaults:

```
set lod_medium 400000      # full geometry: the default drops detail above 5,000 atoms, and every complex is bigger
set lod_low 400000
set bg white               # the default is transparent, which the JPEG step would flatten onto black
set sm relative            # a view tuned at any size renders identically at another
set csd 24                 # cartoon spline subdivisions (default 14)
set outline on
screenshot figs/x.png 1800 1350 300        # 6 x 4.5 in at 300 dpi; up to 8192 x 8192
```

A picture for the conversation does not need any of this - the defaults are made for a terminal, and are faster.

### The final deck: PyMOL, ray-traced

`residudes slides T --final` writes `report/scenes.pml`, ray-traces it and every `report/pml/*.pml` in one sbatch
job, and waits. Write each of your report scenes there under the name of its molterm figure, so the ray-traced
picture replaces it. Start from the deck's settings, and use `face` for a peptide or ligand on its partner
(`report/scenes.pml` defines it after a final build; copy the `python ... python end` block):

```
reinitialize
bg_color white
set ray_opaque_background, 1
set antialias, 2
set ray_trace_mode, 1
set ray_trace_color, 0x404040
set ray_shadows, 1
set ambient_occlusion_mode, 1
set specular, 0.25
set depth_cue, 1
set fog_start, 0.55
set cartoon_fancy_helices, 1
set stick_radius, 0.22
load /abs/path/af3_s20.cif, af3_pS
load /abs/path/pt2_s5.cif, pt2
super pt2 and name CA, af3_pS and name CA
hide everything
show cartoon
color 0xDADADA, chain B
show sticks, chain B and resi 49+56+131+132 and not name N+C+O
show sticks, chain A and resi 7 and not name N+C+O
face chain A, chain B, chain A, 6          # look at A from outside B, zoom on A with 6 A around it
png figs/pep_split.png, width=1800, height=1350, dpi=300, ray=1
```

Never run `pymol` yourself; the command guard refuses it, because the login node is not for ray tracing.

## In a conversation

In `residudes chat` the person is looking at your tmux window. Show the structure in a pane beside your own and
drive it as the conversation goes; they ask in words, you run these:

```sh
residudes view T show M1            # M1-M5 the slate, C0.. a cluster, tpl_6S9K_A a template, slate, or a file
residudes view T show M1 M3         # several: superposed on the first by sequence, one colour each
residudes view T do "color plddt"   # any molterm command in the open pane; "a; b" for several
residudes view T snap               # a PNG of what the pane shows: read it, so you see what they see
residudes view T close              # when they are done
```

"show" again replaces what the pane shows rather than opening another. The pane starts slim, a thin ribbon with a
small chain in ball-and-stick, because a terminal has few dots to spend; molterm's representations are `ribbon`,
`cartoon`, `ballstick`, `wireframe`, `spacefill` and `backbone` (there is no `sticks`). Useful `do` lines: `color plddt`,
`color chain`, `focus chain A and resi 7`, `show ballstick chain B and resi 49+56+131+132`, `align M3 to M1 automap`,
`turn y 90`, `zoom chain A`, `hbonds`, `interface on`. The pane answers on its last line, an error included - read
what `do` returns. `$sele` is cleared after each `do` (the live pane rings every selected atom), so put a `select`
and what uses it in one call, or name it: `select pocket = chain B and resi 49+56+131+132`.

When `show` says the pane is a text renderer (a tmux that cannot pass pixels), a character is two dots: frame the
pocket with the helices around it rather than one residue, and take fine detail from `snap`, which is always pixels.

Then say what the picture shows and what follows from it — the figure is the evidence, the sentence is the claim.
Record the claim with `residudes note` or `residudes evidence` as usual; a picture that is never referred to again
did not need taking.

On tmux older than 3.4 the pane draws with text (molterm's block renderer): tmux cannot pass pixels through. `snap`
is always drawn in pixels, so look at that before describing a detail. Outside tmux there is no pane: render a
PNG with `molterm --no-tui --script` into report/figs and describe it.
