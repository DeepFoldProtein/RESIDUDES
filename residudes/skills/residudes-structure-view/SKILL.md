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

## In a conversation

Render into the target's dossier, next to everything else about it:

```sh
molterm --no-tui --script view.mt "$DDIR/models/model_1.cif"   # screenshot <ddir>/report/figs/...
```

Then say what the picture shows and what follows from it — the figure is the evidence, the
sentence is the claim. Record the claim with `residudes note` or `residudes evidence` as usual;
a picture that is never referred to again did not need taking.
