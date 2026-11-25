# Examples

## OpenFold3 / Protenix

Both models use the same OF3 query JSON format.

### `ubiquitin.json`
Minimal single-chain protein query (no MSA/template paths). Use with:
```bash
bash openfold3/run_inference.sh \
    --query examples/openfold3/ubiquitin.json \
    --checkpoint "$OPENFOLD3_CKPT" \
    --engines-dir "$ENGINES_DIR" \
    --output-dir results/ubiquitin
```

### `ubiquitin.a3m`
Minimal pre-computed MSA in `.a3m` format (5 ubiquitin homologs).
**Note:** when using `--msa-a3m`, the file is staged as `colabfold_main.a3m`
so OF3's MSA parser (which filters by filename stem) picks it up.

### `1ubq.cif`
PDB structure 1UBQ — ubiquitin crystal structure (76 residues). Used as a template CIF example matching the ubiquitin query sequence.

### `ubiquitin_with_msa_template.json`
Query with `main_msa_file_paths` and `template_entry_chain_ids` pre-filled.
Use directly if running from the core/ directory:
```bash
bash openfold3/run_inference.sh \
    --query examples/openfold3/ubiquitin_with_msa_template.json \
    --checkpoint "$OPENFOLD3_CKPT" \
    --engines-dir "$ENGINES_DIR" \
    --output-dir results/ubiquitin_msa_template \
    --no-msa
```

Or use CLI flags with the plain query:
```bash
bash openfold3/run_inference.sh \
    --query examples/openfold3/ubiquitin.json \
    --checkpoint "$OPENFOLD3_CKPT" \
    --engines-dir "$ENGINES_DIR" \
    --output-dir results/ubiquitin_msa_template \
    --no-msa \
    --msa-a3m examples/openfold3/ubiquitin.a3m \
    --template-cif examples/openfold3/1ubq.cif
```

### `multimer.json`
Two-protein heterodimer (PDB 7CNX).

### `protein_ligand.json`
Protein + small molecule ligand query.

---

## Boltz-2

Boltz-2 uses YAML format (version 1).

### `ubiquitin.yaml`
Single-chain ubiquitin query.

### `multimer.yaml`
Two-protein complex.
