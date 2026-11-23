#!/bin/bash
# Parity check: `lantern-qa visualize` vs the CASP17 af3-viz container on the same test target.
#
# Builds WORK_DIR/T1417 with links to targets/T1417/results/A3/af3_msa/{af3 seed 1-2, pt2 seed 1}
# (the real target is never written to), runs the original analyze_target.sh and lantern-qa on it,
# and compares the produced file sets and every CSV.
#
#   sbatch -p normal -c 4 --wrap "bash tests/parity/visualize_t1417.sh /store/deepfold/lantern-qa-parity/viz"
set -uo pipefail
LQA="${LQA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WORK="${1:-/store/deepfold/lantern-qa-parity/viz}"
SRC=/gpfs/deepfold/casp/casp17/targets/T1417/results/A3/af3_msa
W="$WORK/T1417"
mkdir -p "$W/results/A3/af3_msa/af3" "$W/results/A3/af3_msa/pt2"
for s in 1 2; do ln -sfn "$SRC/af3/seed_$s" "$W/results/A3/af3_msa/af3/seed_$s"; done
ln -sfn "$SRC/pt2/seed_1" "$W/results/A3/af3_msa/pt2/seed_1"
cd "$WORK"
rm -rf "$W/visualization" "$W/visualization_orig"
# the container binds only the target dir; the links point into /gpfs
APPTAINER_BIND=/gpfs,/store /gpfs/deepfold/casp/casp17/visualization/analyze_target.sh --skip-msa "$W" > orig.log 2>&1
echo "orig rc=$?"
mv "$W/visualization" "$W/visualization_orig"
"$LQA/bin/lantern-qa" visualize --no-msa "$W" > new.log 2>&1
echo "new rc=$?"
"$LQA/.venv/bin/python" - "$W" <<'PY'
import pathlib, sys
import pandas as pd
W = pathlib.Path(sys.argv[1])
o, n = W / "visualization_orig", W / "visualization"
fo = sorted(p.relative_to(o) for p in o.rglob("*") if p.is_file())
fn = sorted(p.relative_to(n) for p in n.rglob("*") if p.is_file())
print("files orig", len(fo), "new", len(fn), "same set:", fo == fn)
ok = fo == fn
for rel in [f for f in fo if f.suffix == ".csv"]:
    a, b = pd.read_csv(o / rel), pd.read_csv(n / rel)
    num = a.select_dtypes("number").columns
    eq = a.shape == b.shape and a.drop(columns=num).equals(b.drop(columns=num)) and \
        (((a[num] - b[num]).abs().max().max() if len(num) else 0) < 1e-9)
    ok &= bool(eq)
    print(f"{str(rel):55s} rows={len(a):4d}  identical={eq}")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
