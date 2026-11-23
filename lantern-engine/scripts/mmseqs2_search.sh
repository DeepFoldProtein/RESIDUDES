#!/usr/bin/env bash
# Local MMseqs2 MSA search — the ColabFold recipe (UniRef30 + environmental
# DB, PDB100 templates) as run in the CASP17 pipeline (msas/mmseqs2_gpu), for
# one FASTA on this machine.
#
#   scripts/mmseqs2_search.sh <query.fasta> <work_dir>
#
# Environment:
#   MMSEQS         mmseqs binary (default: mmseqs on PATH; may be a wrapper such as
#                  "apptainer exec --nv image.sif mmseqs")
#   DB_DIR         directory holding the databases (required)
#   UNIREF_DB      default $DB_DIR/uniref30_2302_db
#   ENV_DB         default $DB_DIR/colabfold_envdb_202108_db
#   TEMPLATE_DB    default $DB_DIR/pdb100_230517
#   USE_ENV=1      also search the environmental DB and merge
#   USE_TEMPLATES=1  search PDB100 and write m8 template hits
#   USE_GPU=0      1 → --gpu 1 --prefilter-mode 1 (MMseqs2-GPU build and a GPU required)
#   FILTER=1       ColabFold diversity filter (0 → keep everything)
#   THREADS=8
#
# Output (work_dir): a3m/<name>.a3m per query record, m8/<name>.m8 template hits.
# Databases: see https://github.com/sokrypton/ColabFold (setup_databases.sh);
# the GPU prefilter needs the `_db` indexes built with `mmseqs makepaddedseqdb`.
set -euo pipefail

QUERY=${1:?query fasta}
WORK=${2:?work dir}
: "${MMSEQS:=mmseqs}"
: "${DB_DIR:?set DB_DIR to the MMseqs2 database directory}"
: "${UNIREF_DB:=$DB_DIR/uniref30_2302_db}"
: "${ENV_DB:=$DB_DIR/colabfold_envdb_202108_db}"
: "${TEMPLATE_DB:=$DB_DIR/pdb100_230517}"
: "${USE_ENV:=1}"
: "${USE_TEMPLATES:=1}"
: "${USE_GPU:=0}"
: "${FILTER:=1}"
: "${THREADS:=8}"

mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
QDB="$WORK/qdb"
OUT="$WORK/out"
TMP="$WORK/tmp"
mkdir -p "$OUT" "$TMP"

GPU_FLAGS=()
[ "$USE_GPU" = "1" ] && GPU_FLAGS=(--gpu 1 --prefilter-mode 1)
if [ "$FILTER" = "1" ]; then
    EXPAND_FILTER=1; ALIGN_MAX_ACCEPT=100000; FILT_QSC=0.8; FILTER_MSA=1
else
    EXPAND_FILTER=0; ALIGN_MAX_ACCEPT=1000000; FILT_QSC=-20.0; FILTER_MSA=0
fi
run() { echo ">>> $*" >&2; $MMSEQS "$@"; }

run createdb "$QUERY" "$QDB"

# ── UniRef30: iterative profile search, expand to cluster members, realign, filter
run search "$QDB" "$UNIREF_DB" "$OUT/res" "$TMP" --num-iterations 3 -e 0.1 --db-load-mode 2 \
    --max-seqs 10000 -a --k-score 'seq:96,prof:80' --threads "$THREADS" "${GPU_FLAGS[@]}"
for ext in "" .index .dbtype; do cp "$TMP/latest/profile_1$ext" "$OUT/prof_res$ext"; done
cp "$TMP/latest/profile_1.lookup" "$OUT/prof_res.lookup" 2>/dev/null || true
for ext in "" .index .dbtype; do cp "${QDB}_h$ext" "$OUT/prof_res_h$ext"; done
run expandaln "$QDB" "${UNIREF_DB}.idx" "$OUT/res" "${UNIREF_DB}.idx" "$OUT/res_exp" --db-load-mode 2 \
    --expansion-mode 0 -e inf --expand-filter-clusters "$EXPAND_FILTER" --max-seq-id 0.95 --threads "$THREADS"
run align "$OUT/prof_res" "${UNIREF_DB}.idx" "$OUT/res_exp" "$OUT/res_exp_realign" --db-load-mode 2 -e 10 \
    --max-accept "$ALIGN_MAX_ACCEPT" --alt-ali 10 -a --threads "$THREADS"
run filterresult "$QDB" "${UNIREF_DB}.idx" "$OUT/res_exp_realign" "$OUT/res_filt" --db-load-mode 2 --qid 0 \
    --qsc "$FILT_QSC" --diff 0 --max-seq-id 1.0 --filter-min-enable 100 --threads "$THREADS"
run result2msa "$QDB" "${UNIREF_DB}.idx" "$OUT/res_filt" "$OUT/uniref.a3m" --msa-format-mode 6 --db-load-mode 2 \
    --filter-msa "$FILTER_MSA" --filter-min-enable 1000 --diff 3000 --qid 0.0,0.2,0.4,0.6,0.8,1.0 --qsc 0 \
    --max-seq-id 0.95 --threads "$THREADS"
for d in res res_exp res_exp_realign res_filt; do run rmdb "$OUT/$d"; done

# ── environmental DB with the UniRef profile
if [ "$USE_ENV" = "1" ]; then
    run search "$OUT/prof_res" "$ENV_DB" "$OUT/res_env" "$TMP" --num-iterations 3 -e 0.1 --db-load-mode 2 \
        --max-seqs 10000 -a --k-score 'seq:96,prof:80' --threads "$THREADS" "${GPU_FLAGS[@]}"
    run expandaln "$OUT/prof_res" "${ENV_DB}.idx" "$OUT/res_env" "${ENV_DB}.idx" "$OUT/res_env_exp" \
        --db-load-mode 2 --expansion-mode 0 -e inf --threads "$THREADS"
    run align "$TMP/latest/profile_1" "${ENV_DB}.idx" "$OUT/res_env_exp" "$OUT/res_env_realign" --db-load-mode 2 \
        -e 10 --max-accept "$ALIGN_MAX_ACCEPT" --alt-ali 10 -a --threads "$THREADS"
    run filterresult "$QDB" "${ENV_DB}.idx" "$OUT/res_env_realign" "$OUT/res_env_filt" --db-load-mode 2 --qid 0 \
        --qsc "$FILT_QSC" --diff 0 --max-seq-id 1.0 --filter-min-enable 100 --threads "$THREADS"
    run result2msa "$QDB" "${ENV_DB}.idx" "$OUT/res_env_filt" "$OUT/env.a3m" --msa-format-mode 6 --db-load-mode 2 \
        --filter-msa "$FILTER_MSA" --filter-min-enable 1000 --diff 3000 --qid 0.0,0.2,0.4,0.6,0.8,1.0 --qsc 0 \
        --max-seq-id 0.95 --threads "$THREADS"
    for d in res_env res_env_exp res_env_realign res_env_filt; do run rmdb "$OUT/$d"; done
    run mergedbs "$QDB" "$OUT/merged.a3m" "$OUT/uniref.a3m" "$OUT/env.a3m"
    FINAL="$OUT/merged.a3m"
else
    FINAL="$OUT/uniref.a3m"
fi

# ── PDB100 templates (m8 with cigar, same columns as the ColabFold server)
if [ "$USE_TEMPLATES" = "1" ]; then
    run search "$OUT/prof_res" "$TEMPLATE_DB" "$OUT/res_pdb" "$TMP" --db-load-mode 2 -s 7.5 -a -e 0.1 \
        --threads "$THREADS" "${GPU_FLAGS[@]}"
    run convertalis "$OUT/prof_res" "${TEMPLATE_DB}.idx" "$OUT/res_pdb" "$OUT/pdb.m8" \
        --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar \
        --db-output 1 --db-load-mode 2 --threads "$THREADS"
    run rmdb "$OUT/res_pdb"
fi

# ── unpack per query record (named by FASTA header)
mkdir -p "$WORK/a3m"
cp "$QDB.lookup" "$FINAL.lookup"
run unpackdb "$FINAL" "$WORK/a3m" --unpack-suffix .a3m --unpack-name-mode 1 --threads "$THREADS"
if [ "$USE_TEMPLATES" = "1" ]; then
    mkdir -p "$WORK/m8"
    cp "$QDB.lookup" "$OUT/pdb.m8.lookup"
    run unpackdb "$OUT/pdb.m8" "$WORK/m8" --unpack-suffix .m8 --unpack-name-mode 1 --threads "$THREADS"
fi
rm -rf "$TMP"
echo "done: $WORK/a3m" >&2
