#!/usr/bin/env bash
# Check a RESIDUDES setup end to end: the images, the commands inside them, and the databases they
# are given. Every check prints one line; the script exits non-zero if any of them failed.
#
#   containers/verify.sh                      the images in ./images, databases from the site file
#   containers/verify.sh --build              build what is missing first (apptainer, no root)
#   containers/verify.sh --pull               take the published images instead (ghcr.io)
#   containers/verify.sh --af3-db /db --deep  check the databases there, and run one jackhmmer
#   containers/verify.sh --gpu                also the four GPU images, each against the card
#   containers/verify.sh --all                those plus residudes-relax
#   containers/verify.sh --quick              skip the test suites
#
# What it deliberately does not do: submit anything. Schedulers belong to the host, and a setup
# check that queues jobs is a setup check nobody runs twice.
set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
IMAGES="${IMAGES:-$REPO/images}"
GPU=0
ALL=0
AF3_DB="${RESIDUDES_ENGINE_AF3_DB_DIR:-}"
REGISTRY="${REGISTRY:-ghcr.io/deepfoldprotein}"
TAG="${TAG:-latest}"
BUILD=0 PULL=0 DEEP=0 QUICK=0
pass=0 fail=0 skip=0

while [ $# -gt 0 ]; do
    case "$1" in
        --images) IMAGES=$2; shift 2 ;;
        --af3-db) AF3_DB=$2; shift 2 ;;
        --build)  BUILD=1; shift ;;
        --pull)   PULL=1; shift ;;
        --deep)   DEEP=1; shift ;;
        --gpu)    GPU=1; shift ;;
        --all)    ALL=1; shift ;;
        --quick)  QUICK=1; shift ;;
        -h|--help) sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail + 1)); }
note() { printf '  --    %s\n' "$1"; skip=$((skip + 1)); }
head_() { printf '\n%s\n' "$1"; }

run() {   # run <description> <command…>: ok when it exits 0
    local what=$1; shift
    if out=$("$@" 2>&1); then ok "$what"; else bad "$what"; printf '%s\n' "$out" | tail -4 | sed 's/^/        /'; fi
}

# ── the host ────────────────────────────────────────────────────────────────────────────────
head_ "host"
for t in apptainer uv python3 tmux; do          # tmux: `residudes chat` holds its sessions in it
    if command -v "$t" >/dev/null; then ok "$t  $(command -v "$t")"; else bad "$t not found"; fi
done
command -v sbatch >/dev/null && ok "sbatch (jobs run outside the containers)" \
                             || note "no sbatch: this host cannot submit, only run"

# Is this image actually pullable, by the person running this script?
#
# Publishing an image does not make it public: GHCR marks a package private on first push, so a
# pull from outside the organisation fails although CI uploaded it. Only the registry knows, so
# the script asks it, and reports the address it asked about. GHCR answers the anonymous token
# request distinctly, which is what the codes below mean:
#
#   public     a token is issued and the manifest is served     -> pull it
#   needs-auth 401 at the token endpoint: it exists, privately   -> pull only with credentials
#   absent     403 or 404: nothing anonymous can reach           -> build it here
#   offline    no answer at all                                  -> build it here
registry_status() {
    _name=$1
    _host=${REGISTRY%%/*}                       # ghcr.io/deepfoldprotein -> ghcr.io
    _owner=${REGISTRY#*/}                       #                         -> deepfoldprotein
    command -v curl >/dev/null || { echo offline; return; }
    _body=$(curl -s --max-time 15 -w '\n%{http_code}' \
        "https://$_host/token?scope=repository:$_owner/$_name:pull&service=$_host" 2>/dev/null)
    _code=${_body##*$'\n'}
    case "$_code" in
        000|"") echo offline; return ;;
        401)    echo needs-auth; return ;;
        403|404) echo absent; return ;;
    esac
    _tok=$(printf '%s' "$_body" | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
    [ -n "$_tok" ] || { echo absent; return; }
    _m=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
        -H "Authorization: Bearer $_tok" \
        -H 'Accept: application/vnd.oci.image.index.v1+json' \
        -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
        -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
        "https://$_host/v2/$_owner/$_name/manifests/$TAG" 2>/dev/null)
    case "$_m" in
        200) echo public ;;
        401|403) echo needs-auth ;;
        404) echo absent ;;
        000) echo offline ;;
        *) echo "http$_m" ;;
    esac
}

# Are there registry credentials on this machine? A private image is still pullable with them.
have_registry_credentials() {
    [ -n "${APPTAINER_DOCKER_USERNAME:-}${SINGULARITY_DOCKER_USERNAME:-}${GHCR_TOKEN:-}" ] && return 0
    [ -s "$HOME/.docker/config.json" ] && grep -q "${REGISTRY%%/*}" "$HOME/.docker/config.json" 2>/dev/null
}

# ── images ──────────────────────────────────────────────────────────────────────────────────
head_ "images in $IMAGES"
mkdir -p "$IMAGES"
CORE="residudes residudes-qa residudes-msa"
GPU_IMAGES="residudes-qa-cu130 residudes-engine-trt residudes-engine-jax residudes-ef2"
# residudes-relax is CPU, but 2 GB of conda and only useful with the pipeline mounted
EXTRA="residudes-relax"
SET="$CORE"
[ "$GPU" = 1 ] && SET="$SET $GPU_IMAGES"
[ "$ALL" = 1 ] && SET="$CORE $GPU_IMAGES $EXTRA"
for name in $SET; do
    sif="$IMAGES/$name.sif"
    # with neither --pull nor --build, ask the registry and decide from the answer
    if [ ! -f "$sif" ] && [ "$PULL" = 0 ] && [ "$BUILD" = 0 ]; then
        case "$(registry_status "$name")" in
            public)     note "$REGISTRY/$name:$TAG is public: pulling"; PULL_ONE=1 ;;
            needs-auth) if have_registry_credentials; then
                            note "$REGISTRY/$name:$TAG is private: pulling with your credentials"; PULL_ONE=1
                        else
                            note "$REGISTRY/$name:$TAG is private and you have no credentials for ${REGISTRY%%/*}: building"; BUILD_ONE=1
                        fi ;;
            absent)     note "$REGISTRY/$name:$TAG is not published: building"; BUILD_ONE=1 ;;
            offline)    note "${REGISTRY%%/*} unreachable: building $name"; BUILD_ONE=1 ;;
            *)          note "unexpected answer from ${REGISTRY%%/*} for $name: building"; BUILD_ONE=1 ;;
        esac
    else
        PULL_ONE=0; BUILD_ONE=0
    fi
    if [ -f "$sif" ]; then
        ok "$name.sif  $(du -hL "$sif" | cut -f1)"
    elif [ "$PULL" = 1 ] || [ "${PULL_ONE:-0}" = 1 ]; then
        printf '  ..    pulling %s\n' "$name"
        # keep the output: a failed pull used to print nothing but "could not pull", which says
        # neither whether the registry refused it nor whether the disk or the cache is at fault
        log=$(mktemp)
        if apptainer build --force "$sif" "docker://$REGISTRY/$name:$TAG" >"$log" 2>&1; then
            ok "$name.sif  pulled from $REGISTRY  $(du -hL "$sif" | cut -f1)"
        # The large images are one big layer each, and a multi-GB blob over HTTP/2 can die with
        #   conveyor failed to get: error writing layer: stream error: ... PROTOCOL_ERROR
        # which is the transport, not the image. Go disables HTTP/2 with this variable, so the
        # retry asks for HTTP/1.1 rather than simply trying the same thing again.
        elif grep -q 'PROTOCOL_ERROR\|stream error\|unexpected EOF\|connection reset' "$log" &&
             printf '  ..    %s\n' "retrying $name over HTTP/1.1" &&
             GODEBUG=http2client=0 apptainer build --force "$sif" "docker://$REGISTRY/$name:$TAG" >>"$log" 2>&1; then
            ok "$name.sif  pulled over HTTP/1.1  $(du -hL "$sif" | cut -f1)"
        else
            # A pull that cannot finish is not a reason to have no image: the recipe is here, and
            # building fetches many small wheels instead of one 9-11 GB layer, which is what fails.
            note "$name could not be pulled ($(sed -n '$p' "$log" | tail -c 60)); building it instead"
            printf '  ..    building %s\n' "$name"
            if (cd "$REPO/containers/$name" && apptainer build --force "$sif" "$name.def" >>"$log" 2>&1); then
                ok "$name.sif  built after the pull failed  $(du -hL "$sif" | cut -f1)"
            else
                bad "$name.sif  neither pulled nor built"
                sed -n '$p' "$log" | sed 's/^/          /'
                printf '          full log: %s\n' "$log"
                log=""                             # keep it: the message above points at it
            fi
        fi
        [ -n "$log" ] && rm -f "$log" 2>/dev/null || true
    elif [ "$BUILD" = 1 ] || [ "${BUILD_ONE:-0}" = 1 ]; then
        printf '  ..    building %s\n' "$name"
        blog=$(mktemp)
        if (cd "$REPO/containers/$name" && apptainer build --force "$sif" "$name.def" >"$blog" 2>&1); then
            ok "$name.sif  built  $(du -hL "$sif" | cut -f1)"
        else
            bad "$name.sif  build failed"
            sed -n '$p' "$blog" | sed 's/^/          /'
            printf '          full log: %s\n' "$blog"
        fi
    else
        bad "$name.sif missing (--build, or apptainer build from containers/$name)"
    fi
    PULL_ONE=0; BUILD_ONE=0
done

# ── the agent image, on real work ───────────────────────────────────────────────────────────
head_ "agent"
AGENT="$IMAGES/residudes.sif"
if [ -f "$AGENT" ]; then
    run "residudes --version" apptainer run -B "$REPO/residudes:/opt/residudes" "$AGENT" --version
    if [ "$QUICK" = 0 ]; then
        work=$(mktemp -d)
        mkdir -p "$work/targets"
        cp -r "$REPO/residudes/examples/H1388" "$work/targets/"
        printf '[paths]\nwork_root = "/work"\n' > "$work/residudes.toml"
        if apptainer run -B "$REPO/residudes:/opt/residudes" -B "$work:/work" "$AGENT" \
               --config /work/residudes.toml report H1388 -o /work/out.md >/dev/null 2>&1 \
           && cmp -s "$work/out.md" "$REPO/residudes/examples/H1388/analysis.md"; then
            ok "report H1388 reproduces the shipped analysis.md, figures and all"
        else
            bad "report H1388 did not reproduce the shipped analysis.md"
        fi
        rm -rf "$work"

        # The picture tools, exercised rather than looked up. `molterm-mcp` installs the viewer into its own
        # npm vendor directory, so the MCP server found it while a shell did not, and a suite that only asked
        # "is the package installed" passed while `residudes slides` had no renderer. So: draw something.
        work=$(mktemp -d)
        cat > "$work/tiny.pdb" <<'PDB'
ATOM      1  N   ALA A   1      -0.966   0.493   1.500  1.00 80.00           N
ATOM      2  CA  ALA A   1       0.257   0.418   0.692  1.00 80.00           C
ATOM      3  C   ALA A   1      -0.094   0.017  -0.716  1.00 80.00           C
ATOM      4  O   ALA A   1      -1.056  -0.682  -0.923  1.00 80.00           O
ATOM      5  CB  ALA A   1       1.204  -0.620   1.296  1.00 80.00           C
ATOM      6  N   ALA A   2       0.716   0.464  -1.673  1.00 80.00           N
ATOM      7  CA  ALA A   2       0.480   0.231  -3.090  1.00 80.00           C
ATOM      8  C   ALA A   2       1.332  -0.917  -3.605  1.00 80.00           C
ATOM      9  O   ALA A   2       2.335  -1.278  -2.987  1.00 80.00           O
ATOM     10  CB  ALA A   2       0.764   1.495  -3.882  1.00 80.00           C
END
PDB
        printf 'show cartoon\ncolor chain\norient\nscreenshot %s\n' "/work/out.png" > "$work/v.mt"
        if apptainer exec -B "$work:/work" "$AGENT" molterm --no-tui --script /work/v.mt /work/tiny.pdb \
               >"$work/mt.log" 2>&1 && [ -s "$work/out.png" ] \
           && head -c4 "$work/out.png" | grep -q PNG; then
            ok "molterm renders a structure inside the image ($(wc -c <"$work/out.png") bytes of PNG)"
        else
            bad "molterm did not render in the image: $(tail -1 "$work/mt.log" 2>/dev/null)"
        fi
        if apptainer exec "$AGENT" sh -c 'command -v typst >/dev/null && command -v tmux >/dev/null' 2>/dev/null; then
            ok "typst and tmux in the image (the deck, and the chat sessions)"
        else
            bad "typst or tmux missing from the agent image"
        fi
        # Every molterm superposition shells out to USalign, and a missing one does not stop the script: the
        # overlay is then drawn from unaligned models and looks exactly like an aligned one.
        if apptainer exec "$AGENT" sh -c 'USalign -h >/dev/null 2>&1'; then
            ok "USalign in the image (molterm superposes through it)"
        else
            bad "USalign missing: molterm's align/super/alignto all fail, and overlays are drawn unaligned"
        fi
        rm -rf "$work"
    fi
else
    note "agent image missing: skipped"
fi

# ── the QA image ────────────────────────────────────────────────────────────────────────────
head_ "qa"
QA="$IMAGES/residudes-qa.sif"
if [ -f "$QA" ]; then
    run "residudes-qa doctor (CPU image: torch absent by design)" \
        sh -c "apptainer run -B '$REPO/residudes-qa:/opt/residudes-qa' '$QA' doctor | grep -q 'pinned packages'"
    if [ "$QUICK" = 0 ]; then
        run "residudes-qa: 92 tests" apptainer exec -B "$REPO/residudes-qa:/opt/residudes-qa" "$QA" \
            python -m pytest -q /opt/residudes-qa/tests -p no:cacheprovider
    fi
else
    note "qa image missing: skipped"
fi

# ── the MSA image and the databases ─────────────────────────────────────────────────────────
head_ "msa and databases"
MSA="$IMAGES/residudes-msa.sif"
if [ -f "$MSA" ]; then
    run "hmmer, mmseqs and kalign in the image" apptainer exec "$MSA" sh -c \
        'for b in jackhmmer nhmmer hmmsearch hmmbuild hmmalign hmmscan kalign mmseqs; do command -v $b >/dev/null || exit 1; done'
    if [ -n "$AF3_DB" ]; then
        args=(-B "$REPO/residudes-engine:/opt/residudes-engine" -B "$AF3_DB:/db"
              -B "$REPO/residudes-engine/sites:/sites"
              --env RESIDUDES_ENGINE_SITE_FILE=/sites/local.example.toml --env RESIDUDES_ENGINE_AF3_DB_DIR=/db)
        [ "$DEEP" = 1 ] && deep=--deep || deep=
        if out=$(apptainer run "${args[@]}" "$MSA" doctor $deep 2>&1); then :; fi
        if printf '%s' "$out" | grep -q '✗ .*\.fa'; then
            bad "AF3 databases under $AF3_DB"
            printf '%s\n' "$out" | grep '✗' | sed 's/^/        /'
        else
            ok "AF3 databases under $AF3_DB  ($(printf '%s' "$out" | grep -c '^    ✓ ') of 9 present)"
        fi
        [ "$DEEP" = 1 ] && { printf '%s' "$out" | grep -q 'deep check *✓' \
            && ok "jackhmmer runs against a real database" || bad "jackhmmer deep check"; }
    else
        note "no --af3-db given: databases not checked"
    fi
else
    note "msa image missing: skipped"
fi

# ── the GPU images, against the card ────────────────────────────────────────────────────────
if [ "$GPU" = 1 ] || [ "$ALL" = 1 ]; then
    head_ "the GPU images on this machine's card"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        note "no nvidia-smi: these four only mean anything where there is a GPU"
    else
        card=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        ok "card: ${card:-unknown}"

        run_gpu() {   # name, label, python
            [ -f "$IMAGES/$1.sif" ] || { note "$1.sif not here"; return; }
            if out=$(apptainer exec --nv "$IMAGES/$1.sif" "$2" -c "$3" 2>&1 | tail -1); then
                ok "$1: $out"
            else
                bad "$1: $out"
            fi
        }
        # torch has to see the card, not merely import
        run_gpu residudes-qa-cu130 python 'import torch; assert torch.cuda.is_available(); x=torch.randn(512,512,device="cuda"); print("torch", torch.__version__, "| matmul ok", bool((x@x.T).sum()!=0))'
        # and TensorRT has to build, which is what fails when image and driver disagree
        run_gpu residudes-engine-trt python 'import torch,tensorrt as trt; assert torch.cuda.is_available(); print("tensorrt", trt.__version__, "| builder", trt.Builder(trt.Logger(trt.Logger.ERROR)) is not None)'
        run_gpu residudes-engine-jax /opt/venv/bin/python 'import jax; d=jax.devices(); print("jax", jax.__version__, "| devices", [x.platform for x in d])'
        run_gpu residudes-ef2 /opt/venv/bin/python 'import torch; from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| ESMFold2Model ok")'
    fi
fi

# ── the relaxation tools ────────────────────────────────────────────────────────────────────
if [ "$ALL" = 1 ] && [ -f "$IMAGES/residudes-relax.sif" ]; then
    head_ "relaxation tools"
    v=$(apptainer exec "$IMAGES/residudes-relax.sif" gmx --version 2>/dev/null | grep -m1 'GROMACS version' | sed 's/.*: *//')
    [ -n "$v" ] && ok "gmx $v" || bad "gmx did not run"
    missing=
    for tool in tleap antechamber parmchk2 reduce; do
        apptainer exec "$IMAGES/residudes-relax.sif" sh -c "command -v $tool >/dev/null" 2>/dev/null || missing="$missing $tool"
    done
    [ -z "$missing" ] && ok "AmberTools: tleap antechamber parmchk2 reduce" || bad "AmberTools missing:$missing"
    # the pipeline itself is not in the image, by design - say where it has to come from
    # the pipeline is a site's own, so ask the configuration rather than guessing a path
    pipeline=${MINIMIZATION:-$(sed -n 's/^minimization *= *"\(.*\)"/\1/p' \
        "${RESIDUDES_CONFIG:-$REPO/residudes/residudes.toml}" 2>/dev/null | head -1)}
    if [ -n "$pipeline" ] && [ -x "$pipeline" ]; then
        ok "pipeline present to mount: $pipeline"
    else
        note "relax pipeline not configured here (set [tools] minimization in residudes.toml)"
    fi
fi

# With the engine images present, a host venv is optional - the images ARE the runtime. Only
# a setup that has neither is broken, so the verdict depends on which path this machine took.
run_engine_doctor() {
    local images_present=0
    [ -f "$IMAGES/residudes-engine-trt.sif" ] && [ -f "$IMAGES/residudes-engine-jax.sif" ] && images_present=1
    local out rc
    out=$("$REPO/residudes-engine/bin/residudes-engine" doctor 2>&1); rc=$?
    if [ $rc -eq 0 ]; then
        ok "residudes-engine doctor (host venvs)"
    elif [ "$images_present" = 1 ]; then
        note "no host venvs, and none needed: the engine images are here (predict through them)"
    else
        bad "residudes-engine doctor, and no engine images either"
        printf '%s\n' "$out" | sed -n '1,8p' | sed 's/^/        /'
    fi
}

# ── the host's own engine setup ─────────────────────────────────────────────────────────────
head_ "engine on this host"
if [ -x "$REPO/residudes-engine/bin/residudes-engine" ]; then
    run_engine_doctor
else
    note "no engine checkout here"
fi

printf '\n%s passed, %s failed, %s skipped\n' "$pass" "$fail" "$skip"
if [ "$fail" != 0 ]; then
    printf 'to fix these with an agent on this machine: %s/residudes/bin/residudes setup\n' "$REPO"
    printf 'or by hand: %s/docs/setup.md\n' "$REPO"
fi
[ "$fail" = 0 ]
