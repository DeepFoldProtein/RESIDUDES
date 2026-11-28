"""bin/residudes-qa on the image route: with RESIDUDES_QA_IMAGE set it hands the call to apptainer, binding the top
directory of the working directory, of absolute argument paths and of the paths this tool's own variables name.

A fake `apptainer` on PATH prints what it was given, so no image is needed."""
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "bin" / "residudes-qa"


def run(tmp_path, **env):
    fake = tmp_path / "fakebin"
    fake.mkdir(exist_ok=True)
    (fake / "apptainer").write_text('#!/bin/sh\necho "apptainer $*"\n')
    (fake / "apptainer").chmod(0o755)
    base = {"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path)}
    return subprocess.run(["bash", str(LAUNCHER), "--help"], cwd=tmp_path, env={**base, **env},
                          capture_output=True, text=True, timeout=60)


def test_image_alone_runs_the_image(tmp_path):
    # the usual case: RESIDUDES_QA_IMAGE is the only RESIDUDES_QA_* variable. The filter that drops the image from
    # the paths to bind then has nothing left, and its exit status 1 used to end the launcher silently (set -e)
    r = run(tmp_path, RESIDUDES_QA_IMAGE="/images/residudes-qa-cu130.sif")
    assert r.returncode == 0, r.stderr
    assert "apptainer run --nv" in r.stdout and "/images/residudes-qa-cu130.sif --help" in r.stdout
    top = "/" + str(tmp_path).split("/")[1]
    assert f"-B {top}" in r.stdout


def test_paths_named_by_its_variables_are_bound(tmp_path):
    r = run(tmp_path, RESIDUDES_QA_IMAGE="/images/residudes-qa-cu130.sif", RESIDUDES_QA_SOURCE_TREE="/srv/casp/tree")
    assert r.returncode == 0, r.stderr
    binds = r.stdout.split("-B ", 1)[1].split()[0].split(",")
    assert "/srv" in binds and "/images" not in binds
