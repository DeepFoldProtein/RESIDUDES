"""Make `pytest tests` work from this package without installing it.

`bin/residudes-qa` puts src/ on PYTHONPATH and runs the CLI from the uv venv, so the package is
never installed into it. The tests are the one caller that would otherwise have to know that —
including the CLI tests, which spawn `python -m residudes_qa.cluster.cli` and so need the path in
the environment, not only in this process.
"""
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if SRC not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join([SRC, *filter(None, [os.environ.get("PYTHONPATH")])])
