#!/usr/bin/env python3
"""Thin wrapper around :func:`lantern_qa.interface.cli.main` for checkouts without `uv sync`/install.

    .venv/bin/python scripts/interface/run_scoring.py run-all --target-dir T --sto A2 --mode ppi -o OUT
(equivalent: `lantern-qa xgb run-all ...`)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from lantern_qa.interface.cli import main  # noqa: E402

main(sys.argv[1:])
