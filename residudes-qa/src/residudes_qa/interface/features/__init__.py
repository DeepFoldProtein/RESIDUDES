"""residudes_qa.interface.features — per-cif feature extraction modules.

Each module exposes a single-cif worker (``run_*``) and a parallel driver
(``run_*_parallel`` / ``run_*_sequential``) that returns a DataFrame keyed on
``model_id`` (matching the index built by :mod:`residudes_qa.interface.core.discover`).
"""
