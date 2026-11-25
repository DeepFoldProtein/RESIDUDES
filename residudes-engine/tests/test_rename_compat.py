"""Outputs written by earlier versions keep being read: the run-metadata folder has had other names,
and 1.3 M models on disk still carry them."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from residudes_engine.runners._common import META_DIR, META_DIRS, meta_of  # noqa: E402


class RenameCompatTests(unittest.TestCase):
    def test_outputs_written_under_earlier_names_are_found(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            older, old, new, empty = Path(t, "older"), Path(t, "old"), Path(t, "new"), Path(t, "empty")
            (older / ".foldmux").mkdir(parents=True)
            (old / ".lantern-engine").mkdir(parents=True)
            (new / META_DIR).mkdir(parents=True)
            empty.mkdir()
            self.assertEqual(meta_of(older).name, ".foldmux")
            self.assertEqual(meta_of(old).name, ".lantern-engine")
            self.assertEqual(meta_of(new).name, ".residudes-engine")
            self.assertEqual(meta_of(empty).name, ".residudes-engine")      # where a new run writes
            self.assertEqual(META_DIRS, (META_DIR, ".lantern-engine", ".foldmux"))

