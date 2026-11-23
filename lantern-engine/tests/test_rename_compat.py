"""foldmux -> lantern-engine (2026-11): what was written or set under the old name keeps working."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lantern_engine import cli  # noqa: E402
from lantern_engine.runners._common import META_DIR, META_DIRS, meta_of  # noqa: E402


class RenameCompatTests(unittest.TestCase):
    def test_outputs_written_as_foldmux_are_found(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            old, new, empty = Path(t, "old"), Path(t, "new"), Path(t, "empty")
            (old / ".foldmux").mkdir(parents=True)
            (new / META_DIR).mkdir(parents=True)
            empty.mkdir()
            self.assertEqual(meta_of(old).name, ".foldmux")
            self.assertEqual(meta_of(new).name, ".lantern-engine")
            self.assertEqual(meta_of(empty).name, ".lantern-engine")      # where a new run writes
            self.assertIn(".foldmux", META_DIRS)

    def test_foldmux_environment_variables_are_honoured(self) -> None:
        env = {"FOLDMUX_JAX_CUEQ": "1", "LANTERN_ENGINE_JAX_TOKAMAX": "0", "FOLDMUX_JAX_TOKAMAX": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(sys, "stdout"):
                cli.main(["models"])
            self.assertEqual(os.environ.get("LANTERN_ENGINE_JAX_CUEQ"), "1")      # mapped
            self.assertEqual(os.environ.get("LANTERN_ENGINE_JAX_TOKAMAX"), "0")   # the new name wins


if __name__ == "__main__":
    unittest.main()
