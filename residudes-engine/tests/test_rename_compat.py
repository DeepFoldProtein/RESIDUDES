"""foldmux / Bio-TRT -> lantern-engine -> residudes-engine (2026-11): what was written or set under an
earlier name keeps working."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from residudes_engine import cli  # noqa: E402
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

    def test_earlier_environment_variables_are_honoured(self) -> None:
        env = {"FOLDMUX_JAX_CUEQ": "1", "RESIDUDES_ENGINE_JAX_TOKAMAX": "0", "FOLDMUX_JAX_TOKAMAX": "1",
               "LANTERN_ENGINE_JAX_PAIRFORMER_SCAN": "1", "BIO_TRT_JAX_ATTENTION_IMPL": "xla"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(sys, "stdout"):
                cli.main(["models"])
            self.assertEqual(os.environ.get("RESIDUDES_ENGINE_JAX_CUEQ"), "1")      # mapped
            self.assertEqual(os.environ.get("RESIDUDES_ENGINE_JAX_TOKAMAX"), "0")   # the new name wins
            self.assertEqual(os.environ.get("RESIDUDES_ENGINE_JAX_PAIRFORMER_SCAN"), "1")   # lantern-engine too
            self.assertEqual(os.environ.get("RESIDUDES_ENGINE_JAX_ATTENTION_IMPL"), "xla")   # and the BIO_TRT_ names


if __name__ == "__main__":
    unittest.main()
