"""GPU-free tests for the MSA/inputs layer (parsers, A3M handling, JSON assembly)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from residudes_engine import inputs, msa  # noqa: E402


class FastaAndSto(unittest.TestCase):
    def test_parse_fasta_and_sto(self) -> None:
        recs = msa.parse_fasta(">h1 desc\nMKT\nAYI\n>h2\nGGG\n")
        self.assertEqual(recs, [("h1", "MKTAYI"), ("h2", "GGG")])
        self.assertEqual(msa.parse_sto("A2B1", 2), [2, 1])
        self.assertEqual(msa.parse_sto(None, 3), [1, 1, 1])
        with self.assertRaises(msa.MsaError):
            msa.parse_sto("C2", 2)
        with self.assertRaises(msa.MsaError):
            msa.parse_sto("A1A2", 2)

    def test_chains_from_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "q.fasta"
            p.write_text(">alpha\nMKTAYI\n>beta\nGGGS\n")
            chains = msa.chains_from_fasta(p, "A3B1")
        self.assertEqual([(c.letter, c.name, c.copies) for c in chains], [("A", "alpha", 3), ("B", "beta", 1)])


class A3m(unittest.TestCase):
    def test_normalise_puts_query_first_and_dedupes(self) -> None:
        text = ">x\nMKTA\n>y\nMKTA\n>z\nMK-A\n"
        out = msa.normalise_a3m(text, "MKTA", "q")
        self.assertEqual(out.splitlines(), [">q", "MKTA", ">z", "MK-A"])
        self.assertEqual(msa.a3m_depth(out), 2)

    def test_merge_a3ms(self) -> None:
        out = msa.merge_a3ms([">a\nMKTA\n>b\nMK-A\n", ">c\nMK-A\n>d\nMKTT\n"], "MKTA", "q")
        self.assertEqual(out.splitlines(), [">q", "MKTA", ">b", "MK-A", ">d", "MKTT"])

    def test_server_a3m_split(self) -> None:
        text = ">101\nMKTA\n>hit\nMKTT\n\x00>102\nGGGS\n>h2\nGGGA\n"
        blocks = msa._split_server_a3m(text, 2)
        self.assertTrue(blocks[0].startswith(">101") and "MKTT" in blocks[0])
        self.assertTrue(blocks[1].startswith(">102") and "GGGA" in blocks[1])


class Specs(unittest.TestCase):
    def test_ligand(self) -> None:
        self.assertEqual(inputs.parse_ligand_spec("ATP=ccd:ATP"), {"name": "ATP", "count": 1, "ccdCodes": ["ATP"]})
        self.assertEqual(inputs.parse_ligand_spec("MG*3=ccd:MG")["count"], 3)
        self.assertEqual(inputs.parse_ligand_spec("lig=smi:CCO")["smiles"], "CCO")
        with self.assertRaises(inputs.InputError):
            inputs.parse_ligand_spec("x=foo:bar")

    def test_glycan(self) -> None:
        g = inputs.parse_glycan_spec("A29:NAG-NAG-BMA")
        self.assertEqual((g["chain"], g["pos"], g["ccds"], g["prot_atom"], g["inter_atom"]), ("A", 29, ["NAG", "NAG", "BMA"], None, "O4"))
        g = inputs.parse_glycan_spec("B150:A2G@OG1/O3")
        self.assertEqual((g["prot_atom"], g["inter_atom"]), ("OG1", "O3"))

    def test_contact(self) -> None:
        c = inputs.parse_contact_spec("B:104.CA=D:166@max=5,min=1")
        self.assertEqual((c["chain1"], c["pos1"], c["atom1"], c["chain2"], c["pos2"], c["max_distance"], c["min_distance"]),
                         ("B", 104, "CA", "D", 166, 5.0, 1.0))

    def test_multimer_template_spec(self) -> None:
        cif, pairs = inputs.parse_multimer_template_spec("8yiv.cif:C=A,D=B,E=C")
        self.assertEqual((str(cif), pairs), ("8yiv.cif", [("C", "A"), ("D", "B"), ("E", "C")]))
        cif, pairs = inputs.parse_multimer_template_spec("/data/x/8yiv.cif:C=A,D=B")
        self.assertEqual(str(cif), "/data/x/8yiv.cif")


class AssemblyJson(unittest.TestCase):
    def test_ids_msas_glycan_contact(self) -> None:
        chains = [msa.Chain("A", "heavy", "MKTAYIAKQRN", 2), msa.Chain("B", "light", "GGGSTT", 1)]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "unpaired_A.a3m").write_text(">heavy\nMKTAYIAKQRN\n")
            (d / "paired_A.a3m").write_text(">heavy\nMKTAYIAKQRN\n")
            asm = inputs.Assembly(name="t", chains=chains, unpaired={"A": d / "unpaired_A.a3m"}, paired={"A": d / "paired_A.a3m"})
            asm.ligands = [inputs.parse_ligand_spec("ATP=ccd:ATP")]
            asm.glycans = [inputs.parse_glycan_spec("A11:NAG-NAG")]
            asm.contacts = [inputs.parse_contact_spec("A:3=B:2@max=5")]
            doc = asm.build(d)
            self.assertEqual(asm.output_ids(), {"A": ["A", "B"], "B": ["C"]})
            prot = [s["protein"] for s in doc["sequences"] if "protein" in s]
            self.assertEqual(prot[0]["id"], ["A", "B"])
            self.assertEqual(prot[1]["id"], "C")
            self.assertTrue(prot[0]["unpairedMsaPath"].endswith("unpaired_A.a3m"))
            self.assertTrue(prot[0]["pairedMsaPath"].endswith("paired_A.a3m"))
            self.assertNotIn("unpairedMsaPath", prot[1])
            ligs = [s["ligand"] for s in doc["sequences"] if "ligand" in s]
            self.assertEqual(ligs[0]["id"], "D")                      # next free letter after A,B,C
            self.assertEqual(ligs[0]["ccdCodes"], ["ATP"])
            # one glycan entity per copy of chain A (E, F), bonded Asn10 ND2 → C1, NAG O4 → NAG C1
            self.assertEqual([g["id"] for g in ligs[1:]], ["E", "F"])
            self.assertEqual(doc["bondedAtomPairs"][0], [["A", 11, "ND2"], ["E", 1, "C1"]])
            self.assertEqual(doc["bondedAtomPairs"][1], [["E", 1, "O4"], ["E", 2, "C1"]])
            self.assertEqual(doc["bondedAtomPairs"][2], [["B", 11, "ND2"], ["F", 1, "C1"]])
            con = doc["constraint"]["contact"]
            self.assertEqual(len(con), 1)                             # min(copies A=2, B=1) = 1 protomer pairing
            self.assertEqual((con[0]["entity1"], con[0]["entity2"], con[0]["position1"], con[0]["max_distance"]), (1, 2, 3, 5.0))
            self.assertEqual(doc["dialect"], "alphafold3")
            p = inputs.write_input(doc, d / "out", pairing_mode="precomputed", provenance={"x": 1})
            self.assertTrue(p.is_file() and (d / "out" / "pairing_mode.txt").read_text().strip() == "precomputed")
            self.assertEqual(json.loads(p.read_text())["name"], "t")

    def test_glycan_rejects_non_nst_without_atom(self) -> None:
        chains = [msa.Chain("A", "a", "MKTAYI", 1)]
        asm = inputs.Assembly(name="t", chains=chains, glycans=[inputs.parse_glycan_spec("A1:NAG")])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(inputs.InputError):
                asm.build(Path(d))


class Cli(unittest.TestCase):
    def test_parsers_exist(self) -> None:
        from residudes_engine.cli import build_parser
        p = build_parser()
        for cmd in ("msa", "inputs", "prepare"):
            ns = p.parse_args([cmd, "--fasta", "q.fasta"])
            self.assertEqual(ns.fasta, Path("q.fasta"))


if __name__ == "__main__":
    unittest.main()


class Af3DataPipeline(unittest.TestCase):
    """`residudes-engine msa --source af3`: AlphaFold 3's own jackhmmer/hmmsearch pipeline."""

    def _chains(self):
        return [msa.Chain("A", "alpha", "MKTAYIAK"), msa.Chain("B", "beta", "GGGSGGGS")]

    def _fake_run(self, out, payload, rc=0, stderr=""):
        """Stand in for the pipeline: write the data JSON it would write, remember the command."""
        seen = {}

        def run(cmd, capture_output=True, text=True, timeout=None, env=None):
            seen["cmd"] = cmd
            work = Path([c for c in cmd if str(c).startswith("--output_dir=")][0].split("=", 1)[1])
            (work / "query").mkdir(parents=True, exist_ok=True)
            (work / "query" / "query_data.json").write_text(json.dumps(payload))
            return subprocess.CompletedProcess(cmd, rc, "out", stderr)
        return run, seen

    def test_the_data_json_becomes_unpaired_and_paired_msas(self) -> None:
        payload = {"sequences": [
            {"protein": {"id": "A", "sequence": "MKTAYIAK", "unpairedMsa": ">q\nMKTAYIAK\n>h\nMKTAYIAR\n",
                         "pairedMsa": ">p\nMKTAYIAK\n", "templates": [{"mmcif": "data_x"}]}},
            {"protein": {"id": "B", "sequence": "GGGSGGGS", "unpairedMsa": ">q\nGGGSGGGS\n", "templates": []}},
        ]}
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            run, seen = self._fake_run(out, payload)
            with mock.patch.object(msa.subprocess, "run", run), mock.patch.object(msa.shutil, "which", lambda b: f"/u/{b}"):
                res = msa.af3_search(self._chains(), out_dir=out, script="/af3/run_alphafold.py",
                                     python="/venv/bin/python", db_dir="/db/af3", max_template_date="2026-01-01",
                                     log=lambda *a: None)
        self.assertEqual(res.source, "af3")
        self.assertEqual(msa.a3m_depth(res.unpaired["A"]), 2)
        self.assertIn("B", res.unpaired)
        self.assertEqual(res.paired["A"], ">p\nMKTAYIAK\n")
        self.assertEqual(res.meta["templates"], {"A": 1, "B": 0})
        cmd = " ".join(str(c) for c in seen["cmd"])
        self.assertIn("--run_data_pipeline=true", cmd)
        self.assertIn("--run_inference=false", cmd)          # no GPU, no weights: this is the search only
        self.assertIn("--db_dir=/db/af3", cmd)
        self.assertIn("--max_template_date=2026-01-01", cmd)
        for b in msa.AF3_BINARIES:
            self.assertIn(f"--{b}_binary_path=/u/{b}", cmd)

    def test_a_container_runs_the_same_pipeline(self) -> None:
        payload = {"sequences": [{"protein": {"id": "A", "sequence": "MKTAYIAK", "unpairedMsa": ">q\nMKTAYIAK\n"}}]}
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            run, seen = self._fake_run(out, payload)
            with mock.patch.object(msa.subprocess, "run", run), mock.patch.object(msa.shutil, "which", lambda b: f"/u/{b}"):
                msa.af3_search([msa.Chain("A", "alpha", "MKTAYIAK")], out_dir=out, script=None, python=None,
                               db_dir="/db/af3", container="/img/af3.sif", log=lambda *a: None)
        cmd = [str(c) for c in seen["cmd"]]
        self.assertEqual(cmd[:2], ["apptainer", "exec"])
        self.assertIn("/img/af3.sif", cmd)
        self.assertIn(msa.AF3_IN_CONTAINER, cmd)

    def test_a_missing_hmmer_binary_says_which(self) -> None:
        with tempfile.TemporaryDirectory() as t, mock.patch.object(msa.shutil, "which", lambda b: None):
            with self.assertRaises(msa.MsaError) as ctx:
                msa.af3_search(self._chains(), out_dir=Path(t), script="/s.py", python="/p",
                               db_dir="/db", log=lambda *a: None)
        self.assertIn("jackhmmer not found", str(ctx.exception))

    def test_a_failed_pipeline_keeps_its_log_and_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            run, _ = self._fake_run(out, {}, rc=1, stderr="jackhmmer: no such database")
            with mock.patch.object(msa.subprocess, "run", run), mock.patch.object(msa.shutil, "which", lambda b: f"/u/{b}"):
                with self.assertRaises(msa.MsaError) as ctx:
                    msa.af3_search(self._chains(), out_dir=out, script="/s.py", python="/p", db_dir="/db",
                                   log=lambda *a: None)
            self.assertIn("no such database", str(ctx.exception))
            self.assertTrue((out / "msa" / "_af3" / "pipeline.log").is_file()
                            or (out / "_af3" / "pipeline.log").is_file())
