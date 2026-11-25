"""residudes-qa command line.

    residudes-qa doctor                     environment + data check
    residudes-qa data {list,fetch,verify}   data files of data/manifest.toml
    residudes-qa interface <args>           interface scorer: chain-pair features + XGBoost DockQ/ICS/IPS
                                          (discover, features, master, score, run-all, run-ppi-v4, ...)
    residudes-qa cluster <target_dir> ...   MassiveClustering driver on a target's predictions (run_clustering)
    residudes-qa massivecluster <args>      MassiveClustering low-level CLI (comprehensive, network, mpnn, join-xgb, ...)
    residudes-qa visualize <target_dir>     pLDDT/PAE/PDE/distogram/contact/Ramachandran/MSA plots + ranking CSVs
    residudes-qa visualize-msa <target_dir> MSA coverage images only

`interface`, `cluster` and `massivecluster` pass every following argument through unchanged, so the
CASP17 command lines keep working with `xgbdimer` (XGBDimerScorer) / `run_clustering.py` /
`massivecluster` replaced by these subcommands. Slurm fan-out wrappers live in scripts/.
"""
from __future__ import annotations

import argparse
import sys

from residudes_qa import __version__

PASSTHROUGH = {
    "interface": "interface scorer: chain-pair features + XGBoost DockQ/ICS/IPS (ex-XGBDimerScorer)",
    "cluster": "MassiveClustering on a target's predictions (TM / iTM / interface clustering, medoids)",
    "massivecluster": "MassiveClustering low-level commands",
    "visualize": "per-target prediction plots + all_rankings/summary CSVs (ex casp17/visualization)",
    "visualize-msa": "MSA coverage images per chain / MSA source",
}


def _passthrough(name: str, argv: list[str]) -> int:
    if name == "interface":
        from residudes_qa.interface.cli import main as interface_main
        interface_main(argv)
        return 0
    if name == "cluster":
        from residudes_qa.cluster.run_clustering import main as run_main
        return run_main(argv) or 0
    if name == "visualize":
        from residudes_qa.visualize.target import analyze
        return analyze.main(args=argv, prog_name="residudes-qa visualize", standalone_mode=True) or 0
    if name == "visualize-msa":
        from residudes_qa.visualize.msa import main as msa_main   # parses sys.argv itself
        sys.argv = ["residudes-qa visualize-msa", *argv]
        return msa_main() or 0
    from residudes_qa.cluster.cli import main as mc_main      # parses sys.argv itself
    sys.argv = ["residudes-qa massivecluster", *argv]
    return mc_main() or 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in PASSTHROUGH:
        return _passthrough(argv[0], argv[1:])

    ap = argparse.ArgumentParser(prog="residudes-qa", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"residudes-qa {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor", help="check python, pinned packages, torch/GPU, data, optional tools")
    d.add_argument("--json", action="store_true")
    from residudes_qa import data as data_mod
    data_mod.add_parser(sub)
    for name, hlp in PASSTHROUGH.items():
        sub.add_parser(name, help=hlp + " (arguments passed through)")
    args = ap.parse_args(argv)
    if args.cmd == "doctor":
        from residudes_qa.doctor import run
        return run(as_json=args.json)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
