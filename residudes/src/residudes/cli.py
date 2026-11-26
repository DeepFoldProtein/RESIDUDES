"""residudes - a structure-prediction campaign, one dossier per target.

  residudes                          what is in flight, and what is waiting for you
  residudes setup                    install and check this machine, with an agent
  residudes target add H1388 --fasta examples/H1388/H1388.fasta --stoich A1B1   (the shipped example)
  residudes chat T1234               talk to the agent about a target (ctrl-b d leaves it running)
  residudes daemon --all             keep every target moving; it stops where a person is needed

when a target is waiting for you:
  residudes checkpoints T1234        the brief and the conversation so far
  residudes reply T1234 C1 "..."     answer in that thread
  residudes resolve T1234 C1 --outcome proceed | loop:HYPOTHESES
  residudes approve T1234 --what "slate v1"
  residudes package T1234 --server   the CASP files; you submit them

The agent records evidence, runs jobs and scores pools with commands of its own.
They are not hidden, only kept out of the way: `residudes help --all`.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from pathlib import Path

from residudes import __version__
from residudes.config import load
from residudes.dossier import DECISION_TYPES, STAGES, Dossier, list_targets, now


def _actor() -> str:
    return os.environ.get("RESIDUDES_ACTOR") or getpass.getuser()


def _automation_on(cfg, target: str) -> bool:
    """Has a person left a standing mandate on this target? Checked before the agent is refused."""
    try:
        return Dossier.load(cfg.targets_dir, target).automation_allows("approve")
    except Exception:                    # no dossier yet, unreadable, anything: fall back to refusing
        return False


def _sto(d, given: str | None) -> str:
    """--sto, or the stoichiometry given at `target add --stoich`."""
    if given:
        return given
    sto = d.stoichiometry.get("given")
    if not sto:
        raise ValueError(f"give --sto: no stoichiometry was given for {d.target} (`residudes target add --stoich`)")
    decided = [x for x in d.decisions if x.get("type") == "stoichiometry"]
    if decided:                                        # the team's decision may differ from the organisers' (A3 vs A2:3,A1:2)
        m = re.match(r"\s*((?:[A-Z]\d+)+(?::\d+)?(?:,(?:[A-Z]\d+)+:\d+)*)", decided[-1].get("chose", ""))
        if not m or m.group(1) != sto:
            raise ValueError(f"give --sto: {d.target} was given as {sto}, and decision {decided[-1]['id']} chose "
                             f"{decided[-1].get('chose', '')[:60]!r}")
    return sto


def _with(cfg, target: str, fn) -> int:
    with Dossier.locked(cfg.targets_dir, target) as d:
        out = fn(d)
    if out:
        print(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, FileNotFoundError) as exc:      # refused transitions, missing dossiers: one line, exit 1
        print(f"residudes: {exc}", file=sys.stderr)
        return 1


FULL_HELP = """residudes command line.

  residudes setup [--yes] [--project DIR]     first run, together with the user: resources, databases, config, skills
  residudes doctor                            re-check resources, databases and components
  residudes target add T [--fasta F] [--stoich A3] [--deadline-server ISO]
  residudes target show T | target list
  residudes next T                            the next step for T and the skill that covers it
  residudes report T [-o analysis.md]         the target analysis: priors, decisions, which priors each decision rests on
  residudes msa T --depth | --crawl --family PF… | --plm   MSA depth/coverage; deepen a thin MSA (T1331 recipe)
  residudes structures T [--identity 0.5]     public structures of each chain's sequence (evidence: public_structures)
  residudes structures T --foldseek MODEL     the structural half (sbatch); then --read-foldseek TSV
  residudes structures T --hhsearch           profile against profile when there is no model yet (sbatch)
  residudes relax T [--sc-fc N] | --read      GROMACS + Rosetta on the slate's models, under restraints (sbatch)
  residudes slides T                          the same as Typst slides with PyMOL pictures -> report/report.pdf
  residudes render T --pml FILE               render your own PyMOL scenes through sbatch -> report/figs/
  residudes agent step T                      let the agent do that step (Claude Agent SDK)
  residudes daemon --target T.. | --all [--once] [--interval 300]  scheduler: steps every target that can make progress, skips jobs / people

  pool and QA:
  residudes pool T [--dir DIR]                import an existing prediction pool (residudes-engine layout) -> pool.csv
  residudes qa T --sto A3 --what interface    residudes-qa through sbatch (interface | cluster); `residudes jobs T`
  residudes gate T --init --sto A3           draft structural gates from the dossier's site evidence
  residudes gate T --spec F --sto A3 | --show   evaluate gates over the pool (sbatch) / record the result
  residudes slate T --sto A3 | A2:3,A1:2     propose 5 from the models that pass (af3 leads, predictors alternate)
  residudes slate T --models M1,...,M5       record the slate chosen after review (then decide --type selection)
  residudes package T --server|--human       after approval: geometry checks, CASP files via casp17-utils (never submits)
  residudes retro T --native PDB              once the native is out: score the slate models (sbatch, OpenStructure)
  residudes retro T [--scores CSV]            write retro.md from those scores (+ a pool-wide scored table)
  residudes export T DIR                      dossier, step costs, logs, slate, retro, QA summaries -> DIR

  working with a person (checkpoints: PLAN_REVIEW after the analysis, RESULTS_REVIEW after the first round):
  residudes consult T --kind plan|results --ask "question (options; recommendation)" [--brief FILE]
  residudes reply T C1 "text"                 either side of the conversation
  residudes resolve T C1 --outcome proceed|loop:RUNNING|loop:HYPOTHESES      people only
  residudes checkpoints T [C1]                list, or show one conversation
  residudes review T [--add "name: why"]      first-results checks -> triggers (a trigger means a results checkpoint)

  dossier writes (used by the agent and by people):
  residudes note T "text"                     free-form log line ("NEEDS HUMAN: ..." asks for a person)
  residudes evidence T KEY VALUE | @file.json   append a finding (VALUE may be JSON; @file for long JSON)
  residudes decide T --type stoichiometry --chose A3 --evidence "..." [--alt A1 ...] [--confidence high]
  residudes run-record T --job ID --what "af3 seeds 1-128 on af3_msa" [--scheduler slurm|pueue|other] [--state done]
  residudes stage T STAGE [--why "..."]
  residudes approve T --what "slate v1"       people only: lets the target past SLATE_PROPOSED
"""


class _Help(argparse.RawDescriptionHelpFormatter):
    """Leave the agent's own commands out of `-h`.

    argparse formats sub-commands without checking help is SUPPRESS - the check only guards
    top-level actions - so a suppressed sub-command prints "==SUPPRESS==" instead of vanishing.
    This drops them, and `residudes help --all` lists everything.
    """

    def _format_action(self, action):
        return "" if action.help is argparse.SUPPRESS else super()._format_action(action)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="residudes", description=__doc__, formatter_class=_Help)
    ap.add_argument("--version", action="version", version=f"residudes {__version__}")
    ap.add_argument("--config", help="residudes.toml (default: $RESIDUDES_CONFIG or residudes/residudes.toml)")
    # `residudes` with no command shows the conversations running here, because that is the
    # question someone has when they sit down: what is in flight, and what is waiting for me.
    # the usage line is built from every choice, which is the wall this is trying to avoid
    sub = ap.add_subparsers(dest="cmd", required=False, metavar="<command>")

    p = sub.add_parser("setup", help="first-run onboarding, with an agent by default")
    p.add_argument("--manual", action="store_true",
                   help="answer the questions yourself instead of letting an agent do the setup")
    p.add_argument("--harness", choices=("claude", "codex", "ag", "gemini"),
                   help="which agent CLI to use (default: the first one on PATH)")
    p.add_argument("--yes", action="store_true", help="accept every suggestion (non-interactive)")
    p.add_argument("--json", action="store_true", help="only print the findings as JSON")
    p.add_argument("--project", help="link the skills into this project's .claude/skills and .agents/skills")
    p.add_argument("--copy-skills", action="store_true",
                   help="copy the skills instead of symlinking them (a harness that does not follow symlinks); rerun to refresh")
    p = sub.add_parser("help", help="every command, including the agent's own")
    p.add_argument("--all", action="store_true", default=True, help=argparse.SUPPRESS)

    p = sub.add_parser("doctor", help="re-check resources, databases, components")
    p.add_argument("--json", action="store_true")

    t = sub.add_parser("target", help="create, show or list target dossiers").add_subparsers(dest="tcmd", required=True)
    p = t.add_parser("add")
    p.add_argument("target")
    p.add_argument("--fasta")
    p.add_argument("--stoich", help="stoichiometry given by the organisers, e.g. A3 or A1B1C1")
    p.add_argument("--deadline-server")
    p.add_argument("--deadline-human")
    p.add_argument("--review", action="store_true",
                   help="stop at the plan and results checkpoints and at the slate for a person; by default the agent "
                        "answers them itself and runs the target through to a packaged slate (submitting stays yours)")
    p = t.add_parser("show")
    p.add_argument("target")
    p.add_argument("--json", action="store_true")
    t.add_parser("list")

    p = sub.add_parser("next", help="the next step for a target")
    p.add_argument("target")
    p = sub.add_parser("note", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("text")
    p = sub.add_parser("evidence", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("key")
    p.add_argument("value")
    p = sub.add_parser("decide", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--type", required=True, choices=DECISION_TYPES)
    p.add_argument("--chose", required=True)
    p.add_argument("--evidence", nargs="+", required=True)
    p.add_argument("--alt", nargs="*", default=[])
    p.add_argument("--confidence", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--until", help="reversible until (ISO time), e.g. the server deadline")
    p = sub.add_parser("run-record", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--job", required=True)
    p.add_argument("--what", help="what the job runs (required for a new job)")
    p.add_argument("--scheduler", choices=["slurm", "pueue", "other"], default="slurm",
                   help="slurm jobs are polled with squeue/sacct, pueue tasks with the local pueue; other is never polled")
    p.add_argument("--state", help="submitted (default for a new job), or done/failed to close a job residudes cannot poll")
    p = sub.add_parser("stage", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--why", default="")
    p = sub.add_parser("approve", help="people only, unless a person turned automation on")
    p.add_argument("target")
    p.add_argument("--what", required=True)

    p = sub.add_parser("automate", help="people only: let the agent answer its own checkpoints on this target")
    p.add_argument("target")
    p.add_argument("--on", action="store_true", help="grant the mandate")
    p.add_argument("--off", action="store_true", help="revoke it")
    p.add_argument("--show", action="store_true", help="say what is in force (the default)")
    p.add_argument("--until", default="PACKAGED", metavar="STAGE",
                   help="the last stage the mandate covers (default PACKAGED; SUBMITTED is never allowed)")
    p.add_argument("--note", default="", help="why, for the record")

    p = sub.add_parser("pool", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--dir", help="prediction directory (default <targets_root>/<target>)")
    p.add_argument("--refresh", action="store_true", help="re-read the pool even if it was imported before")
    p.add_argument("--threads", type=int, default=8, help="parallel reads (default 8; modest on a shared login node)")
    p = sub.add_parser("qa", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--sto", help="stoichiometry, e.g. A3 (default: the one given at `target add --stoich`)")
    p.add_argument("--what", required=True, choices=["interface", "cluster"])
    p.add_argument("--variants", help="comma-separated MSA variants (default: residudes-qa's default)")
    p.add_argument("--mode", choices=["ppi", "ab"], default="ppi",
                   help="interface: ppi (protein-protein, default) or ab (antibody-antigen: give --ab-chains and --ag-chains)")
    p.add_argument("--ab-chains", help="ab mode: antibody chain ids, e.g. CD")
    p.add_argument("--ag-chains", help="ab mode: antigen chain ids, e.g. AB")
    p = sub.add_parser("jobs", help=argparse.SUPPRESS)
    p.add_argument("target")
    p = sub.add_parser("retro", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--scores", help="CSV of pool models scored against the native")
    p.add_argument("--native", help="native structure: submit an OpenStructure job on the slate models")
    p.add_argument("--mode", choices=["mono", "oligo"], help="scoring mode for --native (default: from its chain count)")
    p = sub.add_parser("slate", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--sto", help="A3, or a hedge like A2:3,A1:2 (slots in that order); default: the given stoichiometry")
    p.add_argument("--models", help="record a chosen slate: five names from a proposal, comma-separated, slot order")
    p.add_argument("--no-interface", action="store_true", help="ignore interface scores (e.g. antibody-antigen)")
    p.add_argument("--gates", help="gate result to use (its --name); default: the most recent for the stoichiometry; none: confidence only")
    p.add_argument("--site", help="keep only models whose census label is this site (e.g. the literature's pocket)")
    p = sub.add_parser("gate", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--spec", help="gates JSON (see `python -m residudes.gate --help` / skill residudes-selection)")
    p.add_argument("--sto", help="stoichiometry, e.g. A3 (default: the one given at `target add --stoich`)")
    p.add_argument("--name", default="default", help="name of this gate set (several can coexist)")
    p.add_argument("--show", action="store_true", help="summarise a finished result and record evidence `gates`")
    p.add_argument("--init", action="store_true", help="draft a spec from the dossier's site evidence (gates/draft_<sto>.json)")
    p.add_argument("--local", action="store_true", help="evaluate here instead of sbatch (small pools only)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--hours", type=int, default=4, help="sbatch time limit (about 0.3-2 s per model per worker)")

    p = sub.add_parser("package", help="the approved slate: geometry checks, then CASP files via casp17-utils (a person submits)")
    p.add_argument("target")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--server", action="store_true", help="the server author code (from casp17-utils' .env)")
    g.add_argument("--human", action="store_true", help="the human author code")
    p.add_argument("--method", default="DeepFold")
    p.add_argument("--casp-id", help="the CASP target id when the dossier name differs (e.g. T1416 for T1416-replay)")
    p.add_argument("--mmcif", action="store_true", help="CASP mmCIF instead of PDB (very large targets)")
    p.add_argument("--check-only", action="store_true", help="the geometry checks only, at any stage")
    p = sub.add_parser("chat", help="talk to the agent about a target in its own session (tmux); no target lists them")
    p.add_argument("target", nargs="?")
    p.add_argument("--harness", choices=("claude", "codex", "ag", "gemini"),
                   help="which agent CLI to start (default: the first one on PATH)")
    p.add_argument("--stop", action="store_true",
                   help="save the conversation into the dossier and end it")
    p.add_argument("--archive", action="store_true",
                   help="save the conversation into the dossier and leave it running")
    p.add_argument("--auto", action="store_true", default=None,
                   help="run the target to PACKAGED without stopping for you: the conversation is asked for each next "
                        "step, and the agent answers its own checkpoints (grants the mandate if there is none)")
    p.add_argument("--no-auto", dest="auto", action="store_false",
                   help="the target moves only when asked; takes back a mandate auto granted; remembered until --auto")
    p.add_argument("--drive", action="store_true", help=argparse.SUPPRESS)   # the --auto driver's own loop

    p = sub.add_parser("sessions", help="the target conversations running on this machine")

    p = sub.add_parser("consult", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--kind", required=True, choices=["plan", "results", "other"])
    p.add_argument("--ask", action="append", default=[], help="a question with its options and your recommendation (repeat)")
    p.add_argument("--brief", help="a markdown brief to put at the top (the analysis is added below it)")
    p = sub.add_parser("reply", help="reply in a checkpoint conversation")
    p.add_argument("target")
    p.add_argument("checkpoint")
    p.add_argument("text")
    p = sub.add_parser("resolve", help="people only: close a checkpoint with a decision")
    p.add_argument("target")
    p.add_argument("checkpoint")
    p.add_argument("--outcome", required=True, choices=["proceed", "loop:HYPOTHESES", "loop:RUNNING"])
    p.add_argument("--note", default="")
    p = sub.add_parser("checkpoints", help="list checkpoints, or show one conversation")
    p.add_argument("target")
    p.add_argument("checkpoint", nargs="?")
    p = sub.add_parser("review", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--add", action="append", default=[], help="a trigger you see yourself: \"name: why\" (repeat)")
    p = sub.add_parser("msa", help=argparse.SUPPRESS)
    p.add_argument("target")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--depth", nargs="*", metavar="A3M", help="depth/coverage of these a3m files (default: the target's MSAs)")
    g.add_argument("--crawl", action="store_true", help="collect the protein family (Pfam/InterPro + hmmsearch) -> msas/crawl")
    g.add_argument("--plm", action="store_true", help="plmMSA: PLM-embedding search + OTalign -> msas/plm_<aligner>")
    p.add_argument("--family", action="append", default=[], help="Pfam PF… or InterPro IPR… (repeat)")
    p.add_argument("--db", action="append", default=[], choices=["uniref90", "mgnify", "uniprot"])
    p.add_argument("--max-members", type=int, default=20000, help="UniProt members per family (cap)")
    p.add_argument("--aligner", default="otalign", choices=["otalign", "plmalign", "plm_blast", "hmm"],
                   help="--plm: the plmMSA aligner; --crawl: otalign (default) or hmm (offline, weaker for remote families)")
    p = sub.add_parser("structures", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--identity", type=float, default=0.9, help="identity cutoff of the search (default 0.9)")
    p.add_argument("--chain", help="one chain letter (default: every chain)")
    p.add_argument("--rows", type=int, default=100, help="entities to ask the search for (default 100)")
    p.add_argument("--limit", type=int, default=25, help="entries to look up and show per chain (default 25)")
    p.add_argument("--foldseek", metavar="MODEL", help="structural search of this model instead (sbatch)")
    p.add_argument("--read-foldseek", metavar="TSV", help="read a finished foldseek table into the dossier")
    p.add_argument("--hhsearch", action="store_true", help="profile-profile search of one chain instead (sbatch)")
    p.add_argument("--read-hhsearch", metavar="HHR", help="read a finished hhsearch .hhr into the dossier")
    p.add_argument("--db", help="--foldseek: the database prefix or directory (default: [databases] in residudes.toml)")
    p = sub.add_parser("relax", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--models", help="comma list of model keys (default: the slate's)")
    p.add_argument("--ssbond", default="auto", help="SSBOND file, or auto to find disulfides by distance (default auto)")
    p.add_argument("--bb-fc", type=int, help="backbone position-restraint force constant, kJ/mol/nm² (default 1000000)")
    p.add_argument("--sc-fc", type=int, help="side-chain position-restraint force constant (default 10000)")
    p.add_argument("--dstres", help="distance restraints file (GROMACS leg)")
    p.add_argument("--ligand", action="store_true", help="ligand-aware route (AmberTools topology) instead of protein-only")
    p.add_argument("--read", action="store_true", help="read the finished runs: geometry before and after, how far it moved")
    p = sub.add_parser("slides", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--final", action="store_true",
                   help="the final deck: ray-trace every picture with PyMOL (sbatch, waited for), the agent's report/pml/*.pml too")
    p.add_argument("--no-render", action="store_true", help="compile only: draw no pictures")
    p.add_argument("--with-retro", action="store_true")
    p = sub.add_parser("render", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("--pml", required=True)
    # the chat agent's: a structure in a pane beside the conversation, opened, driven and closed as the person asks
    p = sub.add_parser("view", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("action", choices=["show", "do", "snap", "close"])
    p.add_argument("rest", nargs=argparse.REMAINDER, help="show: M1 M3 | slate | C0 | tpl_6S9K_A | FILE; do: a molterm command")
    p = sub.add_parser("report", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("-o", "--out", help="write analysis.md here (default: print)")
    p.add_argument("--with-retro", action="store_true", help="include native-derived results")
    p = sub.add_parser("export", help=argparse.SUPPRESS)
    p.add_argument("target")
    p.add_argument("dest")
    p.add_argument("--with-retro", action="store_true", help="include scores against the native (check it is public first)")
    p = sub.add_parser("daemon", help="scheduler: one agent step per target when it can make progress")
    p.add_argument("--target", nargs="*", help="the targets to step")
    p.add_argument("--all", action="store_true", help="every dossier under work_root (a shared folder may hold other people's)")
    p.add_argument("--interval", type=int, default=300, help="seconds between rounds (>= 60)")
    p.add_argument("--once", action="store_true", help="one round, then exit")
    p.add_argument("--exit-when-idle", action="store_true", help="stop when every target waits for a person")
    p.add_argument("--stop", action="store_true", help="ask the running daemon to exit after its current round")
    p.add_argument("--budget", type=float, help="USD per agent step")
    a = sub.add_parser("agent", help="run the agent").add_subparsers(dest="acmd", required=True)
    p = a.add_parser("step", help="one step on one target")
    p.add_argument("target")
    p.add_argument("--budget", type=float, help="USD for this step (default from residudes.toml)")
    p.add_argument("--max-turns", type=int)
    p.add_argument("--model")
    p.add_argument("--quiet", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "setup":
        from residudes.setup import run_setup, run_setup_guided
        # An agent can read the report, decide what is missing and install it; the questions are
        # the fallback, and --manual asks for them. --json and --yes are machine modes: no agent.
        if not (args.manual or args.yes or args.json):
            rc = run_setup_guided(args.config, args.harness)
            if rc != 1:                      # 1 = no agent CLI here, so fall through to the questions
                return rc
        return run_setup(args.config, args.yes, args.json, args.project, args.copy_skills)
    if args.cmd == "doctor":
        from residudes.setup import run_doctor
        return run_doctor(args.config, args.json)

    cfg = load(args.config)
    if args.cmd == "retro":
        from residudes import retro
        ddir = lambda d: Dossier.path(cfg.targets_dir, d.target).parent
        if args.native:
            return _with(cfg, args.target, lambda d: retro.score_native_cmd(cfg, d, ddir(d), Path(args.native), args.mode, _actor()))
        return _with(cfg, args.target, lambda d: retro.run(d, ddir(d), Path(args.scores) if args.scores else None, _actor()))
    if args.cmd in ("pool", "qa", "jobs", "slate"):
        from residudes import ops
        if args.cmd == "pool":
            return _with(cfg, args.target, lambda d: ops.pool_cmd(cfg, d, args.dir, _actor(), args.refresh, args.threads))
        if args.cmd == "qa":
            return _with(cfg, args.target, lambda d: ops.qa_cmd(cfg, d, _sto(d, args.sto), args.what, args.variants, _actor(),
                                                               args.mode, args.ab_chains, args.ag_chains))
        if args.cmd == "jobs":
            return _with(cfg, args.target, ops.jobs_cmd)
        return _with(cfg, args.target, lambda d: ops.slate_cmd(cfg, d, args.sto or (None if args.models else _sto(d, None)),
                                                               not args.no_interface, _actor(), args.models,
                                                               args.gates, args.site))
    if args.cmd == "package":
        from residudes import package
        mode = "server" if args.server else ("human" if args.human else None)
        return _with(cfg, args.target, lambda d: package.package_cmd(cfg, d, mode, args.method, args.casp_id, args.mmcif,
                                                                     args.check_only, _actor()))
    if args.cmd == "gate":
        from residudes import gate
        if args.show:
            return _with(cfg, args.target, lambda d: gate.show_cmd(cfg, d, args.name, _sto(d, args.sto), _actor()))
        if args.init:
            return _with(cfg, args.target, lambda d: gate.init_cmd(cfg, d, _sto(d, args.sto), _actor()))
        if not args.spec:
            print("give --spec FILE (or --show)", file=sys.stderr)
            return 2
        return _with(cfg, args.target, lambda d: gate.gate_cmd(cfg, d, args.spec, _sto(d, args.sto), args.name, args.local, args.workers, _actor(), args.hours))
    if args.cmd == "target":
        if args.tcmd == "add":
            if Dossier.path(cfg.targets_dir, args.target).exists():
                print(f"{args.target} already has a dossier", file=sys.stderr)
                return 1
            d = Dossier(target=args.target)
            if args.fasta:
                d.sequence["fasta"] = str(Path(args.fasta).resolve())
                d.sequence["text"] = Path(args.fasta).read_text()
            if args.stoich:
                d.stoichiometry["given"] = args.stoich
            d.deadlines = {k: v for k, v in (("server", args.deadline_server), ("human", args.deadline_human)) if v}
            d.note(_actor(), "dossier created")
            # Full automation is the default: whoever registers the target grants the mandate, by name and on the
            # record, up to a packaged slate. The checkpoints still happen and are written the same way - the
            # agent answers them - so the reasoning a person would have read is in the dossier either way.
            # Submission is outside any mandate. `--review` keeps a person at each checkpoint instead.
            mode = "with human checkpoints"
            if not args.review:
                try:
                    d.authorise_automation(_actor(), "PACKAGED", "the default at registration")
                    mode = "full automation, up to a packaged slate (submitting stays yours)"
                except ValueError as exc:                  # the agent registering a target cannot grant itself one
                    mode = f"with human checkpoints ({exc})"
            print(f"created {d.save(cfg.targets_dir)}")
            print(f"  {mode}")
            if not args.review:
                print(f"  to stop at the checkpoints instead: residudes automate {args.target} --off")
            return 0
        if args.tcmd == "show":
            d = Dossier.load(cfg.targets_dir, args.target)
            if args.json:
                print(json.dumps(d.__dict__, indent=1, ensure_ascii=False))
                return 0
            goal, skill = d.next_step()
            print(f"{d.target}  stage {d.stage}  (created {d.created})")
            a = d.automation
            print("  mode: " + (f"full automation, granted by {a['by']} up to {a['until']} (submitting stays a person's)"
                                if a else "human checkpoints (a person answers each one and approves the slate)"))
            print(f"  next: {goal}" + (f"  [skill {skill}]" if skill else ""))
            for k, v in d.stoichiometry.items():
                print(f"  stoichiometry {k}: {v}")
            from residudes.slides import _plain
            for k, v in d.evidence.items():                  # the latest entry of each key, as one readable line
                last = v[-1] if v else {}
                more = f" (+{len(v) - 1} earlier)" if len(v) > 1 else ""
                print(f"  evidence {k} [{last.get('by', '?')}{more}]: {_plain(last.get('value'), 150)}")
            for x in d.decisions:
                print(f"  {x['id']} {x['type']}: {x['chose']} ({x['confidence']}) <- {'; '.join(x['evidence'])[:160]}")
            for x in d.runs:
                # a local run (a molterm render, a gate with --local) has no job id: name where it ran instead
                print(f"  run {x.get('job') or x.get('scheduler', 'local')}: {x['what']} [{x['state']}]")
            for x in d.log[-5:]:
                print(f"  {x['time']} {x['actor']}: {x['note'][:160]}")
            return 0
        for d in list_targets(cfg.targets_dir):
            from residudes.daemon import _needs_human
            human = next((x["note"] for x in reversed(d.log) if x["note"].startswith("NEEDS HUMAN")), "") if _needs_human(d) else ""
            print(f"{d.target:10s} {d.stage:15s} decisions {len(d.decisions):2d}  runs {len(d.runs):2d}  {human[:80]}")
        return 0
    if args.cmd == "next":
        d = Dossier.load(cfg.targets_dir, args.target)
        goal, skill = d.next_step()
        print(f"{d.target} at {d.stage}: {goal}" + (f"  [skill {skill}]" if skill else ""))
        return 0
    if args.cmd == "note":
        return _with(cfg, args.target, lambda d: d.note(_actor(), args.text))
    if args.cmd == "evidence":
        raw = args.value
        if raw.startswith("@"):                                # long values from a file: no shell quoting to get wrong
            raw = Path(raw[1:]).read_text()
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        def add(d):
            d.evidence.setdefault(args.key, []).append({"time": now(), "by": _actor(), "value": val})
        return _with(cfg, args.target, add)
    if args.cmd == "decide":
        return _with(cfg, args.target, lambda d: "recorded " + d.decide(_actor(), args.type, args.chose, args.evidence,
                                                                          args.alt, args.confidence, args.until)["id"])
    if args.cmd == "run-record":
        def rec(d):
            from residudes.ops import scheduler
            old = next((r for r in d.runs if str(r.get("job")) == args.job and scheduler(r) == args.scheduler), None)
            if old:                                            # update: e.g. close a pueue task on a remote pod
                old.update({k: v for k, v in (("what", args.what), ("state", args.state)) if v})
                return f"updated {args.scheduler} job {args.job}: {old.get('state')}"
            if not args.what:
                raise ValueError("--what is required for a new job")
            d.runs.append({"time": now(), "by": _actor(), "job": args.job, "scheduler": args.scheduler, "what": args.what,
                           "state": args.state or "submitted"})
        return _with(cfg, args.target, rec)
    if args.cmd is None:
        from residudes import chat
        return chat.list_cmd(cfg)
    if args.cmd == "help":
        print(FULL_HELP)
        return 0
    if args.cmd in ("chat", "sessions"):
        from residudes import chat
        if getattr(args, "drive", False):
            from residudes import auto
            return auto.drive(cfg, args.target)
        return chat.chat_cmd(cfg, getattr(args, "target", None), getattr(args, "harness", None),
                             None, getattr(args, "stop", False), getattr(args, "archive", False),
                             getattr(args, "auto", None))
    if args.cmd == "automate":
        if args.show or not (args.on or args.off):
            d = Dossier.load(cfg.targets_dir, args.target)
            a = d.automation
            if not a:
                print(f"{args.target}: automation off — a person answers checkpoints and approves the slate")
            else:
                print(f"{args.target}: automation on, granted by {a['by']} at {a['time']}, up to {a['until']}"
                      + (f"\n  note: {a['note']}" if a.get("note") else "")
                      + "\n  submission stays a person's move in every case")
            return 0
        if os.environ.get("RESIDUDES_ACTOR") == "agent":
            print("only a person turns automation on or off", file=sys.stderr)
            return 2
        if args.off:
            return _with(cfg, args.target, lambda d: d.revoke_automation(_actor(), args.note)
                         or f"{args.target}: automation off")
        return _with(cfg, args.target,
                     lambda d: f"{args.target}: automation on up to {d.authorise_automation(_actor(), args.until, args.note)['until']}"
                               " (submission still needs a person)")
    if args.cmd == "stage":
        return _with(cfg, args.target, lambda d: d.move(args.stage, _actor(), args.why) or f"{d.target}: now {d.stage}")
    if args.cmd == "approve":
        if os.environ.get("RESIDUDES_ACTOR") == "agent" and not _automation_on(cfg, args.target):
            print("approve is for people only (a person can turn automation on: "
                  f"`residudes automate {args.target} --on`)", file=sys.stderr)
            return 2
        def approve(d):
            from residudes import gate
            props = [x["value"] for x in d.evidence.get("slate_proposal", [])]
            m = re.search(r"\bv(\d+)\b", args.what)               # "slate v2" approves version 2, not the latest
            names_slate = bool(m) or re.search(r"\bslate\b", args.what, re.I) is not None
            if m and not any(str(x.get("version")) == m.group(1) for x in props):
                raise ValueError(f"no slate v{m.group(1)} is recorded (slates: "
                                 + (", ".join(f"v{x.get('version')}" for x in props) or "none") + "); nothing was approved")
            d.approve(_actor(), args.what)
            if d.stage == "SLATE_PROPOSED" and names_slate:     # approving a slate is the person's move to APPROVED
                d.move("APPROVED", _actor(), f"approved: {args.what}")
            if not names_slate:
                return f"approved: {args.what} (not a slate: the stage stays {d.stage})"
            prop = next((x for x in props if m and str(x.get("version")) == m.group(1)), props[-1] if props else {})
            models = prop.get("models", [])
            try:
                fails = gate.proposal_failures(Dossier.path(cfg.targets_dir, d.target).parent, prop)
            except Exception as exc:                   # the approval is saved whatever the warning does
                return f"approved: {args.what}\nnote: the gate check could not run ({type(exc).__name__}: {exc})"
            if not fails:
                return f"approved: {args.what}"
            lines = [f"approved: {args.what}",
                     f"note: slate v{prop.get('version')} has models that fail the structural gates it was made with:"]
            lines += [f"  slot {models.index(m) + 1} {m}: {', '.join(g)}" for m, g in fails.items()]
            d.note(_actor(), "approved with gate failures in the slate: " + "; ".join(f"{m} ({', '.join(g)})" for m, g in fails.items()))
            return "\n".join(lines)
        return _with(cfg, args.target, approve)
    if args.cmd in ("consult", "reply", "resolve", "checkpoints", "review"):
        from residudes import consult
        if args.cmd == "consult":
            return _with(cfg, args.target, lambda d: consult.open_cmd(cfg, d, args.kind, args.ask, args.brief, _actor()))
        if args.cmd == "reply":
            return _with(cfg, args.target, lambda d: d.reply(_actor(), args.checkpoint, args.text) and f"{args.checkpoint}: reply recorded")
        if args.cmd == "resolve":
            if os.environ.get("RESIDUDES_ACTOR") == "agent" and not _automation_on(cfg, args.target):
                print("resolve is for people only (a person can turn automation on: "
                      f"`residudes automate {args.target} --on`)", file=sys.stderr)
                return 2
            return _with(cfg, args.target, lambda d: d.resolve(_actor(), args.checkpoint, args.outcome, args.note)
                         and f"{args.checkpoint} resolved: {args.outcome}; {d.target} now {d.stage}")
        if args.cmd == "checkpoints":
            print(consult.show_cmd(Dossier.load(cfg.targets_dir, args.target), args.checkpoint))
            return 0
        return _with(cfg, args.target, lambda d: consult.review_cmd(cfg, d, args.add, _actor()))
    if args.cmd == "msa":
        from residudes import msa
        if args.crawl:
            return _with(cfg, args.target, lambda d: msa.crawl_cmd(cfg, d, args.family, args.db, args.max_members, _actor(),
                                                                   "hmm" if args.aligner == "hmm" else "otalign"))
        if args.plm:
            return _with(cfg, args.target, lambda d: msa.plm_cmd(cfg, d, args.aligner, _actor()))
        return _with(cfg, args.target, lambda d: msa.depth_cmd(cfg, d, args.depth, _actor()))
    if args.cmd == "structures":
        from residudes import structures
        if args.foldseek:
            return _with(cfg, args.target, lambda d: structures.foldseek_cmd(cfg, d, args.foldseek, _actor(), args.db))
        if args.read_foldseek:
            return _with(cfg, args.target, lambda d: structures.read_foldseek_cmd(
                cfg, d, args.read_foldseek, _actor(), limit=args.limit))
        if args.hhsearch:
            return _with(cfg, args.target, lambda d: structures.hhsearch_cmd(cfg, d, _actor(), args.chain, args.db))
        if args.read_hhsearch:
            return _with(cfg, args.target, lambda d: structures.read_hhsearch_cmd(
                cfg, d, args.read_hhsearch, _actor(), limit=args.limit))
        return _with(cfg, args.target, lambda d: structures.sweep_cmd(
            cfg, d, _actor(), identity=args.identity, rows=args.rows, limit=args.limit, chain=args.chain))
    if args.cmd == "relax":
        from residudes import relax
        if args.read:
            return _with(cfg, args.target, lambda d: relax.read_cmd(cfg, d, _actor()))
        models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
        return _with(cfg, args.target, lambda d: relax.submit_cmd(
            cfg, d, _actor(), models or None, args.ssbond, args.bb_fc, args.sc_fc, args.dstres, args.ligand))
    if args.cmd == "view":                                  # reads the dossier, writes nothing to it: no lock
        from residudes import viewer
        print(viewer.view_cmd(cfg, Dossier.load(cfg.targets_dir, args.target), args.action, args.rest))
        return 0
    if args.cmd in ("slides", "render"):
        from residudes import slides
        if args.cmd == "slides":
            return _with(cfg, args.target, lambda d: slides.slides_cmd(cfg, d, _actor(), not args.no_render, args.with_retro,
                                                                          args.final))
        return _with(cfg, args.target, lambda d: slides.render_cmd(cfg, d, Path(args.pml), _actor()))
    if args.cmd == "report":
        from residudes import report
        print(report.run(Dossier.load(cfg.targets_dir, args.target), args.out, args.with_retro,
                         Dossier.path(cfg.targets_dir, args.target).parent))
        return 0
    if args.cmd == "export":
        from residudes import export
        d = Dossier.load(cfg.targets_dir, args.target)
        print(export.run(d, Dossier.path(cfg.targets_dir, d.target).parent, Path(args.dest), args.with_retro))
        return 0
    if args.cmd == "daemon":
        from residudes import daemon
        if args.stop:
            (cfg.targets_dir.parent / ".daemon.stop").write_text("stop\n")
            print("the daemon will exit after its current round")
            return 0
        if not args.target and not args.all:
            print("name the targets (--target T1 T2) or pass --all: the daemon runs paid agent steps on every dossier it "
                  f"is given, and {cfg.targets_dir} may hold other people's", file=sys.stderr)
            return 2
        return daemon.run(cfg, args.target or None, args.interval, args.once, args.budget, exit_when_idle=args.exit_when_idle)
    if args.cmd == "agent":
        from residudes.agent import step
        res = step(cfg, args.target, args.budget, args.max_turns, args.model, verbose=not args.quiet)
        print(json.dumps(res, indent=1, default=str))
        return 0 if res.get("subtype") in ("success", "skipped") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
