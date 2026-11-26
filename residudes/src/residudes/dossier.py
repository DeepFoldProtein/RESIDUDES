"""One dossier per target: <work_root>/targets/<T>/dossier.json.

The dossier is the agent's whole memory of a target. Every agent step starts by reading it and ends
by writing to it, so a step can be interrupted, repeated or done by a person without losing state.
Every decision is recorded with its evidence and alternatives, so it can be scored once the native
structure is released (stage RETROSPECTIVE).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ["NEW", "IDENTIFIED", "HYPOTHESES", "PLAN_REVIEW", "INPUTS_READY", "RUNNING", "POOL_READY", "RESULTS_REVIEW",
          "SLATE_PROPOSED", "APPROVED", "PACKAGED", "SUBMITTED", "RETROSPECTIVE"]
NEXT = {s: {STAGES[i + 1]} for i, s in enumerate(STAGES[:-1])}
NEXT["PLAN_REVIEW"] |= {"HYPOTHESES"}                  # the person and the agent rework the hypotheses
NEXT["POOL_READY"] |= {"RUNNING"}                      # more sampling
NEXT["RESULTS_REVIEW"] |= {"HYPOTHESES", "RUNNING"}    # the first results call for another round
NEXT["SUBMITTED"] |= {"HYPOTHESES"}                    # a stage-2 reissue
DECISION_TYPES = ["identity", "stoichiometry", "variant", "sampling", "selection", "package", "other"]
CHECKPOINT_KINDS = ("plan", "results", "other")
OUTCOMES = ("proceed", "loop:HYPOTHESES", "loop:RUNNING")

# what the step at each stage is for, and which skill covers it (PLAN_REVIEW and RESULTS_REVIEW depend on the
# checkpoint's state: see Dossier.next_step)
PLAYBOOK = {
    "NEW": ("analyse the target: construct mapping, organism and family (UniProt, Pfam / InterPro), the literature on it "
            "and its partners (the site a partner binds, the chemistry, the oligomeric state, each with its source), "
            "public structures with their identity (`residudes structures T`, and `--hhsearch` for remote homologues), "
            "MSA depth per chain; then write it up in `notes/target.md`",
            "residudes-intake"),
    "IDENTIFIED": ("form stoichiometry hypotheses with evidence", "residudes-stoichiometry"),
    "HYPOTHESES": ("design input variants (MSA, templates, PTMs, ligands) and the sampling plan (seeds, GPU hours) as decisions; then move to PLAN_REVIEW", "residudes-intake"),
    "INPUTS_READY": ("build the agreed inputs and submit the prediction runs with residudes-engine (within the GPU budget), each "
                     "into the layout `residudes pool` reads: `--out results/<sto>/<variant>/<runner>` (e.g. results/A1B1/sep/af3), "
                     "which makes seed_<N>/sample_<M> under it; if predictions already exist under targets_root, import them with "
                     "`residudes pool T` instead and move on", "residudes-ops-safety"),
    "RUNNING": ("check `residudes jobs T`; when every run is finished, `residudes pool T` to import the models and verify the counts per predictor", "residudes-ops-safety"),
    "POOL_READY": ("`residudes qa T --sto S --what interface` and `--what cluster`; draft the structural gates with `residudes gate T --init "
                   "--sto S` (site gates from the evidence), check and edit the draft against what the analysis supports (sites, "
                   "references), and run it with `residudes gate T --spec <draft> --sto S` (sbatch: stop and let the next step continue "
                   "while they run); when they are done, `residudes gate T --show` and move to RESULTS_REVIEW", "residudes-selection"),
    "SLATE_PROPOSED": ("wait for a person to run `residudes approve`", None),
    "APPROVED": ("package the approved slate: `residudes package T --server|--human` (geometry checks, polymer-only TS files "
                 "through casp17-utils); read package/checks.md and report anything a person should look at. If the checks "
                 "show clashes or non-proline cis peptides, `residudes relax T` repairs them under restraints and says what "
                 "changed - a person decides whether to submit the relaxed file. Then the final report: write your own "
                 "report scenes as PyMOL in report/pml/ and run `residudes slides T --final`, which ray-traces every "
                 "picture through sbatch and waits for it", "residudes-selection"),
    "PACKAGED": ("a person submits; record it with `residudes stage T SUBMITTED`", None),
    "SUBMITTED": ("wait for the native structure; on a stage-2 reissue go back to HYPOTHESES", None),
    "RETROSPECTIVE": ("score each recorded decision against the native and write the lessons", None),
}


PERSON_MOVES = ("APPROVED", "SUBMITTED")          # a person approves a slate and submits it; never the agent
# Submission is the one move no mandate can delegate: entering a prediction for a target is an act
# in someone's name, outside this machine, and not undoable. Everything before it can be automated
# by a person who says so in writing - see authorise_automation.
NEVER_AUTOMATED = ("SUBMITTED",)
LOOPS = {("RESULTS_REVIEW", "HYPOTHESES"), ("RESULTS_REVIEW", "RUNNING"), ("PLAN_REVIEW", "HYPOTHESES")}   # via `resolve`
AGENT_ACTORS = ("agent",)


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


_EPOCH = dt.datetime.min.replace(tzinfo=dt.timezone.utc)


def tkey(t: str) -> dt.datetime:
    """A stored time as a comparable instant: entries from hosts in different time zones (a KST login node, a UTC pod)
    compare correctly, which their ISO strings do not. '' and unparsable values sort first."""
    try:
        x = dt.datetime.fromisoformat(t)
        return x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return _EPOCH


# Chain letters in order: A…Z, then AA, AB, … — the ids AlphaFold 3 accepts, and what the
# engine assigns. Past Z, chr(ord("A") + i) hands out "[", "\\" and "]".
CHAIN_LETTERS = [chr(ord("A") + i) for i in range(26)] + [a + b for a in (chr(ord("A") + i) for i in range(26))
                                                          for b in (chr(ord("A") + i) for i in range(26))]


def chain_letter(i: int) -> str:
    return CHAIN_LETTERS[i] if i < len(CHAIN_LETTERS) else f"X{i}"


@dataclass
class Dossier:
    target: str
    stage: str = "NEW"
    created: str = field(default_factory=now)
    sequence: dict = field(default_factory=dict)        # fasta path, chains, construct notes
    stoichiometry: dict = field(default_factory=dict)   # given (e.g. by CASP) and hypotheses
    deadlines: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)        # public_structures, msa, literature, ...
    decisions: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    slate: dict = field(default_factory=dict)
    approvals: list = field(default_factory=list)
    checkpoints: list = field(default_factory=list)     # conversations with a person: plan, results (see consult())
    automation: dict = field(default_factory=dict)      # a person's standing mandate, if any (see authorise_automation)
    log: list = field(default_factory=list)

    # ------------------------------------------------------------------ io
    @staticmethod
    def path(targets_dir: Path, target: str) -> Path:
        return Path(targets_dir) / target / "dossier.json"

    @classmethod
    def load(cls, targets_dir: Path, target: str) -> "Dossier":
        p = cls.path(targets_dir, target)
        if not p.exists():
            raise FileNotFoundError(f"no dossier for {target} at {p} (create it with `residudes target add {target}`)")
        d = json.loads(p.read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    @contextlib.contextmanager
    def locked(cls, targets_dir: Path, target: str):
        """Load, let the caller change, save — holding an exclusive lock, so that the daemon, an agent step and a
        person's `residudes` command never overwrite each other's changes with a stale copy."""
        p = cls.path(targets_dir, target)
        if not p.exists():
            raise FileNotFoundError(f"no dossier for {target} at {p} (create it with `residudes target add {target}`)")
        with open(p.parent / ".dossier.lock", "w") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                d = cls.load(targets_dir, target)
                yield d
                d.save(targets_dir)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def save(self, targets_dir: Path) -> Path:
        p = self.path(targets_dir, self.target)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False))
        os.replace(tmp, p)
        return p

    # ------------------------------------------------------------------ changes
    def note(self, actor: str, text: str) -> None:
        self.log.append({"time": now(), "actor": actor, "note": text})

    def _entered(self, stage: str) -> str:
        """When the dossier last moved into `stage` ('' if never)."""
        return max((x["time"] for x in self.log if x["note"].startswith("stage ") and x["note"].split(":")[0].endswith(f"-> {stage}")),
                   default="", key=tkey)

    def move(self, stage: str, actor: str, why: str = "") -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}; one of {STAGES}")
        if stage != self.stage and stage not in NEXT.get(self.stage, set()):
            raise ValueError(f"{self.target}: {self.stage} -> {stage} is not allowed (next: {sorted(NEXT.get(self.stage, []))})")
        if (stage in PERSON_MOVES or (self.stage, stage) in LOOPS) and actor in AGENT_ACTORS \
                and not (self.automation_allows(stage) and stage not in NEVER_AUTOMATED):
            raise ValueError(f"{self.stage} -> {stage} is a person's move, not the agent's"
                             + (" and no mandate can delegate it" if stage in NEVER_AUTOMATED else "")
                             + (" (a person resolves the checkpoint with `residudes resolve`)" if (self.stage, stage) in LOOPS else ""))
        if stage == "APPROVED" and not self.fresh_approval():
            raise ValueError("APPROVED needs an approval recorded after the slate was proposed: a person runs `residudes approve`")
        if self.stage == "PLAN_REVIEW" and stage == "INPUTS_READY" and not self._agreed("plan"):
            raise ValueError("the prediction plan needs a person's agreement: open a plan checkpoint (`residudes consult T --kind plan`) "
                             "and wait for `residudes resolve T C# --outcome proceed`")
        if self.stage == "RESULTS_REVIEW" and stage == "SLATE_PROPOSED":
            review = self.latest("results_review", since=self._entered("RESULTS_REVIEW"))
            if review is None:
                raise ValueError("record the review of the first results first (`residudes review T`)")
            if review.get("value", {}).get("triggers") and not self._agreed("results", since=review["time"]):
                raise ValueError(f"the first results raised {', '.join(review['value']['triggers'])}: talk it through with a person "
                                 "(`residudes consult T --kind results`) and wait for their decision")
        self.note(actor, f"stage {self.stage} -> {stage}" + (f": {why}" if why else ""))
        self.stage = stage

    # ------------------------------------------------------------------ evidence helpers
    def latest(self, key: str, since: str = "") -> dict | None:
        return next((e for e in reversed(self.evidence.get(key, [])) if tkey(e.get("time", "")) >= tkey(since)), None)

    # ------------------------------------------------------------------ checkpoints: structured conversations with a person
    def consult(self, actor: str, kind: str, questions: list[str], brief: str = "") -> dict:
        """Open a checkpoint: the agent (or a person) lays out the analysis and asks; a person answers and resolves."""
        if kind not in CHECKPOINT_KINDS:
            raise ValueError(f"checkpoint kind must be one of {CHECKPOINT_KINDS}")
        if any(c["status"] == "open" and c["kind"] == kind for c in self.checkpoints):
            raise ValueError(f"a {kind} checkpoint is already open: reply to it instead")
        c = {"id": f"C{len(self.checkpoints) + 1}", "kind": kind, "stage": self.stage, "time": now(), "by": actor,
             "brief": brief, "questions": questions, "status": "open", "thread": [], "resolution": None}
        self.checkpoints.append(c)
        self.note(actor, f"CHECKPOINT {c['id']} ({kind}) opened: " + " | ".join(questions)[:300])
        return c

    def checkpoint(self, cid: str) -> dict:
        c = next((c for c in self.checkpoints if c["id"] == cid), None)
        if c is None:
            raise ValueError(f"no checkpoint {cid}")
        return c

    def reply(self, actor: str, cid: str, text: str) -> dict:
        c = self.checkpoint(cid)
        if c["status"] != "open":
            raise ValueError(f"{cid} is {c['status']}")
        entry = {"time": now(), "by": actor, "text": text}
        c["thread"].append(entry)
        self.note(actor, f"{cid} reply: {text[:300]}")
        return entry

    def resolve(self, actor: str, cid: str, outcome: str, note: str = "") -> dict:
        if actor in AGENT_ACTORS and not self.automation_allows("resolve"):
            raise ValueError("a person resolves a checkpoint, not the agent "
                             "(or a person turns automation on: `residudes automate T --on`)")
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        c = self.checkpoint(cid)
        if c["status"] != "open":
            raise ValueError(f"{cid} is {c['status']}")
        c["status"], c["resolution"] = "resolved", {"time": now(), "by": actor, "outcome": outcome, "note": note,
                                                    **self._automation_mark(actor)}
        self.note(actor, f"{cid} resolved: {outcome}" + (f": {note}" if note else ""))
        if outcome.startswith("loop:"):
            self.move(outcome.split(":", 1)[1], actor, f"{cid}: {note or 'another round'}")
        return c

    def open_checkpoint(self) -> dict | None:
        return next((c for c in self.checkpoints if c["status"] == "open"), None)

    def agents_turn(self, c: dict) -> bool:
        """An open checkpoint waits for a person until a person has written the latest message in it."""
        return bool(c["thread"]) and c["thread"][-1]["by"] not in AGENT_ACTORS + ("residudes",)

    def _agreed(self, kind: str, since: str = "") -> bool:
        since = max(since, self._entered("PLAN_REVIEW" if kind == "plan" else "RESULTS_REVIEW"), key=tkey)
        # A person's resolution counts; so does the agent's when it was made under a mandate, which it carries on the
        # record (`automated`, and who granted it). Without the second clause a mandate let the agent resolve the plan
        # checkpoint and then refused the move that resolution was for, so full automation stopped at PLAN_REVIEW.
        return any(c["kind"] == kind and c["status"] == "resolved" and c["resolution"]["outcome"] == "proceed"
                   and (c["resolution"]["by"] not in AGENT_ACTORS or c["resolution"].get("automated"))
                   and tkey(c["resolution"]["time"]) >= tkey(since) for c in self.checkpoints)

    def decide(self, actor: str, kind: str, chose: str, evidence: list[str], alternatives: list[str] | None = None,
               confidence: str = "medium", reversible_until: str | None = None) -> dict:
        if kind not in DECISION_TYPES:
            raise ValueError(f"decision type must be one of {DECISION_TYPES}")
        if not evidence:
            raise ValueError("a decision needs at least one piece of evidence")
        d = {"id": f"D{len(self.decisions) + 1}", "time": now(), "actor": actor, "type": kind, "chose": chose,
             "alternatives": alternatives or [], "evidence": evidence, "confidence": confidence,
             "reversible_until": reversible_until, "outcome": None}
        self.decisions.append(d)
        return d

    def fresh_approval(self) -> dict | None:
        """The newest approval given after the latest move into SLATE_PROPOSED (a stage-2 reissue needs a new one)."""
        proposed = max((x["time"] for x in self.log if x["note"].startswith("stage ") and x["note"].split(":")[0].endswith("-> SLATE_PROPOSED")),
                       default="", key=tkey)
        return next((a for a in reversed(self.approvals) if tkey(a["time"]) >= tkey(proposed)), None)

    # ------------------------------------------------------- the automation mandate
    def authorise_automation(self, person: str, until: str = "PACKAGED", note: str = "") -> dict:
        """Let the agent take the moves normally reserved for a person, up to but never including
        submission. Only a person can grant this, it names who granted it, and it stops at `until`."""
        if person in AGENT_ACTORS:
            raise ValueError("only a person can turn automation on")
        if until not in STAGES:
            raise ValueError(f"unknown stage {until}; one of {STAGES}")
        if until == "SUBMITTED" or STAGES.index(until) > STAGES.index("PACKAGED"):
            raise ValueError("automation cannot reach SUBMITTED: submitting is a person's act, always")
        self.automation = {"by": person, "time": now(), "until": until, "note": note}
        self.note(person, f"automation on, up to {until}" + (f": {note}" if note else ""))
        return self.automation

    def revoke_automation(self, person: str, note: str = "") -> None:
        if person in AGENT_ACTORS:
            raise ValueError("only a person can turn automation off")
        if not self.automation:
            raise ValueError("automation is not on")
        was = self.automation
        self.automation = {}
        self.note(person, f"automation off (granted by {was.get('by')} at {was.get('time')})"
                          + (f": {note}" if note else ""))

    def automation_allows(self, what: str) -> bool:
        """True when a person's mandate covers this action at the dossier's current stage."""
        a = self.automation
        if not a or what in NEVER_AUTOMATED:
            return False
        until = a.get("until", "PACKAGED")
        return STAGES.index(self.stage) < STAGES.index(until) or self.stage == until

    def _automation_mark(self, actor: str) -> dict:
        """What an automated action records, so that the log never blurs the two."""
        if actor not in AGENT_ACTORS or not self.automation:
            return {}
        return {"automated": True, "mandate": {"by": self.automation.get("by"),
                                               "time": self.automation.get("time")}}

    def approve(self, person: str, what: str) -> None:
        if person in AGENT_ACTORS and not self.automation_allows("approve"):
            raise ValueError("approve is for people only (or a person turns automation on: `residudes automate T --on`)")
        self.approvals.append({"time": now(), "by": person, "what": what, **self._automation_mark(person)})
        self.note(person, f"approved: {what}"
                  + (f" [automated under the mandate from {self.automation.get('by')}]" if person in AGENT_ACTORS else ""))

    def agent_may_act(self) -> bool:
        """Whether a step of the agent's has anything to do now: the one rule the daemon, `residudes agent step`
        and the brief all follow, so they cannot disagree about where a target stops.

        Without a mandate the agent stops at every checkpoint that waits for a person and at the slate. With one
        it answers those itself, on the record, the way a person would have. Two stops hold in every case:
        PACKAGED, because what follows is submission, an act in someone's name that no mandate delegates; and
        SUBMITTED, which waits for the native."""
        if self.stage in ("PACKAGED", "SUBMITTED"):
            return False
        mandate = self.automation_allows(self.stage)
        c = self.open_checkpoint()
        if c and not self.agents_turn(c) and not mandate:
            return False
        if self.stage == "SLATE_PROPOSED" and not mandate:
            return False
        return True

    def next_step(self) -> tuple[str, str | None]:
        c = self.open_checkpoint()
        mandate = self.automation_allows(self.stage)
        by = self.automation.get("by", "") if mandate else ""
        if c and mandate and not self.agents_turn(c):
            # The checkpoint was opened as for a person, with the same brief; under the mandate the agent answers it.
            return (f"automation is on (granted by {by}): answer checkpoint {c['id']} ({c['kind']}) yourself, as the "
                    f"person it was written for would. Read it (`residudes checkpoints {self.target} {c['id']}`), weigh "
                    "the options against the evidence, write your answer and the reasons with `residudes reply "
                    f"{self.target} {c['id']} \"...\"`, then `residudes resolve {self.target} {c['id']} --outcome "
                    "proceed` (or `loop:HYPOTHESES` / `loop:RUNNING` if your own reading finds the plan or the pool "
                    "wanting). Write NEEDS HUMAN instead only when the evidence cannot decide it", "residudes-consult")
        if c and self.agents_turn(c):
            return (f"a person replied in checkpoint {c['id']} ({c['kind']}): read the thread (`residudes checkpoints {self.target} {c['id']}`), "
                    f"answer with `residudes reply {self.target} {c['id']} \"...\"`, and revise the plan or decisions if they asked; "
                    + ("then resolve it yourself under the mandate" if mandate else "do not resolve it yourself"), "residudes-consult")
        if c:
            return (f"wait for a person in checkpoint {c['id']} ({c['kind']})", None)
        if self.stage == "SLATE_PROPOSED" and mandate:
            return (f"automation is on (granted by {by}): approve the slate as a person would - read it against the "
                    "structural gates and the decisions once more (`residudes slate T --show`, `residudes gate T --show`), "
                    f"then `residudes approve {self.target} \"<why these five>\"` and `residudes stage {self.target} "
                    "APPROVED`. Write NEEDS HUMAN instead if a gate fails on model 1 or the slate contradicts a decision",
                    "residudes-selection")
        if self.stage == "PLAN_REVIEW":
            if self._agreed("plan"):
                return ("the plan is agreed: move to INPUTS_READY", None)
            return ("talk the prediction plan through with a person: build the slide report (`residudes slides T`, add your own "
                    "slides in report/agent.typ), write the brief (analysis, the priors behind each "
                    "decision, the plan, the open questions with options and your recommendation, a pointer to report/report.pdf) and open "
                    f"`residudes consult {self.target} --kind plan --brief FILE --ask ...`; then stop", "residudes-consult")
        if self.stage == "RESULTS_REVIEW":
            review = self.latest("results_review", since=self._entered("RESULTS_REVIEW"))
            if review is None:
                return ("review the first results: `residudes review T` for the standard checks, then add what you see "
                        "(architectures, gates, disagreement with the priors); anything unusual goes to a person with "
                        f"`residudes consult {self.target} --kind results`", "residudes-consult")
            if review.get("value", {}).get("triggers") and not self._agreed("results", since=review["time"]):
                return (f"the review raised {', '.join(review['value']['triggers'])}: rebuild the slide report with pictures of "
                        "what you saw (`residudes slides T`, your own PyMOL scenes via `residudes render`), open a results checkpoint "
                        "with the options (more sampling, a new variant, another stoichiometry, proceed) and stop", "residudes-consult")
            return ("`residudes slate T --sto S`, review it against the structural gates, record the five with `residudes slate "
                    "T --models ...` and `residudes decide --type selection`, rebuild the slide report (`residudes slides T`) so the "
                    "person sees the five structures, move to SLATE_PROPOSED", "residudes-selection")
        return PLAYBOOK[self.stage]


def list_targets(targets_dir: Path) -> list[Dossier]:
    out = []
    for p in sorted(Path(targets_dir).glob("*/dossier.json")):
        try:
            out.append(Dossier.load(targets_dir, p.parent.name))
        except (json.JSONDecodeError, TypeError):
            continue
    return out
