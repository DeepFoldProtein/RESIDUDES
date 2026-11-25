"""`residudes daemon`: the scheduler. One process, no LLM, polls at most once a minute.

Each round, for every dossier (or the ones named):
  - where the dossier says the agent may not act (Dossier.agent_may_act): skip. Without a mandate that is every
    checkpoint waiting for a person and the slate; with one (the default from `target add`) the agent answers those
    itself, and only PACKAGED (submission is a person's act) and SUBMITTED stop it. RETROSPECTIVE: skip
  - an open checkpoint where a person wrote last: run an agent step so the agent answers (back and forth)
  - a chat session open for that target: skip, because a person is working on it there
  - a "NEEDS HUMAN" note newer than the last step: skip until a person writes a newer note
  - at RUNNING / POOL_READY with recorded jobs (Slurm or local pueue) not finished: refresh their state, skip
    (other stages go on while jobs run; e.g. a QA job submitted early does not hold up input design)
  - otherwise: run one agent step (residudes.agent.step), within the per-step budget
A target that makes no stage progress in `--max-idle-steps` consecutive steps is parked with a NEEDS HUMAN note.
An error on one target (agent CLI, Slurm, a broken dossier) is logged to that dossier; the other targets go on.
Every dossier change happens under the dossier's file lock, re-read fresh.
This replaces one watcher per job (which once froze GPFS) with a single loop: one daemon per work_root (a second one
refuses to start), and one agent step per target at a time (a step lock; a second step returns `busy`).
"""
from __future__ import annotations

import fcntl
import os
import time

from residudes import ops
from residudes.agent import step
from residudes.config import Config
from residudes.dossier import Dossier, list_targets

WAITS_FOR_JOBS = ("RUNNING", "POOL_READY")                     # stages whose work needs the jobs' results


def _waiting_on_jobs(cfg: Config, target: str) -> bool:
    """Refresh the recorded jobs (re-reading the dossier under its lock) and say whether any is still active."""
    d = Dossier.load(cfg.targets_dir, target)
    if d.stage not in WAITS_FOR_JOBS or not any(ops.tracked(r) for r in d.runs):
        return False
    with Dossier.locked(cfg.targets_dir, target) as d:
        ops.jobs_cmd(d)
        return any(ops.active(r) for r in d.runs if ops.tracked(r))


def _needs_human(d: Dossier) -> bool:
    for x in reversed(d.log):
        if x["note"].startswith("NEEDS HUMAN"):
            return True
        if x["actor"] not in ("agent", "residudes"):              # a person wrote after the request
            return False
    return False


def api_error(res: dict) -> str | None:
    """The first line of an API error that ended an agent step (a safeguard refusal, an outage), else None.
    A step that only ran out of turns or budget is not one: it counts as an idle step, as before."""
    text = str(res.get("result") or "").strip()
    if text.startswith("API Error") or res.get("subtype") == "error_during_execution":
        return (text.splitlines()[0] if text else str(res.get("subtype")))[:200]
    return None                     # error_max_turns / error_max_budget_usd are limits, not errors: an idle step


def _step_through_conversation(cfg: Config, d: Dossier) -> str:
    """Ask the target's session to take the next step.

    One session per target: the agent that works on it and the conversation a person joins are
    the same agent, so there is one context and a person can watch or interrupt. The cost is the
    accounting - a step asked this way has no turn count, no price and no structured result,
    because tmux reports none of that. The dossier is the record, which is where the work belongs.
    """
    from residudes import chat
    if not chat.ensure(cfg, d.target):
        return f"{d.target}: no session and none could be started (tmux? an agent CLI?)"
    goal, skill = d.next_step()
    ask = (f"Next step for {d.target} ({d.stage}): {goal}"
           + (f" Follow the {skill} skill." if skill else "")
           + " Record what you find and decide in the dossier as usual.")
    if chat.send(d.target, ask):
        return f"{d.target}: asked the open conversation to take the next step"
    return f"{d.target}: in conversation, but it did not accept the step"


def _round_target(cfg: Config, d: Dossier, budget: float | None, max_idle_steps: int, idle: dict,
                  ) -> str:
    if d.stage == "RETROSPECTIVE":
        return f"{d.target}: {d.stage} (waiting for a person)"
    c = d.open_checkpoint()
    if not d.agent_may_act():                                # the dossier decides, mandate included
        if c and not d.agents_turn(c):
            return f"{d.target}: waiting for a person at checkpoint {c['id']} ({c['kind']})"
        return f"{d.target}: {d.stage} (waiting for a person)"
    if _needs_human(d) and not c:
        return f"{d.target}: NEEDS HUMAN"
    if _waiting_on_jobs(cfg, d.target):
        return f"{d.target}: waiting on jobs"
    before = d.stage
    # A step runs for minutes, and the round's lines are printed when it ends; without this a fresh daemon.log
    # stays empty long enough to look like a dead daemon.
    print(time.strftime("%H:%M:%S"), f"{d.target}: step at {d.stage} started", flush=True)
    res = step(cfg, d.target, budget=budget, verbose=False)
    after = Dossier.load(cfg.targets_dir, d.target).stage
    if res.get("subtype") in ("busy", "skipped"):              # nothing ran: not an idle step
        return f"{d.target}: {res.get('subtype')}"
    err = api_error(res)
    if err:                                                    # a refusal or an outage: retrying every round only costs
        with Dossier.locked(cfg.targets_dir, d.target) as dd:
            dd.note("residudes", f"NEEDS HUMAN: the agent step ended with an API error ({err}); a person decides whether "
                               "to retry, change the model, or take the step over")
        idle[d.target] = 0
        return f"{d.target}: error {err} (parked: NEEDS HUMAN)"
    idle[d.target] = 0 if after != before else idle.get(d.target, 0) + 1
    line = f"{d.target}: step {res.get('subtype')} ${res.get('cost_usd')} {before} -> {after}"
    if idle[d.target] >= max_idle_steps:
        with Dossier.locked(cfg.targets_dir, d.target) as dd:
            dd.note("residudes", f"NEEDS HUMAN: {max_idle_steps} agent steps without stage progress at {after}")
        idle[d.target] = 0                                     # after the person's answer it gets the full allowance again
    return line


def round_once(cfg: Config, targets: list[str] | None, budget: float | None, max_idle_steps: int,
               idle: dict) -> list[str]:
    report = []
    for d in list_targets(cfg.targets_dir):
        if targets and d.target not in targets:
            continue
        try:
            d = Dossier.load(cfg.targets_dir, d.target)        # fresh: earlier targets in this round may have taken minutes
            report.append(_round_target(cfg, d, budget, max_idle_steps, idle))
        except Exception as exc:                               # one target's failure never stops the others
            report.append(f"{d.target}: error {type(exc).__name__}: {exc}")
            try:
                with Dossier.locked(cfg.targets_dir, d.target) as dd:
                    dd.note("residudes", f"daemon error: {type(exc).__name__}: {str(exc)[:300]}")
            except Exception:
                pass
    return report


def run(cfg: Config, targets: list[str] | None, interval: int, once: bool, budget: float | None, max_idle_steps: int = 3,
        exit_when_idle: bool = False) -> int:
    interval = max(60, interval)
    idle: dict = {}
    cfg.targets_dir.mkdir(parents=True, exist_ok=True)
    lock = open(cfg.targets_dir.parent / ".daemon.lock", "a+")      # one daemon per work_root
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        print(f"another residudes daemon is running on {cfg.targets_dir.parent} ({lock.read().strip() or 'pid unknown'}); not starting a second one",
              flush=True)
        return 1
    lock.seek(0); lock.truncate(); lock.write(f"pid {os.getpid()} since {time.strftime('%H:%M:%S')}"); lock.flush()
    while True:
        lines = round_once(cfg, targets, budget, max_idle_steps, idle)
        for line in lines:
            print(time.strftime("%H:%M:%S"), line, flush=True)
        if once:
            return 0
        stop = cfg.targets_dir.parent / ".daemon.stop"
        if stop.exists():                                    # `residudes daemon --stop`: finish the round, then exit
            stop.unlink()
            print(time.strftime("%H:%M:%S"), "stop requested: exiting after the round", flush=True)
            return 0
        if exit_when_idle and not any(": step " in l or "waiting on jobs" in l or ": error " in l for l in lines):
            print(time.strftime("%H:%M:%S"), "nothing left for the agent: every target waits for a person", flush=True)
            return 0
        time.sleep(interval)
