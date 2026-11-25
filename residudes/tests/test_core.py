import json

import pytest

from residudes import guard
from residudes.dossier import Dossier, STAGES


def _to_slate(d):
    """Walk a dossier from its stage to SLATE_PROPOSED the way a campaign does: the person agrees the plan, the first
    results raise nothing."""
    for st in ("IDENTIFIED", "HYPOTHESES", "PLAN_REVIEW"):
        if STAGES.index(d.stage) < STAGES.index(st):
            d.move(st, "agent")
    c = d.consult("agent", "plan", ["A3 and A1 hedge, 128 seeds each?"])
    d.resolve("person", c["id"], "proceed")
    for st in ("INPUTS_READY", "RUNNING", "POOL_READY", "RESULTS_REVIEW"):
        d.move(st, "agent")
    d.evidence["results_review"] = [{"time": "9999-12-31T00:00:00+00:00", "by": "agent", "value": {"triggers": []}}]
    d.move("SLATE_PROPOSED", "agent")


@pytest.mark.parametrize("cmd", [
    "rm -rf /data/x", "cd a && rm -fr b", "rm --recursive old", "find / -name x", "find /gpfs -name '*.cif'",
    "du -sh /store", "scancel -u deepfold", "pueue clean", "pueue kill --all", "git push -f origin main",
    "casp17 submit model_1.pdb", "curl https://predictioncenter.org/casp17/submit",
    "while true; do squeue -u me; sleep 5; done",
    # found by review: flag splitting, absolute paths, python, option values, globs, other loop shapes, approval
    "rm -f -r X", "/bin/rm -rf X", "sudo rm -rf x", "python3 -c 'import shutil; shutil.rmtree(\"x\")'",
    "find . -name '*.tmp' -delete", "find qa -exec rm {} +", "du -d 1 /gpfs", "du -sh /gpfs/*", "ls -R /store",
    "while sleep 5; do squeue; done", "watch -n 5 squeue", "until false; do pueue status; sleep 30s; done",
    "RESIDUDES_ACTOR=bob residudes approve T --what v1", "env -u RESIDUDES_ACTOR residudes approve T --what v1",
    "python -m casp17_utils.cli submit x.pdb", "uv run python -c 'import casp17_utils.net'",
    "bin/residudes approve T1417 --what x", "export RESIDUDES_ACTOR=me", "echo x && residudes approve T --what y",
])
def test_guard_blocks(cmd):
    assert guard.check(cmd), cmd


@pytest.mark.parametrize("cmd", [
    "rm model.cif", "find targets/T1417/results -name model.cif", "du -sh targets/T1417", "scancel 12345",
    "pueue status", "git push origin feat/x", "sbatch run.sh", "ls -la /data/casp", "mv old /data/trash/",
    "residudes target show T1417", "residudes slate T1417 --sto A3", "du -sh /data/casp17/targets/T1417",
    "until ! squeue -h -j 1 | grep -q .; do sleep 60; done", "ls /data/casp17/targets/T1417/",
    "rm -f qa/tmp.csv", "grep -r approve skills/",
])
def test_guard_allows(cmd):
    assert guard.check(cmd) is None, cmd


def test_dossier_flow(tmp_path):
    d = Dossier(target="T9999")
    d.save(tmp_path)
    d = Dossier.load(tmp_path, "T9999")
    assert d.stage == "NEW"
    with pytest.raises(ValueError):
        d.move("POOL_READY", "t")                       # cannot skip stages
    with pytest.raises(ValueError):
        d.decide("t", "stoichiometry", "A3", [])         # evidence required
    dec = d.decide("t", "stoichiometry", "A3", ["n-sweep peak at A3 on af3 and pt2"], ["A1"], "high")
    assert dec["id"] == "D1"
    _to_slate(d)
    assert d.stage == "SLATE_PROPOSED"
    with pytest.raises(ValueError):
        d.move("APPROVED", "agent")                      # a person must approve first
    d.approve("person", "slate v1")
    d.move("APPROVED", "person")
    d.save(tmp_path)
    saved = json.loads((tmp_path / "T9999" / "dossier.json").read_text())
    assert saved["stage"] == "APPROVED" and saved["decisions"][0]["alternatives"] == ["A1"]


@pytest.mark.asyncio
async def test_hook_shape():
    out = await guard.pre_tool_use({"tool_name": "Bash", "tool_input": {"command": "rm -rf x"}}, None, None)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert await guard.pre_tool_use({"tool_name": "Bash", "tool_input": {"command": "ls"}}, None, None) == {}


def test_daemon_needs_human():
    from residudes.daemon import _needs_human
    d = Dossier(target="T")
    d.note("agent", "did intake")
    assert not _needs_human(d)
    d.note("agent", "NEEDS HUMAN: which construct?")
    assert _needs_human(d)
    d.note("residudes", "agent step: success")
    assert _needs_human(d)                         # bookkeeping lines do not clear the request
    d.note("person", "use the full-length construct")
    assert not _needs_human(d)                     # a person answered


def test_export_leaves_out_native_results(tmp_path):
    from residudes import export
    d = Dossier(target="T")
    d.decide("agent", "selection", "slate v1", ["z"])
    d.decisions[0]["outcome"] = {"model1": {"lddt": 0.9}}
    d.evidence["retrospective"] = [{"value": {"model1_lddt": 0.9}}]
    d.evidence["pool"] = [{"value": {"n_models": 1}}]
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "retro.md").write_text("lddt 0.9")
    export.run(d, tmp_path / "src", tmp_path / "out")
    j = json.loads((tmp_path / "out" / "dossier.json").read_text())
    assert "retrospective" not in j["evidence"] and j["decisions"][0]["outcome"] is None and "pool" in j["evidence"]
    assert not (tmp_path / "out" / "retro.md").exists()
    assert d.decisions[0]["outcome"] is not None                 # the dossier itself is untouched


def test_export_cuts_at_first_native_result(tmp_path):
    from residudes import export
    d = Dossier(target="T")
    d.note("agent", "intake done")
    d.evidence["native"] = [{"time": "2099-01-01T00:00:00+00:00", "value": {"path": "n.pdb"}}]
    d.log.append({"time": "2099-01-02T00:00:00+00:00", "actor": "agent", "note": "lesson: model 1 lDDT 0.82, field #5"})
    d.evidence["lessons"] = [{"time": "2099-01-02T00:00:00+00:00", "value": "model 1 lDDT 0.82"}]
    (tmp_path / "src" / "agent_logs").mkdir(parents=True)
    (tmp_path / "src" / "agent_logs" / "20990103-000000_RETROSPECTIVE.log").write_text("lDDT 0.82")
    export.run(d, tmp_path / "src", tmp_path / "out")
    j = json.loads((tmp_path / "out" / "dossier.json").read_text())
    assert [x["note"] for x in j["log"]] == ["intake done"] and "lessons" not in j["evidence"] and "native" not in j["evidence"]
    assert not list((tmp_path / "out" / "agent_logs").glob("*"))


def test_stage2_needs_new_approval_and_agent_cannot_approve():
    d = Dossier(target="T")
    _to_slate(d)
    with pytest.raises(ValueError):
        d.approve("agent", "slate v1")
    d.approve("person", "slate v1")
    with pytest.raises(ValueError):
        d.move("APPROVED", "agent")                      # a person's move
    d.move("APPROVED", "person")
    d.move("PACKAGED", "agent")
    d.move("SUBMITTED", "person")
    d.move("HYPOTHESES", "agent", "stage-2 reissue")
    for s in ("INPUTS_READY", "RUNNING", "POOL_READY", "SLATE_PROPOSED"):
        d.log.append({"time": "2099-01-01T00:00:00+00:00", "actor": "agent", "note": f"stage x -> {s}"})
        d.stage = s
    with pytest.raises(ValueError):
        d.move("APPROVED", "person")                     # the stage-1 approval is stale


def test_daemon_isolates_errors_and_resets_idle(tmp_path, monkeypatch):
    from residudes import daemon
    from residudes.config import Config
    cfg = Config(work_root=str(tmp_path))
    for t in ("T1", "T2"):
        Dossier(target=t).save(cfg.targets_dir)

    def fake_step(cfg, target, budget=None, verbose=False):
        if target == "T1":
            raise RuntimeError("claude CLI exited 1")
        return {"subtype": "success", "cost_usd": 0.1}       # no stage progress
    monkeypatch.setattr(daemon, "step", fake_step)
    idle: dict = {}
    rep = daemon.round_once(cfg, None, None, 1, idle)
    assert any(l.startswith("T1: error RuntimeError") for l in rep) and any(l.startswith("T2: step success") for l in rep)
    assert "daemon error" in Dossier.load(cfg.targets_dir, "T1").log[-1]["note"]
    assert Dossier.load(cfg.targets_dir, "T2").log[-1]["note"].startswith("NEEDS HUMAN") and idle["T2"] == 0


def test_report_links_priors_to_decisions():
    from residudes import report
    d = Dossier(target="T")
    d.log.append({"time": "2026-01-01T10:00:00+09:00", "actor": "agent", "note": "stage NEW -> IDENTIFIED"})
    d.evidence["msa"] = [{"time": "2026-01-01T09:00:00+09:00", "by": "agent", "value": {"depth": 3}}]
    d.evidence["stoichiometry_evidence"] = [{"time": "2026-01-01T10:30:00+09:00", "by": "agent", "value": "A2 peak"}]
    d.evidence["gates_A2"] = [{"time": "2026-01-01T11:00:00+09:00", "by": "agent", "value": "ok"}]
    d.decisions = [
        {"id": "D1", "time": "2026-01-01T10:00:00+09:00", "type": "variant", "chose": "custom MSA", "evidence": ["msa"], "alternatives": []},
        {"id": "D2", "time": "2026-01-01T10:40:00+09:00", "type": "stoichiometry", "chose": "A2",
         "evidence": ["n-sweep peaks at A2", "OPERATOR said UNK"], "alternatives": ["A1"]},
        {"id": "D3", "time": "2026-01-01T11:00:00+09:00", "type": "selection", "chose": "v2",
         "evidence": ["per D2; gates_A2.csv; af3_msa models only"], "alternatives": []},
    ]
    lk = report.links(d)
    assert lk["D1"] == {"evidence": ["msa"], "decisions": [], "checkpoints": [], "person": False, "stage": "NEW"}   # same-second move comes after
    assert lk["D2"]["evidence"] == ["stoichiometry_evidence"] and lk["D2"]["person"] and lk["D2"]["stage"] == "IDENTIFIED"
    assert lk["D3"]["evidence"] == ["gates_A2"] and lk["D3"]["decisions"] == ["D2"]           # not msa (af3_msa)
    md = report.build(d)
    assert "## Prior → decision map" in md and "| `msa` | ● |  |  |" in md


def test_plan_checkpoint_is_a_conversation():
    d = Dossier(target="T")
    for st in ("IDENTIFIED", "HYPOTHESES", "PLAN_REVIEW"):
        d.move(st, "agent")
    with pytest.raises(ValueError):
        d.move("INPUTS_READY", "agent")                  # no plan agreed yet
    assert "talk the prediction plan through" in d.next_step()[0]
    c = d.consult("agent", "plan", ["A2 primary with an A1 hedge (rec.), or A1 only?"])
    assert d.next_step()[0].startswith("wait for a person")
    d.reply("person", c["id"], "why not A3? the family has trimers")
    assert d.agents_turn(c) and d.next_step()[1] == "residudes-consult"       # the agent answers
    d.reply("agent", c["id"], "A3 collapses on all predictors (ipTM <= 0.25); keeping A2 + A1")
    assert not d.agents_turn(c)
    with pytest.raises(ValueError):
        d.resolve("agent", c["id"], "proceed")           # only a person resolves
    d.resolve("person", c["id"], "proceed", "ok")
    d.move("INPUTS_READY", "agent")


def test_results_checkpoint_loops_back():
    d = Dossier(target="T")
    _to_slate(d)
    d.stage = "RESULTS_REVIEW"
    d.log.append({"time": "9999-01-01", "actor": "agent", "note": "stage POOL_READY -> RESULTS_REVIEW"})
    d.evidence["results_review"].append({"time": "9999-01-02", "by": "agent", "value": {"triggers": ["architecture_split: X vs Y"]}})
    with pytest.raises(ValueError):
        d.move("SLATE_PROPOSED", "agent")                # a trigger needs a person
    with pytest.raises(ValueError):
        d.move("RUNNING", "agent")                       # looping back is the person's call
    c = d.consult("agent", "results", ["X or Y? (a) 64 more seeds of both (rec.), (b) proceed with X"])
    d.resolve("person", c["id"], "loop:RUNNING", "more seeds")
    assert d.stage == "RUNNING"


def test_review_checks_find_triggers(tmp_path):
    import csv as _csv
    from residudes import consult
    rows = []
    for sto, base in (("A2", {"af3": 0.8, "of3": 0.5}), ("A3", {"af3": 0.5, "of3": 0.82})):
        for run, b in base.items():
            for seed in range(5 if run == "af3" else 2):         # of3 is missing models
                rows.append({"sto": sto, "variant": "v", "runner": run, "seed": seed, "sample": 0, "ranking_score": b,
                             "iptm": b, "ptm": b, "plddt": 65, "has_clash": False, "path": "x", "rs_z_runner": 0})
    with open(tmp_path / "pool.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    num, trig = consult.checks(tmp_path / "pool.csv", None)
    names = {t.split(":")[0] for t in trig}
    assert {"stoichiometry_split", "low_plddt", "missing_models"} <= names and "low_confidence" not in names


def test_guard_blocks_resolve():
    assert guard.check("residudes resolve T1437 C1 --outcome proceed")


def test_daemon_waits_at_checkpoint_until_a_person_replies(tmp_path, monkeypatch):
    from residudes import daemon
    from residudes.config import Config
    cfg = Config(work_root=str(tmp_path))
    d = Dossier(target="T", stage="PLAN_REVIEW")
    c = d.consult("agent", "plan", ["A2 or A1?"])
    d.save(cfg.targets_dir)
    calls = []
    monkeypatch.setattr(daemon, "step", lambda cfg, t, budget=None, verbose=False: calls.append(t) or {"subtype": "success"})
    assert daemon.round_once(cfg, None, None, 3, {}) == ["T: waiting for a person at checkpoint C1 (plan)"] and not calls
    with Dossier.locked(cfg.targets_dir, "T") as dd:
        dd.reply("person", c["id"], "why A2?")
    daemon.round_once(cfg, None, None, 3, {})
    assert calls == ["T"]                                    # the agent answers the person


def test_one_step_per_target_and_one_daemon(tmp_path, monkeypatch):
    import fcntl
    from residudes import agent, daemon
    from residudes.config import Config
    cfg = Config(work_root=str(tmp_path))
    Dossier(target="T").save(cfg.targets_dir)
    with open(cfg.targets_dir / "T" / ".step.lock", "w") as fh:     # a step is running
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert agent.step(cfg, "T", verbose=False) == {"subtype": "busy"}
    with open(tmp_path / ".daemon.lock", "a+") as fh:                # a daemon is running
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert daemon.run(cfg, None, 60, once=True, budget=None) == 1


def test_slides_deck_and_scenes(tmp_path):
    import csv as _csv
    from residudes import slides
    from residudes.config import Config
    cfg = Config(work_root=str(tmp_path))
    d = Dossier(target="T")
    d.evidence["msa"] = [{"time": "t", "by": "agent", "value": {"depth": 3, "note": 'a "quoted" #hash $dollar'}}]
    d.save(cfg.targets_dir)
    ddir = cfg.targets_dir / "T"
    with open(ddir / "pool.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["sto", "variant", "runner", "seed", "sample", "ranking_score", "iptm", "ptm", "plddt",
                                            "has_clash", "path", "rs_z_runner"])
        w.writeheader()
        for sto in ("A1", "A3"):
            w.writerow({"sto": sto, "variant": "v", "runner": "af3", "seed": 1, "sample": 0, "ranking_score": 0.8, "iptm": 0.8,
                        "ptm": 0.8, "plddt": 90, "has_clash": False, "path": f"/x/{sto}.cif", "rs_z_runner": 0})
    models = slides.figure_models(d, ddir)
    assert [m["label"] for m in models] == ["A1", "A3"]
    pml = slides.scenes_pml(models)
    assert "png figs/A3_plddt.png" in pml and "overlay" not in pml          # no superposition across stoichiometries
    mt = slides.scenes_mt(models)
    assert "screenshot figs/A3_plddt.png 1800 1350 300" in mt and "overlay" not in mt
    assert "set lod_medium 400000" in mt and "set bg white" in mt           # report quality, not terminal defaults
    assert "rename A1 A1" in mt or "rename A1.cif A1" in mt or "rename A1 " in mt   # named on arrival
    assert "color chain\n" in mt                                            # a scheme takes no selection in molterm
    typ = slides.build_typ(d, ddir)
    assert '#"' in typ and '\\"quoted\\"' in typ                             # dossier text goes in as Typst strings
    assert "table.cell(fill:" in typ                                         # the confidence heat table


def test_scenes_mt_overlay_uses_molterm_hex():
    """PyMOL takes 0xRRGGBB and molterm does not: it refuses the slot colours unless they are written #RRGGBB."""
    from residudes import slides

    models = [{"label": f"m{i}", "name": f"m{i}", "path": f"/x/m{i}.cif", "sto": "A2"} for i in range(2)]
    mt = slides.scenes_mt(models)
    assert "color #0072B2 m0" in mt and "0x" not in mt
    assert "align m1 to m0 automap" in mt                                   # chains paired by sequence


def test_guard_sends_pymol_to_sbatch():
    assert guard.check("pymol -cq scenes.pml") and guard.check("/opt/venv/bin/pymol -cq x.pml")
    assert guard.check("python -m pymol -cq x.pml")
    assert guard.check("residudes render T1416 --pml report/pml/x.pml") is None


def test_export_keeps_a_native_free_deck(tmp_path):
    from residudes import export
    d = Dossier(target="T")
    d.evidence["native"] = [{"time": "2000-01-01T00:00:00+00:00", "value": {}}]        # native scored long ago
    d.evidence["report"] = [{"time": "2099-01-01T00:00:00+00:00", "value": {"with_retro": False}}]
    rep = tmp_path / "src" / "report"
    (rep / "figs").mkdir(parents=True)
    for f in ("report.typ", "report.pdf", "scenes.pml", "figs/M1_chains.jpg"):
        (rep / f).write_text("x")
    export.run(d, tmp_path / "src", tmp_path / "out")
    assert (tmp_path / "out" / "report" / "report.pdf").exists() and (tmp_path / "out" / "report" / "figs" / "M1_chains.jpg").exists()
    (rep / "agent.typ").write_text("== lessons: model 1 lDDT 0.82")                     # written after the native
    export.run(d, tmp_path / "src", tmp_path / "out2")
    assert not (tmp_path / "out2" / "report" / "report.pdf").exists()                   # the deck would include it
    assert (tmp_path / "out2" / "report" / "figs" / "M1_chains.jpg").exists()           # structures are still fine


def test_a2m_rows_reindexed_on_the_query():
    from residudes.msa import a2m_to_a3m
    # HMM match columns: 6; the query has residues at columns 1-3,5-6 and an insertion 'k' after column 2
    q = "MAkC-DE"                     # query: M A (ins k) C, deletion at col 4, D E  -> query residues M A K C D E (6)
    # homolog "MAqCWDE": insertion q after col 2, residue W at col 4 where the query has none
    a3 = a2m_to_a3m(q, "MA" + "q" + "CWDE" + "FGHIKLMNPQ")   # extra matches beyond the query's columns are ignored safely
    assert a3 is None or len([c for c in a3 if not c.islower()]) == 6
    a3 = a2m_to_a3m("MAkCDEFGHIKLMN", "MAqCDEFGHIKLMN")
    assert [c for c in a3 if not c.islower()] == list("MA-CDEFGHIKLMN")      # query's 'k' gets a gap; homolog's 'q' is an insertion
    assert "q" in a3


def test_depth_counts_coverage(tmp_path):
    from residudes.msa import depth
    p = tmp_path / "x.a3m"
    p.write_text(">q\nACDEFGHIKLMNPQRSTVWY\n>a\nACDEFGHIKL----------\n>b\nACDEFGHIKLMNPQRSTVWY\n>c\n----------MNPQRSTVWY\n")
    s = depth(p)
    assert s["n"] == 3 and s["n_cov50"] == 3 and s["coverage"][0] == 2 and s["coverage"][15] == 2


def test_job_states_normalised():
    from residudes.ops import active
    assert not active({"state": "CANCELLED+"}) and not active({"state": "CANCELLED by 1234"})
    assert active({"state": "RUNNING"}) and active({"state": "submitted"}) and not active({"state": "COMPLETED"})


def test_daemon_parks_a_target_after_an_api_error(tmp_path, monkeypatch):
    """A refusal (e.g. the model's safeguards) or an outage ends the step with an API error: park the target for a
    person instead of paying for the same refusal every round (seen live on 2026-11-22)."""
    from residudes import daemon
    from residudes.config import Config
    cfg = Config(work_root=str(tmp_path))
    Dossier(target="T3").save(cfg.targets_dir)
    calls = []

    def refused(cfg, target, budget=None, verbose=False):
        calls.append(target)
        return {"subtype": "success", "is_error": True, "cost_usd": 0.5,
                "result": "API Error: the model's safeguards flagged this session\nDetails: ..."}
    monkeypatch.setattr(daemon, "step", refused)
    idle: dict = {}
    rep = daemon.round_once(cfg, None, None, 3, idle)
    assert rep == ["T3: error API Error: the model's safeguards flagged this session (parked: NEEDS HUMAN)"]
    assert Dossier.load(cfg.targets_dir, "T3").log[-1]["note"].startswith("NEEDS HUMAN: the agent step ended with an API error")
    assert daemon.round_once(cfg, None, None, 3, idle) == ["T3: NEEDS HUMAN"] and calls == ["T3"]   # not retried
    assert daemon.api_error({"subtype": "success", "result": "done"}) is None


def test_a_step_that_runs_out_of_budget_is_unfinished_not_failed(tmp_path, monkeypatch, capsys):
    """The SDK raises when max_budget_usd is reached. Everything the agent recorded through `residudes` is already in
    the dossier, so the step is unfinished: report it, log it, and let the next step carry on (H2338, 2026-11-23)."""
    import asyncio
    from residudes import agent
    from residudes.dossier import Dossier

    class Cfg:
        targets_dir = tmp_path / "targets"
        agent_usd_per_step = 2.0
        agent_max_turns = 60
        model = "m"
    d = Dossier(target="T14")
    Dossier.path(Cfg.targets_dir, "T14").parent.mkdir(parents=True)
    d.save(Cfg.targets_dir)

    async def boom(*a, **k):
        raise RuntimeError("Claude Code returned an error result: Reached maximum budget ($2) (exit code: 1)")
    monkeypatch.setattr(agent, "_step", boom)
    res = asyncio.run(agent._run(d, Cfg, 2.0, 60, "m", False))
    assert res["subtype"] == "budget" and res["is_error"] is False and res["cost_usd"] == 2.0

    async def other(*a, **k):
        raise RuntimeError("the socket closed")
    monkeypatch.setattr(agent, "_step", other)
    with pytest.raises(RuntimeError):                     # anything else is still an error
        asyncio.run(agent._run(d, Cfg, 2.0, 60, "m", False))


def test_report_front_sections_and_slate_scores(tmp_path):
    """The report reads the target out of the evidence instead of dumping it: what it is, its sequence, its MSA
    depth, the structures a search returned, and the slate with residudes-qa's predicted DockQ."""
    from residudes import report
    d = Dossier(target="T9", sequence={"text": ">T9,subunit1|\nMKTAYIAKQRQISFVK\n>T9,subunit2|\nGGGGGGGGCCHHHHHH\n"},
                stoichiometry={"given": "A1B1"})
    d.evidence["construct"] = [{"time": "2026-01-01T09:00:00+09:00", "by": "agent",
                                "value": {"length": 32, "header": "Bacillus subtilis", "tag": "C-terminal His6"}}]
    d.evidence["identity"] = [{"time": "2026-01-01T09:05:00+09:00", "by": "agent",
                               "value": {"family": "SnoaL-like (IPR032710)", "uniprot": "P12345"}}]
    d.evidence["msa"] = [{"time": "2026-01-01T09:10:00+09:00", "by": "agent",
                          "value": {"depth": {"uniref90": 6, "bfd": 1}, "assessment": "single-sequence regime"}}]
    d.evidence["templates"] = [{"time": "2026-01-01T09:15:00+09:00", "by": "agent",
                                "value": {"best": "6w3w_A 37% id, qcov 99/122", "checked": False}}]
    (tmp_path / "slate_v1.json").write_text(json.dumps(
        {"sto": "A1B1", "chosen": True, "n_candidates": {"A1B1": 1600}, "used_interface": True,
         "models": [{"slot": 1, "model": "m1", "runner": "af3", "ranking_score": 0.8,
                     "rs_z_runner": 1.7, "xgb_pred": 0.7511, "cluster": "c0"},
                    {"slot": 2, "model": "m2", "runner": "of3", "ranking_score": 0.75, "xgb_pred": None}]}))

    md = report.build(d, False, tmp_path)
    assert "| chains | T9,subunit1 — 16 aa, T9,subunit2 — 16 aa |" in md      # both chains, from the FASTA
    assert "| `construct.header` | Bacillus subtilis |" in md and "| `identity.family` | SnoaL-like (IPR032710) |" in md
    assert "| `uniref90` | 6 |" in md and "single-sequence regime" in md
    assert "| `6w3w_A` | 37% id | qcov 99/122 | `templates.best` |" in md        # entry, identity, coverage
    assert "`templates.checked`" not in md.split("## Every prior")[0]          # a bare flag is not a structure
    assert "8× G" in md and "| A1B1 1600 candidates" not in md                 # runs counted; dict counts formatted
    slate = md.split("## Slates")[1]
    assert "A1B1 1600 candidates" in slate and "interface scores used" in slate
    assert "| 1 | `m1` | af3 | 0.800 | +1.70 | 0.751 | c0 |" in slate          # our calibrated score in the table
    assert "| 2 | `m2` | of3 | 0.750 | – | – | – |" in slate                   # and a dash where it was not computed


def test_plan_gaps_lists_the_basics_that_were_never_recorded(tmp_path):
    """A person approving a plan should see which basics are missing, not infer it from a short brief. Each gap may
    be right for the target - a design has no organism - so it is a list, not a block."""
    from residudes import consult
    d = Dossier(target="T9")
    assert consult.plan_gaps(d, None) == [
        "no `identity`: what the chains are was never recorded",
        "no `msa_depth`: the MSAs were never measured",
        "no `templates` or `public_structures`: the structures that exist were never looked for "
        "(`residudes structures T`)"]

    d.evidence["identity"] = [{"time": "2026-01-01T09:00:00+09:00", "by": "agent",
                               "value": {"name": "a kinase, Homo sapiens", "family": "Pfam PF08757"}}]
    d.evidence["msa"] = [{"time": "2026-01-01T09:01:00+09:00", "by": "agent", "value": {"depth": {"uniref90": 6}}}]
    d.evidence["templates"] = [{"time": "2026-01-01T09:02:00+09:00", "by": "agent", "value": {"best": "6w3w 37% id"}}]
    d.evidence["binding_site"] = [{"time": "2026-01-01T09:03:00+09:00", "by": "agent",
                                   "value": {"what": "the pocket is Lys49 and Arg56"}}]        # prose, no field
    assert consult.plan_gaps(d, tmp_path) == [
        "`binding_site` has no residues field: a gate drafted from it is read out of the prose",
        "no `notes/target.md`: there is no write-up for a person to read"]

    d.evidence["binding_site"][0]["value"]["residues"] = [49, 56]
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "target.md").write_text("# T9")
    assert consult.plan_gaps(d, tmp_path) == []


def test_the_plan_brief_carries_the_figures_a_person_decides_from(tmp_path):
    """A checkpoint is where a person reads the analysis, so opening one draws the figures and links them from the
    brief. The brief sits one level under them, so the links have to climb out of checkpoints/."""
    from residudes import consult
    from residudes.config import Config
    cfg = Config(work_root=str(tmp_path))
    ddir = cfg.targets_dir / "T9"
    ddir.mkdir(parents=True)
    d = Dossier(target="T9", sequence={"text": ">T9,a|\nPKGLSASLPDL\n>T9,b|\n" + "GAMKT" * 20 + "\n"})
    d.evidence["motif"] = [{"time": "2026-01-01T09:00:00+09:00", "by": "agent",
                            "value": {"chain": "A", "residues": [7], "what": "pS-X-P", "source": "PMID 9428519"}}]
    out = consult.open_cmd(cfg, d, "plan", ["SEP at A7, or unmodified only?"], None, "agent")
    brief = (ddir / "checkpoints" / "C1_plan.md").read_text()
    assert "## The figures" in brief
    assert "](../report/figs/motif.svg)" in brief and "](../report/figs/sites.svg)" in brief
    assert (ddir / "report" / "figs" / "motif.svg").exists()
    assert "![" not in brief.split("## The analysis behind it")[0]       # links in the list, images in the analysis
    assert "](../report/figs/motif.svg)" in brief.split("## The analysis behind it")[1]
    assert "C1 open (plan)" in out


def test_config_accepts_the_paths_key_this_one_was_called_in_2026(tmp_path):
    """foldmux -> residudes_engine: a configuration written in 2026 still loads."""
    from residudes import config
    p = tmp_path / "residudes.toml"
    p.write_text('[paths]\nfoldmux = "/older/engine"\n')
    assert config.load(str(p)).residudes_engine == "/older/engine"
    p.write_text('[paths]\nfoldmux = "/older/engine"\nresidudes_engine = "/new/engine"\n')
    assert config.load(str(p)).residudes_engine == "/new/engine"      # the current key wins


def test_the_config_file_is_residudes_toml(tmp_path, monkeypatch):
    from residudes import config
    monkeypatch.setattr(config, "DEFAULT_PATH", tmp_path / "residudes.toml")
    monkeypatch.delenv("RESIDUDES_CONFIG", raising=False)
    assert config.config_path() == tmp_path / "residudes.toml"
    (tmp_path / "residudes.toml").write_text('[paths]\nwork_root = "/new/work"\n')
    assert config.load().work_root == "/new/work"


def test_catalog_paths_come_from_the_site_config(tmp_path):
    """A machine's own layout belongs in residudes.toml, not in the catalog everyone shares."""
    from residudes import catalog
    c = tmp_path / "catalog.toml"
    c.write_text('[[component]]\nname = "foldseek PDB 2026"\nkind = "database"\n'
                 'any_of = ["{foldseek_pdb_2026}"]\nfix = "rebuild it"\n')
    unset = catalog.check_all(c)[0]
    assert unset["status"] == "unconfigured" and "[databases] foldseek_pdb_2026" in unset["note"]
    assert unset["fix"] == "rebuild it"

    db = tmp_path / "db"; db.mkdir()
    ok = catalog.check_all(c, {"foldseek_pdb_2026": str(db)})[0]
    assert ok["status"] == "ok" and ok["fix"] == ""
    gone = catalog.check_all(c, {"foldseek_pdb_2026": str(tmp_path / "nowhere")})[0]
    assert gone["status"] == "missing"


def test_the_msa_the_depth_command_picks(tmp_path):
    """residudes-engine writes unpaired_<L>.a3m beside paired_<L>.a3m, and `paired` sorts first."""
    from residudes import msa
    from residudes.config import Config
    from residudes.dossier import Dossier
    cfg = Config(work_root=str(tmp_path))
    d = Dossier(target="T1")
    root = Dossier.path(cfg.targets_dir, "T1").parent / "msas"
    for sub, names in (("af3", ["bfd.a3m", "mgnify.a3m", "templates_hmmsearch.a3m", "uniprot.a3m", "uniref90.a3m"]),
                       ("mmseqs2_gpu", ["envdb.a3m", "merged.a3m", "uniref30.a3m"]),
                       ("engine", ["paired_A.a3m", "unpaired_A.a3m", "unpaired_B.a3m"]),
                       ("odd", ["templates_x.a3m", "whatever.a3m"])):
        (root / sub).mkdir(parents=True)
        for n in names:
            (root / sub / n).write_text(">q\nMKT\n")
    got = [p.parent.name + "/" + p.name for p in msa.target_msas(cfg, d)]
    assert got == ["af3/uniref90.a3m", "engine/unpaired_A.a3m", "engine/unpaired_B.a3m",
                   "mmseqs2_gpu/merged.a3m", "odd/whatever.a3m"]


def test_chain_letters_do_not_run_past_z_into_punctuation(tmp_path):
    """A 27-record FASTA used to get "[" as a chain id; AlphaFold 3 wants A…Z, AA, AB, …"""
    from residudes import structures
    from residudes.dossier import CHAIN_LETTERS, Dossier, chain_letter
    assert CHAIN_LETTERS[:3] == ["A", "B", "C"]
    assert chain_letter(25) == "Z" and chain_letter(26) == "AA" and chain_letter(27) == "AB"
    assert chain_letter(len(CHAIN_LETTERS)) == f"X{len(CHAIN_LETTERS)}"      # never punctuation
    p = tmp_path / "many.fasta"
    p.write_text("".join(f">c{i}\nMKT\n" for i in range(28)))
    d = Dossier(target="T1")
    d.sequence = {"fasta": str(p)}
    letters = [c[0] for c in structures.chains(d)]
    assert letters[25:] == ["Z", "AA", "AB"]


def test_the_first_stage_names_the_command_that_finds_public_structures():
    """`residudes next` is what the agent reads to know what to do; a capability it cannot see is not one."""
    from residudes.dossier import PLAYBOOK
    goal, skill = PLAYBOOK["NEW"]
    assert "residudes structures T" in goal and "--hhsearch" in goal and skill == "residudes-intake"


def test_render_mt_distrusts_a_script_that_had_a_failed_command(tmp_path, monkeypatch):
    """The published image taught this one. molterm 0.50 exits 0 where 0.53 exits 1, and either way it carries
    on after a failure: with USalign absent every superposition failed and the overlay was written from
    unaligned models, which looks exactly like an aligned one. So the renderer reads the output, not $?."""
    import subprocess as sp

    from residudes import slides
    from residudes.dossier import Dossier

    ddir = tmp_path / "targets" / "T1"
    (ddir / "report" / "figs").mkdir(parents=True)
    d = Dossier(target="T1")
    scene = ddir / "report" / "scenes.mt"
    scene.write_text("load /x/a.cif\n")

    class Cfg:
        extra = {"tools": {"molterm": "/usr/bin/true"}}
        targets_dir = tmp_path / "targets"

    out = ("scenes.mt:18: `align b to a automap`: Align failed: USalign failed (exit 32512): "
           "sh: 1: USalign: not found\nSaved 1800x1350 to figs/a_chains.png\nScript: 1 command(s) failed\n")
    monkeypatch.setattr(sp, "run", lambda *a, **k: sp.CompletedProcess(a[0], 0, out, ""))   # exit 0, as 0.50 does
    said = slides.render_mt(Cfg(), d, scene, "t")
    assert "not trustworthy" in said and "USalign" in said
    assert any("failed" in x["note"] for x in d.log)

    monkeypatch.setattr(sp, "run", lambda *a, **k: sp.CompletedProcess(a[0], 0, "Saved 1800x1350 to figs/a.png\n", ""))
    assert "not trustworthy" not in slides.render_mt(Cfg(), d, scene, "t")
