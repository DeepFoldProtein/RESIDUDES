"""The automation mandate: what a person can delegate, and what stays theirs."""
import pytest

from residudes.dossier import Dossier, now


def _d(tmp_path, stage="SLATE_PROPOSED"):
    d = Dossier(target="T9999", stage=stage)
    d.save(tmp_path)
    return d


def test_agent_cannot_approve_without_a_mandate(tmp_path):
    d = _d(tmp_path)
    with pytest.raises(ValueError, match="people only"):
        d.approve("agent", "slate v1")


def test_agent_cannot_grant_itself_the_mandate(tmp_path):
    d = _d(tmp_path)
    with pytest.raises(ValueError, match="only a person"):
        d.authorise_automation("agent")


def test_a_person_grants_it_and_the_agent_may_then_approve(tmp_path):
    d = _d(tmp_path)
    d.authorise_automation("delibae", until="PACKAGED", note="overnight run")
    d.approve("agent", "slate v1")
    assert d.approvals[-1]["automated"] is True
    assert d.approvals[-1]["mandate"]["by"] == "delibae"
    assert "automated under the mandate from delibae" in d.log[-1]["note"]


def test_submission_is_never_automatable(tmp_path):
    d = _d(tmp_path, stage="PACKAGED")
    d.authorise_automation("delibae")
    assert d.automation_allows("SUBMITTED") is False
    with pytest.raises(ValueError, match="no mandate can delegate it"):
        d.move("SUBMITTED", "agent")


def test_the_mandate_cannot_be_written_to_reach_submission(tmp_path):
    d = _d(tmp_path)
    with pytest.raises(ValueError, match="submitting is a person's act"):
        d.authorise_automation("delibae", until="SUBMITTED")


def test_it_stops_at_the_stage_it_names(tmp_path):
    d = _d(tmp_path, stage="APPROVED")
    d.authorise_automation("delibae", until="SLATE_PROPOSED")
    assert d.automation_allows("approve") is False          # past the stage the person named
    with pytest.raises(ValueError, match="people only"):
        d.approve("agent", "slate v1")


def test_a_person_can_revoke_it(tmp_path):
    d = _d(tmp_path)
    d.authorise_automation("delibae")
    d.revoke_automation("delibae", "back to hand")
    assert d.automation == {}
    with pytest.raises(ValueError, match="people only"):
        d.approve("agent", "slate v1")
    with pytest.raises(ValueError, match="only a person"):
        d.revoke_automation("agent")


def test_resolve_follows_the_same_rule(tmp_path):
    d = _d(tmp_path, stage="PLAN_REVIEW")
    d.checkpoints.append({"id": "C1", "kind": "plan", "status": "open", "asks": [], "thread": []})
    with pytest.raises(ValueError, match="a person resolves"):
        d.resolve("agent", "C1", "proceed")
    d.authorise_automation("delibae")
    c = d.resolve("agent", "C1", "proceed")
    assert c["resolution"]["automated"] is True
    assert c["resolution"]["mandate"]["by"] == "delibae"


def test_the_round_trip_keeps_the_mandate(tmp_path):
    d = _d(tmp_path)
    d.authorise_automation("delibae", note="why")
    d.save(tmp_path)
    again = Dossier.load(tmp_path, "T9999")
    assert again.automation["by"] == "delibae"
    assert again.automation_allows("approve") is True


# --------------------------------------------------------------------- full automation as the default
def test_where_the_agent_may_act(tmp_path):
    """One rule for the daemon, the step and the brief. Without a mandate: stop at the slate and at a checkpoint
    waiting for a person. With one: go on. PACKAGED and SUBMITTED stop it in every case."""
    d = _d(tmp_path, stage="SLATE_PROPOSED")
    assert not d.agent_may_act()
    d.authorise_automation("delibae")
    assert d.agent_may_act()
    assert "approve" in d.next_step()[0] and "automation is on" in d.next_step()[0]

    p = _d(tmp_path, stage="PLAN_REVIEW")
    p.consult("agent", "plan", ["seeds?"], brief="b")
    assert not p.agent_may_act()                       # the checkpoint waits for a person
    p.authorise_automation("delibae")
    assert p.agent_may_act()
    goal = p.next_step()[0]
    assert "answer checkpoint C1" in goal and "resolve" in goal

    for stage in ("PACKAGED", "SUBMITTED"):
        s = _d(tmp_path, stage=stage)
        s.authorise_automation("delibae") if stage == "PACKAGED" else None
        assert not s.agent_may_act()


def test_the_agent_walks_a_target_to_a_packaged_slate_under_the_mandate(tmp_path):
    """The whole chain, every move made by the agent: the plan checkpoint opened and answered, the slate approved,
    the slate packaged. Then the one move it cannot make."""
    d = Dossier(target="T9999")
    d.authorise_automation("delibae", note="the default at registration")
    for stage in ("IDENTIFIED", "HYPOTHESES", "PLAN_REVIEW"):
        d.move(stage, "agent")
    c = d.consult("agent", "plan", ["which seeds?"], brief="the plan")
    d.resolve("agent", c["id"], "proceed")
    for stage in ("INPUTS_READY", "RUNNING", "POOL_READY", "RESULTS_REVIEW"):
        d.move(stage, "agent")
    # what `residudes review T` records: the standard checks, none raising a trigger
    d.evidence.setdefault("results_review", []).append({"time": now(), "by": "agent",
                                                        "value": {"checks": 7, "triggers": []}})
    d.move("SLATE_PROPOSED", "agent")
    d.approve("agent", "these five pass every gate")
    d.move("APPROVED", "agent")
    d.move("PACKAGED", "agent")
    assert d.stage == "PACKAGED" and not d.agent_may_act()
    with pytest.raises(ValueError, match="no mandate can delegate it"):
        d.move("SUBMITTED", "agent")
    assert d.approvals[-1]["automated"] is True        # the log never blurs who decided


def test_target_add_grants_the_mandate_unless_asked_for_review(tmp_path, monkeypatch):
    import subprocess
    import sys

    cfg = tmp_path / "residudes.toml"
    cfg.write_text(f'[paths]\nwork_root = "{tmp_path / "work"}"\n')
    env = {"RESIDUDES_ACTOR": "delibae", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}

    def add(*extra):
        return subprocess.run([sys.executable, "-m", "residudes", "--config", str(cfg), "target", "add", *extra],
                              capture_output=True, text=True, env=env)

    r = add("T1")
    assert r.returncode == 0 and "full automation" in r.stdout
    assert Dossier.load(tmp_path / "work" / "targets", "T1").automation["by"] == "delibae"

    r = add("T2", "--review")
    assert r.returncode == 0 and "with human checkpoints" in r.stdout
    assert not Dossier.load(tmp_path / "work" / "targets", "T2").automation


def test_the_daemon_steps_at_the_slate_when_the_mandate_covers_it(tmp_path, monkeypatch):
    from residudes import daemon

    called = []
    monkeypatch.setattr(daemon, "step", lambda cfg, t, **k: called.append(t) or {"subtype": "success"})
    monkeypatch.setattr(daemon, "_waiting_on_jobs", lambda cfg, t: False)

    class Cfg:
        targets_dir = tmp_path

    d = _d(tmp_path, stage="SLATE_PROPOSED")
    assert "waiting for a person" in daemon._round_target(Cfg(), d, None, 5, {})
    assert called == []
    d.authorise_automation("delibae")
    d.save(tmp_path)
    daemon._round_target(Cfg(), d, None, 5, {})
    assert called == ["T9999"]


def test_needs_human_stops_the_daemon_even_with_a_checkpoint_open(tmp_path, monkeypatch):
    """Under a mandate the agent answers the checkpoint itself; when it writes NEEDS HUMAN instead, the daemon
    must stop, not step it again (H1388 took two more steps before this)."""
    from residudes import daemon

    called = []
    monkeypatch.setattr(daemon, "step", lambda cfg, t, **k: called.append(t) or {"subtype": "success"})
    monkeypatch.setattr(daemon, "_waiting_on_jobs", lambda cfg, t: False)

    class Cfg:
        targets_dir = tmp_path

    d = _d(tmp_path, stage="PLAN_REVIEW")
    d.authorise_automation("delibae")
    d.consult("agent", "plan", ["seeds?"], brief="b")
    d.note("agent", "NEEDS HUMAN: resolve was refused")
    d.save(tmp_path)
    assert "NEEDS HUMAN" in daemon._round_target(Cfg(), d, None, 5, {})
    assert called == []
    d.reply("delibae", d.checkpoints[0]["id"], "go ahead")              # a person answers in the checkpoint
    d.save(tmp_path)
    daemon._round_target(Cfg(), d, None, 5, {})
    assert called == ["T9999"]
