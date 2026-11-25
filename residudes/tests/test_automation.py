"""The automation mandate: what a person can delegate, and what stays theirs."""
import pytest

from residudes.dossier import Dossier


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
