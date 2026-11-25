"""The conversation sessions: what `residudes` and `residudes chat` promise."""
from residudes import chat


def test_the_brief_names_the_target_and_the_dossier():
    """A session cannot start without it, and nothing else in the tests starts one."""
    text = chat.BRIEF.format(target="T1234", ddir="/w/targets/T1234")
    assert "T1234" in text and "/w/targets/T1234" in text
    assert "never approve" in text                      # the rules hold inside a conversation too


def test_one_session_holds_a_window_per_target():
    """`residudes chat T` and `tmux select-window -t residudes:T` have to mean the same window."""
    assert chat.SESSION == "residudes"                  # one session, so `tmux ls` stays one line
    assert chat.LEAVE == "ctrl-b d"                     # tmux's own keys, not ones we invented
    assert chat.SWITCH == "ctrl-b w"


def test_ago_reads_a_session_timestamp():
    assert chat._ago("not a date") == ""
    assert chat._ago("2020-01-01 00:00").endswith("d")


def test_the_daemon_works_through_the_target_window(monkeypatch):
    """One agent per target: the daemon asks that window rather than starting a second agent."""
    from residudes import daemon
    from residudes.dossier import Dossier
    sent = {}
    monkeypatch.setattr(chat, "ensure", lambda cfg, t, h=None: True)
    monkeypatch.setattr(chat, "send", lambda t, text: sent.update(target=t, text=text) or True)
    d = Dossier(target="T1234", stage="NEW")
    out = daemon._step_through_conversation(None, d)
    assert sent["target"] == "T1234"
    assert "T1234" in sent["text"] and "dossier" in sent["text"]
    assert "asked" in out


def test_help_shows_a_person_the_short_list(capsys):
    """`-h` is for a person: the agent's recording commands stay out of it but keep working."""
    from residudes import cli
    try:
        cli._main(["-h"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "residudes chat T1234" in out and "waiting for you" in out
    for hidden in ("evidence", "run-record", "gate"):
        assert f"    {hidden} " not in out            # not listed
    assert "residudes help --all" in out              # but findable


def test_a_session_brings_the_structure_tools(tmp_path):
    """A conversation about a shape needs to be able to show it."""
    import json
    path = chat.write_mcp_config(tmp_path)
    assert path and path.name == ".mcp.json"
    cfg = json.loads(path.read_text())
    assert "molterm" in cfg["mcpServers"]
    assert chat.write_mcp_config(tmp_path) is None          # someone else's file is left alone
