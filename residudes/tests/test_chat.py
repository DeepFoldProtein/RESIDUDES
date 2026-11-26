"""The conversation sessions: what `residudes` and `residudes chat` promise."""
from residudes import chat


def test_the_brief_names_the_target_and_the_dossier():
    """A session cannot start without it, and nothing else in the tests starts one."""
    text = chat.BRIEF.format(target="T1234", ddir="/w/targets/T1234")
    assert "T1234" in text and "/w/targets/T1234" in text
    assert "never submit" in text and "unless the" in text   # the rules hold inside a conversation too
    assert "[auto]" in text and "nobody answers in between" in text   # and what auto's lines ask of it


def test_one_session_holds_a_window_per_target():
    """`residudes chat T` and `tmux select-window -t residudes:T` have to mean the same window."""
    assert chat.SESSION == "residudes"                  # one session, so `tmux ls` stays one line
    assert chat.LEAVE == "ctrl-b d"                     # tmux's own keys, not ones we invented
    assert chat.SWITCH == "ctrl-b w"


def test_ago_reads_a_session_timestamp():
    assert chat._ago("not a date") == ""
    assert chat._ago("2020-01-01 00:00").endswith("d")


def test_auto_types_only_into_a_quiet_conversation():
    """Auto asks for the next step, never over the agent, a dialog or a person's typing, and never waits on a person."""
    from residudes import auto
    from residudes.dossier import Dossier
    d = Dossier(target="T1234", stage="NEW")
    quiet = {"busy": False, "dialog": False, "typed": "", "quiet": 120.0}
    what, text = auto.decide(d, quiet)
    assert what == "ask" and text.startswith("[auto] Next step for T1234 (NEW)") and "\n" not in text
    for change, why in ((dict(busy=True), "working"), (dict(dialog=True), "dialog"), (dict(typed="hm"), "typing"),
                        (dict(quiet=5.0), "recent")):
        assert why in auto.decide(d, {**quiet, **change})[1]
    assert auto.decide(Dossier(target="T1234", stage="PACKAGED"), quiet) == ("done", "PACKAGED")
    d.note("agent", "NEEDS HUMAN: two stoichiometries the pool cannot separate")
    what, text = auto.decide(d, quiet)                       # auto has no pause: the agent is told to decide it
    assert what == "ask" and "will not be answered" in text and "do not write NEEDS HUMAN" in text


def test_auto_tells_typing_from_a_suggestion():
    """Claude Code draws its suggested prompt grey after the ❯; what a person types is in the normal colour."""
    from residudes import auto
    assert auto.typed("\x1b[38;5;246m❯\u00a0\x1b[39m\n") == ""                                  # empty
    assert auto.typed("\x1b[39m❯\u00a0draft text\n") == "draft text"                               # typed
    assert auto.typed("\x1b[39m❯\u00a0\x1b[2mshow the slate superposed\x1b[22m\n") == ""        # suggested
    assert auto.typed("\x1b[39m❯\u00a0\x1b[38;5;246mshow the slate\x1b[39m\n") == ""
    assert auto.typed("no prompt here") == ""


def test_auto_is_the_default_and_by_hand_is_remembered(tmp_path, monkeypatch):
    """--no-auto is remembered; the list says who moves a target on; the daemon defers; only a person turns it on."""
    import json
    import os
    from residudes import auto, daemon
    from residudes.config import Config
    from residudes.dossier import Dossier
    cfg = Config(work_root=str(tmp_path))
    Dossier(target="T1234").save(cfg.targets_dir)
    monkeypatch.setattr(chat, "_tmux", lambda *a, **k: __import__("subprocess").CompletedProcess(a, 1, "", ""))
    assert not auto.by_hand(cfg, "T1234") and auto.status(cfg, "T1234") == ""
    print(auto.stop(cfg, "T1234", remember=True))
    assert auto.by_hand(cfg, "T1234")                              # --no-auto is remembered
    rec = cfg.targets_dir / "T1234" / auto.STATE
    rec.write_text(json.dumps({"pid": os.getpid(), "since": "now"}))
    monkeypatch.setattr(auto, "running", lambda c, t: {"pid": 1} if t == "T1234" else None)
    assert auto.status(cfg, "T1234") == "auto"
    d = Dossier.load(cfg.targets_dir, "T1234")
    assert "conversation (auto)" in daemon._round_target(cfg, d, None, 3, {})   # one agent per target
    assert "only a person" in auto.start(cfg, "T1234", "agent")                  # it hands over the checkpoints
    assert "▶ auto" in chat._moves({"moves": "auto", "waiting": ""})
    assert "✓ auto done" in chat._moves({"moves": "done", "waiting": "you (submit)"})
    assert "✗ auto stuck" in chat._moves({"moves": "stuck", "waiting": ""})
    assert "paused" in chat._moves({"moves": "daemon", "waiting": "you (C1)"})      # only the daemon pauses
    assert "manual" in chat._moves({"moves": "", "waiting": ""})


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
