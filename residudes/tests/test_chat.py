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
                        (dict(quiet=5.0), "recent"), (dict(ready=False), "starting")):
        assert why in auto.decide(d, {**quiet, **change})[1]
    assert auto.decide(Dossier(target="T1234", stage="PACKAGED"), quiet) == ("done", "PACKAGED")
    d.note("agent", "NEEDS HUMAN: two stoichiometries the pool cannot separate")
    what, text = auto.decide(d, quiet)                       # auto has no pause: the agent is told to decide it
    assert what == "ask" and "will not be answered" in text and "do not write NEEDS HUMAN" in text


def test_auto_ends_stuck_when_the_agent_cli_pauses_the_session():
    """The paused screen keeps redrawing: read as recent activity, auto waited on it for good."""
    from residudes import auto
    from residudes.dossier import Dossier
    screen = """  Session paused
  Opus 5.5's safeguards flagged this session. You may be seeing this for the
  first time ...
  Details: `[bio]`
  ❯ 1. Switch to Opus 5
    2. Edit prompt and retry with Opus 5.5"""
    assert auto.halted(screen) == "its safeguards paused the session [bio]"
    assert auto.halted("❯ draft text\n esc to interrupt") == ""
    p = {"busy": False, "dialog": False, "typed": "", "quiet": 0.5, "ready": True, "halted": auto.halted(screen)}
    what, why = auto.decide(Dossier(target="T1234", stage="NEW"), p)
    assert what == "done" and why.startswith("stuck at NEW: the agent CLI paused") and "a person decides" in why


def test_auto_tells_typing_from_a_suggestion():
    """Claude Code draws its suggested prompt grey after the ❯; what a person types is in the normal colour."""
    from residudes import auto
    assert auto.typed("\x1b[38;5;246m❯\u00a0\x1b[39m\n") == ""                                  # empty
    assert auto.typed("\x1b[39m❯\u00a0draft text\n") == "draft text"                               # typed
    assert auto.typed("\x1b[39m❯\u00a0\x1b[2mshow the slate superposed\x1b[22m\n") == ""        # suggested
    assert auto.typed("\x1b[39m❯\u00a0\x1b[38;5;246mshow the slate\x1b[39m\n") == ""
    assert auto.typed("no prompt here") == ""
    chip = "\x1b[39m❯\u00a0\x1b[38;5;74m[⧉ In auto.py] \x1b[2m\x1b[39m[auto] Advance T1364\x1b[0m\n"
    assert auto.typed(chip) == ""                          # an editor's file chip plus a suggestion: nobody typing


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


def test_the_daemon_goes_on_at_once_after_a_step_that_moved_a_target():
    from residudes import daemon
    moved = ["T1431: step success $0.41 IDENTIFIED -> HYPOTHESES", "H1388: waiting on jobs"]
    assert daemon.next_wait(moved, 300) == 60
    assert daemon.next_wait(["T1431: step success $0.3 RUNNING -> RUNNING", "H1388: NEEDS HUMAN"], 300) == 300
    assert daemon.next_wait(["H1388: waiting on jobs"], 300) == 300


def test_agent_sessions_switch_models_when_a_message_is_flagged():
    """A flagged message paused an unattended chat on a dialog (the person's own setting was false)."""
    import json
    from residudes import setup
    argv = dict((c, f) for c, _n, f in setup.HARNESS_CLI)["claude"]("hello")
    assert argv[:2] == ["claude", "--settings"] and argv[-1] == "hello"
    assert json.loads(argv[2]) == {"switchModelsOnFlag": True}


def test_a_note_can_come_from_stdin_or_a_file(tmp_path, monkeypatch):
    """Backticks in a double-quoted note were run by the shell; stdin and @file keep the text as written."""
    import io
    from residudes import cli
    from residudes.dossier import Dossier
    cfgf = tmp_path / "residudes.toml"
    cfgf.write_text(f'[paths]\nwork_root = "{tmp_path / "work"}"\n')
    monkeypatch.setenv("RESIDUDES_CONFIG", str(cfgf))
    Dossier(target="T9").save(tmp_path / "work" / "targets")
    monkeypatch.setattr("sys.stdin", io.StringIO("the `pocket` evidence: $HOME stays\n"))
    assert cli.main(["note", "T9", "-"]) == 0
    (tmp_path / "n.txt").write_text("from a file")
    assert cli.main(["note", "T9", f"@{tmp_path / 'n.txt'}"]) == 0
    notes = [x["note"] for x in Dossier.load(tmp_path / "work" / "targets", "T9").log]
    assert "the `pocket` evidence: $HOME stays" in notes and "from a file" in notes


def test_a_step_may_not_start_a_background_watcher():
    """A step's background watcher kept its process, and the daemon, waiting on a queued job."""
    import asyncio
    from residudes import guard
    hook = guard.hook(None)
    out = asyncio.run(hook({"tool_name": "Bash", "tool_input": {"command": "while squeue -j 1; do sleep 60; done",
                                                                  "run_in_background": True}}, None, None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "end the step" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert asyncio.run(hook({"tool_name": "Bash", "tool_input": {"command": "ls"}}, None, None)) == {}


def test_the_mandate_says_how_to_approve_and_the_guard_lets_the_usage_through():
    """M1370: told `approve T "<why>"` (no such argument), then refused even `residudes approve -h`."""
    from residudes import guard
    from residudes.dossier import Dossier
    d = Dossier(target="T7", stage="SLATE_PROPOSED")
    d.automation = {"on": True, "by": "a-person", "up_to": "PACKAGED"}
    goal, _ = d.next_step()
    assert 'residudes approve T7 --why "' in goal and "residudes stage" not in goal
    assert guard.check("residudes approve -h", None) is None
    assert guard.check("residudes approve T7 --why x", None) is not None          # no mandate: still refused
    assert guard.check("residudes approve T7 --why x", "T7") is None
