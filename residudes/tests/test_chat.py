"""The conversation sessions: what `residudes` and `residudes chat` promise."""
from residudes import chat


def test_the_brief_names_the_target_and_the_dossier():
    """A session cannot start without it, and nothing else in the tests starts one."""
    text = chat.BRIEF.format(target="T1234", ddir="/w/targets/T1234")
    assert "T1234" in text and "/w/targets/T1234" in text
    assert "never approve" in text                      # the rules hold inside a conversation too


def test_sessions_are_named_for_their_target():
    """`residudes chat T` and `tmux attach -t residudes-T` have to mean the same session."""
    assert chat.PREFIX + "T1234" == "residudes-T1234"
    assert chat.LEAVE == "ctrl-b d"                     # tmux's own key, not one we invented


def test_ago_reads_a_session_timestamp():
    assert chat._ago("not a date") == ""
    assert chat._ago("2020-01-01 00:00").endswith("d")


def test_the_daemon_steps_aside_for_a_conversation(monkeypatch):
    """Two agents on one dossier duplicate work; a chat session means a person is on it."""
    from residudes import daemon
    monkeypatch.setattr(chat, "sessions", lambda *a, **k: [{"target": "T1234"}])
    assert daemon._in_conversation("T1234") is True
    assert daemon._in_conversation("T9999") is False
