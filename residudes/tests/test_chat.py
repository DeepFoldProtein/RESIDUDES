"""The conversation sessions: what `residudes` and `residudes chat` promise."""
from residudes import chat


def test_the_brief_names_the_target_and_the_dossier():
    """A session cannot start without it, and nothing else in the tests starts one."""
    text = chat.BRIEF.format(target="T1234", ddir="/w/targets/T1234")
    assert "T1234" in text and "/w/targets/T1234" in text
    assert "never approve" in text                      # the rules hold inside a conversation too


def test_every_backend_name_resolves():
    assert chat.backend("builtin") == "builtin"
    assert chat.backend(None) == "builtin"              # ours by default: nothing to install
    assert chat.backend("tmux") in ("tmux", "builtin")  # tmux when it is installed


def test_ago_reads_a_session_timestamp():
    assert chat._ago("not a date") == ""
    assert chat._ago("2020-01-01 00:00").endswith("d")
