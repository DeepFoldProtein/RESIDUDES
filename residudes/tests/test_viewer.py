"""The structure pane the chat agent opens beside the conversation (viewer.py), and the conversation pane it must
never be mistaken for (chat.agent_pane)."""
import subprocess as sp

from residudes import chat, viewer


def test_the_pane_superposes_several_models_and_faces_a_peptide_in_one():
    one = viewer.script([{"label": "M1", "path": "/r/sep/af3/seed_20/sample_0/model.cif"}], ("A",), "block", "/v/view.cam")
    assert one.startswith("set renderer block\nload /r/sep/af3/seed_20/sample_0/model.cif\nrename model M1\n")
    assert "show ribbon\ncolor chain" in one and "show ballstick chain A" in one and "spacefill" not in one   # slim
    assert "camera load /v/view.cam\nzoom M1" in one
    px = viewer.script([{"label": "M1", "path": "/r/model.cif"}], ("A",), None, None)   # pixels: a grey cartoon
    assert "show cartoon\ncolor #BBBBBB\nshow ballstick chain A\ncolor #D55E00 chain A" in px and "ribbon" not in px
    assert "set cartoon_mode pymol\nhide all" in px                                   # PyMOL's cartoon, not molterm's
    assert one.endswith("zoom M1\nselect clear\n")                 # no selection halo over the live picture
    two = viewer.script([{"label": "M1", "path": "/a/model.cif"}, {"label": "M3", "path": "/b/model.cif"}], (), None, None)
    assert two.startswith("set renderer auto\n")                         # not left to ~/.molterm/init.mt
    assert "rename model M3" in two                                       # same stem, each renamed on arrival
    assert "align M3 to M1 automap" in two and "color #D55E00 M3" in two and "orient M1" in two


def test_text_renderer_unless_tmux_passes_pixels(monkeypatch):
    monkeypatch.delenv("RESIDUDES_VIEW_RENDERER", raising=False)
    monkeypatch.setattr(viewer, "_tmux", lambda *a: sp.CompletedProcess(a, 0, "bpaste,clipboard,focus\n", ""))
    assert viewer.renderer() == "block"                    # tmux 3.2a: sixel would be swallowed, the pane blank
    monkeypatch.setattr(viewer, "_tmux", lambda *a: sp.CompletedProcess(a, 0, "clipboard,sixel,focus\n", ""))
    assert viewer.renderer() is None
    monkeypatch.setenv("RESIDUDES_VIEW_RENDERER", "braille")
    assert viewer.renderer() == "braille"


def test_outside_tmux_the_agent_is_told_so_rather_than_failing(monkeypatch):
    monkeypatch.delenv("TMUX", raising=False)
    assert "not inside tmux" in viewer.show(None, None, ["M1"])
    assert "no structure is open" in viewer.do(None, None, "color plddt")


def test_the_daemon_types_into_the_conversation_not_the_structure(monkeypatch):
    """With the structure pane focused, the window's active pane is molterm; a prompt sent to the window would be
    typed into the viewer. The conversation's pane is addressed by its own tag, or as the pane that is not the view."""
    tagged = "%0\t1\t\n%2\t\tH1388\n"
    monkeypatch.setattr(chat, "_tmux", lambda *a, **k: sp.CompletedProcess(a, 0, tagged, ""))
    assert chat.agent_pane("H1388") == "%0"
    untagged = "%5\t\tH1388\n%4\t\t\n"                     # a window from before the tag: the one that is not the view
    monkeypatch.setattr(chat, "_tmux", lambda *a, **k: sp.CompletedProcess(a, 0, untagged, ""))
    assert chat.agent_pane("H1388") == "%4"


def test_target_show_lists_a_run_that_has_no_job_id(tmp_path, monkeypatch, capsys):
    """Found by the chat agent on H1388: a molterm render is recorded locally, without a job id, and
    `residudes target show` died on it with KeyError: 'job'."""
    from residudes import cli
    from residudes.dossier import Dossier
    cfg = tmp_path / "residudes.toml"
    cfg.write_text(f'[paths]\nwork_root = "{tmp_path}"\n')
    monkeypatch.setenv("RESIDUDES_CONFIG", str(cfg))
    d = Dossier(target="T")
    d.runs.append({"time": "t", "by": "a", "scheduler": "local", "what": "render: scenes.mt", "state": "done"})
    d.save(tmp_path / "targets")
    assert cli.main(["target", "show", "T"]) == 0
    assert "run local: render: scenes.mt [done]" in capsys.readouterr().out


def test_tmux_is_the_binary_of_the_server_we_are_in(monkeypatch, tmp_path):
    exe = tmp_path / "tmux-3.8"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(chat.os, "readlink", lambda p: str(exe) if p == "/proc/4242/exe" else "")
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/sixel,4242,0")
    assert chat.tmux_bin() == str(exe)                     # a 3.2a client would be dropped by this server
    monkeypatch.delenv("TMUX")
    assert chat.tmux_bin() != str(exe)                     # outside tmux: whatever is on PATH


def test_do_answers_with_molterm_then_drops_the_selection(monkeypatch):
    sent = []
    def fake(*a):
        sent.append(a)
        if a[0] == "list-panes":
            return sp.CompletedProcess(a, 0, "%0\t\n%2\tH1388\n", "")
        return sp.CompletedProcess(a, 0, "NORMAL M1\nSelected 79 atoms\n" if a[0] == "capture-pane" else "", "")
    monkeypatch.setattr(viewer, "_tmux", fake)
    monkeypatch.setattr(viewer, "here", lambda: "%0")
    monkeypatch.setattr(viewer.time, "sleep", lambda s: None)
    assert viewer.do(None, None, "select chain A; show ballstick sele") == "molterm: Selected 79 atoms"
    typed = [a[-1] for a in sent if a[:1] == ("send-keys",) and "-l" in a]
    assert typed == [":select chain A", ":show ballstick sele", ":select clear"]   # the reply is read before the clear
    assert sent[-1][-1] == "C-l"                           # and molterm redraws, or a sixel pane is left blank


def test_the_conversation_agent_stays_off_the_persons_editor(tmp_path):
    """Started from an editor's terminal, the agent CLI would attach to that editor and take it from the person."""
    from residudes.config import Config
    env = chat._agent_env(Config(work_root=str(tmp_path)))
    assert "CLAUDE_CODE_AUTO_CONNECT_IDE=false" in env
