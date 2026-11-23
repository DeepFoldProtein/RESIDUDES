"""One lantern for every harness: Claude Code, Codex, Antigravity and the Agent SDK runner read the same skills and
the same instructions. These tests check what each harness's documentation requires:

  Claude Code   .claude/skills/<name>/SKILL.md; CLAUDE.md (imports AGENTS.md with `@AGENTS.md`); .claude/settings.json
  Codex         .agents/skills/<name>/SKILL.md with `name` and `description`; AGENTS.md (32 KiB default limit)
  Antigravity   .agents/skills (current; legacy .agent/skills); `description` required, `name` defaults to the folder;
                AGENTS.md without frontmatter, truncated past 24,000 bytes
  Agent SDK     lantern agent step appends AGENTS.md to its system prompt; skills from .claude/skills
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILLS = sorted(p.parent.name for p in (ROOT / "skills").glob("*/SKILL.md"))
HARNESS_DIRS = {"Claude Code": ".claude/skills", "Codex": ".agents/skills", "Antigravity": ".agents/skills",
                "Antigravity (legacy)": ".agent/skills"}


def _frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md must start with YAML frontmatter"
    out = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def test_there_are_skills():
    assert {"lantern-intake", "lantern-stoichiometry", "lantern-consult", "lantern-selection"} <= set(SKILLS)


@pytest.mark.parametrize("harness,sub", HARNESS_DIRS.items())
def test_every_harness_finds_every_skill(harness, sub):
    d = ROOT / sub
    assert d.exists(), f"{harness}: {sub} missing"
    found = sorted(p.parent.name for p in d.glob("*/SKILL.md"))
    assert found == SKILLS, f"{harness} sees {found}"


@pytest.mark.parametrize("name", SKILLS)
def test_skill_frontmatter_meets_every_spec(name):
    fm = _frontmatter((ROOT / "skills" / name / "SKILL.md").read_text())
    assert fm.get("name") == name                          # Codex requires it; Antigravity defaults to the folder name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64     # Agent Skills naming
    desc = fm.get("description", "")
    assert 20 <= len(desc) <= 1024 and "<" not in desc     # required by all three; spec limit 1024
    assert "Use " in desc or "use " in desc                # says when to use it: what harnesses match on


def test_agents_md_is_the_shared_instruction_file():
    text = (ROOT / "AGENTS.md").read_text()
    assert not text.startswith("---")                      # Antigravity: AGENTS.md takes no frontmatter
    assert len(text.encode()) < 24_000                     # Antigravity truncates rule files past 24,000 bytes (Codex: 32 KiB)
    for name in SKILLS:
        assert f"`{name}`" in text, f"AGENTS.md does not list {name}"
    for rule in ("lantern approve", "lantern resolve", "LANTERN_ACTOR", "PLAN_REVIEW", "RESULTS_REVIEW"):
        assert rule in text


def test_claude_md_imports_agents_md():
    assert (ROOT / "CLAUDE.md").read_text().splitlines()[0].strip() == "@AGENTS.md"


def test_claude_code_settings_keep_the_session_an_agent():
    s = json.loads((ROOT / ".claude" / "settings.json").read_text())
    assert s["env"]["LANTERN_ACTOR"] == "agent"
    deny = " ".join(s["permissions"]["deny"])
    assert "lantern approve" in deny and "lantern resolve" in deny


def test_sdk_runner_uses_agents_md():
    from lantern.agent import _prompt
    from lantern.config import Config
    from lantern.dossier import Dossier
    system, _ = _prompt(Dossier(target="T"), Config(work_root="/tmp/lantern-test"))
    assert "Work it out with a person" in system and "lantern-consult" in system


def test_setup_links_every_harness(tmp_path):
    from lantern.setup import HARNESS_SKILL_DIRS, link_skills
    link_skills(tmp_path)
    for sub in HARNESS_SKILL_DIRS:
        assert sorted(p.parent.name for p in (tmp_path / sub).glob("*/SKILL.md")) == SKILLS


REPO = ROOT.parent


@pytest.mark.parametrize("sub", [".claude/skills", ".agents/skills", ".agent/skills"])
def test_skills_found_from_the_repo_root(sub):
    """Codex walks from the repo root down to the working folder and Antigravity reads only the workspace folder
    and its parents, so a harness opened at RESIDUDES/ needs the links (and AGENTS.md) at the root as well."""
    found = sorted(p.parent.name for p in (REPO / sub).glob("*/SKILL.md"))
    assert found == SKILLS


def test_root_agents_md_points_to_lantern():
    text = (REPO / "AGENTS.md").read_text()
    assert "lantern/AGENTS.md" in text and not text.startswith("---") and len(text.encode()) < 24_000
    assert (REPO / "CLAUDE.md").read_text().strip() == "@AGENTS.md"


def test_modeller_script_writes_a_multichain_pir():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bam", ROOT / "skills/lantern-modeller/scripts/build_assembly_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                               # imports only argparse/re: no MODELLER, no gemmi needed
    text = m.pir("tmpl", "tmpl.pdb", [("A", "MKV"), ("B", "MKV"), ("C", "MKV")], "MKVL", 3)
    assert "structureX:tmpl.pdb:FIRST:A:LAST:C::::" in text and "MKVL/MKVL/MKVL*" in text and "MKV/MKV/MKV*" in text
    with pytest.raises(ValueError):
        m.pir("tmpl", "tmpl.pdb", [("A", "MKV")], "MKVL", 3)      # the template must have one chain per copy


def _bam():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bam", ROOT / "skills/lantern-modeller/scripts/build_assembly_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_modeller_script_maps_a_hetero_complex(tmp_path):
    """A released complex with gaps (H2422/H2423): one FASTA record per distinct chain, each template chain gets the
    record it matches best, and the mmCIF carries each chain's own full sequence."""
    pytest.importorskip("gemmi")
    m = _bam()
    fa = tmp_path / "t.fasta"
    fa.write_text(">heavy chain\nMKVLAEGG\n>light\nGSHHTWQR\n")
    recs = m.read_fasta_all(fa)
    assert recs == [("heavy", "MKVLAEGG"), ("light", "GSHHTWQR")]
    chains = [("A", "MKVLAE"), ("B", "SHHTWQ"), ("C", "KVLAEG"), ("D", "GSHHTW")]   # observed residues only
    plan = m.assign(chains, recs)
    assert [p[0] for p in plan] == ["heavy", "light", "heavy", "light"] and all(p[2] == 1.0 for p in plan)
    assert m.assign(chains, recs, {"A": "light"})[0][0] == "light"
    with pytest.raises(ValueError):
        m.assign(chains, recs, {"A": "nanobody"})
    text = m.pir("tmpl", "t.pdb", chains, [p[1] for p in plan])
    assert "MKVLAEGG/GSHHTWQR/MKVLAEGG/GSHHTWQR*" in text and "MKVLAE/SHHTWQ/KVLAEG/GSHHTW*" in text
    # the CIF writer: two chains of different sequence -> two entities, each with its full sequence
    lines, n = [], 1
    for ch, seq in (("A", "MKV"), ("B", "GSH")):
        for i, aa in enumerate(seq, start=1):
            res = {"M": "MET", "K": "LYS", "V": "VAL", "G": "GLY", "S": "SER", "H": "HIS"}[aa]
            lines.append(f"ATOM  {n:5d}  CA  {res} {ch}{i:4d}    {3.8 * i:8.3f}{10.0 * (ch == 'B'):8.3f}   0.000  1.00 50.00           C")
            n += 1
        lines.append("TER")
    pdb = tmp_path / "m.pdb"
    pdb.write_text("\n".join(lines + ["END"]) + "\n")
    out = m.to_af3_cif(pdb, ["MKVLAEGG", "GSHHTWQR"], tmp_path / "best.cif")
    import gemmi
    doc = gemmi.cif.read(str(out))
    seqs = {}
    for e, mon in zip(doc[0].find_values("_entity_poly_seq.entity_id"), doc[0].find_values("_entity_poly_seq.mon_id")):
        seqs.setdefault(e, []).append(mon)
    assert sorted(gemmi.one_letter_code(v) for v in seqs.values()) == ["GSHHTWQR", "MKVLAEGG"]


def test_setup_can_copy_skills_for_a_harness_that_does_not_follow_symlinks(tmp_path):
    from lantern import setup
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".agents" / "skills" / "mine").mkdir(parents=True)               # a user's own skill: never touched
    (proj / ".agents" / "skills" / "mine" / "SKILL.md").write_text("---\nname: mine\ndescription: x\n---\n")
    done = setup.link_skills(proj, copy=True)
    copied = proj / ".agents" / "skills" / "lantern-selection"
    assert copied.is_dir() and not copied.is_symlink() and (copied / "SKILL.md").exists()
    assert (copied / setup.COPY_MARKER).exists() and len(done) >= 7 * 3
    (copied / "SKILL.md").write_text("stale")
    setup.link_skills(proj, copy=True)                                        # a rerun refreshes lantern's copies
    assert (copied / "SKILL.md").read_text().startswith("---")
    assert (proj / ".agents" / "skills" / "mine" / "SKILL.md").read_text().startswith("---\nname: mine")
    other = tmp_path / "proj2"
    setup.link_skills(other)                                                  # the default stays a symlink
    assert (other / ".agents" / "skills" / "lantern-selection").is_symlink()


def test_always_on_instructions_fit_antigravity_budgets():
    """Antigravity: each AGENTS.md is truncated past 24,000 bytes; all always-on rules share 20,000 tokens."""
    files = [ROOT / "AGENTS.md", REPO / "AGENTS.md"]
    assert all(len(f.read_text().encode()) < 24_000 for f in files)
    assert sum(len(f.read_text()) for f in files) / 4 < 20_000 * 0.5            # ~4 characters a token, with room


def test_copy_skills_replaces_lanterns_own_links_only(tmp_path):
    from lantern import setup
    proj = tmp_path / "p"
    setup.link_skills(proj)                                                   # first the default: symlinks
    d = proj / ".agents" / "skills"
    assert (d / "lantern-selection").is_symlink()
    (d / "elsewhere").symlink_to(tmp_path)                                    # someone else's link: left alone
    setup.link_skills(proj, copy=True)
    assert not (d / "lantern-selection").is_symlink() and (d / "lantern-selection" / setup.COPY_MARKER).exists()
    assert (d / "elsewhere").is_symlink()


def test_doctor_finds_broken_and_missing_skill_links(tmp_path):
    from lantern import setup
    setup.link_skills(tmp_path)
    d = tmp_path / ".agents" / "skills"
    (d / "lantern-selection").unlink()                                        # one skill gone
    (d / "lantern-old").symlink_to(tmp_path / "nowhere")                      # a dangling link
    rows = {r["folder"]: r for r in setup.skill_check([tmp_path])}
    r = rows[str(d)]
    assert r["missing"] == ["lantern-selection"] and r["broken"] == ["lantern-old"] and r["found"] == r["of"] - 1
    assert rows[str(tmp_path / ".claude" / "skills")]["found"] == r["of"]
