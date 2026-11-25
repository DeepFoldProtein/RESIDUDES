"""`residudes structures`: the public-structure sweep, with RCSB answered from fixtures.

The network is never touched here — `search` and `entry_meta` go through two one-line helpers,
and these tests replace those.
"""
import json

import pytest

from residudes import structures
from residudes.dossier import Dossier

FASTA = ">H1388,subunit1|\nPKGLSASLPDL\n>H1388,subunit2|\nMVDREQLVQKARLAEQAERYDDMAAAMKAVTE\n"


def _dossier(tmp_path, text=FASTA):
    p = tmp_path / "target.fasta"
    p.write_text(text)
    d = Dossier(target="H1388")
    d.sequence = {"fasta": str(p)}
    return d


def _result(identifier, identity, alignment_length, query_length):
    return {"identifier": identifier, "services": [{"service_type": "sequence", "nodes": [
        {"match_context": [{"sequence_identity": identity, "evalue": 0.0, "alignment_length": alignment_length,
                            "query_length": query_length, "query_beg": 1, "query_end": alignment_length}]}]}]}


def test_chains_are_lettered_in_file_order(tmp_path):
    got = structures.chains(_dossier(tmp_path))
    assert [(c[0], len(c[2])) for c in got] == [("A", 11), ("B", 32)]
    assert got[1][1] == "H1388,subunit2|"


def test_a_dossier_without_a_sequence_says_so(tmp_path):
    d = Dossier(target="T0000")
    with pytest.raises(ValueError, match="no sequence"):
        structures.chains(d)


def test_search_reads_identity_and_coverage_from_the_alignment(monkeypatch):
    payload = {}

    def fake_post(url, body, timeout=120):
        payload.update(body)
        return {"result_set": [_result("6A5S_2", 1.0, 32, 32), _result("1abc_1", 0.97, 16, 32)]}

    monkeypatch.setattr(structures, "_post", fake_post)
    hits = structures.search("M" * 32, identity=0.9, rows=50)
    assert payload["query"]["parameters"]["identity_cutoff"] == 0.9
    assert payload["request_options"]["paginate"]["rows"] == 50
    assert [h["entry"] for h in hits] == ["6A5S", "1ABC"]          # entry upper-cased, entity split off
    assert [h["entity"] for h in hits] == ["2", "1"]
    assert hits[0]["coverage"] == 1.0 and hits[1]["coverage"] == 0.5
    assert structures.is_near_identical(hits[0])                   # 100 % over the whole chain
    assert not structures.is_near_identical(hits[1])               # 97 %, but only half of it


def test_entry_metadata_that_cannot_be_read_is_not_fatal(monkeypatch):
    def boom(url, timeout=60):
        raise OSError("no network")
    monkeypatch.setattr(structures, "_get_json", boom)
    assert structures.entry_meta("6A5S") == {"method": None, "resolution": None, "released": None, "title": None}


def test_entry_metadata_is_read_from_the_entry_record(monkeypatch):
    monkeypatch.setattr(structures, "_get_json", lambda url, timeout=60: {
        "rcsb_entry_info": {"experimental_method": "X-ray", "resolution_combined": [2.1]},
        "rcsb_accession_info": {"initial_release_date": "2018-03-07T00:00:00Z"},
        "struct": {"title": "14-3-3 gamma with a phosphopeptide"}})
    assert structures.entry_meta("6A5S") == {"method": "X-ray", "resolution": 2.1, "released": "2018-03-07",
                                             "title": "14-3-3 gamma with a phosphopeptide"}


def _patch_rcsb(monkeypatch, per_chain):
    """per_chain: {query sequence length: [result, …]} so each chain gets its own answer."""
    monkeypatch.setattr(structures, "_post",
                        lambda url, body, timeout=120: {"result_set": per_chain.get(
                            len(body["query"]["parameters"]["value"]), [])})
    monkeypatch.setattr(structures, "_get_json", lambda url, timeout=60: {
        "rcsb_entry_info": {"experimental_method": "X-ray", "resolution_combined": [2.1]},
        "rcsb_accession_info": {"initial_release_date": "2018-03-07T00:00:00Z"},
        "struct": {"title": "a title"}})
    monkeypatch.setattr(structures, "template_db_dates", lambda cfg=None: [
        {"name": "PDB mmCIF (templates)", "status": "stale", "date": "2026-04-25"}])


def test_the_sweep_records_evidence_and_names_the_near_identical_entries(tmp_path, monkeypatch):
    _patch_rcsb(monkeypatch, {32: [_result("6A5S_2", 1.0, 32, 32), _result("1ABC_1", 0.91, 20, 32)]})
    d = _dossier(tmp_path)
    out = structures.sweep_cmd(None, d, "agent", identity=0.9, limit=10)

    ev = d.evidence["public_structures"][-1]
    assert ev["by"] == "agent" and ev["value"]["identity_cutoff"] == 0.9
    assert ev["value"]["near_identical"] == {"B": ["6A5S"]}        # the partial-coverage hit is not one
    assert [h["entry"] for h in ev["value"]["per_chain"]["B"]] == ["6A5S", "1ABC"]
    assert ev["value"]["per_chain"]["A"] == []
    assert json.dumps(ev)                                          # the dossier has to serialise

    assert "chain A (H1388,subunit1|): 0 entities" in out and "no public structure of this sequence" in out
    assert "* 6A5S" in out and "2018-03-07" in out and "X-ray" in out
    assert "near-identical" in out and "B: 6A5S" in out
    assert "AlphaFold 3 filters these out" in out                  # why the sweep is a step of its own
    assert "template db: PDB mmCIF (templates) — stale (2026-04-25" in out


def test_nothing_found_says_so_instead_of_warning(tmp_path, monkeypatch):
    _patch_rcsb(monkeypatch, {})
    d = _dossier(tmp_path)
    out = structures.sweep_cmd(None, d, "agent")
    assert "no near-identical public structure for any chain" in out
    assert "AlphaFold 3 filters" not in out
    assert d.evidence["public_structures"][-1]["value"]["near_identical"] == {}


def test_one_chain_can_be_asked_for(tmp_path, monkeypatch):
    _patch_rcsb(monkeypatch, {32: [_result("6A5S_2", 1.0, 32, 32)]})
    d = _dossier(tmp_path)
    out = structures.sweep_cmd(None, d, "agent", chain="b")
    assert "chain B" in out and "chain A" not in out
    assert set(d.evidence["public_structures"][-1]["value"]["per_chain"]) == {"B"}
    with pytest.raises(ValueError, match="no chain Z"):
        structures.sweep_cmd(None, d, "agent", chain="Z")


# ------------------------------------------------------------------ foldseek: the structural half

def _db(tmp_path, name="pdb2026", extra=()):
    d = tmp_path / "fsdb"
    d.mkdir(exist_ok=True)
    for suffix in ("", "_ca", "_h", "_ss"):
        (d / f"{name}{suffix}.dbtype").write_text("x")
    for other in extra:
        (d / f"{other}.dbtype").write_text("x")
    return d


def test_db_prefix_takes_a_prefix_or_a_directory(tmp_path):
    d = _db(tmp_path)
    assert structures.db_prefix(d) == d / "pdb2026"               # one database in a directory
    assert structures.db_prefix(d / "pdb2026") == d / "pdb2026"   # already the prefix
    with pytest.raises(ValueError, match="several foldseek databases"):
        structures.db_prefix(_db(tmp_path, extra=("afdb",)))
    with pytest.raises(FileNotFoundError):
        structures.db_prefix(tmp_path / "nowhere")
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no foldseek database in"):
        structures.db_prefix(empty)


def test_the_newest_configured_database_wins(tmp_path, monkeypatch):
    from residudes.config import Config
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    new, old = _db(tmp_path / "a", "pdb2026"), _db(tmp_path / "b", "pdb")
    cfg = Config(extra={"databases": {"foldseek_pdb": str(old), "foldseek_pdb_2026": str(new)}})
    prefix, how = structures.foldseek_db(cfg)
    assert prefix.name == "pdb2026" and how == "[databases] foldseek_pdb_2026"
    assert structures.foldseek_db(Config(extra={"databases": {"foldseek_pdb": str(old)}}))[1] == "[databases] foldseek_pdb"
    assert structures.foldseek_db(cfg, str(old))[1] == "--db"      # --db wins over both

    import residudes.catalog as catalog_mod                        # nothing configured: fall back to the catalog
    monkeypatch.setattr(catalog_mod, "load", lambda path=None, paths=None: [
        {"name": "foldseek PDB", "any_of": [str(old)], "unset": []}])
    assert structures.foldseek_db(Config(extra={})) == (old / "pdb", "catalog: foldseek PDB")
    monkeypatch.setattr(catalog_mod, "load", lambda path=None, paths=None: [])
    with pytest.raises(FileNotFoundError, match="no foldseek database configured"):
        structures.foldseek_db(Config(extra={}))


def test_entry_ids_survive_the_spellings_foldseek_uses():
    assert structures._entry_of("6a5s_A") == ("6A5S", "A")
    assert structures._entry_of("/db/pdb6a5s.ent.gz_B") == ("6A5S", "B")
    assert structures._entry_of("6A5S.cif.gz_C") == ("6A5S", "C")
    assert structures._entry_of("8ARV") == ("8ARV", None)


TSV = (
    "model\t6a5s_A\t0.98\t230\t233\t240\t1e-60\t800\t0.95\t0.99\n"
    "model\t6a5s_B\t0.98\t228\t233\t240\t1e-59\t790\t0.94\t0.99\n"   # same entry, worse: dropped
    "model\t8arv_A\t0.21\t210\t233\t250\t1e-20\t300\t0.86\t0.90\n"   # low identity, same fold
    "model\t1abc_A\t0.15\t90\t233\t200\t1e-03\t60\t0.41\t0.30\n"
)


def test_a_foldseek_table_becomes_one_row_per_entry(tmp_path):
    p = tmp_path / "fs.tsv"; p.write_text(TSV)
    hits = structures.read_foldseek(p)
    assert [h["entry"] for h in hits] == ["6A5S", "8ARV", "1ABC"]   # best bits first, one per entry
    assert hits[0]["chain"] == "A" and hits[0]["tm"] == 0.95
    assert hits[0]["coverage"] == round(230 / 233, 3)


def test_reading_foldseek_flags_what_the_sequence_sweep_missed(tmp_path, monkeypatch):
    monkeypatch.setattr(structures, "_get_json", lambda url, timeout=60: {
        "rcsb_entry_info": {"experimental_method": "X-ray", "resolution_combined": [2.1]},
        "rcsb_accession_info": {"initial_release_date": "2026-01-09T00:00:00Z"}, "struct": {"title": "t"}})
    p = tmp_path / "fs.tsv"; p.write_text(TSV)
    d = _dossier(tmp_path)
    d.evidence["public_structures"] = [{"time": "2026-11-25T00:00:00+00:00", "by": "agent",
                                        "value": {"per_chain": {"B": [{"entry": "6A5S"}]}}}]
    out = structures.read_foldseek_cmd(None, d, str(p), "agent")

    ev = d.evidence["public_structures"][-1]["value"]
    assert ev["source"] == "foldseek" and ev["close"] == ["6A5S", "8ARV"]
    assert ev["new_to_the_sequence_sweep"] == ["8ARV"]      # 21 % identical: no sequence search finds it
    assert "found by structure, not by sequence: 8ARV" in out
    assert "* 8ARV" in out and "0.86" in out


def test_a_foldseek_search_is_submitted_with_the_columns_the_reader_expects(tmp_path, monkeypatch):
    from residudes import ops
    from residudes.config import Config
    from residudes.dossier import Dossier
    sent = {}

    def fake_sbatch(cmd, name, log, gpu, hours, after=None, cpus=8):
        sent.update(cmd=cmd, name=name, gpu=gpu)
        return "4242"

    monkeypatch.setattr(ops, "_sbatch", fake_sbatch)
    monkeypatch.setattr("residudes.msa.tool", lambda cfg, name: "/bin/foldseek")
    db = _db(tmp_path)
    cfg = Config(work_root=str(tmp_path / "work"), extra={"databases": {"foldseek_pdb_2026": str(db)}})
    d = _dossier(tmp_path)
    Dossier.path(cfg.targets_dir, d.target).parent.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.cif"; model.write_text("data_x")

    out = structures.foldseek_cmd(cfg, d, str(model), "agent")
    assert sent["gpu"] is False and sent["name"] == "residudes-foldseek-H1388"
    assert f"easy-search {model} {db / 'pdb2026'}" in sent["cmd"]
    assert "--format-output " + ",".join(structures.FOLDSEEK_COLUMNS) in sent["cmd"]
    assert "--alignment-type 1" in sent["cmd"]                  # TM-align: alntmscore is only filled then
    assert d.runs[-1]["job"] == "4242" and d.runs[-1]["state"] == "submitted"
    assert "--read-foldseek" in out

    with pytest.raises(FileNotFoundError, match="no model at"):
        structures.foldseek_cmd(cfg, d, str(tmp_path / "gone.cif"), "agent")


# ------------------------------------------------------------------ hhsearch: profile against profile

HHR = """Query         H1388_B
Match_columns 233
No_of_seqs    412 out of 1823

 No Hit                             Prob E-value P-value  Score    SS Cols Query HMM  Template HMM
  1 6a5s_A 14-3-3 gamma            100.0 1.2E-60 2.3E-65  450.2  21.3  230    2-232     1-234 (240)
  2 8arv_A BifA EAL domain          99.2 4.0E-20 7.1E-25  210.4  18.0  205    6-228     3-210 (250)
  3 1abc_B something else           45.0     2.0 3.1E-05   40.1   5.0   60   10-70     11-72  (120)
"""


def _hhdb(tmp_path, name="pdb70"):
    d = tmp_path / "hh"
    d.mkdir(exist_ok=True)
    for part in ("_hhm.ffdata", "_a3m.ffdata", "_cs219.ffdata"):
        (d / f"{name}{part}").write_text("x")
    return d / name


def test_an_hh_database_is_a_prefix_and_its_absence_points_at_foldseek(tmp_path):
    from residudes.config import Config
    prefix = _hhdb(tmp_path)
    cfg = Config(extra={"databases": {"hhsearch_db": str(prefix)}})
    assert structures.hh_db(cfg) == (prefix, "[databases] hhsearch_db")
    assert structures.hh_db(cfg, str(prefix))[1] == "--db"
    with pytest.raises(FileNotFoundError, match="--foldseek"):
        structures.hh_db(Config(extra={}))
    with pytest.raises(FileNotFoundError, match="no HH-suite database at"):
        structures.hh_db(Config(extra={"databases": {"hhsearch_db": str(tmp_path / "nope")}}))


def test_the_query_is_the_chain_s_msa_when_there_is_one(tmp_path, monkeypatch):
    from residudes.config import Config
    from residudes.dossier import Dossier
    cfg = Config(work_root=str(tmp_path / "work"))
    d = _dossier(tmp_path)
    Dossier.path(cfg.targets_dir, d.target).parent.mkdir(parents=True, exist_ok=True)

    q, what = structures.hh_query(cfg, d, "B")               # no MSA: the bare sequence, and it says so
    assert q.name == "query_B.fasta" and "no MSA yet" in what
    assert q.read_text().startswith(">H1388_B\nMVDRE")

    a3m = tmp_path / "unpaired_B.a3m"; a3m.write_text(">q\nMVD\n")
    monkeypatch.setattr("residudes.msa.target_msas", lambda cfg, d: [a3m])
    q2, what2 = structures.hh_query(cfg, d, "B")
    assert q2 == a3m and what2 == "MSA unpaired_B.a3m"
    with pytest.raises(ValueError, match="no chain Z"):
        structures.hh_query(cfg, d, "Z")


def test_an_hhr_table_becomes_one_row_per_entry(tmp_path):
    p = tmp_path / "x.hhr"; p.write_text(HHR)
    hits = structures.read_hhsearch(p)
    assert [h["entry"] for h in hits] == ["6A5S", "8ARV", "1ABC"]
    assert hits[0]["prob"] == 100.0 and hits[0]["cols"] == 230 and hits[0]["query_range"] == [2, 232]
    assert hits[1]["evalue"] == 4.0e-20 and hits[1]["template_length"] == 250


def test_reading_hhsearch_names_the_remote_homologues(tmp_path, monkeypatch):
    monkeypatch.setattr(structures, "_get_json", lambda url, timeout=60: {})
    p = tmp_path / "x.hhr"; p.write_text(HHR)
    d = _dossier(tmp_path)
    d.evidence["public_structures"] = [{"time": "2026-11-25T00:00:00+00:00", "by": "agent",
                                        "value": {"per_chain": {"B": [{"entry": "6A5S"}]}}}]
    out = structures.read_hhsearch_cmd(None, d, str(p), "agent")
    ev = d.evidence["public_structures"][-1]["value"]
    assert ev["source"] == "hhsearch" and ev["likely"] == ["6A5S", "8ARV"]     # 45 % is not one
    assert ev["new_to_the_sequence_sweep"] == ["8ARV"]
    assert "found by profile, not by sequence: 8ARV" in out
    assert "a template candidate, not a match" in out


def test_an_hhsearch_is_submitted_against_the_configured_database(tmp_path, monkeypatch):
    from residudes import ops
    from residudes.config import Config
    from residudes.dossier import Dossier
    sent = {}
    monkeypatch.setattr(ops, "_sbatch", lambda cmd, name, log, gpu, hours, after=None, cpus=8: sent.update(
        cmd=cmd, name=name) or "77")
    monkeypatch.setattr("residudes.msa.tool", lambda cfg, name: f"/bin/{name}")
    prefix = _hhdb(tmp_path)
    cfg = Config(work_root=str(tmp_path / "work"), extra={"databases": {"hhsearch_db": str(prefix)}})
    d = _dossier(tmp_path)
    Dossier.path(cfg.targets_dir, d.target).parent.mkdir(parents=True, exist_ok=True)
    out = structures.hhsearch_cmd(cfg, d, "agent", chain="B")
    assert sent["name"] == "residudes-hhsearch-H1388"
    assert "/bin/hhsearch -i " in sent["cmd"] and f"-d {prefix}" in sent["cmd"] and "hhsearch_B.hhr" in sent["cmd"]
    assert d.runs[-1]["job"] == "77" and "--read-hhsearch" in out


# ------------------------------------------------------------------ what the report makes of it

def test_the_report_renders_all_three_searches_not_only_the_last():
    from residudes import report
    ev = {"public_structures": [
        {"value": {"identity_cutoff": 0.9, "near_identical": {"B": ["6A5S", "3UZD"]},
                   "per_chain": {"A": [], "B": [{"entry": "6A5S", "identity": 1.0, "coverage": 1.0,
                                                 "method": "X-ray", "resolution": 2.1, "released": "2018-03-07",
                                                 "title": "14-3-3 gamma"}]}}},
        {"value": {"source": "foldseek", "close": ["8ARV"], "new_to_the_sequence_sweep": ["8ARV"],
                   "hits": [{"entry": "8ARV", "chain": "A", "tm": 0.86, "coverage": 0.9, "method": "X-ray",
                             "resolution": 1.9, "released": "2026-01-09", "title": "BifA EAL"}]}},
        {"value": {"source": "hhsearch", "likely": ["9XYZ"], "new_to_the_sequence_sweep": [],
                   "hits": [{"entry": "9XYZ", "prob": 99.2, "coverage": 0.8}]}},
    ]}
    out = "\n".join(report._templates(ev))
    assert "| `6a5s` | sequence B | 100% |" in out        # the sequence sweep is not lost behind the later ones
    assert "| `8arv`_A | foldseek | TM 0.86 |" in out
    assert "| `9xyz` | hhsearch | prob 99 |" in out
    assert "came back from foldseek and not from the sequence sweep" in out
    assert "AlphaFold 3 filters those out" in out and "`6a5s`, `3uzd`" in out


def test_hand_written_structure_evidence_still_renders_as_before():
    from residudes import report
    ev = {"public_structures": [{"value": {"best": "6A5S at 2.10 A, 100% identity"}}]}
    out = "\n".join(report._templates(ev))
    assert "| entry | identity | coverage | search | what it reported |" in out
    assert "6A5S at 2.10 A" in out


def test_the_key_points_count_every_search_and_name_what_only_structure_found():
    from residudes import report
    from residudes.dossier import Dossier
    d = Dossier(target="T0000")
    d.evidence["public_structures"] = [
        {"value": {"per_chain": {"B": [{"entry": "6A5S"}]}}},
        {"value": {"source": "foldseek", "hits": [{"entry": "8ARV"}], "new_to_the_sequence_sweep": ["8ARV"]}},
    ]
    line = dict(report.key_points(d, None))["Structures that exist"]
    assert "**2** PDB entries" in line and "`6a5s`" in line and "`8arv`" in line
    assert "`8arv` only by structure or profile" in line


def test_the_deck_gets_a_sentence_not_a_blob():
    from residudes import report
    ev = {"public_structures": [
        {"value": {"per_chain": {"B": [{"entry": "6A5S"}, {"entry": "3UZD"}]},
                   "near_identical": {"B": ["6A5S", "3UZD"]}}},
        {"value": {"source": "foldseek", "hits": [{"entry": "8ARV"}], "new_to_the_sequence_sweep": ["8ARV"]}},
    ]}
    line = report.sweep_summary(ev)
    assert line == ("3 PDB entries; near-identical 6a5s (B), 3uzd (B); only by structure or profile: 8arv")
    assert report.sweep_summary({"public_structures": [{"value": {"best": "6A5S by hand"}}]}) is None
    assert report.sweep_summary({}) is None
