import csv
import json
from pathlib import Path

from residudes import pool, retro, slate
from residudes.dossier import Dossier


def _make_pool(root: Path):
    """2 runners x 3 seeds x 2 samples at A3; runner 'hi' has inflated ranking scores."""
    for runner, base in (("af3", 0.6), ("hi", 0.9)):
        for seed in (1, 2, 3):
            for smp in (0, 1):
                d = root / "results" / "A3" / "af3_msa" / runner / f"seed_{seed}" / f"sample_{smp}"
                d.mkdir(parents=True)
                (d / "summary.json").write_text(json.dumps({"ranking_score": base + 0.01 * seed + 0.001 * smp,
                                                            "iptm": base, "ptm": base, "plddt": 80}))
                (d / "model.cif").write_text("data_x\n")


def test_pool_scan_zscores(tmp_path):
    _make_pool(tmp_path / "T")
    rows = pool.scan(tmp_path / "T")
    assert len(rows) == 12
    by = {(r["runner"], r["seed"], r["sample"]): r for r in rows}
    # z is within the runner: the best af3 model has the same z as the best 'hi' model despite lower raw score
    assert abs(by[("af3", "3", "1")]["rs_z_runner"] - by[("hi", "3", "1")]["rs_z_runner"]) < 1e-9
    s = pool.summarize(rows)
    assert s["A3/af3_msa/af3"]["n"] == 6


def test_slate_mixes_runners(tmp_path):
    _make_pool(tmp_path / "T")
    rows = pool.scan(tmp_path / "T")
    pool.write_csv(rows, tmp_path / "pool.csv")
    s = slate.propose(tmp_path / "pool.csv", "A3")
    runners = [m["runner"] for m in s["models"]]
    assert len(s["models"]) == 5
    assert set(runners) == {"af3", "hi"} and max(runners.count(r) for r in set(runners)) <= 3


def test_retro_scores_slate(tmp_path):
    _make_pool(tmp_path / "T")
    rows = pool.scan(tmp_path / "T")
    pool.write_csv(rows, tmp_path / "pool.csv")
    s = slate.propose(tmp_path / "pool.csv", "A3")
    j, _ = slate.write(s, tmp_path, 1)
    scores = tmp_path / "scores.csv"
    with open(scores, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "lddt", "dockq_wave", "rank_m1", "n_field"])
        for r in rows:
            q = 0.9 if r["runner"] == "af3" else 0.5          # the inflated runner is the wrong one
            w.writerow([f"T__A3__af3_msa__{r['runner']}__s{r['seed']}_{r['sample']}", q, q, 1 if q > 0.8 else 40, 78])
    d = Dossier(target="T")
    d.evidence["slate_proposal"] = [{"value": {"json": str(j)}}]
    d.decide("t", "selection", "slate v1", ["test"])
    out = retro.run(d, tmp_path, scores, "t")
    assert "retrospective" in out and d.decisions[0]["outcome"]["model1"]["lddt"] in (0.9, 0.5)
    assert json.loads((tmp_path / "retro.json").read_text())["n_field"] == 78


def test_hedged_slate_and_qa_keys(tmp_path):
    _make_pool(tmp_path / "T")
    for runner, base in (("af3", 0.5), ("hi", 0.8)):     # an A1 part of the pool
        for seed in (1, 2):
            d = tmp_path / "T" / "results" / "A1" / "notpl" / runner / f"seed_{seed}" / "sample_0"
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"ranking_score": base + 0.01 * seed, "ptm": base, "plddt": 80}))
            (d / "model.cif").write_text("data_x\n")
    rows = pool.scan(tmp_path / "T")
    pool.write_csv(rows, tmp_path / "pool.csv")
    qa = tmp_path / "qa"                                   # clusters for A3 only; every af3 seed in one cluster
    cdir = qa / "cluster_A3" / "A3" / "af3_msa" / "tm" / "results"
    cdir.mkdir(parents=True)
    with open(cdir / "tm_clustering.csv", "w") as fh:
        fh.write("cluster,structure\n" + "".join(f"{0 if r == 'af3' else s},{r}_seed{s}_sample{m}.pdb\n"
                                                  for r in ("af3", "hi") for s in (1, 2, 3) for m in (0, 1)))
    s = slate.propose(tmp_path / "pool.csv", "A3:3,A1:2", qa)
    stos = [m["sto"] for m in s["models"]]
    assert stos == ["A3", "A3", "A3", "A1", "A1"]
    assert all(m["model"].startswith(m["sto"] + "__") for m in s["models"])
    top3 = s["models"][:3]
    assert len({m["cluster"] for m in top3}) == 3          # one per cluster in slots 1-3
    assert all(m["cluster"] is None for m in s["models"][3:])  # A3 clusters never leak onto A1 models
    try:
        slate.propose(tmp_path / "pool.csv", "A3:3,A1:1", qa)
        assert False, "counts must add up to 5"
    except ValueError:
        pass


def test_retro_from_direct_native_scores(tmp_path):
    _make_pool(tmp_path / "T")
    rows = pool.scan(tmp_path / "T")
    pool.write_csv(rows, tmp_path / "pool.csv")
    s = slate.propose(tmp_path / "pool.csv", "A3")
    j, _ = slate.write(s, tmp_path, 1)
    (tmp_path / "retro_native").mkdir()
    for m in s["models"][:4]:                              # one model's OST run failed -> stays unscored
        (tmp_path / "retro_native" / f"{m['model']}.json").write_text(json.dumps({"status": "SUCCESS", "lddt": 0.7, "tm_score": 0.8}))
    (tmp_path / "retro_native" / f"{s['models'][4]['model']}.json").write_text(json.dumps({"status": "FAILURE"}))
    d = Dossier(target="T")
    d.evidence["slate_proposal"] = [{"value": {"json": str(j)}}]
    out = retro.run(d, tmp_path, None, "t")
    r = json.loads((tmp_path / "retro.json").read_text())
    assert [x["source"] for x in r["slots"]] == ["direct"] * 4 + ["unscored"]
    assert r["slots"][0]["lddt"] == 0.7 and "unscored" in out


def test_chosen_slate_and_versions(tmp_path):
    _make_pool(tmp_path / "T")
    rows = pool.scan(tmp_path / "T")
    pool.write_csv(rows, tmp_path / "pool.csv")
    v1 = slate.propose(tmp_path / "pool.csv", "A3")
    j1, _ = slate.write(v1, tmp_path, 1)
    names = ["A3__af3_msa__af3__s1_0", "A3__af3_msa__hi__s3_1", "A3__af3_msa__af3__s2_0", "A3__af3_msa__hi__s1_0", "A3__af3_msa__af3__s3_0"]
    v2 = slate.from_models(tmp_path / "pool.csv", names)
    assert v2["chosen"] and [m["model"] for m in v2["models"]] == names
    j2, md = slate.write(v2, tmp_path, 2)
    assert "chosen by review" in md.read_text()
    try:
        slate.from_models(tmp_path / "pool.csv", names[:4] + ["A3__af3_msa__nope__s1_0"])
        assert False
    except ValueError:
        pass
    scores = tmp_path / "scores.csv"
    with open(scores, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "lddt", "ranking"])
        for r in rows:
            w.writerow([f"T__A3__af3_msa__{r['runner']}__s{r['seed']}_{r['sample']}", 0.9 if r["runner"] == "af3" else 0.5, 0])
    d = Dossier(target="T")
    d.evidence["slate_proposal"] = [{"by": "agent", "value": {"version": 1, "json": str(j1)}},
                                    {"by": "agent", "value": {"version": 2, "json": str(j2)}}]
    retro.run(d, tmp_path, scores, "t")
    r = json.loads((tmp_path / "retro.json").read_text())
    assert r["slate"].startswith("v2") and r["slots"][0]["lddt"] == 0.9
    labels = [c["slate"] for c in r["compare"]]
    assert labels[0].startswith("v1") and labels[1].startswith("v2") and any(l.startswith("baseline") for l in labels)


def test_hedged_slate_caps_runners_across_parts(tmp_path):
    for sto in ("A2", "A1"):                              # af3 is best in both stoichiometries
        for runner, base in (("af3", 0.9), ("of3", 0.5), ("pt2", 0.4)):
            for seed in (1, 2, 3):
                d = tmp_path / "T" / "results" / sto / "v" / runner / f"seed_{seed}" / "sample_0"
                d.mkdir(parents=True)
                (d / "summary.json").write_text(json.dumps({"ranking_score": base + 0.01 * seed, "ptm": base, "plddt": 80}))
                (d / "model.cif").write_text("data_x\n")
    rows = pool.scan(tmp_path / "T")
    for r in rows:                                        # make af3 the top z in both parts
        if r["runner"] == "af3":
            r["rs_z_runner"] += 5
    pool.write_csv(rows, tmp_path / "pool.csv")
    s = slate.propose(tmp_path / "pool.csv", "A2:3,A1:2")
    runners = [m["runner"] for m in s["models"]]
    assert runners.count("af3") <= 2, runners


def test_pool_keeps_chains_apart(tmp_path):
    for chain in ("A", "B"):
        d = tmp_path / "T" / chain / "results" / "A1" / "v" / "af3" / "seed_1" / "sample_0"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"ranking_score": 0.8 if chain == "A" else 0.3, "ptm": 0.8, "plddt": 80}))
        (d / "model.cif").write_text("data_x\n")
    rows = pool.scan(tmp_path / "T")
    assert sorted(r["sto"] for r in rows) == ["A1@A", "A1@B"]
    pool.write_csv(rows, tmp_path / "pool.csv")
    s = slate.from_models(tmp_path / "pool.csv", ["A1@B__v__af3__s1_0"])
    assert s["models"][0]["path"].startswith(str(tmp_path / "T" / "B"))
