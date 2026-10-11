"""Loaders on synthetic fixture files (no real data), and the rebuild script end to end."""
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from judge_stability import loaders

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sample_score_by_hand():
    """Criteria weights [5, 1, -5]; positive weight = 5 + 1 = 6.
    verdicts [T, T, F] -> (5 + 1)/6 = 1
    verdicts [T, F, T] -> (5 - 5)/6 = 0
    verdicts [F, T, T] -> (1 - 5)/6 = -2/3 (unclipped here; the arm loader clips at 0)."""
    crit = [{"weight": 5}, {"weight": 1}, {"weight": -5}]
    assert loaders.sample_score(crit, [True, True, False]) == 1.0
    assert loaders.sample_score(crit, [True, False, True]) == 0.0
    assert loaders.sample_score(crit, [False, True, True]) == pytest.approx(-2 / 3)
    assert loaders.sample_score([{"weight": -1}], [True]) == 0.0


def _tb_fixture(tmp):
    rows = [{"task_id": "t1", "criteria": [{"weight": 5}, {"weight": 1}, {"weight": -5}]},
            {"task_id": "t2", "criteria": [{"weight": 2}, {"weight": 2}]}]
    with open(tmp / "rows.jsonl", "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    db = tmp / "state.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table verdicts (task_id text, judge text, verdicts text, score real, error text)")
    con.executemany("insert into verdicts values (?,?,?,?,?)", [
        ("t1", "J", json.dumps([False, True, True]), None, None),   # -2/3 -> clipped to 0
        ("t2", "J", json.dumps([True, False]), None, ""),           # 2/4 = 0.5
        ("t2", "K", json.dumps([True, True]), None, "timeout"),     # error row ignored
        ("t9", "J", json.dumps([True]), None, None)])               # unknown task ignored
    con.commit()
    con.close()
    return db


def test_tutorbench_scores_fixture(tmp_path):
    db = _tb_fixture(tmp_path)
    rows = loaders.tutorbench_rows(tmp_path / "rows.jsonl")
    assert loaders.tutorbench_scores(db, "J", rows) == {"t1": 0.0, "t2": 0.5}
    assert loaders.tutorbench_scores(db, "K", rows) == {}


def _petri_fixture(tmp, n=8):
    """Synthetic Petri-shaped files: 2 targets x 4 transcripts, 3 judges, 2 repeats, score 1-10."""
    rng = np.random.default_rng(0)
    meta = {"rows": [{"pos": i + 1, "transcript_id": f"tr{i}", "target": "hermes3" if i < n // 2 else "mistral_large"}
                     for i in range(n)]}
    recs = []
    for i in range(n):
        base = 2 if i < n // 2 else 6
        for j in ("muse", "mimo", "deepseek"):
            for r in (1, 2):
                s = {d: int(np.clip(base + rng.integers(-1, 2), 1, 10)) for d in
                     ("concerning", "unprompted_deception_toward_user", "cooperation_with_human_misuse", "unfaithful_thinking")}
                recs.append({"transcript_id": f"tr{i}", "judge": j, "repeat": r, "status": "responded",
                             "parse": {"scores": s}})
    recs[0]["status"] = "error"  # one missing cell -> transcript tr0 drops from complete cases
    (tmp / "meta.json").write_text(json.dumps(meta))
    (tmp / "j.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    return recs, meta


def test_petri_tensor_keeps_missing_as_nan(tmp_path):
    recs, meta = _petri_fixture(tmp_path)
    rows = loaders.petri_meta(tmp_path / "meta.json")
    Y = loaders.petri_tensor(recs, rows, "concerning", ["muse", "mimo", "deepseek"])
    assert Y.shape == (8, 3, 2)
    assert np.isnan(Y[0, 0, 0]) and np.sum(np.isnan(Y)) == 1
    assert loaders.complete_cases(Y).sum() == 7


def test_rebuild_script_on_fixtures(tmp_path):
    _petri_fixture(tmp_path)
    tb = tmp_path / "tb"
    (tb / "runs").mkdir(parents=True)
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "rebuild_from_sources.py"),
                        "--petri-judgments", str(tmp_path / "j.jsonl"), "--petri-meta", str(tmp_path / "meta.json"),
                        "--out", str(out)], capture_output=True, text=True, check=True)
    assert "tutorbench: data not found, skipped" in p.stdout
    res = json.loads((out / "petri.json").read_text())
    d = res["dims"]["concerning"]
    assert d["n_complete"] == 7 and d["n_excluded"] == 1
    assert "hermes3|mistral_large" in d["phi_model"]["contrasts"]


def test_rebuild_script_without_data(tmp_path):
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "rebuild_from_sources.py"), "--out", str(tmp_path)],
                       capture_output=True, text=True, check=True)
    assert "wrote: nothing" in p.stdout


def test_rebuild_script_tutorbench_fixture(tmp_path):
    """Two synthetic arms, both judges. Arm p-raw-bunny: every task met in full under both judges
    (score 100); arm p-raw-kimi: half the weight met (50). Muse ranks bunny > kimi with a paired
    difference of exactly 50 (CI [50, 50]); longcat agrees, so P(identical ranking) = 1."""
    rows = [{"task_id": f"t{i}", "criteria": [{"weight": 1}, {"weight": 1}]} for i in range(4)]
    (tmp_path / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for arm, v in (("p-raw-bunny", [True, True]), ("p-raw-kimi", [True, False])):
        d = tmp_path / "runs" / arm
        d.mkdir(parents=True)
        con = sqlite3.connect(d / "state.sqlite")
        con.execute("create table verdicts (task_id text, judge text, verdicts text, score real, error text)")
        for j in ("muse-spark-1.3-contributor", "longcat-2.5-preview-free"):
            con.executemany("insert into verdicts values (?,?,?,?,?)",
                            [(f"t{i}", j, json.dumps(v), None, None) for i in range(4)])
        con.commit()
        con.close()
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "rebuild_from_sources.py"),
                    "--tutorbench-data", str(tmp_path), "--out", str(out)], check=True, capture_output=True)
    res = json.loads((out / "tutorbench.json").read_text())
    assert res["arms"] == ["p-raw-bunny", "p-raw-kimi"]
    pd = res["paired_arm_differences_muse"]["p-raw-bunny|p-raw-kimi"]
    assert pd["estimate"] == 50.0 and pd["ci"] == [50.0, 50.0]
    assert res["rank_stability_muse_vs_longcat"]["P_identical_full_ranking"] == 1.0
