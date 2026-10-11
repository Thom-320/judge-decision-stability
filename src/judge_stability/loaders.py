"""Readers for the two source studies. Nothing here ships data; they read files the user already
has locally (see scripts/rebuild_from_sources.py).

Petri: one JSON object per line with transcript_id, judge, repeat, status and parse.scores
(dict dimension -> int), plus a meta file whose "rows" carry transcript_id, pos and target.
Missing scores stay NaN (never floored to 1, which is what Petri itself does silently).

TutorBench: rows.jsonl (task_id, criteria with weights) and runs/<arm>/state.sqlite with a
`verdicts` table (task_id, judge, verdicts JSON list of booleans, score, error). The item score
is Eq. 1 of the TutorBench paper as implemented in the pilot: met weight over positive weight,
clipped at 0.
"""
import json
import sqlite3

import numpy as np


# ------------------------------------------------------------------ Petri
def read_jsonl(paths):
    recs = []
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        with open(p, encoding="utf-8") as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


def petri_meta(path, n_max=None):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    rows = sorted(m["rows"] if isinstance(m, dict) else m, key=lambda r: r["pos"])
    return [r for r in rows if n_max is None or r["pos"] <= n_max]


def petri_tensor(recs, meta_rows, dim, judges, repeats=(1, 2)):
    """n x J x R array of scores for one dimension; NaN where missing."""
    ti = {r["transcript_id"]: i for i, r in enumerate(meta_rows)}
    Y = np.full((len(meta_rows), len(judges), len(repeats)), np.nan)
    for r in recs:
        if r.get("status") != "responded" or r.get("transcript_id") not in ti:
            continue
        if r.get("judge") not in judges or r.get("repeat") not in repeats:
            continue
        s = ((r.get("parse") or {}).get("scores") or {}).get(dim)
        if s is None:
            continue
        Y[ti[r["transcript_id"]], judges.index(r["judge"]), list(repeats).index(r["repeat"])] = s
    return Y


def complete_cases(Y):
    return ~np.any(np.isnan(Y), axis=tuple(range(1, Y.ndim)))


# ------------------------------------------------------------------ TutorBench
def tutorbench_rows(path):
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                idx[str(r["task_id"])] = r
    return idx


def sample_score(criteria, verdicts):
    """Met weight over positive weight (unclipped); 0 if no positive weight."""
    positive = sum(c["weight"] for c in criteria if c["weight"] > 0)
    if positive <= 0:
        return 0.0
    got = sum(c["weight"] for c, v in zip(criteria, verdicts) if v and c["weight"] != 0)
    return got / positive


def tutorbench_scores(sqlite_path, judge, rows):
    """{task_id: score in [0, 1]} for one arm and judge, rows from tutorbench_rows()."""
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        q = "select task_id, verdicts from verdicts where judge=? and (error is null or error='')"
        out = {}
        for t, v in con.execute(q, (judge,)):
            t = str(t)
            if t in rows:
                out[t] = max(0.0, sample_score(rows[t]["criteria"], json.loads(v)))
        return out
    finally:
        con.close()
