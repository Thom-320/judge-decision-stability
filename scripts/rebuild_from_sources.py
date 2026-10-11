#!/usr/bin/env python3
"""Rebuild phi / ICC / D-study results from the source studies when their data are on this machine.

Nothing read here is committed: outputs go to results/local/ (gitignored). Each source is skipped,
with a message, when its files are absent.

Petri (petri-judge-stability pilot, cheap judges; transcripts are never read, only judge scores):
  --petri-judgments  pilot/judgments_final.jsonl   (judge outputs; not redistributed here)
  --petri-meta       pilot/sample_meta.json
TutorBench (Scale AI items, licence unclear; raw judge outputs not shipped):
  --tutorbench-data  directory with rows.jsonl and runs/<arm>/state.sqlite

Example:
  python scripts/rebuild_from_sources.py \
      --petri-judgments ../petri-judge-stability/pilot/judgments_final.jsonl \
      --petri-meta ../petri-judge-stability/pilot/sample_meta.json \
      --tutorbench-data /path/to/tutorbench/data
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from judge_stability import (decisions, dstudy_contrast, flip_rate, icc21, paired_bootstrap,  # noqa: E402
                             pairwise_sign_flip_rate, rank_stability, ranking_flip_rate)
from judge_stability import loaders  # noqa: E402

PETRI_DIMS = ["concerning", "unprompted_deception_toward_user", "cooperation_with_human_misuse",
              "unfaithful_thinking"]
PETRI_JUDGES = ["muse", "mimo", "deepseek"]
PETRI_THRESHOLDS = (1.5, 2.0, 3.0)
TB_JUDGES = {"muse": "muse-spark-1.3-contributor", "longcat": "longcat-2.5-preview-free"}
TB_ARMS = ["p-raw-bunny", "p-raw-muse", "p-raw-kimi"]  # the manta arm is dropped (see README)
SEED = 20261010


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def petri(judgments, meta, n_max):
    recs = loaders.read_jsonl(judgments)
    rows = loaders.petri_meta(meta, n_max)
    targets = np.array([r["target"] for r in rows])
    out = {"n_transcripts": len(rows), "judges": PETRI_JUDGES, "dims": {}}
    for dim in PETRI_DIMS:
        Yall = loaders.petri_tensor(recs, rows, dim, PETRI_JUDGES)
        ok = loaders.complete_cases(Yall)
        Y, tg = Yall[ok], targets[ok]
        d = {"n_complete": int(ok.sum()), "n_excluded": int((~ok).sum())}
        if Y.shape[0] < 6:
            out["dims"][dim] = {**d, "note": "too few complete transcripts"}
            continue
        d["icc21_interjudge_rep1"] = icc21(Y[:, :, 0])
        # item-level pass/fail flips (score >= c) across judges and across repeats
        d["item_flip_by_judge"] = {str(c): flip_rate(Y >= c, axis=1) for c in PETRI_THRESHOLDS}
        d["item_flip_by_repeat"] = {str(c): flip_rate(Y >= c, axis=2) for c in PETRI_THRESHOLDS}
        gl = sorted(set(tg.tolist()))
        means = np.array([[Y[tg == g, j, :].mean() for g in gl] for j in range(Y.shape[1])])
        d["target_ranking_flip_across_judges"] = ranking_flip_rate(means)
        d["target_pair_sign_flip_across_judges"] = {f"{gl[a]}|{gl[b]}": v for (a, b), v in
                                                    pairwise_sign_flip_rate(means).items()}
        res, gs = decisions(Y, tg, gl, PETRI_THRESHOLDS)
        d["phi_model"] = res
        d["dstudy"] = {k: {kk: vv for kk, vv in dstudy_contrast(gs, *k.split("|")).items() if kk != "grid"}
                       for k in res["contrasts"]}
        out["dims"][dim] = d
    return out


def tutorbench(data_dir):
    rows = loaders.tutorbench_rows(os.path.join(data_dir, "rows.jsonl"))
    sc = {}
    for arm in TB_ARMS:
        p = os.path.join(data_dir, "runs", arm, "state.sqlite")
        if os.path.exists(p):
            for short, judge in TB_JUDGES.items():
                sc[(arm, short)] = {t: 100 * v for t, v in loaders.tutorbench_scores(p, judge, rows).items()}
    arms = [a for a in TB_ARMS if all((a, j) in sc for j in TB_JUDGES)]
    out = {"arms": arms, "judges": list(TB_JUDGES)}
    if len(arms) < 2:
        return {**out, "note": "fewer than two arms scored by both judges"}
    sA = {a: sc[(a, "muse")] for a in arms}
    sB = {a: sc[(a, "longcat")] for a in arms}
    out["rank_stability_muse_vs_longcat"] = rank_stability(sA, sB, B=2000, seed=SEED)
    out["rank_stability_muse_vs_longcat"].pop("units", None)
    out["paired_arm_differences_muse"] = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            r = paired_bootstrap(sA[a], sA[b], B=2000, seed=SEED)
            r.pop("units")
            out["paired_arm_differences_muse"][f"{a}|{b}"] = r
    out["judge_difference_by_arm"] = {}
    for a in arms:
        r = paired_bootstrap(sA[a], sB[a], B=2000, seed=SEED)
        r.pop("units")
        out["judge_difference_by_arm"][a] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--petri-judgments")
    ap.add_argument("--petri-meta")
    ap.add_argument("--petri-n", type=int, default=120)
    ap.add_argument("--tutorbench-data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "local"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    done = []
    if a.petri_judgments and a.petri_meta and os.path.exists(a.petri_judgments) and os.path.exists(a.petri_meta):
        res = petri(a.petri_judgments, a.petri_meta, a.petri_n)
        with open(os.path.join(a.out, "petri.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=_json)
        done.append("petri")
    else:
        print("petri: data not found, skipped")
    if a.tutorbench_data and os.path.exists(os.path.join(a.tutorbench_data, "rows.jsonl")):
        res = tutorbench(a.tutorbench_data)
        with open(os.path.join(a.out, "tutorbench.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=_json)
        done.append("tutorbench")
    else:
        print("tutorbench: data not found, skipped")
    print("wrote:", ", ".join(done) if done else "nothing", "->", os.path.abspath(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
