"""Paired bootstrap with a fixed seed, checked by hand, plus the pilot's ordering pitfall."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from judge_stability import (cluster_bootstrap_mean, exact_bootstrap_means, paired_bootstrap,
                             percentile_ci, rank_stability)

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def test_constant_differences_give_a_point_interval():
    """a - b = 2 on every unit -> every resample mean is 2 -> estimate 2, CI [2, 2]."""
    a = {"u1": 5.0, "u2": 7.0, "u3": 3.0}
    b = {"u1": 3.0, "u2": 5.0, "u3": 1.0}
    r = paired_bootstrap(a, b, B=500, seed=1)
    assert r["estimate"] == 2.0 and r["ci"] == [2.0, 2.0] and r["n"] == 3


def test_pairing_removes_between_unit_spread():
    """b varies a lot across units (0, 50, 100), a = b + 1. Paired differences are all 1, so the
    paired CI is exactly [1, 1]; an unpaired comparison of the two means would be about +-50 wide."""
    b = {"x": 0.0, "y": 50.0, "z": 100.0}
    a = {k: v + 1 for k, v in b.items()}
    assert paired_bootstrap(a, b, B=500, seed=3)["ci"] == [1.0, 1.0]


def test_only_shared_units_are_used():
    """Unit 'c' has no partner: n = 2 and the estimate is ((1-0) + (3-0))/2 = 2."""
    r = paired_bootstrap({"a": 1.0, "b": 3.0, "c": 99.0}, {"a": 0.0, "b": 0.0}, B=200, seed=0)
    assert r["n"] == 2 and r["estimate"] == 2.0 and r["units"] == ["a", "b"]


def test_exact_bootstrap_distribution_by_hand():
    """Sample [0, 1]: the 2^2 = 4 resamples (0,0) (0,1) (1,0) (1,1) have means 0, .5, .5, 1.
    Sample [0, 0, 3]: 27 resamples; the mean is k (the number of times 3 is drawn), k ~ Bin(3, 1/3):
    P(0) = 8/27, P(1) = 12/27, P(2) = 6/27, P(3) = 1/27."""
    assert exact_bootstrap_means([0, 1]) == [0.0, 0.5, 0.5, 1.0]
    m = exact_bootstrap_means([0, 0, 3])
    assert [m.count(k) for k in (0.0, 1.0, 2.0, 3.0)] == [8, 12, 6, 1]


def test_monte_carlo_matches_exact_distribution():
    """Diffs [0, 1] (a - b per unit): P(resample mean = 0.5) = 1/2 exactly. With B = 20000 the
    Monte Carlo share has SD sqrt(.25/20000) = 0.0035; tolerance 0.02 is about 5.7 SD."""
    rng = np.random.default_rng(11)
    d = np.array([0.0, 1.0])
    means = d[rng.integers(0, 2, size=(20000, 2))].mean(axis=1)
    assert abs(np.mean(means == 0.5) - 0.5) < 0.02
    # the library uses the same draw scheme: same seed, same draws
    r = paired_bootstrap({"u0": 0.0, "u1": 1.0}, {"u0": 0.0, "u1": 0.0}, B=20000, seed=11)
    assert r["ci"] == percentile_ci(means)


def test_percentile_ci_by_hand():
    """Values 0..100 (101 values): 2.5th and 97.5th percentiles with linear interpolation are 2.5 and 97.5."""
    assert percentile_ci(np.arange(101)) == pytest.approx([2.5, 97.5])


def test_same_seed_same_result_any_insertion_order():
    """Fixed seed: the result is a function of (data, seed, B) only. Reversing the dict insertion
    order changes nothing because units are sorted before drawing."""
    rng = np.random.default_rng(0)
    ids = [f"task-{i:03d}" for i in range(40)]
    a = {u: float(rng.normal()) for u in ids}
    b = {u: float(rng.normal()) for u in ids}
    r1 = paired_bootstrap(a, b, B=1000, seed=20261010)
    r2 = paired_bootstrap(dict(reversed(list(a.items()))), dict(reversed(list(b.items()))), B=1000, seed=20261010)
    assert r1["ci"] == r2["ci"]
    assert paired_bootstrap(a, b, B=1000, seed=20261011)["ci"] != r1["ci"]


def test_sets_are_rejected():
    with pytest.raises(TypeError):
        paired_bootstrap({"a", "b"}, {"a": 1.0, "b": 2.0})


_CHILD = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, {src!r})
    import numpy as np
    from judge_stability import paired_bootstrap
    ids = ["t%d-%s" % (i, "xyzw"[i % 4]) for i in range(30)]
    shared = set(ids)                      # the pilot iterated a set like this one
    rng = np.random.default_rng(5)
    vals = dict(zip(ids, rng.normal(size=30).tolist()))
    # the pilot's pattern: build the list of differences by iterating the set
    pilot_order = [t for t in shared]
    d = np.array([vals[t] for t in pilot_order])
    pilot_draws = d[np.random.default_rng(17).integers(0, 30, (500, 30))].mean(axis=1)
    pilot_ci = [float(np.percentile(pilot_draws, 2.5)), float(np.percentile(pilot_draws, 97.5))]
    ours = paired_bootstrap({{t: vals[t] for t in shared}}, {{t: 0.0 for t in shared}}, B=500, seed=17)["ci"]
    print(json.dumps({{"order": pilot_order, "pilot_ci": pilot_ci, "ours": ours}}))
""")


def test_pilot_set_order_bug_cannot_happen_here():
    """The TutorBench pilot's analyze.py built the paired-difference list by iterating a set of
    task ids (section 4, `for t in tasks_m`). String hashing is randomized per process
    (PYTHONHASHSEED), so the same bootstrap seed resampled different values and the CI moved.
    Here the child process reproduces that pattern under several hash seeds: the pilot-style CI
    changes with the hash seed, ours does not."""
    script = _CHILD.format(src=SRC)
    import json
    outs = []
    for hs in ("0", "1", "2", "3", "4", "5"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        p = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outs.append(json.loads(p.stdout))
    assert len({tuple(o["order"]) for o in outs}) > 1, "set order should vary with PYTHONHASHSEED"
    assert len({tuple(o["pilot_ci"]) for o in outs}) > 1, "the pilot pattern is hash-dependent"
    assert len({tuple(o["ours"]) for o in outs}) == 1, "ours must not depend on the hash seed"


def test_cluster_bootstrap_by_hand():
    """Clusters u1 = [1, 3] and u2 = [2, 2]: both have 2 values with mean 2, so every resample of
    clusters has mean 2 -> estimate 2, CI [2, 2]; 4 values in 2 clusters."""
    r = cluster_bootstrap_mean({"u1": [1, 3], "u2": [2, 2]}, B=300, seed=2)
    assert (r["estimate"], r["ci"], r["n_units"], r["n_values"]) == (2.0, [2.0, 2.0], 2, 4)


def _scores(order):
    """Every unit gives the systems fixed scores, so every resample has the same ranking."""
    return {s: {f"u{i}": float(v) for i in range(5)} for s, v in order.items()}


def test_rank_stability_identical_judges():
    """Both judges: A = 3, B = 2, C = 1 on every unit -> ranking A > B > C in every resample:
    P(identical) = P(same top) = 1 and every pair keeps its sign."""
    s = _scores({"A": 3, "B": 2, "C": 1})
    r = rank_stability(s, s, B=200, seed=0)
    assert r["P_identical_full_ranking"] == 1.0 and r["P_same_top"] == 1.0
    assert set(r["P_same_sign"].values()) == {1.0}
    assert r["rank_a_point"] == ("A", "B", "C")


def test_rank_stability_swapped_top_pair():
    """Judge 2 swaps A and B on every unit (A = 2, B = 3, C = 1): rankings A>B>C vs B>A>C always
    differ -> P(identical) = 0, P(same top) = 0; pair A|B never keeps its sign (0), A|C and B|C
    always do (1). Ranking phi for the judge swap = 1 - 0 = 1."""
    r = rank_stability(_scores({"A": 3, "B": 2, "C": 1}), _scores({"A": 2, "B": 3, "C": 1}), B=200, seed=0)
    assert r["P_identical_full_ranking"] == 0.0 and r["P_same_top"] == 0.0
    assert r["P_same_sign"] == {"A|B": 0.0, "A|C": 1.0, "B|C": 1.0}
