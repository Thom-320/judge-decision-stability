"""Paired bootstrap with a fixed seed and a canonical unit order.

Pitfall fixed here (from the TutorBench pilot, code/analyze.py): the pilot built a list of
paired differences by iterating a Python `set` of task ids. Set order of strings depends on
PYTHONHASHSEED, so the same seed drew different resamples on different runs and the reported
CI of one contrast moved between [-3.20, +3.66] and [-3.27, +3.66]. Every function in this
module therefore takes data keyed by unit id (a Mapping), and orders units with `sorted()`
before any random draw. Plain sets are rejected; sequences are accepted only where order is
given by the caller (exact_bootstrap_means, percentile_ci).

Random draws use numpy's Generator (PCG64) seeded by `seed`, so a result is a pure function of
(data, seed, B) and does not depend on dict insertion order or on the hash seed.
"""
import itertools
from collections.abc import Mapping, Set

import numpy as np


def _units(*maps):
    for m in maps:
        if isinstance(m, Set) or not isinstance(m, Mapping):
            raise TypeError("pass data as a Mapping {unit_id: value}; sets have no stable order")
    common = set(maps[0]).intersection(*[set(m) for m in maps[1:]])
    return sorted(common, key=lambda u: (str(type(u)), u))


def percentile_ci(draws, level=0.95):
    """Percentile interval (numpy 'linear' interpolation), ignoring nan."""
    a = (1 - level) / 2
    d = np.asarray(draws, float)
    return [float(np.nanpercentile(d, 100 * a)), float(np.nanpercentile(d, 100 * (1 - a)))]


def exact_bootstrap_means(values):
    """All n**n equally likely resamples of a tiny sample (n <= 7): sorted list of their means.
    Used to check the Monte Carlo bootstrap against a distribution computable by hand."""
    v = list(values)
    n = len(v)
    if n > 7:
        raise ValueError("exact enumeration is only for n <= 7")
    return sorted(float(np.mean([v[i] for i in idx])) for idx in itertools.product(range(n), repeat=n))


def paired_bootstrap(a, b, B=2000, seed=0, level=0.95):
    """Mean paired difference a[u] - b[u] over the units present in both, with a unit bootstrap.

    a, b: Mapping unit_id -> score (e.g. two judges, or two systems scored by the same judge).
    The same resampled units are used for both sides (paired)."""
    units = _units(a, b)
    if len(units) < 2:
        raise ValueError("need at least two shared units")
    d = np.array([a[u] - b[u] for u in units], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    draws = d[idx].mean(axis=1)
    return {"n": len(d), "estimate": float(d.mean()), "ci": percentile_ci(draws, level),
            "B": B, "seed": seed, "units": units}


def cluster_bootstrap_mean(values_by_unit, B=2000, seed=0, level=0.95):
    """Mean of all values, resampling whole clusters (units). values_by_unit: Mapping unit -> list."""
    units = _units(values_by_unit)
    vals = [np.asarray(values_by_unit[u], float) for u in units]
    rng = np.random.default_rng(seed)
    n = len(units)
    draws = np.empty(B)
    for k in range(B):
        pick = rng.integers(0, n, n)
        draws[k] = np.concatenate([vals[i] for i in pick]).mean()
    allv = np.concatenate(vals)
    return {"n_units": n, "n_values": int(allv.size), "estimate": float(allv.mean()),
            "ci": percentile_ci(draws, level), "B": B, "seed": seed}


def rank_stability(scores_a, scores_b, B=2000, seed=0):
    """Ranking of systems under two judges, with a paired unit bootstrap (same resample for both).

    scores_a, scores_b: Mapping system -> Mapping unit -> score. Units used are those scored for
    every system by both judges; systems are ordered by sorted name, ties are broken by that order.
    Returns point rankings, P(identical full ranking), P(same top system) and, per system pair,
    P(same sign of the difference under both judges). 1 - P(identical full ranking) is the
    ranking phi for a judge swap on resampled item sets."""
    systems = sorted(set(scores_a) & set(scores_b))
    units = _units(*[scores_a[s] for s in systems], *[scores_b[s] for s in systems])
    n = len(units)
    if n < 2 or len(systems) < 2:
        raise ValueError("need >= 2 shared units and >= 2 systems")
    A = np.array([[scores_a[s][u] for s in systems] for u in units], float)  # n x S
    Bm = np.array([[scores_b[s][u] for s in systems] for u in units], float)
    rank = lambda m: tuple(systems[i] for i in np.argsort(-m, kind="stable"))  # noqa: E731
    pairs = list(itertools.combinations(range(len(systems)), 2))
    rng = np.random.default_rng(seed)
    same_rank = same_top = 0
    same_sign = np.zeros(len(pairs))
    for _ in range(B):
        ix = rng.integers(0, n, n)
        ma, mb = A[ix].mean(axis=0), Bm[ix].mean(axis=0)
        ra, rb = rank(ma), rank(mb)
        same_rank += ra == rb
        same_top += ra[0] == rb[0]
        for k, (i, j) in enumerate(pairs):
            same_sign[k] += (ma[i] - ma[j] > 0) == (mb[i] - mb[j] > 0)
    return {"n_units": n, "systems": systems,
            "rank_a_point": rank(A.mean(axis=0)), "rank_b_point": rank(Bm.mean(axis=0)),
            "P_identical_full_ranking": same_rank / B, "P_same_top": same_top / B,
            "P_same_sign": {f"{systems[i]}|{systems[j]}": float(same_sign[k] / B) for k, (i, j) in enumerate(pairs)},
            "B": B, "seed": seed}
