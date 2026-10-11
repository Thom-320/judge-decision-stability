"""phi: the probability that a decision changes when the judge, the rubric wording or the resample
changes.

Two estimators:
  * design-based (`flip_rate`, `ranking_flip_rate`, `pairwise_sign_flip_rate`,
    `system_threshold_flip`): count how often the decision differs between two observed
    conditions of a facet (two judges, two rubric wordings, two repeats), other facets fixed;
  * model-based (`phi_from`, `decisions`): plug the G-study variance into a normal model,
    phi = 2 p (1 - p), p = Phi(-|Delta| / sd), which is the probability that two independent
    panels land on different sides of the decision boundary.

Known limit of the model-based estimator: the plug-in |Delta| is biased upward, so phi is biased
downward near Delta = 0 (under a true null the plug-in phi averages about 1/3, not 0.5).
"""
import itertools

import numpy as np
from scipy.stats import norm


def phi_from(delta, sd):
    """phi = 2 p (1 - p), p = Phi(-|delta| / sd). sd = 0: 0 if delta != 0, else 0.5."""
    if sd <= 0:
        return 0.0 if delta != 0 else 0.5
    p = norm.cdf(-abs(delta) / sd)
    return float(2 * p * (1 - p))


def z_for_phi(eps):
    """|Delta|/sd needed for phi <= eps: p = (1 - sqrt(1 - 2 eps)) / 2, z = -Phi^{-1}(p)."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    p = (1 - np.sqrt(1 - 2 * eps)) / 2
    return float(-norm.ppf(p))


def flip_rate(decisions, axis=1):
    """Share of pairs of conditions along `axis` whose decisions differ, averaged over all other
    positions (items, other facets). decisions: array of booleans/labels; axis: the facet that
    changes (e.g. judges), every other axis is held fixed."""
    D = np.moveaxis(np.asarray(decisions), axis, -1)
    K = D.shape[-1]
    if K < 2:
        raise ValueError("need at least two conditions along the facet axis")
    pairs = list(itertools.combinations(range(K), 2))
    diff = np.stack([D[..., i] != D[..., j] for i, j in pairs], axis=-1)
    return float(diff.mean())


def _rank(row):
    # stable descending order; ties broken by system index so the result is deterministic
    return tuple(np.argsort(-np.asarray(row, float), kind="stable").tolist())


def ranking_flip_rate(means):
    """means: conditions x systems (e.g. per-judge mean score of each system). Share of pairs of
    conditions whose full ranking differs."""
    M = np.asarray(means, float)
    ranks = [_rank(r) for r in M]
    pairs = list(itertools.combinations(range(len(ranks)), 2))
    return float(np.mean([ranks[i] != ranks[j] for i, j in pairs]))


def pairwise_sign_flip_rate(means):
    """For each pair of systems (a, b), the share of pairs of conditions in which sign(a - b)
    differs. Returns {(a, b): phi}."""
    M = np.asarray(means, float)
    K, S = M.shape
    cpairs = list(itertools.combinations(range(K), 2))
    out = {}
    for a, b in itertools.combinations(range(S), 2):
        s = np.sign(M[:, a] - M[:, b])
        out[(a, b)] = float(np.mean([s[i] != s[j] for i, j in cpairs]))
    return out


def system_threshold_flip(Y, c, facet_axis=1):
    """System-level pass/fail: the system passes under a condition if its mean over items
    (axis 0) is >= c. Y: n x (facet axes...). Returns the share of pairs of conditions along
    `facet_axis` (an axis of Y, >= 1) whose pass/fail differs, other facets fixed."""
    Y = np.asarray(Y, float)
    if facet_axis < 1:
        raise ValueError("axis 0 is the item axis; the facet axis must be >= 1")
    M = Y.mean(axis=0)
    return flip_rate(M >= c, axis=facet_axis - 1)


# ------------------------------------------------------------------ model-based, all decisions
def decisions(Y, groups, group_order=None, thresholds=()):
    """Model-based phi for every pairwise contrast between groups (and optional thresholds),
    for each facet, at the observed design. Returns (results, gstudy)."""
    from .dstudy import FACETS, contrast_var, threshold_var
    from .gstudy import gstudy, pair_gj

    gs = gstudy(Y, groups, group_order)
    out = {"contrasts": {}, "thresholds": {}}
    gl = gs["groups"]
    for a, b in itertools.combinations(gl, 2):
        delta = gs["group_means"][a] - gs["group_means"][b]
        row = {"delta": float(delta)}
        for f in FACETS:
            sd = float(np.sqrt(contrast_var(gs, a, b, facet=f)))
            row[f"sd_{f}"] = sd
            row[f"phi_{f}"] = phi_from(delta, sd)
        sgj = pair_gj(gs, a, b)
        row["s_gj_pair"] = float(sgj)
        row["phi_full_pair_gj"] = phi_from(delta, float(np.sqrt(contrast_var(gs, a, b, s_gj=sgj))))
        out["contrasts"][f"{a}|{b}"] = row
    for c in thresholds:
        for a in gl:
            delta = gs["group_means"][a] - c
            row = {"delta": float(delta)}
            for f in FACETS:
                sd = float(np.sqrt(threshold_var(gs, a, facet=f)))
                row[f"sd_{f}"] = sd
                row[f"phi_{f}"] = phi_from(delta, sd)
            out["thresholds"][f"{a}>={c}"] = row
    return out, gs
