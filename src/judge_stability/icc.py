"""Agreement statistics for comparison with phi (ported from petri-judge-stability pilot/scripts/analyze.py).

ICC(2,1) and Krippendorff's alpha measure how well the scores track each other across the whole
scale. They say nothing about whether a decision near a threshold or between two close systems
survives a change of judge: a high ICC is compatible with a large phi (see README and
tests/test_icc.py::test_high_icc_with_flipping_decisions).
"""
import numpy as np


def icc21(X):
    """Shrout-Fleiss ICC(2,1): two-way random effects, absolute agreement, single rater.
    X: n items x k raters. Returns nan when undefined (n < 3, constant data)."""
    X = np.asarray(X, float)
    n, k = X.shape
    if n < 3 or np.all(X == X.flat[0]):
        return np.nan
    g = X.mean()
    rm, cm = X.mean(axis=1), X.mean(axis=0)
    SSR = k * np.sum((rm - g) ** 2)
    SSC = n * np.sum((cm - g) ** 2)
    SST = np.sum((X - g) ** 2)
    SSE = SST - SSR - SSC
    MSR, MSC, MSE = SSR / (n - 1), SSC / (k - 1), SSE / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return float((MSR - MSE) / den) if den > 0 else np.nan


def kalpha_interval(X):
    """Krippendorff's alpha, interval metric, complete data. X: n units x k raters."""
    X = np.asarray(X, float)
    n, k = X.shape
    if np.all(X == X.flat[0]):
        return np.nan
    N = n * k
    per_unit = 2 * k * np.sum(X ** 2, axis=1) - 2 * np.sum(X, axis=1) ** 2
    Do = np.sum(per_unit / (k - 1)) / N
    allv = X.ravel()
    De = (2 * N * np.sum(allv ** 2) - 2 * np.sum(allv) ** 2) / (N * (N - 1))
    return float(1 - Do / De) if De > 0 else np.nan


def varcomp(Y):
    """Method-of-moments variance components for a balanced, fully crossed n x J x R design
    (item x judge x repeat, one observation per cell). Returns (raw, shares)."""
    Y = np.asarray(Y, float)
    n, J, R = Y.shape
    if min(n, J, R) < 2:
        raise ValueError("varcomp needs at least 2 items, 2 judges and 2 repeats")
    m = Y.mean()
    yt, yj, yr = Y.mean(axis=(1, 2)), Y.mean(axis=(0, 2)), Y.mean(axis=(0, 1))
    ytj, ytr, yjr = Y.mean(axis=2), Y.mean(axis=1), Y.mean(axis=0)
    SS_A = J * R * np.sum((yt - m) ** 2)
    SS_B = n * R * np.sum((yj - m) ** 2)
    SS_C = n * J * np.sum((yr - m) ** 2)
    SS_AB = R * np.sum((ytj - yt[:, None] - yj[None, :] + m) ** 2)
    SS_AC = J * np.sum((ytr - yt[:, None] - yr[None, :] + m) ** 2)
    SS_BC = n * np.sum((yjr - yj[:, None] - yr[None, :] + m) ** 2)
    SS_T = np.sum((Y - m) ** 2)
    SS_ABC = SS_T - SS_A - SS_B - SS_C - SS_AB - SS_AC - SS_BC
    df = {"A": n - 1, "B": J - 1, "C": R - 1, "AB": (n - 1) * (J - 1), "AC": (n - 1) * (R - 1),
          "BC": (J - 1) * (R - 1), "ABC": (n - 1) * (J - 1) * (R - 1)}
    SS = {"A": SS_A, "B": SS_B, "C": SS_C, "AB": SS_AB, "AC": SS_AC, "BC": SS_BC, "ABC": SS_ABC}
    MS = {k: SS[k] / df[k] for k in df}
    s_abc = MS["ABC"]
    raw = {"item": (MS["A"] - MS["AB"] - MS["AC"] + s_abc) / (J * R),
           "judge": (MS["B"] - MS["AB"] - MS["BC"] + s_abc) / (n * R),
           "repeat": (MS["C"] - MS["AC"] - MS["BC"] + s_abc) / (n * J),
           "item_x_judge": (MS["AB"] - s_abc) / R, "item_x_repeat": (MS["AC"] - s_abc) / J,
           "judge_x_repeat": (MS["BC"] - s_abc) / n, "residual_3way": s_abc}
    raw = {k: float(v) for k, v in raw.items()}
    pos = {k: max(v, 0.0) for k, v in raw.items()}
    tot = sum(pos.values())
    shares = {k: (v / tot if tot > 0 else np.nan) for k, v in pos.items()}
    return raw, shares
