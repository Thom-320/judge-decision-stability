"""Nested G-study for judge scores (ported from petri-judge-stability, branch v2-build, analysis/phi.py).

Score model: transcripts/items t nested in groups g (systems, targets or arms), crossed with
judges j; repeats r nested in item x judge cells:

    X_gtjr = mu + a_g + t_{t:g} + c_j + (ac)_gj + (tc)_{tj:g} + e_{gtjr}

Variance components by ANOVA / expected mean squares (method of moments). Item-level components
are estimated separately per group (no homogeneity assumption). Within group h (balanced
n_h x J x R):

    s_e(h)  = MS_e
    s_tj(h) = (MS_tj - MS_e) / R          (if R == 1: MS_tj, confounded with s_e)
    s_t(h)  = (MS_t - MS_tj) / (J R)

Across groups, from the G x J table of group-by-judge means M_gj, whose cells carry noise
v_g = s_tj(g)/n_g + s_e(g)/(n_g R):

    E[SS_int] = (G-1)(J-1) s_gj + (J-1)(1-1/G) sum_g v_g
    E[SS_col] = G(J-1) s_j + (J-1) s_gj + (J-1)/G sum_g v_g

Negative estimates are truncated at 0; the raw values are kept with the suffix `_raw`.
"""
import numpy as np


def within_group(Yh):
    """Components for one group. Yh: n x J x R array (complete cases)."""
    Yh = np.asarray(Yh, float)
    n, J, R = Yh.shape
    m = Yh.mean()
    yt, yj, ytj = Yh.mean(axis=(1, 2)), Yh.mean(axis=(0, 2)), Yh.mean(axis=2)
    ss_t = J * R * np.sum((yt - m) ** 2)
    ss_tj = R * np.sum((ytj - yt[:, None] - yj[None, :] + m) ** 2)
    ss_e = np.sum((Yh - ytj[:, :, None]) ** 2)
    ms_t = ss_t / (n - 1)
    ms_tj = ss_tj / ((n - 1) * (J - 1)) if J > 1 else 0.0
    ms_e = ss_e / (n * J * (R - 1)) if R > 1 else np.nan
    s_e = ms_e
    s_tj = (ms_tj - ms_e) / R if R > 1 else ms_tj
    s_t = (ms_t - ms_tj) / (J * R)
    return {"n": n, "s_t_raw": float(s_t), "s_tj_raw": float(s_tj), "s_e": float(s_e),
            "s_t": max(float(s_t), 0.0), "s_tj": max(float(s_tj), 0.0), "mean": float(m),
            "judge_means": yj}


def _cell_noise(P, R):
    s_e = P["s_e"] if (R > 1 and np.isfinite(P["s_e"])) else 0.0
    return P["s_tj"] / P["n"] + s_e / (P["n"] * R)


def gstudy(Y, groups, group_order=None):
    """Y: n x J x R scores (complete cases); groups: length-n labels.

    Groups are processed in `group_order` or, by default, in sorted order, so the output never
    depends on hash order. Groups with fewer than 2 items are dropped."""
    Y = np.asarray(Y, float)
    groups = np.asarray(groups)
    gl = list(group_order) if group_order is not None else sorted(set(groups.tolist()))
    gl = [g for g in gl if np.sum(groups == g) >= 2]
    n, J, R = Y.shape
    per = {g: within_group(Y[groups == g]) for g in gl}
    G = len(gl)
    out = {"groups": gl, "J": J, "R": R, "per_group": per,
           "group_means": {g: per[g]["mean"] for g in gl}}
    if G >= 2 and J >= 2:
        M = np.array([per[g]["judge_means"] for g in gl])
        v = np.array([_cell_noise(per[g], R) for g in gl])
        rm, cm, gm = M.mean(axis=1), M.mean(axis=0), M.mean()
        ss_int = np.sum((M - rm[:, None] - cm[None, :] + gm) ** 2)
        s_gj = (ss_int - (J - 1) * (1 - 1 / G) * v.sum()) / ((G - 1) * (J - 1))
        ss_col = G * np.sum((cm - gm) ** 2)
        s_j = (ss_col / (J - 1) - max(s_gj, 0.0) - v.sum() / G) / G
    else:
        s_gj, s_j = np.nan, np.nan
    out.update({"s_gj_raw": float(s_gj), "s_j_raw": float(s_j),
                "s_gj": max(float(s_gj), 0.0) if np.isfinite(s_gj) else np.nan,
                "s_j": max(float(s_j), 0.0) if np.isfinite(s_j) else np.nan,
                "df_gj": (G - 1) * (J - 1), "df_j": J - 1})
    return out


def pair_gj(gs, a, b):
    """Pair-specific group x judge variance from the J per-judge differences d_j = M_aj - M_bj:
    E[var_j(d_j)] = 2 s_gj + v_a + v_b (J - 1 df; a sensitivity check on the pooled s_gj)."""
    A, B = gs["per_group"][a], gs["per_group"][b]
    J, R = gs["J"], gs["R"]
    if J < 2:
        return np.nan
    d = np.asarray(A["judge_means"]) - np.asarray(B["judge_means"])
    return max((np.var(d, ddof=1) - _cell_noise(A, R) - _cell_noise(B, R)) / 2, 0.0)
