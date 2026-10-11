"""Synthetic score tensors drawn from the G-study model (no real judge output).

    X = mu_g + t + c_j + (gc)_gj + (tc)_tj + e,   rounded/clipped optionally.
"""
import numpy as np


def simulate(group_means, n_per_group, J, R, s_t=1.0, s_j=0.0, s_gj=0.0, s_tj=0.0, s_e=0.0, seed=0):
    """Returns (Y, groups): Y is (G*n) x J x R, groups the label of each row (sorted group order)."""
    rng = np.random.default_rng(seed)
    gl = sorted(group_means)
    c = rng.normal(0, np.sqrt(s_j), J)
    Ys, labels = [], []
    for g in gl:
        gc = rng.normal(0, np.sqrt(s_gj), J)
        t = rng.normal(0, np.sqrt(s_t), n_per_group)
        tc = rng.normal(0, np.sqrt(s_tj), (n_per_group, J))
        e = rng.normal(0, np.sqrt(s_e), (n_per_group, J, R))
        Ys.append(group_means[g] + t[:, None, None] + (c + gc)[None, :, None] + tc[:, :, None] + e)
        labels += [g] * n_per_group
    return np.concatenate(Ys), np.array(labels)
