"""Decision-level D-study: error variance of a decision for a panel of J judges, R repeats and
n items per group, and the cheapest panel whose phi stays below a target.

Contrast "A > B" (judge main effect cancels):
    Var = s_t(A)/n_A + s_t(B)/n_B + 2 s_gj/J + s_tj(A)/(n_A J) + s_tj(B)/(n_B J)
          + s_e(A)/(n_A J R) + s_e(B)/(n_B J R)
Threshold "mean(A) >= c" (judge main effect does not cancel):
    Var = s_t(A)/n_A + (s_j + s_gj)/J + s_tj(A)/(n_A J) + s_e(A)/(n_A J R)

Facets say what is re-drawn between the two panels whose decisions are compared:
  "full"        items, judges and repeats;
  "judges"      judges and repeats, same items;
  "repeats"     repeats only (same items and judges);
  "items"       items (and repeats), same judges.
phi = 2 p (1 - p), p = Phi(-|Delta| / sd), with the plug-in Delta.
"""
import numpy as np

from .phi import phi_from

FACETS = ("full", "judges", "repeats", "items")


def _e(P):
    s = P.get("s_e", 0.0)
    return s if s is not None and np.isfinite(s) else 0.0


def threshold_var_components(s_t, s_tj, s_e, s_j, s_gj, n, J, R, facet="full"):
    t = s_t / n
    jj = (s_j + s_gj) / J
    tj = s_tj / (n * J)
    e = s_e / (n * J * R)
    return {"full": t + jj + tj + e, "judges": jj + tj + e, "repeats": e, "items": t + tj + e}[facet]


def contrast_var_components(A, B, s_gj, n_a, n_b, J, R, facet="full"):
    """A, B: dicts with s_t, s_tj, s_e for each group."""
    t = A["s_t"] / n_a + B["s_t"] / n_b
    gj = 2 * s_gj / J
    tj = A["s_tj"] / (n_a * J) + B["s_tj"] / (n_b * J)
    e = _e(A) / (n_a * J * R) + _e(B) / (n_b * J * R)
    return {"full": t + gj + tj + e, "judges": gj + tj + e, "repeats": e, "items": t + tj + e}[facet]


def contrast_var(gs, a, b, J=None, R=None, n_a=None, n_b=None, facet="full", s_gj=None):
    A, B = gs["per_group"][a], gs["per_group"][b]
    return contrast_var_components(A, B, gs["s_gj"] if s_gj is None else s_gj,
                                   n_a or A["n"], n_b or B["n"], J or gs["J"], R or gs["R"], facet)


def threshold_var(gs, a, J=None, R=None, n_a=None, facet="full"):
    A = gs["per_group"][a]
    return threshold_var_components(A["s_t"], A["s_tj"], _e(A), gs["s_j"], gs["s_gj"],
                                    n_a or A["n"], J or gs["J"], R or gs["R"], facet)


def _cheapest(grid, eps):
    best = {}
    for e in eps:
        ok = [g for g in grid if g["phi"] <= e]
        best[str(e)] = min(ok, key=lambda g: (g["calls"], g["J"], g["R"])) if ok else None
    return best


D_J = (1, 2, 3, 4, 6, 8)
D_R = (1, 2, 3, 4)
EPS = (0.05, 0.10)


def dstudy_threshold(delta, s_t, s_tj, s_e, s_j, s_gj, n, Js=D_J, Rs=D_R, eps=EPS, facet="full"):
    """How many judges and repeats a threshold decision needs, at a fixed number n of items.

    Returns the grid of phi over (J, R), the cheapest (fewest calls J*R*n, then fewest judges)
    design reaching each eps, and the floor phi with infinitely many repeats for each J."""
    grid = []
    for J in Js:
        for R in Rs:
            sd = np.sqrt(threshold_var_components(s_t, s_tj, s_e, s_j, s_gj, n, J, R, facet))
            grid.append({"J": J, "R": R, "calls": J * R * n, "phi": phi_from(delta, sd)})
    floor = {str(J): phi_from(delta, np.sqrt(threshold_var_components(s_t, s_tj, 0.0, s_j, s_gj, n, J, 1, facet)))
             for J in Js}
    return {"delta": float(delta), "grid": grid, "cheapest": _cheapest(grid, eps),
            "phi_limit_R_infinite_by_J": floor}


def dstudy_contrast(gs, a, b, Js=D_J, Rs=D_R, Ns=(10, 20, 40, 100, 200, 400, 1000), eps=EPS):
    """phi over (J, R, n per group) for the contrast a vs b; cheapest design for each eps, and the
    judge floor (infinitely many items) for each J."""
    delta = gs["group_means"][a] - gs["group_means"][b]
    grid = []
    for J in Js:
        for R in Rs:
            for n in Ns:
                sd = np.sqrt(contrast_var(gs, a, b, J=J, R=R, n_a=n, n_b=n))
                grid.append({"J": J, "R": R, "n": n, "calls": J * R * 2 * n, "phi": phi_from(delta, sd)})
    floor = {str(J): phi_from(delta, np.sqrt(2 * gs["s_gj"] / J)) for J in Js}
    return {"delta": float(delta), "grid": grid, "cheapest": _cheapest(grid, eps),
            "phi_limit_n_infinite_by_J": floor}
