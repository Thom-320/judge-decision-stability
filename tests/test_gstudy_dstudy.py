"""G-study components and decision-level D-study, checked by hand."""
import numpy as np
import pytest

from judge_stability import (contrast_var, decisions, dstudy_contrast, dstudy_threshold, gstudy,
                             phi_from, within_group)
from judge_stability.synthetic import simulate


def test_within_group_truncates_negative_component():
    """Y[item, judge, repeat]: item 1 = [[1,3],[1,3]], item 2 = [[3,5],[3,5]] (2 x 2 x 2).
    cell means: item 1 [2,2], item 2 [4,4]; item means 2, 4; judge means 3, 3; m = 3.
    SS_t = J R ((2-3)^2 + (4-3)^2) = 4*2 = 8 -> MS_t = 8/1 = 8.
    SS_tj = 0 (no item x judge pattern) -> MS_tj = 0.
    SS_e = 8 cells, each 1 away from its cell mean -> 8; df = n J (R-1) = 4 -> MS_e = 2.
    s_e = 2; s_tj = (0 - 2)/R = -1 -> truncated to 0; s_t = (8 - 0)/(J R) = 2."""
    Y = np.array([[[1, 3], [1, 3]], [[3, 5], [3, 5]]], float)
    w = within_group(Y)
    assert w["s_e"] == pytest.approx(2.0)
    assert w["s_tj_raw"] == pytest.approx(-1.0) and w["s_tj"] == 0.0
    assert w["s_t"] == pytest.approx(2.0)


def _two_groups():
    """Group A: items [1,3] and [3,5] (judge 2 always +2); group B: items [1,1] and [3,3]. R = 1."""
    Y = np.array([[1, 3], [3, 5], [1, 1], [3, 3]], float)[:, :, None]
    return Y, np.array(["A", "A", "B", "B"])


def test_gstudy_two_groups_by_hand():
    """Group A: item means 2, 4 -> s_t = J R * 2 / (J R) = 2 (SS_t = 2*(1+1) = 4, MS_t = 4,
    MS_tj = 0); judge means 2, 4; group mean 3. Group B: s_t = 2, judge means 2, 2, mean 2.
    s_tj = 0 in both, R = 1 so the cell noise v_g = 0.
    M = [[2,4],[2,2]], row means 3, 2, column means 2, 3, grand 2.5.
    Interaction residuals: -0.5, 0.5, 0.5, -0.5 -> SS_int = 1 -> s_gj = 1/((G-1)(J-1)) = 1.
    SS_col = G((2-2.5)^2 + (3-2.5)^2) = 1 -> s_j = (1/1 - 1 - 0)/2 = 0."""
    Y, g = _two_groups()
    gs = gstudy(Y, g)
    assert gs["per_group"]["A"]["s_t"] == pytest.approx(2.0)
    assert gs["per_group"]["B"]["s_t"] == pytest.approx(2.0)
    assert gs["s_gj"] == pytest.approx(1.0)
    assert gs["s_j"] == pytest.approx(0.0, abs=1e-12)
    assert gs["group_means"] == {"A": 3.0, "B": 2.0}


def test_contrast_phi_by_facet_by_hand():
    """Same data, contrast A > B, delta = 3 - 2 = 1, n = 2 per group, J = 2, R = 1.
    full:    s_t/n + s_t/n + 2 s_gj/J = 1 + 1 + 1 = 3 -> sd 1.7321, z = 0.57735,
             Phi(-0.57735) = 0.281851 -> phi = 2(0.281851)(0.718149) = 0.404820
    judges:  2 s_gj / J = 1 -> z = 1 -> phi = 0.266968
    items:   1 + 1 = 2 -> sd 1.41421, z = 0.70711, Phi(-0.70711) = 0.239750 -> phi = 0.364540
    repeats: 0 -> phi = 0 (delta != 0)."""
    Y, g = _two_groups()
    res, _ = decisions(Y, g)
    r = res["contrasts"]["A|B"]
    assert r["delta"] == pytest.approx(1.0)
    assert r["phi_full"] == pytest.approx(0.404820, abs=1e-5)
    assert r["phi_judges"] == pytest.approx(0.266968, abs=1e-5)
    assert r["phi_items"] == pytest.approx(0.364540, abs=1e-5)
    assert r["phi_repeats"] == 0.0


def test_gstudy_does_not_depend_on_row_order():
    """Shuffling rows (and their labels) leaves every component unchanged: groups are processed in
    sorted order, never in hash or first-seen order."""
    Y, g = _two_groups()
    p = np.array([3, 0, 2, 1])
    a, b = gstudy(Y, g), gstudy(Y[p], g[p])
    assert a["groups"] == b["groups"] == ["A", "B"]
    assert a["s_gj"] == pytest.approx(b["s_gj"])
    assert contrast_var(a, "A", "B") == pytest.approx(contrast_var(b, "A", "B"))


def test_dstudy_threshold_judges_by_hand():
    """Threshold decision, only judge variance: s_j = 1, all else 0, delta = mean - c = 1, n = 10.
    Var = (s_j + s_gj)/J = 1/J.
    J = 1: sd 1      -> z 1     -> phi 0.266968
    J = 3: sd 0.5774 -> z 1.732 -> Phi(-1.732) = 0.041632 -> phi = 2(.041632)(.958368) = 0.079798
    J = 4: sd 0.5    -> z 2     -> phi 0.044465 <= 0.05.
    Repeats do not help (s_e = 0), so the cheapest design for eps = 0.05 is J = 4, R = 1,
    calls = 4 * 1 * 10 = 40; for eps = 0.10 it is J = 3 (0.0798 <= 0.10), calls 30."""
    d = dstudy_threshold(1.0, s_t=0, s_tj=0, s_e=0, s_j=1.0, s_gj=0, n=10, Js=(1, 2, 3, 4), Rs=(1, 2))
    phi = {(x["J"], x["R"]): x["phi"] for x in d["grid"]}
    assert phi[(1, 1)] == pytest.approx(0.266968, abs=1e-6)
    assert phi[(3, 1)] == pytest.approx(0.079798, abs=1e-5)
    assert phi[(4, 1)] == pytest.approx(0.044465, abs=1e-6)
    assert phi[(4, 2)] == phi[(4, 1)]
    assert d["cheapest"]["0.05"] == {"J": 4, "R": 1, "calls": 40, "phi": phi[(4, 1)]}
    assert d["cheapest"]["0.1"]["J"] == 3


def test_dstudy_threshold_repeats_by_hand():
    """Only resample noise: s_e = 4, n = 1, delta = 1. Var = s_e/(n J R) = 4/(J R).
    phi <= 0.05 needs z >= 1.9494 -> sd <= 0.51298 -> Var <= 0.26315 -> J R >= 15.2 -> J R = 16.
    Grid J in {1,2,4,8}, R in {1,2,4}: J R = 16 at (4,4) and (8,2), both 16 calls; the tie goes to
    fewer judges -> (4, 4). With infinitely many repeats the floor is phi = 0 for every J."""
    d = dstudy_threshold(1.0, s_t=0, s_tj=0, s_e=4.0, s_j=0, s_gj=0, n=1, Js=(1, 2, 4, 8), Rs=(1, 2, 4))
    best = d["cheapest"]["0.05"]
    assert (best["J"], best["R"], best["calls"]) == (4, 4, 16)
    assert all(v == 0.0 for v in d["phi_limit_R_infinite_by_J"].values())


def test_dstudy_contrast_judge_floor():
    """With infinitely many items the contrast variance is 2 s_gj / J. Two-group data above:
    s_gj = 1, delta = 1. J = 2 -> Var 1 -> phi 0.266968; J = 8 -> Var 0.25, sd 0.5 -> phi 0.044465."""
    Y, g = _two_groups()
    d = dstudy_contrast(gstudy(Y, g), "A", "B", Js=(2, 8))
    assert d["phi_limit_n_infinite_by_J"]["2"] == pytest.approx(0.266968, abs=1e-6)
    assert d["phi_limit_n_infinite_by_J"]["8"] == pytest.approx(0.044465, abs=1e-6)


def test_synthetic_recovery_statistical():
    """Not a hand check: a seeded simulation with large n recovers the components it was built
    from (s_t = 1, s_tj = 0.5, s_e = 0.25) within 15 %."""
    Y, g = simulate({"A": 0.0, "B": 0.5}, n_per_group=3000, J=3, R=2, s_t=1.0, s_tj=0.5, s_e=0.25, seed=7)
    gs = gstudy(Y, g)
    for grp in ("A", "B"):
        P = gs["per_group"][grp]
        assert P["s_t"] == pytest.approx(1.0, rel=0.15)
        assert P["s_tj"] == pytest.approx(0.5, rel=0.15)
        assert P["s_e"] == pytest.approx(0.25, rel=0.15)
    assert phi_from(0.5, 0.0) == 0.0
