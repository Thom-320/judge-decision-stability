"""phi, checked by hand. Normal-table values used: Phi(-1) = 0.158655, Phi(-2) = 0.022750."""
import numpy as np
import pytest

from judge_stability import (flip_rate, pairwise_sign_flip_rate, phi_from, ranking_flip_rate,
                             system_threshold_flip, z_for_phi)


def test_phi_from_hand_values():
    """phi = 2 p (1 - p), p = Phi(-|delta|/sd).
    delta = 0          -> p = 0.5      -> phi = 2(0.5)(0.5) = 0.5 (a coin flip)
    sd = 0, delta = 1  -> no noise     -> phi = 0
    delta = 1, sd = 1  -> p = 0.158655 -> phi = 2(0.158655)(0.841345) = 0.266968
    delta = 2, sd = 1  -> p = 0.022750 -> phi = 2(0.022750)(0.977250) = 0.044465
    sign of delta does not matter (|delta|)."""
    assert phi_from(0.0, 1.0) == 0.5
    assert phi_from(1.0, 0.0) == 0.0
    assert phi_from(1.0, 1.0) == pytest.approx(0.266968, abs=1e-6)
    assert phi_from(2.0, 1.0) == pytest.approx(0.044465, abs=1e-6)
    assert phi_from(-2.0, 1.0) == phi_from(2.0, 1.0)


def test_z_for_phi_inverts_phi():
    """phi = 0.05  <=>  p^2 - p + 0.025 = 0  <=>  p = (1 - sqrt(0.9))/2 = (1 - 0.948683)/2 = 0.025658,
    and Phi^{-1}(0.025658) = -1.9494 (between Phi(-1.96)=0.0250 and Phi(-1.95)=0.0256)."""
    z = z_for_phi(0.05)
    assert 1.94 < z < 1.96
    assert phi_from(z, 1.0) == pytest.approx(0.05, abs=1e-12)
    with pytest.raises(ValueError):
        z_for_phi(0.5)


def test_flip_rate_three_judges():
    """Items x judges, pass = 1.
    item 1: [1, 1, 0] -> pairs (J1,J2) same, (J1,J3) differ, (J2,J3) differ -> 2/3
    item 2: [1, 1, 1] -> 0/3
    mean over the 6 (item, pair) cells = 2/6 = 1/3."""
    D = np.array([[1, 1, 0], [1, 1, 1]], bool)
    assert flip_rate(D, axis=1) == pytest.approx(1 / 3)


def test_flip_rate_separates_facets():
    """One item, decisions[judge, wording] = [[1, 0], [1, 0]] (shape 1 x 2 x 2).
    Change the judge, wording fixed: wording 1: 1 vs 1 same; wording 2: 0 vs 0 same -> phi_judge = 0.
    Change the wording, judge fixed: judge 1: 1 vs 0 differ; judge 2: 1 vs 0 differ -> phi_wording = 1."""
    D = np.array([[[1, 0], [1, 0]]], bool)
    assert flip_rate(D, axis=1) == 0.0
    assert flip_rate(D, axis=2) == 1.0


def test_flip_rate_needs_two_conditions():
    with pytest.raises(ValueError):
        flip_rate(np.ones((3, 1)), axis=1)


def test_ranking_flip_rate_and_pair_signs():
    """Rows = 3 judges, columns = systems A, B, C (mean scores).
    J1 [3, 2, 1] -> A > B > C;  J2 [3, 2, 1] -> A > B > C;  J3 [2, 3, 1] -> B > A > C.
    Judge pairs: (1,2) same, (1,3) differ, (2,3) differ -> ranking phi = 2/3.
    Pair A|B: signs +, +, - -> 2 of 3 judge pairs differ -> 2/3; A|C and B|C: always + -> 0."""
    M = [[3, 2, 1], [3, 2, 1], [2, 3, 1]]
    assert ranking_flip_rate(M) == pytest.approx(2 / 3)
    s = pairwise_sign_flip_rate(M)
    assert s[(0, 1)] == pytest.approx(2 / 3)
    assert s[(0, 2)] == 0.0 and s[(1, 2)] == 0.0


def test_system_threshold_flip():
    """Y[item, judge, repeat], system passes if mean over items >= 2.5.
    judge 1: items [2, 4] -> mean 3 -> pass;   judge 2: items [1, 3] -> mean 2 -> fail.
    One judge pair, decisions differ -> phi_judge = 1.
    With two identical repeats per judge, changing the repeat never flips -> phi_repeat = 0."""
    Y = np.array([[[2, 2], [1, 1]], [[4, 4], [3, 3]]], float)  # 2 items x 2 judges x 2 repeats
    assert system_threshold_flip(Y, 2.5, facet_axis=1) == 1.0
    assert system_threshold_flip(Y, 2.5, facet_axis=2) == 0.0
    with pytest.raises(ValueError):
        system_threshold_flip(Y, 2.5, facet_axis=0)
