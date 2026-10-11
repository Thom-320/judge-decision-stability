"""ICC(2,1), alpha and variance components, checked by hand."""
import numpy as np
import pytest

from judge_stability import flip_rate, icc21, kalpha_interval, varcomp


def test_icc21_constant_offset():
    """X = [[1,2],[2,3],[3,4]]: judge 2 = judge 1 + 1 on every item.
    grand = 15/6 = 2.5; row means 1.5, 2.5, 3.5; column means 2, 3.
    SSR = 2(1 + 0 + 1) = 4; SSC = 3(0.25 + 0.25) = 1.5; SST = 2.25+.25+.25+.25+.25+2.25 = 5.5;
    SSE = 5.5 - 4 - 1.5 = 0.  MSR = 4/2 = 2, MSC = 1.5/1 = 1.5, MSE = 0.
    ICC = (2 - 0) / (2 + 0 + 2(1.5 - 0)/3) = 2/3."""
    assert icc21([[1, 2], [2, 3], [3, 4]]) == pytest.approx(2 / 3)


def test_high_icc_with_flipping_decisions():
    """Why ICC is not enough. X = [[0,0],[10,10],[4,6]], pass if score >= 5.
    grand = 30/6 = 5; row means 0, 10, 5; column means 14/3, 16/3.
    SSR = 2(25 + 25 + 0) = 100; SSC = 3(1/9 + 1/9) = 2/3; SST = 25+25+25+25+1+1 = 102;
    SSE = 102 - 100 - 2/3 = 4/3. MSR = 50, MSC = 2/3, MSE = (4/3)/2 = 2/3.
    ICC = (50 - 2/3) / (50 + 2/3 + 2(2/3 - 2/3)/3) = (148/3)/(152/3) = 37/38 = 0.9737.
    Decisions: item 1 fail/fail, item 2 pass/pass, item 3 fail/pass -> phi_judge = 1/3.
    ICC 0.97 and yet a third of pass/fail decisions change with the judge."""
    X = np.array([[0, 0], [10, 10], [4, 6]], float)
    assert icc21(X) == pytest.approx(37 / 38)
    assert flip_rate(X >= 5, axis=1) == pytest.approx(1 / 3)


def test_icc21_undefined_cases():
    assert np.isnan(icc21([[1, 1], [1, 1], [1, 1]]))
    assert np.isnan(icc21([[1, 2], [2, 3]]))  # n < 3


def test_kalpha_identical_raters_is_one():
    """Two raters agree on every unit: observed disagreement Do = 0 -> alpha = 1 - 0/De = 1."""
    assert kalpha_interval([[1, 1], [2, 2], [3, 3]]) == pytest.approx(1.0)


def test_varcomp_item_only():
    """Y[item, judge, repeat]: item 1 scores 0 everywhere, item 2 scores 2 everywhere (2 x 2 x 2).
    m = 1; item means 0, 2: SS_A = J R ((0-1)^2 + (2-1)^2) = 4 * 2 = 8, MS_A = 8/1 = 8.
    Every other SS is 0, so s_item = (8 - 0 - 0 + 0)/(J R) = 8/4 = 2, all others 0, share 1."""
    Y = np.zeros((2, 2, 2))
    Y[1] = 2
    raw, sh = varcomp(Y)
    assert raw["item"] == pytest.approx(2.0)
    assert sh["item"] == pytest.approx(1.0)
    assert all(abs(v) < 1e-12 for k, v in raw.items() if k != "item")


def test_varcomp_needs_two_repeats():
    with pytest.raises(ValueError):
        varcomp(np.zeros((3, 2, 1)))
