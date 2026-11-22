#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python - <<'PY'
import math
import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from judge_stability import flip_rate, icc21, phi_from


readme = " ".join(Path("README.md").read_text(encoding="utf-8").split())


def published(pattern):
    match = re.search(pattern, readme)
    if match is None:
        raise SystemExit(f"Cannot locate README claim matching {pattern!r}")
    return match[1]


def matches_decimal(value, displayed):
    # Allow half of the last published decimal place, including rounding ties.
    expected = Decimal(displayed)
    tolerance = Decimal("0.5").scaleb(expected.as_tuple().exponent)
    return value.is_finite() and abs(value - expected) <= tolerance


# The three-item example in tests/test_icc.py::test_high_icc_with_flipping_decisions.
scores = np.array([[0, 0], [10, 10], [4, 6]])
icc = icc21(scores)
icc_fraction = published(r"ICC = (\d+/\d+) =")
icc_display = published(r"ICC = \d+/\d+ = (\d+\.\d+)")
flip = flip_rate(scores >= 5, axis=1)
flip_display = published(r"yet (one decision in three) flips")

# Under the normal null with known sd, Delta_hat / sd is standard normal.
null_plugin, _ = quad(lambda z: 2 * norm.pdf(z) * phi_from(z, 1), 0, np.inf)
null_display = published(r"about (\d+/\d+) instead of")
null_truth = phi_from(0, 1)
null_truth_display = published(r"about \d+/\d+ instead of (\d+\.\d+)")

rows = [
    (f"Three-item ICC {icc_fraction} = {icc_display}", f"{icc:.8f}",
     math.isclose(icc, float(Fraction(icc_fraction)), rel_tol=0, abs_tol=1e-12)
     and matches_decimal(Decimal(str(icc)), icc_display)),
    (f"Three-item flip rate ({flip_display})", f"{flip:.8f}",
     math.isclose(flip, 1 / 3, rel_tol=0, abs_tol=1e-12)),
    (f"Mean plug-in phi under the normal null (about {null_display})", f"{null_plugin:.8f}",
     math.isclose(null_plugin, float(Fraction(null_display)), rel_tol=0, abs_tol=1e-8)),
    (f"True phi under the normal null ({null_truth_display})", f"{null_truth:.8f}",
     matches_decimal(Decimal(str(null_truth)), null_truth_display)),
]

# The pilot inputs and results tables are absent from this repository.
# Synthetic fixtures have no bearing on these historical claims.
petri = [
    published(r"(ICC \d+\.\d+) vs"),
    published(r"vs (\d+ % identical order) on cooperation"),
    published(r"(\d+ of \d+ transcripts complete)"),
]
for claim in petri:
    rows.append((f"Petri {claim}", "Missing pilot scores and sample metadata", None))

interval = r"\[[+-]?\d+\.\d+, [+-]?\d+\.\d+\]"
tutorbench = [
    published(r"identical under both judges in (\d+ %) of paired resamples"),
    published(r"of paired resamples \((\d+ tasks)\)"),
    published(rf"contrast was reported as ({interval} and {interval})"),
]
for claim in tutorbench:
    rows.append((f"TutorBench {claim}", "Missing pilot scores and paired task inputs", None))

print("| README figure | Recalculated | Match |")
print("|---|---|---|")
for claim, actual, match in rows:
    status = {True: "Yes", False: "No", None: "Unverifiable"}[match]
    print(f"| {claim} | {actual} | {status} |")

if any(match is False for _, _, match in rows):
    raise SystemExit(2)
if any(match is None for _, _, match in rows):
    raise SystemExit(1)
PY
