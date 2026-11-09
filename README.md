# judge-decision-stability

[![tests](https://github.com/Thom-320/judge-decision-stability/actions/workflows/tests.yml/badge.svg)](https://github.com/Thom-320/judge-decision-stability/actions/workflows/tests.yml)

When an LLM judge scores a system, someone then makes a decision: the system passes or fails a
safety threshold, or system A is ranked above system B. This repository measures how often that
decision would change if you swapped the judge, reworded the rubric, or simply asked the same
judge again. It is a small Python package (`src/judge_stability/`) that unifies the analysis code
of two pilot studies: Petri judge stability and TutorBench judge agreement.

Status: research code, pilot stage. No data is included (see "What is not included").

## What phi measures

**phi = the probability that a pass/fail or ranking decision changes when one facet of the
judging setup changes**: the judge, the rubric wording, or the resample (a repeat of the same call).

Two estimators are provided.

- **Design-based (observed flips).** Compare the decisions made under two observed conditions of
  a facet while every other facet stays fixed, and count disagreements:
  `flip_rate` (item-level pass/fail), `system_threshold_flip` (system-level pass/fail on the mean),
  `ranking_flip_rate` (full ranking of systems) and `pairwise_sign_flip_rate` (A above B or not).
- **Model-based (G-study plug-in).** Estimate variance components (items nested in systems,
  crossed with judges, repeats nested in item x judge cells), compute the error variance of the
  decision for a panel of J judges, R repeats and n items, and set
  `phi = 2 p (1 - p)` with `p = Phi(-|Delta| / sd)`, the probability that two independent panels
  land on different sides of the decision boundary (`decisions`, `gstudy`).

The **decision-level D-study** (`dstudy_threshold`, `dstudy_contrast`) inverts this: how many
judges and repeats does a threshold decision need before phi drops under 5 % or 10 %, and what is
the floor that no number of repeats (or items) can remove? A judge-by-system interaction, for
example, is not averaged away by asking the same judge more times.

A **paired bootstrap with a fixed seed** (`paired_bootstrap`, `cluster_bootstrap_mean`,
`rank_stability`) gives intervals for differences between judges or systems on the same items.

## Why ICC is not enough

ICC(2,1) and Krippendorff's alpha (`icc21`, `kalpha_interval`, `varcomp`, included for
comparison) measure how well scores track each other across the whole scale. A decision only
depends on what happens near the threshold or between two close systems. The test
`tests/test_icc.py::test_high_icc_with_flipping_decisions` works through a three-item example by
hand: ICC = 37/38 = 0.97, yet one decision in three flips when the judge changes. The Petri pilot
shows the same pattern on real data: for "cooperation with human misuse" the inter-judge ICC was
0.81, but the three judges gave the same order of target models in only 34 % of bootstrap draws.

ICC also mixes facets. phi is reported per facet (judge, wording, repeat, items), so it tells you
what to buy: more judges, more repeats, or more items.

## How the two source studies use it

| | Petri judge stability | TutorBench judge agreement |
|---|---|---|
| Decision | Is a target model's mean score on a Petri dimension above a threshold? Is target A more concerning than target B? | Which answer-generation arm ranks first? Is arm A better than arm B? |
| Facets | 3 judges x 2 repeats on 120 public transcripts (pilot); Petri 2.0's own judges in v2 | 2 judges (plus a third on one arm) and 5 prompt-format perturbations |
| Pilot result (not reproduced here, data not included) | ICC 0.81 vs 34 % identical order on cooperation with misuse; 105 of 120 transcripts complete (cheap judges, not Petri's official judges) | Full arm ranking identical under both judges in 77 % of paired resamples (57 tasks) |
| Code here | `loaders.petri_tensor`, `decisions`, `dstudy_contrast`, `flip_rate` | `loaders.tutorbench_scores`, `rank_stability`, `paired_bootstrap` |

`scripts/rebuild_from_sources.py` rebuilds both analyses when the data are already on your
machine and writes them to `results/local/` (gitignored). Each source is skipped with a message
when its files are absent. The TutorBench "manta" arm is left out by default; the pilot plan drops it.

## A pitfall from the pilots that cannot happen here

The TutorBench pilot built a list of paired differences by iterating a Python `set` of task ids
and then bootstrapped it with a fixed seed. Set order for strings changes with `PYTHONHASHSEED`,
so the "fixed-seed" interval moved between runs (one contrast was reported as [-3.20, +3.66] and
[-3.27, +3.66]). Here every bootstrap takes data keyed by unit id, sorts the units before any
random draw, rejects plain sets, and uses numpy's seeded `Generator`. The test
`test_pilot_set_order_bug_cannot_happen_here` reruns the pilot pattern under six hash seeds,
checks that the pilot-style interval changes, and checks that this package's interval does not.

## Install and test

```bash
uv venv && uv pip install -e ".[test]"     # or: pip install -e ".[test]"
.venv/bin/python -m pytest -q              # 39 tests, no network, no data needed
```

Every numeric expectation in the tests is worked out by hand in the test docstring (sums of
squares, normal-table values, exact bootstrap enumerations). One test is a seeded simulation and
says so.

## What is not included, and why

- **No TutorBench items or rubrics.** TutorBench is a Scale AI dataset and its licence for
  redistribution is unclear, so no prompts, answers, rubric criteria or task ids are shipped.
- **No raw judge outputs** (Muse Spark, MiMo, DeepSeek Flash, LongCat):
  the terms for redistributing these outputs were not established.
- **No Petri transcripts.** The public corpus is CC-BY-4.0, but this package does not need it;
  only judge scores are read, and only from your own local copy.
- **No results tables.** Without the data they cannot be checked here; the rebuild script
  regenerates them locally.

All fixtures in `tests/` are synthetic.

## Known limits

- With 3 judges the judge components have 2 (and 4) degrees of freedom; model-based phi is
  imprecise and the bootstrap over items does not include judge-sampling uncertainty.
- The plug-in |Delta| is biased upward, so model-based phi is biased downward near Delta = 0
  (about 1/3 instead of 0.5 under a true null).
- Pilot judges were cheap models, not the judges the benchmarks use; pilot numbers do not transfer.

## AI-use statement

The code was ported and restructured from the author's two pilot repositories with the help of
an AI coding assistant (Claude, Anthropic), which also drafted the tests and this README. The
author chose the methods, and the hand computations in the test docstrings were checked against
the code by running the test suite. No AI system was used as a judge or data source in this
repository.

## Licence

MIT (see `LICENSE`). The statistical code is original to the author's pilots; nothing from
Petri (`meridianlabs-ai/inspect_petri`, MIT) or TutorBench is vendored.
