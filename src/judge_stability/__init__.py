"""Decision stability of LLM judges.

phi = probability that a pass/fail or ranking decision changes when the judge, the rubric
wording or the resample changes. See README.md.
"""
from .bootstrap import (exact_bootstrap_means, paired_bootstrap, percentile_ci,
                        cluster_bootstrap_mean, rank_stability)
from .dstudy import (contrast_var, dstudy_contrast, dstudy_threshold, threshold_var)
from .gstudy import gstudy, within_group
from .icc import icc21, kalpha_interval, varcomp
from .phi import (decisions, flip_rate, pairwise_sign_flip_rate, phi_from, ranking_flip_rate,
                  system_threshold_flip, z_for_phi)

__version__ = "0.1.0"
__all__ = [
    "phi_from", "z_for_phi", "flip_rate", "ranking_flip_rate", "pairwise_sign_flip_rate",
    "system_threshold_flip", "decisions", "gstudy", "within_group", "contrast_var", "threshold_var",
    "dstudy_contrast", "dstudy_threshold", "icc21", "kalpha_interval", "varcomp", "paired_bootstrap",
    "cluster_bootstrap_mean", "rank_stability", "exact_bootstrap_means", "percentile_ci",
]
