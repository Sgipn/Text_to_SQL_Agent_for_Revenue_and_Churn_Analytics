"""Delta Method confidence intervals for ratio metrics (e.g. ARM).

ARM is a ratio of sums (SUM(revenue) / SUM(subscribers)), so it is a ratio
estimator: its sampling distribution is non-linear, and Var(R)/Var(S) alone
do not give its variance. This implements the first-order Taylor expansion
(Delta Method) approximation from the project proposal's statistical
foundations section:

    Var(R_bar/S_bar) ~= (1/mu_S^2) Var(R_bar)
                       + (mu_R^2/mu_S^4) Var(S_bar)
                       - 2 (mu_R/mu_S^3) Cov(R_bar, S_bar)

`numerator`/`denominator` must be row-per-independent-sampling-unit arrays
(e.g. one row per month) -- not a single pre-aggregated total, since the
variance terms above are estimated across units.

The critical value uses Student's t distribution with n-1 degrees of
freedom, not a fixed normal z-score. They converge for large n (our typical
case is ~24 months), but a WHERE filter can narrow a slice down to just a
handful of periods, where a normal approximation understates uncertainty --
the interval would look falsely precise right when the estimate is least
reliable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

_SUPPORTED_CONFIDENCE_LEVELS = (0.90, 0.95, 0.99)


@dataclass
class RatioConfidenceInterval:
    estimate: float
    standard_error: float
    lower: float
    upper: float
    n_units: int
    confidence_level: float


def ratio_confidence_interval(
    numerator,
    denominator,
    confidence_level: float = 0.95,
) -> RatioConfidenceInterval:
    """Delta Method CI for the ratio-of-sums estimator sum(numerator)/sum(denominator).

    Raises ValueError if there are fewer than 2 sampling units, the arrays
    are misaligned or contain NaN, the confidence level isn't supported, or
    the mean of `denominator` is zero (the ratio -- and its variance -- are
    undefined).
    """
    if confidence_level not in _SUPPORTED_CONFIDENCE_LEVELS:
        supported = ", ".join(str(level) for level in _SUPPORTED_CONFIDENCE_LEVELS)
        raise ValueError(f"Unsupported confidence_level {confidence_level}; use one of: {supported}.")

    r = np.asarray(numerator, dtype=float)
    s = np.asarray(denominator, dtype=float)
    if r.shape != s.shape:
        raise ValueError("numerator and denominator must be the same length.")
    if np.isnan(r).any() or np.isnan(s).any():
        raise ValueError("numerator and denominator must not contain NaN/null values.")

    n = len(r)
    if n < 2:
        raise ValueError("Delta Method variance requires at least 2 independent sampling units.")

    r_bar = r.mean()
    s_bar = s.mean()
    if s_bar == 0:
        raise ValueError("Cannot compute a ratio confidence interval when the denominator's mean is zero.")

    estimate = r.sum() / s.sum()

    var_r_bar = r.var(ddof=1) / n
    var_s_bar = s.var(ddof=1) / n
    cov_r_bar_s_bar = np.cov(r, s, ddof=1)[0, 1] / n

    variance = (
        (1 / s_bar**2) * var_r_bar
        + (r_bar**2 / s_bar**4) * var_s_bar
        - 2 * (r_bar / s_bar**3) * cov_r_bar_s_bar
    )
    # Floating-point noise can push a near-zero true variance slightly negative.
    variance = max(variance, 0.0)
    standard_error = math.sqrt(variance)

    critical_value = stats.t.ppf(1 - (1 - confidence_level) / 2, df=n - 1)
    margin = critical_value * standard_error
    return RatioConfidenceInterval(
        estimate=estimate,
        standard_error=standard_error,
        lower=estimate - margin,
        upper=estimate + margin,
        n_units=n,
        confidence_level=confidence_level,
    )
