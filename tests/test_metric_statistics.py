import unittest

import numpy as np

from app.services.metric_statistics import ratio_confidence_interval


class RatioConfidenceIntervalTests(unittest.TestCase):
    def test_estimate_equals_ratio_of_sums(self) -> None:
        revenue = [100.0, 200.0, 150.0]
        subscribers = [10.0, 20.0, 15.0]

        ci = ratio_confidence_interval(revenue, subscribers)

        self.assertAlmostEqual(ci.estimate, sum(revenue) / sum(subscribers))
        self.assertEqual(ci.n_units, 3)

    def test_interval_brackets_the_estimate(self) -> None:
        revenue = [100.0, 220.0, 90.0, 205.0, 118.0]
        subscribers = [10.0, 21.0, 9.0, 20.0, 12.0]

        ci = ratio_confidence_interval(revenue, subscribers)

        self.assertLess(ci.lower, ci.estimate)
        self.assertGreater(ci.upper, ci.estimate)
        self.assertGreaterEqual(ci.standard_error, 0.0)

    def test_wider_interval_for_noisier_data(self) -> None:
        subscribers = [100.0] * 12
        low_noise_revenue = [1000 + i for i in range(12)]
        high_noise_revenue = [1000 + i * 200 * ((-1) ** i) for i in range(12)]

        tight_ci = ratio_confidence_interval(low_noise_revenue, subscribers)
        wide_ci = ratio_confidence_interval(high_noise_revenue, subscribers)

        self.assertLess(tight_ci.upper - tight_ci.lower, wide_ci.upper - wide_ci.lower)

    def test_matches_known_delta_method_formula(self) -> None:
        # Hand-computed against the proposal's formula for a small, exact example.
        r = np.array([10.0, 20.0, 30.0, 40.0])
        s = np.array([2.0, 4.0, 5.0, 9.0])
        n = len(r)
        r_bar, s_bar = r.mean(), s.mean()
        var_r_bar = r.var(ddof=1) / n
        var_s_bar = s.var(ddof=1) / n
        cov_bar = np.cov(r, s, ddof=1)[0, 1] / n
        expected_variance = (
            (1 / s_bar**2) * var_r_bar
            + (r_bar**2 / s_bar**4) * var_s_bar
            - 2 * (r_bar / s_bar**3) * cov_bar
        )

        ci = ratio_confidence_interval(r, s)

        self.assertAlmostEqual(ci.standard_error, expected_variance**0.5, places=10)

    def test_raises_with_fewer_than_two_units(self) -> None:
        with self.assertRaises(ValueError):
            ratio_confidence_interval([100.0], [10.0])

    def test_raises_on_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            ratio_confidence_interval([100.0, 200.0], [10.0])

    def test_raises_on_zero_mean_denominator(self) -> None:
        with self.assertRaises(ValueError):
            ratio_confidence_interval([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def test_raises_on_nan_input(self) -> None:
        with self.assertRaises(ValueError):
            ratio_confidence_interval([100.0, float("nan"), 150.0], [10.0, 20.0, 15.0])

    def test_uses_wider_critical_value_for_small_samples_than_normal_z(self) -> None:
        # With only 3 units (df=2), the t-distribution critical value is
        # well above the normal z=1.96 used for a naive large-sample
        # approximation -- the interval should reflect that extra
        # uncertainty, not understate it.
        revenue = [100.0, 140.0, 90.0]
        subscribers = [10.0, 10.5, 9.8]

        ci = ratio_confidence_interval(revenue, subscribers)
        implied_critical_value = (ci.upper - ci.estimate) / ci.standard_error

        self.assertGreater(implied_critical_value, 1.96)

    def test_critical_value_approaches_normal_z_for_large_samples(self) -> None:
        rng = np.random.default_rng(1)
        subscribers = rng.normal(200, 10, 100)
        revenue = 15 * subscribers + rng.normal(0, 5, 100)

        ci = ratio_confidence_interval(revenue, subscribers)
        implied_critical_value = (ci.upper - ci.estimate) / ci.standard_error

        self.assertAlmostEqual(implied_critical_value, 1.96, delta=0.05)

    def test_raises_on_unsupported_confidence_level(self) -> None:
        with self.assertRaises(ValueError):
            ratio_confidence_interval([100.0, 200.0, 150.0], [10.0, 20.0, 15.0], confidence_level=0.5)

    def test_higher_confidence_level_widens_interval(self) -> None:
        revenue = [100.0, 220.0, 90.0, 205.0, 118.0]
        subscribers = [10.0, 21.0, 9.0, 20.0, 12.0]

        ci_90 = ratio_confidence_interval(revenue, subscribers, confidence_level=0.90)
        ci_99 = ratio_confidence_interval(revenue, subscribers, confidence_level=0.99)

        self.assertLess(ci_90.upper - ci_90.lower, ci_99.upper - ci_99.lower)


if __name__ == "__main__":
    unittest.main()
