"""Generates synthetic subscription billing data for the semantic metric repository.

Produces one row per user per active billing month, with the edge cases that
make ARM a non-additive ratio metric: free trial months, mid-tenure plan
changes, promotional discounts, and mid-month (prorated) cancellations.
Users who never churn within the window are intentionally left active through
the last month (right-censored), matching how a real subscriber base looks.
"""
from __future__ import annotations

import calendar
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEED = 42
N_USERS = 800
N_MONTHS = 24
START_YEAR, START_MONTH = 2024, 1

REGIONS = ["US", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

PLAN_PRICES = {"Basic": 8.0, "Standard": 14.0, "Premium": 20.0}
PLAN_NAMES = list(PLAN_PRICES.keys())
PLAN_WEIGHTS = [0.30, 0.45, 0.25]

TRIAL_PROB = 0.6
PROMO_PROB = 0.4
PROMO_DISCOUNT = 0.2
PROMO_MONTHS = 3
MONTHLY_CHURN_HAZARD = 0.04
PLAN_CHANGE_PROB_PER_MONTH = 0.03
MID_MONTH_CHURN_PROB = 0.5

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "subscriptions.csv"


def _month_bounds(year: int, month: int) -> tuple[dt.date, dt.date]:
    days_in_month = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), dt.date(year, month, days_in_month)


def _add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + offset
    return total // 12, total % 12 + 1


@dataclass
class UserProfile:
    user_id: str
    region_id: str
    plan_type: str
    signup_month_idx: int
    has_trial: bool
    has_promo: bool
    churn_month_idx: int | None


def _build_user_profiles(rng: np.random.Generator) -> list[UserProfile]:
    profiles = []
    # Weight signups toward earlier months so the base looks like a growing,
    # already-established subscriber population rather than everyone joining on day one.
    signup_weights = np.linspace(3.0, 1.0, N_MONTHS)
    signup_weights /= signup_weights.sum()

    for i in range(N_USERS):
        signup_month_idx = int(rng.choice(N_MONTHS, p=signup_weights))
        region_id = rng.choice(REGIONS, p=REGION_WEIGHTS)
        plan_type = rng.choice(PLAN_NAMES, p=PLAN_WEIGHTS)
        has_trial = rng.random() < TRIAL_PROB
        has_promo = rng.random() < PROMO_PROB

        churn_month_idx = None
        month_idx = signup_month_idx
        while month_idx < N_MONTHS:
            if rng.random() < MONTHLY_CHURN_HAZARD:
                churn_month_idx = month_idx
                break
            month_idx += 1

        profiles.append(
            UserProfile(
                user_id=f"user_{i:05d}",
                region_id=region_id,
                plan_type=plan_type,
                signup_month_idx=signup_month_idx,
                has_trial=has_trial,
                has_promo=has_promo,
                churn_month_idx=churn_month_idx,
            )
        )
    return profiles


def _generate_rows(rng: np.random.Generator, profiles: list[UserProfile]) -> list[dict]:
    rows = []

    for profile in profiles:
        current_plan = profile.plan_type
        last_active_month_idx = (
            profile.churn_month_idx if profile.churn_month_idx is not None else N_MONTHS - 1
        )
        paid_month_count = 0

        for month_offset in range(profile.signup_month_idx, last_active_month_idx + 1):
            year, month = _add_months(START_YEAR, START_MONTH, month_offset)
            period_start, period_end = _month_bounds(year, month)

            if month_offset > profile.signup_month_idx and rng.random() < PLAN_CHANGE_PROB_PER_MONTH:
                current_plan = rng.choice(PLAN_NAMES, p=PLAN_WEIGHTS)

            is_trial_month = profile.has_trial and month_offset == profile.signup_month_idx
            is_paid_tier = not is_trial_month

            is_churn_month = month_offset == profile.churn_month_idx
            if is_churn_month and rng.random() < MID_MONTH_CHURN_PROB:
                days_in_month = (period_end - period_start).days + 1
                active_days = int(rng.integers(1, days_in_month))
                period_end = period_start + dt.timedelta(days=active_days - 1)
                proration_fraction = active_days / days_in_month
            else:
                proration_fraction = 1.0

            if is_paid_tier:
                base_price = PLAN_PRICES[current_plan]
                discount = (
                    PROMO_DISCOUNT
                    if profile.has_promo and paid_month_count < PROMO_MONTHS
                    else 0.0
                )
                net_revenue_usd = round(base_price * (1 - discount) * proration_fraction, 2)
                paid_month_count += 1
            else:
                net_revenue_usd = 0.0

            rows.append(
                {
                    "user_id": profile.user_id,
                    "region_id": profile.region_id,
                    "plan_type": current_plan,
                    "billing_period_start": period_start.isoformat(),
                    "billing_period_end": period_end.isoformat(),
                    "net_revenue_usd": net_revenue_usd,
                    "is_paid_tier": is_paid_tier,
                }
            )

    return rows


def generate(output_path: Path = OUTPUT_PATH, seed: int = SEED) -> Path:
    rng = np.random.default_rng(seed)
    profiles = _build_user_profiles(rng)
    rows = _generate_rows(rng, profiles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id",
        "region_id",
        "plan_type",
        "billing_period_start",
        "billing_period_end",
        "net_revenue_usd",
        "is_paid_tier",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


if __name__ == "__main__":
    path = generate()
    print(f"Wrote synthetic subscription data to {path}")
