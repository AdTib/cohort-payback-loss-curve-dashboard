"""
Synthetic data generator for a fintech PFM (personal finance management) subscription business.

Produces two CSVs matching the reusable engine's input schema:
  - customers.csv:            customer_id, signup_date, acquisition_channel, initial_plan, cac
  - subscription_events.csv:  customer_id, month, mrr

Design intent (see README.md "Methodology" for the full writeup):
  - 4 acquisition channels with deliberately different CAC/retention economics, so the
    dashboard tells a real story instead of a flat blob.
  - A hazard-rate churn model: high monthly churn probability early, decaying toward a
    per-channel floor -- this produces the classic steep-then-flattening retention curve.
  - A per-cohort "improvement factor" that lowers churn hazard for later signup cohorts,
    so the business is visibly getting better at retention over time.
  - Occasional upsell/downgrade plan-tier moves for active customers, so NRR diverges
    from GRR.
  - A fixed calendar observation window (not a rolling window per customer), so recent
    cohorts are naturally truncated -- exactly like a real billing export -- which forces
    the metrics engine to handle partially-observed cohorts correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

SEED = 42
N_CUSTOMERS = 2000
N_SIGNUP_MONTHS = 18                 # signup cohorts: month 0 .. 17
OBSERVATION_MONTHS = 24              # total calendar months of billing history collected
SIGNUP_START = pd.Timestamp("2024-01-01")

PLAN_TIERS = ["Basic", "Plus", "Premium"]
PLAN_MRR = {"Basic": 15.0, "Plus": 29.0, "Premium": 59.0}

# Per-channel economics. Hazard = monthly probability an active customer churns.
# peak_hazard applies at tenure month 1, decaying geometrically toward floor_hazard.
CHANNELS = {
    "organic_referral": {
        "share": 0.30,
        "cac_mean": 65, "cac_sd": 15, "cac_clip": (25, 130),
        "peak_hazard": 0.10, "floor_hazard": 0.045, "decay": 0.85,
        "upsell_p": 0.030, "downgrade_p": 0.020,
        "plan_mix": {"Basic": 0.50, "Plus": 0.35, "Premium": 0.15},
    },
    "paid_social": {
        "share": 0.30,
        "cac_mean": 300, "cac_sd": 40, "cac_clip": (180, 460),
        "peak_hazard": 0.22, "floor_hazard": 0.060, "decay": 0.75,
        "upsell_p": 0.015, "downgrade_p": 0.030,
        "plan_mix": {"Basic": 0.65, "Plus": 0.30, "Premium": 0.05},
    },
    "content_seo": {
        "share": 0.25,
        "cac_mean": 150, "cac_sd": 25, "cac_clip": (90, 260),
        "peak_hazard": 0.08, "floor_hazard": 0.020, "decay": 0.85,
        "upsell_p": 0.050, "downgrade_p": 0.015,
        "plan_mix": {"Basic": 0.25, "Plus": 0.45, "Premium": 0.30},
    },
    "paid_search": {
        "share": 0.15,
        "cac_mean": 200, "cac_sd": 30, "cac_clip": (120, 330),
        "peak_hazard": 0.14, "floor_hazard": 0.050, "decay": 0.80,
        "upsell_p": 0.025, "downgrade_p": 0.020,
        "plan_mix": {"Basic": 0.40, "Plus": 0.40, "Premium": 0.20},
    },
}

# Later cohorts churn less: hazard multiplier shrinks linearly from 1.0 (first cohort)
# down to a floor as signup_month_index increases, floor reached at last cohort.
IMPROVEMENT_FLOOR = 0.75


def _cohort_improvement_factor(signup_month_index: int) -> float:
    frac = signup_month_index / max(N_SIGNUP_MONTHS - 1, 1)
    return 1.0 - (1.0 - IMPROVEMENT_FLOOR) * frac


def _signups_per_month(n_customers: int, n_months: int) -> np.ndarray:
    """Linear ramp in signup volume (early-stage growth), summing to n_customers."""
    weights = np.arange(1, n_months + 1, dtype=float)
    counts = np.round(n_customers * weights / weights.sum()).astype(int)
    counts[-1] += n_customers - counts.sum()
    return counts


def generate_customers(rng: np.random.Generator) -> pd.DataFrame:
    counts_per_month = _signups_per_month(N_CUSTOMERS, N_SIGNUP_MONTHS)
    channel_names = list(CHANNELS.keys())
    channel_probs = np.array([CHANNELS[c]["share"] for c in channel_names])
    channel_probs = channel_probs / channel_probs.sum()

    rows = []
    cid = 0
    for month_idx, n in enumerate(counts_per_month):
        signup_date = SIGNUP_START + pd.DateOffset(months=month_idx)
        channels_for_month = rng.choice(channel_names, size=n, p=channel_probs)
        for channel in channels_for_month:
            cfg = CHANNELS[channel]
            cac = rng.normal(cfg["cac_mean"], cfg["cac_sd"])
            cac = float(np.clip(cac, *cfg["cac_clip"]))
            plan = rng.choice(list(cfg["plan_mix"].keys()), p=list(cfg["plan_mix"].values()))
            rows.append({
                "customer_id": f"CUST{cid:05d}",
                "signup_date": signup_date,
                "signup_month_index": month_idx,
                "acquisition_channel": channel,
                "initial_plan": plan,
                "cac": round(cac, 2),
            })
            cid += 1

    return pd.DataFrame(rows)


def _simulate_customer_mrr(rng: np.random.Generator, cfg: dict, plan: str,
                            improvement_factor: float, n_tenure_months: int) -> list[float]:
    """Returns MRR for tenure months 0..n_tenure_months-1 (0 = signup month)."""
    plan_idx = PLAN_TIERS.index(plan)
    mrr_series = [PLAN_MRR[plan]]
    churned = False

    for t in range(1, n_tenure_months):
        if churned:
            mrr_series.append(0.0)
            continue

        hazard = cfg["floor_hazard"] + (cfg["peak_hazard"] - cfg["floor_hazard"]) * (cfg["decay"] ** (t - 1))
        hazard = float(np.clip(hazard * improvement_factor, 0.0, 0.95))

        if rng.random() < hazard:
            churned = True
            mrr_series.append(0.0)
            continue

        r = rng.random()
        if r < cfg["upsell_p"] and plan_idx < len(PLAN_TIERS) - 1:
            plan_idx += 1
        elif r < cfg["upsell_p"] + cfg["downgrade_p"] and plan_idx > 0:
            plan_idx -= 1

        base = PLAN_MRR[PLAN_TIERS[plan_idx]]
        noise = rng.normal(1.0, 0.03)
        mrr_series.append(round(max(base * noise, 1.0), 2))

    return mrr_series


def generate_subscription_events(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for row in customers.itertuples(index=False):
        cfg = CHANNELS[row.acquisition_channel]
        improvement = _cohort_improvement_factor(row.signup_month_index)
        n_tenure_months = OBSERVATION_MONTHS - row.signup_month_index
        if n_tenure_months <= 0:
            continue

        mrr_series = _simulate_customer_mrr(rng, cfg, row.initial_plan, improvement, n_tenure_months)

        for t, mrr in enumerate(mrr_series):
            event_month = row.signup_date + pd.DateOffset(months=t)
            rows.append({"customer_id": row.customer_id, "month": event_month, "mrr": mrr})

    return pd.DataFrame(rows)


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    customers = generate_customers(rng)
    events = generate_subscription_events(customers, rng)
    customers = customers.drop(columns=["signup_month_index"])
    return customers, events


def main():
    import os
    customers, events = generate()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)

    customers_path = os.path.join(out_dir, "customers.csv")
    events_path = os.path.join(out_dir, "subscription_events.csv")

    customers.to_csv(customers_path, index=False)
    events.to_csv(events_path, index=False)

    print(f"Wrote {len(customers)} customers to {customers_path}")
    print(f"Wrote {len(events)} subscription events to {events_path}")


if __name__ == "__main__":
    main()
