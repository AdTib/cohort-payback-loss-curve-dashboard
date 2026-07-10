"""
The reusable calculation layer. Every function here takes the panel produced by
engine.loaders.build_customer_panel (or a filtered slice of it) and returns a plain
DataFrame -- nothing here is aware of Streamlit or the synthetic generator, so it
works unchanged against a real company's export.

See README.md "Methodology" for the definitions and the reasoning behind each choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GROSS_MARGIN = 0.75          # fintech PFM SaaS gross margin assumption (post infra/support/processing)
MIN_COHORT_COVERAGE = 0.20           # a (cohort, tenure) cell needs >= this fraction of the cohort observed
LTV_CONVERGENCE_THRESHOLD = 0.001    # stop summing the retention tail once survival drops below this
LTV_MAX_HORIZON_MONTHS = 240         # hard cap so a pathological (near-zero-churn) channel can't loop forever


# ---------------------------------------------------------------------------
# 1. Cohort retention
# ---------------------------------------------------------------------------

def cohort_retention_table(panel: pd.DataFrame) -> pd.DataFrame:
    """
    % of each signup cohort with MRR > 0 at tenure month N.
    Rows = cohort_month, columns = tenure_month, values = retention % (0-100).
    Cells for (cohort, tenure) combinations that haven't happened yet in the data
    are simply absent (NaN) -- callers should not treat NaN as 0% retention.
    """
    cohort_sizes = panel.groupby("cohort_month")["customer_id"].nunique()

    active = (
        panel.groupby(["cohort_month", "tenure_month"])["is_active"]
        .sum()
        .reset_index(name="n_active")
    )
    active["cohort_size"] = active["cohort_month"].map(cohort_sizes)
    active["retention_pct"] = 100 * active["n_active"] / active["cohort_size"]

    table = active.pivot(index="cohort_month", columns="tenure_month", values="retention_pct")
    table.index = table.index.astype(str)
    return table


def blended_retention_curve(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Single retention curve (% active at tenure N) pooled across all cohorts in the
    panel, restricted to tenures with at least MIN_COHORT_COVERAGE of the population
    observed. Used for the channel-comparison retention chart.
    """
    total_customers = panel["customer_id"].nunique()
    grouped = panel.groupby("tenure_month").agg(
        n_observed=("customer_id", "nunique"),
        n_active=("is_active", "sum"),
    ).reset_index()
    grouped["coverage"] = grouped["n_observed"] / total_customers
    grouped["retention_pct"] = 100 * grouped["n_active"] / grouped["n_observed"]
    return grouped[grouped["coverage"] >= MIN_COHORT_COVERAGE][
        ["tenure_month", "retention_pct", "coverage", "n_observed"]
    ]


# ---------------------------------------------------------------------------
# 2. CAC payback
# ---------------------------------------------------------------------------

def _avg_cumulative_margin_by_tenure(panel: pd.DataFrame, gross_margin: float) -> pd.DataFrame:
    """
    For each tenure month t, the average cumulative gross margin per customer,
    computed only over customers who have actually been observed for t+1 months
    (i.e. right-censored cohorts don't drag the average down with phantom zeros).
    """
    df = panel.sort_values(["customer_id", "tenure_month"]).copy()
    df["margin"] = df["mrr"] * gross_margin
    df["cum_margin"] = df.groupby("customer_id")["margin"].cumsum()

    total_customers = df["customer_id"].nunique()
    out = df.groupby("tenure_month").agg(
        avg_cum_margin=("cum_margin", "mean"),
        n_observed=("customer_id", "nunique"),
    ).reset_index()
    out["coverage"] = out["n_observed"] / total_customers
    return out[out["coverage"] >= MIN_COHORT_COVERAGE]


def cac_payback_by_channel(panel: pd.DataFrame, customers: pd.DataFrame,
                            gross_margin: float = DEFAULT_GROSS_MARGIN) -> pd.DataFrame:
    """
    Payback month per channel: the smallest tenure month t at which average
    cumulative gross margin per customer >= average CAC for that channel.
    Returns one row per channel with payback_month (None if not reached within
    the observed, sufficiently-covered window) plus the full margin/CAC ratio
    curve as a list, for plotting.
    """
    avg_cac = customers.groupby("acquisition_channel")["cac"].mean()

    rows = []
    curves = {}
    for channel, cac in avg_cac.items():
        sub = panel[panel["acquisition_channel"] == channel]
        curve = _avg_cumulative_margin_by_tenure(sub, gross_margin).copy()
        curve["cac"] = cac
        curve["payback_ratio"] = curve["avg_cum_margin"] / cac
        curves[channel] = curve

        reached = curve[curve["payback_ratio"] >= 1.0]
        payback_month = int(reached["tenure_month"].iloc[0]) if len(reached) else None

        rows.append({
            "acquisition_channel": channel,
            "avg_cac": round(cac, 2),
            "payback_month": payback_month,
            "max_observed_tenure": int(curve["tenure_month"].max()) if len(curve) else 0,
        })

    summary = pd.DataFrame(rows).sort_values("acquisition_channel").reset_index(drop=True)
    summary["payback_month"] = summary["payback_month"].astype("Int64")  # nullable int: preserves None as pd.NA
    return summary, curves


# ---------------------------------------------------------------------------
# 3. LTV and LTV:CAC
# ---------------------------------------------------------------------------

def _fit_tail_hazard(retention_curve: pd.DataFrame, n_tail_points: int = 3) -> float:
    """
    Monthly churn hazard implied by the last few reliably-observed points of a
    retention curve, used to extrapolate survival past the observed window.
    hazard_t = 1 - S(t)/S(t-1); we average the last n_tail_points such hazards.
    """
    s = retention_curve.sort_values("tenure_month")
    survival = s["retention_pct"].to_numpy() / 100.0
    if len(survival) < 2:
        return 0.05  # fallback: arbitrary small-sample default, documented in README
    hazards = 1 - (survival[1:] / np.clip(survival[:-1], 1e-9, None))
    hazards = hazards[-n_tail_points:]
    return float(np.clip(np.mean(hazards), 0.001, 0.5))


def expected_active_months(retention_curve: pd.DataFrame) -> float:
    """
    Expected number of "retained months" per customer (i.e. sum of survival
    probability over all future months), using observed retention where available
    and a geometric extrapolation (constant tail hazard) beyond that.
    """
    s = retention_curve.sort_values("tenure_month")
    observed_survival = dict(zip(s["tenure_month"], s["retention_pct"] / 100.0))
    last_t = int(s["tenure_month"].max())
    last_survival = observed_survival[last_t]
    tail_hazard = _fit_tail_hazard(s)

    total = sum(observed_survival.values())
    survival = last_survival
    t = last_t
    while survival >= LTV_CONVERGENCE_THRESHOLD and (t - last_t) < LTV_MAX_HORIZON_MONTHS:
        survival *= (1 - tail_hazard)
        total += survival
        t += 1
    return float(total)


def ltv_by_channel(panel: pd.DataFrame, customers: pd.DataFrame,
                    gross_margin: float = DEFAULT_GROSS_MARGIN) -> pd.DataFrame:
    """
    LTV(channel) = expected_active_months(channel) * ARPA(channel) * gross_margin,
    where ARPA is the average MRR among a channel's *active* customer-months
    (so upsell/downgrade noise is reflected but churned $0 months are excluded).
    Modeled to convergence (effectively "to infinity") rather than a fixed window --
    see README for why.
    """
    rows = []
    for channel in sorted(customers["acquisition_channel"].unique()):
        sub = panel[panel["acquisition_channel"] == channel]
        curve = blended_retention_curve(sub)
        exp_months = expected_active_months(curve)
        arpa = sub.loc[sub["is_active"], "mrr"].mean()
        ltv = exp_months * arpa * gross_margin
        rows.append({
            "acquisition_channel": channel,
            "expected_active_months": round(exp_months, 2),
            "arpa": round(arpa, 2),
            "ltv": round(ltv, 2),
        })
    return pd.DataFrame(rows)


def ltv_to_cac_by_channel(panel: pd.DataFrame, customers: pd.DataFrame,
                           gross_margin: float = DEFAULT_GROSS_MARGIN) -> pd.DataFrame:
    ltv = ltv_by_channel(panel, customers, gross_margin)
    cac = customers.groupby("acquisition_channel")["cac"].mean().rename("avg_cac").reset_index()
    out = ltv.merge(cac, on="acquisition_channel")
    out["ltv_to_cac"] = out["ltv"] / out["avg_cac"]
    return out


# ---------------------------------------------------------------------------
# 4. GRR / NRR (loss curves)
# ---------------------------------------------------------------------------

def grr_nrr_table(panel: pd.DataFrame, group_col: str = "cohort_month") -> pd.DataFrame:
    """
    GRR/NRR relative to each group's (e.g. each cohort's) tenure-0 revenue base.

    NRR(t) = sum(mrr_t) / sum(mrr_0)                          -- can exceed 100%
    GRR(t) = sum(min(mrr_t, mrr_0)) / sum(mrr_0)               -- capped at 100% by construction,
                                                                   since upsells (mrr_t > mrr_0) are
                                                                   clipped back down to mrr_0 and only
                                                                   churn/downgrade (mrr_t < mrr_0) can move it
    """
    df = panel.copy()
    mrr0 = (
        df[df["tenure_month"] == 0]
        .set_index("customer_id")["mrr"]
        .rename("mrr_0")
    )
    df = df.merge(mrr0, on="customer_id", how="left")
    df["capped_mrr"] = np.minimum(df["mrr"], df["mrr_0"])

    base = df[df["tenure_month"] == 0].groupby(group_col)["mrr"].sum().rename("base_mrr")

    agg = df.groupby([group_col, "tenure_month"]).agg(
        total_mrr=("mrr", "sum"),
        capped_mrr=("capped_mrr", "sum"),
    ).reset_index()
    agg = agg.merge(base, on=group_col)
    agg["nrr_pct"] = 100 * agg["total_mrr"] / agg["base_mrr"]
    agg["grr_pct"] = 100 * agg["capped_mrr"] / agg["base_mrr"]

    agg[group_col] = agg[group_col].astype(str)
    return agg[[group_col, "tenure_month", "grr_pct", "nrr_pct"]]
