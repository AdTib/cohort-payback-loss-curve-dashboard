"""
The reusable calculation layer. Every function here takes the panel produced by
engine.loaders.build_customer_panel (or a filtered slice of it) and returns a plain
DataFrame -- nothing here is aware of Streamlit or the synthetic generator, so it
works unchanged against a real company's export.

See README.md "Methodology" for the definitions and the reasoning behind each choice.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from engine import survival

DEFAULT_GROSS_MARGIN = 0.75          # fintech PFM SaaS gross margin assumption (post infra/support/processing)
MIN_COHORT_COVERAGE = 0.20           # a (cohort, tenure) cell needs >= this fraction of the cohort observed


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

    KNOWN LIMITATION, found during Phase 2 real-data validation (segmenting
    Telco by Contract type): this coverage filter assumes a customer's panel
    naturally extends to a given tenure_month independent of whether they
    churned -- true for a real calendar-bounded dataset (the synthetic demo,
    Online Retail II), where every customer in a cohort is tracked out to the
    same global observed month regardless of outcome. It is NOT true for
    Telco, where each customer's reconstructed panel length is derived from
    their own tenure (see Phase 0), so only customers who already survived to
    month N even have a row at month N. The result: this function implicitly
    conditions retention-at-N on "already survived that long," which biases
    the curve toward looking healthier than it is, at every N -- the same
    survivorship-bias root cause as the cohort_retention_table finding from
    Phase 0. Telco payback numbers inherit a milder version of the same
    distortion; this is still not patched here, and is why Telco's primary
    retention view is the pooled Kaplan-Meier curve from engine.survival
    (built directly off each customer's own tenure + churn outcome, not
    bucketed into cohorts at all) rather than this function.

    As of Phase 3, this function is no longer what LTV is built on --
    ltv_by_channel now fits a parametric survival model directly to each
    customer's (duration, censored?) pair via engine.survival, which doesn't
    have this coverage-conditioning problem. This function is still used for
    the empirical retention-curve chart/comparison.
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
                            gross_margin: float = DEFAULT_GROSS_MARGIN,
                            segment_col: str = "acquisition_channel") -> pd.DataFrame:
    """
    Payback month per segment: the smallest tenure month t at which average
    cumulative gross margin per customer >= average CAC for that segment.

    segment_col defaults to acquisition_channel but accepts any customer-level
    categorical column present in both `customers` and `panel` -- e.g. a real
    field (Telco's Contract type, InternetService, PaymentMethod) when there's
    no real acquisition channel to segment by. CAC itself is still whatever's
    in the `cac` column, so if segmenting by a non-channel field on a dataset
    where CAC is assumed (see Phase 0/1), the payback numbers inherit that
    same "assumed CAC" caveat -- segmenting by Contract type doesn't make a
    fabricated CAC real, it just changes which population it's averaged over.

    Returns one row per segment value with payback_month (None if not reached
    within the observed, sufficiently-covered window) plus the full
    margin/CAC ratio curve as a dict keyed by segment value, for plotting.
    """
    avg_cac = customers.groupby(segment_col)["cac"].mean()

    rows = []
    curves = {}
    for segment_value, cac in avg_cac.items():
        sub = panel[panel[segment_col] == segment_value]
        curve = _avg_cumulative_margin_by_tenure(sub, gross_margin).copy()
        curve["cac"] = cac
        curve["payback_ratio"] = curve["avg_cum_margin"] / cac
        curves[segment_value] = curve

        reached = curve[curve["payback_ratio"] >= 1.0]
        payback_month = int(reached["tenure_month"].iloc[0]) if len(reached) else None

        rows.append({
            segment_col: segment_value,
            "avg_cac": round(cac, 2),
            "payback_month": payback_month,
            "max_observed_tenure": int(curve["tenure_month"].max()) if len(curve) else 0,
        })

    summary = pd.DataFrame(rows).sort_values(segment_col).reset_index(drop=True)
    summary["payback_month"] = summary["payback_month"].astype("Int64")  # nullable int: preserves None as pd.NA
    return summary, curves


def payback_ratio_bridge(curve_a: pd.DataFrame, cac_a: float, curve_b: pd.DataFrame, cac_b: float) -> dict:
    """
    Exact bridge decomposition (Key Insights driver decomposition, Phase 5) of
    why segment A's payback-ratio is ahead of segment B's, at the latest
    tenure month both have coverage-valid data for. curve_a/curve_b are the
    per-segment curves cac_payback_by_channel already returns (avg_cum_margin
    by tenure_month), so this doesn't recompute anything, only re-arranges
    numbers already on screen elsewhere in the app.

    Standard FP&A "bridge"/waterfall technique: swap one input at a time and
    attribute the resulting movement to that input.
      ratio_a            = margin_a(t) / cac_a
      ratio_b_actual      = margin_b(t) / cac_b
      ratio_b_with_a_cac = margin_b(t) / cac_a          -- B's margin, A's CAC

      cac_effect     = ratio_b_with_a_cac - ratio_b_actual   (movement from the CAC swap alone)
      margin_effect  = ratio_a - ratio_b_with_a_cac           (whatever's left: retention/margin)

    cac_effect + margin_effect == ratio_a - ratio_b_actual EXACTLY (telescopes
    algebraically) -- not an approximation, not a regression, just arithmetic,
    which is what makes it defensible to say out loud: "if paid_social had
    organic's CAC, its ratio would close by X; the remaining gap is retention."
    """
    t = min(curve_a["tenure_month"].max(), curve_b["tenure_month"].max())
    margin_a = curve_a.set_index("tenure_month").loc[t, "avg_cum_margin"]
    margin_b = curve_b.set_index("tenure_month").loc[t, "avg_cum_margin"]

    ratio_a = margin_a / cac_a
    ratio_b_actual = margin_b / cac_b
    ratio_b_with_a_cac = margin_b / cac_a

    cac_effect = ratio_b_with_a_cac - ratio_b_actual
    margin_effect = ratio_a - ratio_b_with_a_cac

    return {
        "reference_month": int(t),
        "ratio_a": float(ratio_a),
        "ratio_b": float(ratio_b_actual),
        "total_gap": float(ratio_a - ratio_b_actual),
        "cac_effect": float(cac_effect),
        "margin_effect": float(margin_effect),
    }


def bootstrap_payback_ci(panel: pd.DataFrame, customers: pd.DataFrame,
                          segment_col: str = "acquisition_channel",
                          gross_margin: float = DEFAULT_GROSS_MARGIN,
                          n_boot: int = 300, seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """
    Bootstrapped confidence interval on the payback month from cac_payback_by_channel:
    resample customers within a segment with replacement, recompute payback month
    the same way (same coverage rule, same >= 1.0 crossing), repeat n_boot times,
    take percentiles. Not every resample necessarily crosses 1.0 within the
    observed window -- pct_reached reports how often it did, and the CI is only
    computed from the replicates that did reach it (a replicate that never
    reaches payback doesn't have a "payback month" to include in the interval).

    Implementation note: this resamples from a precomputed (customer x tenure_month)
    matrix of cumulative margin via plain numpy indexing, not by rebuilding a
    resampled DataFrame per replicate -- rebuilding the panel per bootstrap
    iteration is the obvious way to write this and is roughly two orders of
    magnitude slower on real dataset sizes (thousands of customers), which
    matters when this needs to run per segment inside a live Streamlit app.
    """
    rng = np.random.default_rng(seed)
    df = panel.sort_values(["customer_id", "tenure_month"]).copy()
    df["margin"] = df["mrr"] * gross_margin
    df["cum_margin"] = df.groupby("customer_id")["margin"].cumsum()

    rows = []
    for segment_value in sorted(customers[segment_col].dropna().unique()):
        seg_customers = customers[customers[segment_col] == segment_value]
        cust_ids = seg_customers["customer_id"].to_numpy()
        n = len(cust_ids)
        if n == 0:
            continue
        cac_by_customer = seg_customers.set_index("customer_id")["cac"].reindex(cust_ids).to_numpy()

        sub = df[df[segment_col] == segment_value]
        pivot = sub.pivot(index="customer_id", columns="tenure_month", values="cum_margin").reindex(cust_ids)
        tenure_months = pivot.columns.to_numpy()
        mat = pivot.to_numpy()

        boot_paybacks = np.full(n_boot, np.nan)
        idx_matrix = rng.integers(0, n, size=(n_boot, n))
        for b in range(n_boot):
            idx = idx_matrix[b]
            sample_mat = mat[idx]
            sample_cac = cac_by_customer[idx].mean()
            coverage = np.mean(~np.isnan(sample_mat), axis=0)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                # a resample can by chance include zero customers observed at a
                # given tenure_month (all-NaN column) -- nanmean on that column
                # is a real NaN, correctly excluded below by the coverage check,
                # not a bug, so the "empty slice" warning is silenced here.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_cum = np.nanmean(sample_mat, axis=0)
                ratio = avg_cum / sample_cac
            valid = coverage >= MIN_COHORT_COVERAGE
            reached = np.where(valid & (ratio >= 1.0))[0]
            if len(reached):
                boot_paybacks[b] = tenure_months[reached[0]]

        pct_reached = float(np.mean(~np.isnan(boot_paybacks)))
        reached_vals = boot_paybacks[~np.isnan(boot_paybacks)]
        min_replicates_for_ci = max(5, int(0.1 * n_boot))
        if len(reached_vals) >= min_replicates_for_ci:
            lo, hi = np.percentile(reached_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            lo, hi = np.nan, np.nan

        rows.append({
            segment_col: segment_value,
            "payback_ci_lo": lo,
            "payback_ci_hi": hi,
            "pct_bootstrap_reached_payback": round(pct_reached, 3),
            "n_bootstrap": n_boot,
        })

    return pd.DataFrame(rows)


def bootstrap_payback_curve_ci(panel: pd.DataFrame, customers: pd.DataFrame,
                                segment_col: str = "acquisition_channel",
                                gross_margin: float = DEFAULT_GROSS_MARGIN,
                                n_boot: int = 300, seed: int = 42, alpha: float = 0.05) -> dict:
    """
    Like bootstrap_payback_ci, but returns a CI band around the payback-ratio
    *curve itself* at every tenure month, not just around the single crossing
    point -- for shading a confidence band on the payback chart (Phase 5),
    the same way the Kaplan-Meier curve already shows one (Phase 3). Reuses
    the identical resampling matrix as bootstrap_payback_ci; the two are kept
    as separate functions rather than one doing double duty, since "the
    distribution of the crossing month" and "the distribution of the ratio at
    each month" are different questions with different NaN-handling (the
    former only counts replicates that crossed; the latter has a value at
    every month for every replicate, regardless of whether that replicate
    ever crossed 1.0).

    Returns {segment_value: DataFrame[tenure_month, ratio_lo, ratio_mid, ratio_hi]}.
    """
    rng = np.random.default_rng(seed)
    df = panel.sort_values(["customer_id", "tenure_month"]).copy()
    df["margin"] = df["mrr"] * gross_margin
    df["cum_margin"] = df.groupby("customer_id")["margin"].cumsum()

    out = {}
    for segment_value in sorted(customers[segment_col].dropna().unique()):
        seg_customers = customers[customers[segment_col] == segment_value]
        cust_ids = seg_customers["customer_id"].to_numpy()
        n = len(cust_ids)
        if n == 0:
            continue
        cac_by_customer = seg_customers.set_index("customer_id")["cac"].reindex(cust_ids).to_numpy()

        sub = df[df[segment_col] == segment_value]
        pivot = sub.pivot(index="customer_id", columns="tenure_month", values="cum_margin").reindex(cust_ids)
        tenure_months = pivot.columns.to_numpy()
        mat = pivot.to_numpy()

        idx_matrix = rng.integers(0, n, size=(n_boot, n))
        boot_ratios = np.full((n_boot, len(tenure_months)), np.nan)
        for b in range(n_boot):
            idx = idx_matrix[b]
            sample_mat = mat[idx]
            sample_cac = cac_by_customer[idx].mean()
            coverage = np.mean(~np.isnan(sample_mat), axis=0)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_cum = np.nanmean(sample_mat, axis=0)
                ratio = avg_cum / sample_cac
            ratio = np.where(coverage >= MIN_COHORT_COVERAGE, ratio, np.nan)
            boot_ratios[b] = ratio

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN column beyond every replicate's coverage
            lo = np.nanpercentile(boot_ratios, 100 * alpha / 2, axis=0)
            mid = np.nanpercentile(boot_ratios, 50, axis=0)
            hi = np.nanpercentile(boot_ratios, 100 * (1 - alpha / 2), axis=0)

        out[segment_value] = pd.DataFrame({
            "tenure_month": tenure_months, "ratio_lo": lo, "ratio_mid": mid, "ratio_hi": hi,
        }).dropna()

    return out


# ---------------------------------------------------------------------------
# 3. LTV and LTV:CAC
# ---------------------------------------------------------------------------

def ltv_by_channel(panel: pd.DataFrame, customers: pd.DataFrame,
                    gross_margin: float = DEFAULT_GROSS_MARGIN,
                    segment_col: str = "acquisition_channel") -> pd.DataFrame:
    """
    LTV(segment) = expected_active_months(segment) * ARPA(segment) * gross_margin,
    where ARPA is the average MRR among a segment's *active* customer-months
    (so upsell/downgrade noise is reflected but churned $0 months are excluded).

    expected_active_months comes from a parametric survival model (shifted-beta-
    geometric or Weibull hazard, whichever fits better by log-likelihood) fit to
    every customer's actual (duration, censored?) pair via engine.survival --
    see that module's docstring. This replaces an earlier version that
    extrapolated a hazard rate off the last 3 points of the empirical retention
    curve, which blew up to 700x+ LTV:CAC on real, noisy data because a flat
    noisy tail implies a near-zero hazard. A closed-form survival function
    doesn't have that failure mode; see engine/survival.py and Phase 3 in the
    README for the validation this fit went through before being trusted here.

    expected_active_months and ltv are BOUNDED estimates: computed strictly
    through extrapolation_horizon_months (3x the segment's own observed max
    tenure -- see EXTRAPOLATION_CAP_MULTIPLE), not projected to infinity.
    Report them as-is; they're the defensible value through that horizon, not
    a guess at full lifetime value.

    Two separate confidence signals, deliberately not merged into one:
      fit_confidence:            can the fitted curve itself be trusted at
                                  all (enough customers, not too censored)?
      extrapolation_confidence:  has the curve converged by the capped
                                  horizon, or does the bounded estimate above
                                  likely understate true lifetime value
                                  because there's real tail left uncounted?
    A segment can be fit_confidence="ok" and extrapolation_confidence=
    "low_confidence" at the same time -- that's not a contradiction, it means
    "trust this number through month X, and we're deliberately not claiming
    further than that."
    """
    rows = []
    for segment_value in sorted(customers[segment_col].dropna().unique()):
        sub = panel[panel[segment_col] == segment_value]
        durations = survival.extract_durations(sub)
        fit = survival.fit_best_parametric(
            durations["duration"].to_numpy(), durations["event_observed"].to_numpy()
        )
        max_observed_tenure = int(sub["tenure_month"].max())
        exp = survival.expected_active_periods(fit["survival_fn"], max_observed_tenure)
        censoring_fraction = 1 - durations["event_observed"].mean()
        arpa = sub.loc[sub["is_active"], "mrr"].mean()
        ltv = exp["value"] * arpa * gross_margin
        rows.append({
            segment_col: segment_value,
            "expected_active_months": round(exp["value"], 2),
            "extrapolation_horizon_months": exp["horizon_used"],
            "arpa": round(arpa, 2),
            "ltv": round(ltv, 2),
            "survival_model": fit["model"],
            "fit_confidence": survival.fit_confidence(len(durations), censoring_fraction),
            "extrapolation_confidence": survival.extrapolation_confidence(exp["survival_at_horizon"]),
        })
    return pd.DataFrame(rows)


def ltv_to_cac_by_channel(panel: pd.DataFrame, customers: pd.DataFrame,
                           gross_margin: float = DEFAULT_GROSS_MARGIN,
                           segment_col: str = "acquisition_channel") -> pd.DataFrame:
    ltv = ltv_by_channel(panel, customers, gross_margin, segment_col)
    cac = customers.groupby(segment_col)["cac"].mean().rename("avg_cac").reset_index()
    out = ltv.merge(cac, on=segment_col)
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


# ---------------------------------------------------------------------------
# 5. Monte Carlo sensitivity on the payback period
# ---------------------------------------------------------------------------

def monte_carlo_payback_sensitivity(panel: pd.DataFrame, customers: pd.DataFrame,
                                     segment_col: str = "acquisition_channel",
                                     gross_margin: float = DEFAULT_GROSS_MARGIN,
                                     cac_shock_pct: float = 20.0, retention_shift_pts: float = 5.0,
                                     n_sims: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Simulates the distribution of the payback month under uncertainty in CAC
    and retention, rather than recomputing one or two fixed scenarios.

    Each simulation draws one CAC shock (uniform in +/-cac_shock_pct%) and one
    retention shift (uniform in +/-retention_shift_pts percentage points), and
    the SAME n_sims draws are reused across every segment -- not redrawn per
    segment -- so "how does paid_social's payback move under a +15% CAC shock"
    and "how does organic's" are answers to the literal same simulated
    scenario, which is what makes a cross-segment sensitivity comparison
    meaningful rather than comparing two differently-randomized processes.

    Mechanically this reuses Phase 3's coverage-filtered baseline curve
    (_avg_cumulative_margin_by_tenure) rather than re-touching the raw panel
    per simulation: CAC is shocked multiplicatively, and the retention shift
    is applied as an ADDITIVE dollar adjustment to cumulative margin --
    shift_pts/100 * t * ARPA * gross_margin -- on the reasoning that X points
    more (or fewer) of the customer base being active adds roughly X% of
    ARPA's worth of extra revenue each month, compounding linearly with
    elapsed tenure t.

    An earlier version of this instead scaled cumulative margin by the ratio
    shifted_retention(t) / observed_retention(t). That's wrong two ways: (1)
    it divides by zero whenever a segment's retention curve hits exactly 0%
    at some tenure month -- a real, valid state (an early micro-cohort fully
    churned by month N), not an edge case -- which silently turned every
    simulation into a false "payback reached" at that month; caught because
    a zero-shock run didn't reproduce cac_payback_by_channel's point estimate
    exactly, which it must (see tests/test_monte_carlo.py). (2) more
    fundamentally, cumulative margin is a STOCK (it includes revenue already
    banked by customers who've since churned) while retention_pct(t) is an
    INSTANTANEOUS rate, so scaling one by the other conflates two different
    kinds of quantity even before the division blows up. The additive
    ARPA-based adjustment avoids both problems: it's always finite, and it's
    scaling a dollar amount by a dollar rate instead of by a percentage.
    perturbed_margin is floored at 0 (a large enough negative shift can't
    imply negative revenue).

    The baseline curve is already coverage-filtered by MIN_COHORT_COVERAGE
    (see Phase 0), so tenure months without enough observed customers are
    already excluded from the simulation, consistent with the payback and
    LTV logic elsewhere in this module.

    fit_confidence carries forward engine.survival's data-sufficiency signal
    (same n_customers/censoring_fraction check LTV uses) -- a segment that's
    already too thin or too censored to trust at the point-estimate level
    doesn't get to look more precise just because it's now a distribution;
    the flag says so explicitly rather than presenting a falsely precise
    histogram on top of an already-shaky base.

    Returns (summary_df, distributions) where distributions maps segment
    value -> the raw array of n_sims simulated payback months (NaN for
    simulations that never crossed breakeven), for plotting.
    """
    rng = np.random.default_rng(seed)
    cac_shocks = rng.uniform(-cac_shock_pct / 100, cac_shock_pct / 100, size=n_sims)
    retention_shifts = rng.uniform(-retention_shift_pts, retention_shift_pts, size=n_sims)

    rows = []
    distributions = {}
    for segment_value in sorted(customers[segment_col].dropna().unique()):
        seg_customers = customers[customers[segment_col] == segment_value]
        seg_panel = panel[panel[segment_col] == segment_value]
        avg_cac = seg_customers["cac"].mean()
        arpa = seg_panel.loc[seg_panel["is_active"], "mrr"].mean()

        margin_curve = _avg_cumulative_margin_by_tenure(seg_panel, gross_margin)

        durations = survival.extract_durations(seg_panel)
        censoring_fraction = 1 - durations["event_observed"].mean()
        fit_conf = survival.fit_confidence(len(durations), censoring_fraction)

        if len(margin_curve) == 0 or pd.isna(arpa):
            rows.append({
                segment_col: segment_value, "n_customers": len(seg_customers),
                "pct_sims_reached_payback": np.nan, "median_payback_month": np.nan,
                "p10_payback_month": np.nan, "p90_payback_month": np.nan,
                "fit_confidence": fit_conf,
            })
            distributions[segment_value] = np.full(n_sims, np.nan)
            continue

        tenure_months = margin_curve["tenure_month"].to_numpy()
        base_margin = margin_curve["avg_cum_margin"].to_numpy()

        shocked_cac = avg_cac * (1 + cac_shocks)                                                          # (n_sims,)
        dollar_adjustment = (retention_shifts[:, None] / 100) * tenure_months[None, :] * arpa * gross_margin  # (n_sims, n_t)
        perturbed_margin = np.clip(base_margin[None, :] + dollar_adjustment, 0, None)                       # (n_sims, n_t)
        ratio = perturbed_margin / shocked_cac[:, None]                                                      # (n_sims, n_t)

        reached_mask = ratio >= 1.0
        any_reached = reached_mask.any(axis=1)
        first_idx = np.argmax(reached_mask, axis=1)  # garbage where any_reached is False, discarded below
        sim_payback = np.where(any_reached, tenure_months[first_idx], np.nan)

        reached_vals = sim_payback[any_reached]
        rows.append({
            segment_col: segment_value,
            "n_customers": len(seg_customers),
            "pct_sims_reached_payback": round(float(any_reached.mean()), 3),
            "median_payback_month": float(np.median(reached_vals)) if len(reached_vals) else np.nan,
            "p10_payback_month": float(np.percentile(reached_vals, 10)) if len(reached_vals) else np.nan,
            "p90_payback_month": float(np.percentile(reached_vals, 90)) if len(reached_vals) else np.nan,
            "fit_confidence": fit_conf,
        })
        distributions[segment_value] = sim_payback

    return pd.DataFrame(rows), distributions
