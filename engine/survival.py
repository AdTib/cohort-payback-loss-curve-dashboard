"""
Phase 3: proper survival analysis, replacing two things from earlier phases:

  1. The cohort-month retention view for datasets like Telco, where the signup
     date had to be reconstructed from tenure (see Phase 0). Bucketing those
     customers into fake monthly cohorts bakes in survivorship bias -- an old
     cohort can only exist because its members already lived that long. The
     fix isn't a better bucketing scheme, it's not bucketing at all: Kaplan-Meier
     estimates one pooled survival curve directly from (duration, censored?)
     pairs, and handles right-censoring correctly by construction.

  2. The old expected_active_months in engine/metrics.py, which extrapolated
     LTV by fitting a hazard rate to the last 3 observed retention points and
     projecting it forward geometrically. Fine on smooth synthetic curves,
     but it blew up to 700x+ LTV:CAC on real, noisy Online Retail II data
     (documented in metrics.py's git history / Phase 1 report) because a
     near-flat noisy tail implies a near-zero hazard, which compounds into a
     multi-decade extrapolation. The fix: fit a real parametric survival
     model (shifted-beta-geometric or Weibull hazard) by maximum likelihood
     against every customer's actual (duration, censored?) pair, not just the
     last few points of an aggregate curve. A closed-form survival function
     doesn't care how noisy the tail of the empirical curve looks.

Both pieces consume the same primitive: one (duration, event_observed) pair
per customer, extracted from the panel engine.loaders.build_customer_panel
produces. That means this module works on any dataset the engine already
supports (synthetic demo, Telco, Retail, a real prospect's export) -- nothing
here is Telco-specific or Retail-specific, only which function you reach for
depends on how you want to use it (see engine/metrics.py's Phase 3 wiring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MIN_CUSTOMERS_FOR_PARAMETRIC_FIT = 30   # below this, an MLE fit is unstable -- flag, don't fake precision
MAX_CENSORING_FOR_HIGH_CONFIDENCE = 0.70  # if >70% of a segment is still right-censored, flag as low-confidence
ABSOLUTE_MAX_HORIZON_MONTHS = 240       # hard safety-net cap, independent of any segment's own data

# A fitted curve is only asked to extrapolate a bounded multiple of what it
# actually observed. This is the second time an unbounded extrapolation has
# produced an indefensible number (see Phase 1's tail-hazard blowup, then this
# module's own first version: shifted-beta-geometric fit content_seo's real
# 23-month observation window and, left uncapped, projected 22% of customers
# still retained after 20 years -- a mathematically valid consequence of the
# fitted curve, and not a number anyone could defend out loud). The pattern
# both times was the same: nothing in the code stopped a model from being
# trusted arbitrarily far past the data that constrained it. EXTRAPOLATION_CAP_MULTIPLE
# fixes the horizon at a small, statable multiple of each segment's own
# observed max tenure instead of a single global constant, so "how far are we
# extrapolating" is always relative to what that specific segment actually showed.
EXTRAPOLATION_CAP_MULTIPLE = 3

# If the fitted survival curve still hasn't decayed below this by the capped
# horizon, a meaningful share of the LTV estimate is coming from unobserved
# extrapolation rather than data -- flagged low-confidence rather than shown
# at face value, on top of the sample-size and censoring conditions.
MAX_RESIDUAL_SURVIVAL_AT_CAP = 0.05


# ---------------------------------------------------------------------------
# Duration extraction -- the shared input to everything below
# ---------------------------------------------------------------------------

def extract_durations(panel: pd.DataFrame) -> pd.DataFrame:
    """
    One row per customer: duration = number of periods retained before churn
    (their first inactive tenure_month), event_observed = True if that churn
    was actually observed in the data, False if the customer is still active
    at the last tenure_month we have for them (right-censored -- we don't
    know when, or if, they'll eventually churn).

    Convention: a customer active at tenure_month 0..2 and inactive from 3
    onward has duration=3, event_observed=True ("retained for 3 periods").
    A customer active through their entire observed window (last tenure_month
    = T) has duration=T+1, event_observed=False.
    """
    p = panel.sort_values(["customer_id", "tenure_month"])

    last_t = p.groupby("customer_id")["tenure_month"].max().rename("last_t")
    last_active = (
        p.merge(last_t, on="customer_id")
        .query("tenure_month == last_t")
        .set_index("customer_id")["is_active"]
    )

    inactive_rows = p[~p["is_active"]]
    first_inactive = inactive_rows.groupby("customer_id")["tenure_month"].min()

    out = pd.DataFrame({"last_t": last_t, "still_active_at_end": last_active})
    out["duration"] = out.index.map(first_inactive).astype("Float64")
    out["event_observed"] = out["duration"].notna()
    # right-censored: never went inactive (or reactivated past their last inactive
    # spell in blended mode) -- duration is "at least this many periods"
    censored_mask = out["still_active_at_end"]
    out.loc[censored_mask, "duration"] = out.loc[censored_mask, "last_t"] + 1
    out.loc[censored_mask, "event_observed"] = False
    out["duration"] = out["duration"].astype(int)

    return out.reset_index()[["customer_id", "duration", "event_observed"]]


# ---------------------------------------------------------------------------
# Kaplan-Meier -- the Telco / "no reliable cohort" validation case
# ---------------------------------------------------------------------------

def kaplan_meier(durations: np.ndarray, event_observed: np.ndarray) -> pd.DataFrame:
    """
    Standard Kaplan-Meier estimator. Returns one row per distinct event time t
    with survival_prob = S(t) = P(retained beyond t). S(0) = 1 is implicit,
    not included as a row. n_at_risk/n_events are kept for transparency and
    for the low-confidence flag (see confidence_flag).
    """
    durations = np.asarray(durations)
    event_observed = np.asarray(event_observed, dtype=bool)

    event_times = np.sort(np.unique(durations[event_observed]))
    survival = 1.0
    rows = []
    for t in event_times:
        n_at_risk = int((durations >= t).sum())
        n_events = int(((durations == t) & event_observed).sum())
        if n_at_risk == 0:
            continue
        survival *= (1 - n_events / n_at_risk)
        rows.append({"t": int(t), "survival_prob": survival, "n_at_risk": n_at_risk, "n_events": n_events})
    return pd.DataFrame(rows, columns=["t", "survival_prob", "n_at_risk", "n_events"])


def _survival_at(km_curve: pd.DataFrame, t_grid: np.ndarray) -> np.ndarray:
    """Step-function lookup: S(t) for arbitrary t, holding the last known value."""
    if len(km_curve) == 0:
        return np.ones(len(t_grid))
    ts = km_curve["t"].to_numpy()
    ss = km_curve["survival_prob"].to_numpy()
    idx = np.searchsorted(ts, t_grid, side="right") - 1
    out = np.where(idx < 0, 1.0, ss[np.clip(idx, 0, len(ss) - 1)])
    return out


def bootstrap_km_ci(durations: np.ndarray, event_observed: np.ndarray,
                     t_grid: np.ndarray, n_boot: int = 400, seed: int = 42,
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    Nonparametric bootstrap CI for the KM curve: resample customers with
    replacement, recompute KM, repeat n_boot times, take percentiles at each
    t in t_grid. This is the standard way to get a CI around a KM estimate
    without assuming a parametric form for the curve itself.
    """
    durations = np.asarray(durations)
    event_observed = np.asarray(event_observed, dtype=bool)
    n = len(durations)
    rng = np.random.default_rng(seed)

    boot_curves = np.empty((n_boot, len(t_grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        curve = kaplan_meier(durations[idx], event_observed[idx])
        boot_curves[b] = _survival_at(curve, t_grid)

    lo = np.percentile(boot_curves, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_curves, 100 * (1 - alpha / 2), axis=0)
    point = _survival_at(kaplan_meier(durations, event_observed), t_grid)
    return pd.DataFrame({"t": t_grid, "survival_prob": point, "ci_lo": lo, "ci_hi": hi})


# ---------------------------------------------------------------------------
# Parametric fits -- shifted-beta-geometric and Weibull hazard
# ---------------------------------------------------------------------------

def sbg_survival(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Shifted-beta-geometric survival function (Fader & Hardie). Models each
    customer as having their own constant per-period *retention* probability
    theta, with theta varying across customers as Beta(alpha, beta) --
    population-level heterogeneity in a simple constant-hazard-per-individual
    model. S(t) = P(a customer survives past period t) = E_theta[theta^t]
    = B(alpha+t, beta) / B(alpha, beta), which expands to the closed form
    below: S(0)=1, S(t) = product_{i=1}^{t} (alpha + i - 1) / (alpha + beta + i - 1).
    This product form is numerically stable for the tenure ranges here
    (avoids needing the Beta function / potential overflow in a factorial form).

    Verified against scipy.integrate.quad's numerical E[theta^t] and against
    a synthetic dataset simulated directly from a known (alpha, beta) --
    see tests/test_survival.py -- because an earlier version of this formula
    had alpha and beta swapped, which fit silently but recovered the wrong
    parameters. Get the closed form wrong and the MLE still "converges", it
    just converges to nonsense -- which is exactly why a fit is only trusted
    here after it's demonstrated to recover known parameters from simulated data.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.ones_like(t)
    max_t = int(t.max()) if len(t) else 0
    factor = 1.0
    factors_by_i = {0: 1.0}
    for i in range(1, max_t + 1):
        factor *= (alpha + i - 1) / (alpha + beta + i - 1)
        factors_by_i[i] = factor
    return np.array([factors_by_i[int(ti)] for ti in t])


def weibull_survival(t: np.ndarray, k: float, lam: float) -> np.ndarray:
    """Discretized Weibull survival: S(t) = exp(-(t/lam)^k), evaluated at integer months."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return np.exp(-np.power(np.clip(t, 0, None) / lam, k))


def _discrete_loglik(survival_fn, params, durations, event_observed) -> float:
    """
    Discrete-time log-likelihood shared by both parametric models:
    P(churn exactly at duration d) = S(d-1) - S(d) for an observed event,
    P(survive at least to duration d) = S(d) for a censored observation.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(event_observed, dtype=bool)

    s_d = survival_fn(d, *params)
    s_d_minus_1 = survival_fn(np.clip(d - 1, 0, None), *params)
    s_d_minus_1 = np.where(d == 0, 1.0, s_d_minus_1)

    p_event = np.clip(s_d_minus_1 - s_d, 1e-12, None)
    p_censored = np.clip(s_d, 1e-12, None)

    ll = np.where(e, np.log(p_event), np.log(p_censored))
    return float(ll.sum())


def fit_sbg(durations, event_observed) -> dict:
    """MLE fit of (alpha, beta) via bounded numerical optimization."""
    def neg_ll(params):
        return -_discrete_loglik(sbg_survival, params, durations, event_observed)

    result = minimize(neg_ll, x0=[1.0, 1.0], method="L-BFGS-B", bounds=[(1e-3, 500), (1e-3, 500)])
    alpha, beta = result.x
    return {
        "model": "shifted-beta-geometric", "params": {"alpha": float(alpha), "beta": float(beta)},
        "loglik": -float(result.fun), "converged": bool(result.success),
        "survival_fn": lambda t: sbg_survival(t, alpha, beta),
    }


def fit_weibull(durations, event_observed) -> dict:
    """MLE fit of (k, lambda) via bounded numerical optimization."""
    def neg_ll(params):
        return -_discrete_loglik(weibull_survival, params, durations, event_observed)

    result = minimize(neg_ll, x0=[1.0, 10.0], method="L-BFGS-B", bounds=[(1e-3, 50), (1e-3, 2000)])
    k, lam = result.x
    return {
        "model": "weibull", "params": {"k": float(k), "lambda": float(lam)},
        "loglik": -float(result.fun), "converged": bool(result.success),
        "survival_fn": lambda t: weibull_survival(t, k, lam),
    }


def fit_best_parametric(durations, event_observed) -> dict:
    """
    Fits both candidate models and keeps whichever has the higher log-likelihood.
    Both have 2 free parameters, so comparing raw log-likelihood is equivalent
    to comparing AIC here (the parameter-count penalty is identical either way).
    """
    sbg = fit_sbg(durations, event_observed)
    weib = fit_weibull(durations, event_observed)
    sbg_wins = sbg["loglik"] >= weib["loglik"]
    best = dict(sbg if sbg_wins else weib)
    best["alternative_loglik"] = weib["loglik"] if sbg_wins else sbg["loglik"]
    best["alternative_model"] = "weibull" if sbg_wins else "shifted-beta-geometric"
    return best


def expected_active_periods(survival_fn, max_observed_tenure: int,
                             cap_multiple: int = EXTRAPOLATION_CAP_MULTIPLE,
                             absolute_max: int = ABSOLUTE_MAX_HORIZON_MONTHS) -> dict:
    """
    Sum of a fitted survival function over 0..horizon, where horizon is capped
    at cap_multiple times the segment's own observed max tenure (never more
    than absolute_max regardless). The parametric replacement for the old
    noisy tail-hazard extrapolation, with the extrapolation itself now bounded
    -- see EXTRAPOLATION_CAP_MULTIPLE for why that cap exists at all.

    Returns a dict, not just the summed value, because "how far did we
    extrapolate and how much of the curve was still alive when we cut it off"
    is exactly what confidence_flag needs to catch a fit that's technically
    valid but not yet trustworthy:
      value:               the expected-active-periods estimate itself
      horizon_used:         the actual capped horizon (months)
      survival_at_horizon: S(horizon) -- how much of the population the model
                            still says is "active" at the cutoff. Close to 0
                            means the curve had genuinely converged by then;
                            not close to 0 means a meaningful share of `value`
                            is unconverged extrapolation, not observed signal.
    """
    horizon = min(absolute_max, cap_multiple * max(int(max_observed_tenure), 1))
    t = np.arange(0, horizon + 1)
    s = survival_fn(t)
    return {
        "value": float(np.sum(s)),
        "horizon_used": int(horizon),
        "survival_at_horizon": float(s[-1]),
    }


# ---------------------------------------------------------------------------
# Confidence flagging -- two separate questions, deliberately not one field.
#
# "low_confidence" used to conflate two different claims: (a) there isn't
# enough data to trust the fitted curve at all, versus (b) the fit is fine,
# but a bounded estimate built from it (e.g. expected_active_periods, capped
# at 3x observed tenure) doesn't cover the whole tail, so it may understate
# the true unbounded value. Those need different reactions: (a) means don't
# trust the number; (b) means trust the number *through its stated horizon*
# and don't claim further. Reporting only one flag meant a well-supported
# bounded estimate (fit_confidence=ok) got tarred with the same
# "low_confidence" label as a genuinely unreliable fit, when the honest
# statement was "here's the defensible value through month X" rather than
# "we don't know this segment's value."
# ---------------------------------------------------------------------------

def fit_confidence(n_customers: int, censoring_fraction: float) -> str:
    """
    "low_confidence" if the sample is too thin for a stable MLE fit, or too
    heavily right-censored to trust the fitted curve at all. Independent of
    how far anything built on that curve is later extrapolated -- see
    extrapolation_confidence for that separate question.
    """
    if n_customers < MIN_CUSTOMERS_FOR_PARAMETRIC_FIT:
        return "low_confidence"
    if censoring_fraction > MAX_CENSORING_FOR_HIGH_CONFIDENCE:
        return "low_confidence"
    return "ok"


def extrapolation_confidence(survival_at_horizon: float) -> str:
    """
    "low_confidence" if the fitted curve still hasn't converged by the capped
    extrapolation horizon (see EXTRAPOLATION_CAP_MULTIPLE) -- meaning a
    bounded estimate built from it is a defensible value *through that
    horizon*, but likely understates the true unbounded lifetime value, since
    there's real, uncounted survival probability beyond it. A segment can
    have plenty of data and a well-constrained fit (fit_confidence="ok") and
    still score low_confidence here -- that isn't a bad fit, it's an honest
    boundary on what a capped estimate can claim.
    """
    return "low_confidence" if survival_at_horizon > MAX_RESIDUAL_SURVIVAL_AT_CAP else "ok"
