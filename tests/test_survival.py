"""
Unit tests for Phase 3's survival module.

test_kaplan_meier_matches_textbook_example is a classic hand-worked KM example
(6 subjects, one censored) verified by hand in the module docstring / PR notes:
  durations = [2, 3, 5, 5, 8, 9], event_observed = [T, T, T, T, F, T]
  S(2) = 5/6
  S(3) = 5/6 * 4/5
  S(5) = 5/6 * 4/5 * 2/4   (two events tie at t=5)
  S(9) = ... * 0            (last subject at risk churns)

test_fit_recovers_known_parameters generates synthetic data from a shifted-beta-
geometric with KNOWN alpha/beta, fits it back, and checks the recovered
parameters are close -- this is the standard way to validate an MLE
implementation (the only way you actually know a fit is correct is to give it
data where you already know the right answer) before trusting it on noisy real
data. This must pass before Phase 3 relies on fit_sbg/fit_weibull for anything.
"""

import numpy as np
import pytest

from engine.loaders import build_customer_panel, load_customers, load_subscription_events
from engine.survival import (
    bootstrap_km_ci,
    extrapolation_confidence,
    fit_confidence,
    expected_active_periods,
    extract_durations,
    fit_best_parametric,
    fit_sbg,
    fit_weibull,
    kaplan_meier,
    sbg_survival,
)


def test_kaplan_meier_matches_textbook_example():
    durations = np.array([2, 3, 5, 5, 8, 9])
    event_observed = np.array([True, True, True, True, False, True])
    curve = kaplan_meier(durations, event_observed).set_index("t")

    assert curve.loc[2, "survival_prob"] == pytest.approx(5 / 6)
    assert curve.loc[3, "survival_prob"] == pytest.approx((5 / 6) * (4 / 5))
    assert curve.loc[5, "survival_prob"] == pytest.approx((5 / 6) * (4 / 5) * (2 / 4))
    assert curve.loc[9, "survival_prob"] == pytest.approx(0.0)
    # censoring at t=8 shouldn't appear as an event row and shouldn't move survival
    assert 8 not in curve.index


def test_bootstrap_km_ci_contains_point_estimate_and_widens_with_time():
    """
    Sanity checks that don't depend on exact bootstrap values (which are
    stochastic by nature): the point estimate itself must fall inside its own
    CI at every grid point, and ci_lo <= ci_hi always.

    Width should generally grow with t as the risk set thins -- but only
    while survival is still meaningfully away from 0. Bounded proportions get
    *less* uncertain near a boundary (a curve that's already collapsed to
    ~0% retained can't have much variance left either), so a naive "no churn
    ever, straight to a hard cutoff" simulation drives survival to 0 quickly
    and the width shrinks again at the tail -- not a bug, just the wrong
    regime to test the "thinning risk set" property in. Real censored data
    (staggered entry / administrative cutoffs, like Telco's still-active
    customers) keeps a meaningful risk set alive without survival collapsing,
    which is what actually produces the widening this test checks for, so
    that's what's simulated here instead of pure churn-to-zero.
    """
    rng = np.random.default_rng(11)
    n = 800
    p_i = rng.beta(8.0, 1.5, size=n)  # slow, gradual churn
    max_true = 60
    true_churn = np.zeros(n, dtype=int)
    for i in range(n):
        t = 1
        while t <= max_true and rng.random() < p_i[i]:
            t += 1
        true_churn[i] = t
    follow_up = rng.integers(5, 26, size=n)  # staggered entry -> administrative right-censoring
    durations = np.minimum(true_churn, follow_up)
    event_observed = true_churn <= follow_up

    t_grid = np.array([1, 5, 10, 15, 20, 25])
    result = bootstrap_km_ci(durations, event_observed, t_grid, n_boot=200, seed=5)

    assert (result["ci_lo"] <= result["survival_prob"]).all()
    assert (result["survival_prob"] <= result["ci_hi"]).all()
    assert (result["ci_lo"] <= result["ci_hi"]).all()

    width = result["ci_hi"] - result["ci_lo"]
    assert width.iloc[-1] > width.iloc[0]  # wider at t=25 (thin, censored risk set) than t=1 (nearly everyone)


CUSTOMERS_CSV = """customer_id,signup_date,acquisition_channel,initial_plan,cac
C1,2024-01-01,A,Basic,100
C2,2024-01-01,A,Basic,100
C3,2024-01-01,A,Basic,100
"""
EVENTS_CSV = """customer_id,month,mrr
C1,2024-01-01,50
C1,2024-02-01,50
C1,2024-03-01,0
C2,2024-01-01,50
C2,2024-02-01,0
C3,2024-01-01,50
C3,2024-02-01,50
C3,2024-03-01,50
"""


def test_extract_durations(tmp_path):
    cust_path = tmp_path / "customers.csv"
    events_path = tmp_path / "subscription_events.csv"
    cust_path.write_text(CUSTOMERS_CSV)
    events_path.write_text(EVENTS_CSV)
    customers = load_customers(cust_path)
    events = load_subscription_events(events_path)
    panel = build_customer_panel(customers, events)

    d = extract_durations(panel).set_index("customer_id")
    assert d.loc["C1", "duration"] == 2 and d.loc["C1", "event_observed"] == True   # noqa: E712
    assert d.loc["C2", "duration"] == 1 and d.loc["C2", "event_observed"] == True   # noqa: E712
    assert d.loc["C3", "duration"] == 3 and d.loc["C3", "event_observed"] == False  # noqa: E712 (censored)


def test_fit_recovers_known_parameters():
    """Generate data from a known sBG, confirm the fitter recovers it. Validates
    the MLE machinery itself before it's ever pointed at real data."""
    true_alpha, true_beta = 1.0, 3.0
    rng = np.random.default_rng(123)
    n = 4000
    max_t = 30

    p_i = rng.beta(true_alpha, true_beta, size=n)  # each customer's own constant churn prob
    durations = np.zeros(n, dtype=int)
    event_observed = np.zeros(n, dtype=bool)
    for i in range(n):
        t = 1
        while t <= max_t and rng.random() < p_i[i]:
            t += 1
        if t <= max_t:
            durations[i] = t
            event_observed[i] = True
        else:
            durations[i] = max_t
            event_observed[i] = False

    fit = fit_sbg(durations, event_observed)
    assert fit["converged"]
    # Loose tolerance -- this is a stochastic simulation, not exact arithmetic.
    assert fit["params"]["alpha"] == pytest.approx(true_alpha, rel=0.35)
    assert fit["params"]["beta"] == pytest.approx(true_beta, rel=0.35)

    # The fitted survival curve should track the true one reasonably closely.
    t_check = np.array([1, 5, 10, 20])
    true_s = sbg_survival(t_check, true_alpha, true_beta)
    fit_s = fit["survival_fn"](t_check)
    assert np.allclose(fit_s, true_s, atol=0.08)


def test_weibull_fit_converges_on_its_own_generating_process():
    true_k, true_lam = 0.6, 8.0  # k<1 -> decreasing hazard, the usual churn shape
    rng = np.random.default_rng(7)
    n = 4000
    raw = rng.weibull(true_k, size=n) * true_lam
    max_t = 40
    durations = np.clip(np.ceil(raw), 1, max_t).astype(int)
    event_observed = raw <= max_t

    fit = fit_weibull(durations, event_observed)
    assert fit["converged"]
    assert fit["params"]["k"] == pytest.approx(true_k, rel=0.35)
    assert fit["params"]["lambda"] == pytest.approx(true_lam, rel=0.35)


def test_expected_active_periods_is_capped_at_3x_observed_tenure():
    """A steep, well-converged curve: cap shouldn't bind, and survival_at_horizon should be ~0."""
    fn = lambda t: sbg_survival(t, 1.0, 3.0)
    result = expected_active_periods(fn, max_observed_tenure=20)
    assert result["horizon_used"] == 60  # 3x the 3-constant cap * 20 months observed
    assert np.isfinite(result["value"])
    assert 0 < result["value"] < 61
    assert result["survival_at_horizon"] < 0.01


def test_expected_active_periods_flags_unconverged_tail_at_cap():
    """A near-flat, barely-decaying curve: the cap should bind and leave visible residual survival."""
    fn = lambda t: sbg_survival(t, 20.0, 0.3)  # alpha >> beta -> very slow decay, long "immortal" tail
    result = expected_active_periods(fn, max_observed_tenure=10)
    assert result["horizon_used"] == 30
    assert result["survival_at_horizon"] > 0.3  # still far from converged when the cap hits


def test_fit_best_parametric_reports_the_correct_alternative():
    """
    Regression test: an earlier version compared `best is sbg` AFTER already
    reassigning `best = dict(best)`, which breaks object identity, so the
    "alternative" model/loglik was silently wrong regardless of which model
    actually won. Use data clearly better suited to one model (a strong
    early-heavy churn shape with a long flat tail, textbook sBG) to make sure
    the winner and the reported alternative are both correct.
    """
    rng = np.random.default_rng(99)
    n = 3000
    true_alpha, true_beta = 0.8, 2.5
    p_i = rng.beta(true_alpha, true_beta, size=n)
    max_t = 24
    durations = np.zeros(n, dtype=int)
    event_observed = np.zeros(n, dtype=bool)
    for i in range(n):
        t = 1
        while t <= max_t and rng.random() < p_i[i]:
            t += 1
        if t <= max_t:
            durations[i], event_observed[i] = t, True
        else:
            durations[i], event_observed[i] = max_t, False

    sbg = fit_sbg(durations, event_observed)
    weib = fit_weibull(durations, event_observed)
    best = fit_best_parametric(durations, event_observed)

    winner_is_sbg = sbg["loglik"] >= weib["loglik"]
    assert best["model"] == ("shifted-beta-geometric" if winner_is_sbg else "weibull")
    expected_alt_loglik = weib["loglik"] if winner_is_sbg else sbg["loglik"]
    assert best["alternative_loglik"] == pytest.approx(expected_alt_loglik)
    assert best["alternative_loglik"] != pytest.approx(best["loglik"])


def test_fit_confidence_thresholds():
    assert fit_confidence(n_customers=10, censoring_fraction=0.1) == "low_confidence"  # too few
    assert fit_confidence(n_customers=1000, censoring_fraction=0.9) == "low_confidence"  # too censored
    assert fit_confidence(n_customers=1000, censoring_fraction=0.1) == "ok"


def test_extrapolation_confidence_thresholds():
    """
    Independent of fit_confidence: a fit can be well-supported by plenty of
    data and still leave a meaningful uncounted tail at the capped horizon.
    """
    assert extrapolation_confidence(survival_at_horizon=0.36) == "low_confidence"  # content_seo's real case
    assert extrapolation_confidence(survival_at_horizon=0.02) == "ok"              # curve had converged by the cap
