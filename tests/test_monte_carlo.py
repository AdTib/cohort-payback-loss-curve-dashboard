"""
Unit tests for Phase 4's Monte Carlo payback sensitivity.

Reuses the same 4-customer fixture as tests/test_metrics.py (see that file's
docstring for the hand-verified numbers each channel's curve is built from):
  Channel A (C1, C2, C3): payback never reached (point estimate is <NA>)
  Channel B (C4):          payback reached at exactly tenure month 2

test_zero_shock_recovers_phase3_point_estimate is the most important test
here: with cac_shock_pct=0 and retention_shift_pts=0, every simulated draw is
identical to the unperturbed baseline, so the simulation must reproduce
cac_payback_by_channel's point estimate exactly, not approximately. An
earlier version of the retention-shift mechanism failed this test outright
(channel A came back as "100% of sims reach payback at month 2" when the true
answer is "never reaches payback") because it divided by a retention value
that was exactly 0% at one tenure month -- see the comment in
engine/metrics.py's monte_carlo_payback_sensitivity for the full story. This
test is what would catch that regression again.
"""

import numpy as np
import pandas as pd
import pytest

from engine.loaders import build_customer_panel, load_customers, load_subscription_events
from engine.metrics import cac_payback_by_channel, monte_carlo_payback_sensitivity

CUSTOMERS_CSV = """customer_id,signup_date,acquisition_channel,initial_plan,cac
C1,2024-01-01,A,Basic,250
C2,2024-01-01,A,Basic,250
C3,2024-02-01,A,Basic,250
C4,2024-01-01,B,Plus,500
"""

EVENTS_CSV = """customer_id,month,mrr
C1,2024-01-01,100
C1,2024-02-01,100
C1,2024-03-01,0
C2,2024-01-01,100
C2,2024-02-01,0
C3,2024-02-01,100
C3,2024-03-01,100
C4,2024-01-01,200
C4,2024-02-01,200
C4,2024-03-01,300
"""


@pytest.fixture
def panel(tmp_path):
    cust_path = tmp_path / "customers.csv"
    events_path = tmp_path / "subscription_events.csv"
    cust_path.write_text(CUSTOMERS_CSV)
    events_path.write_text(EVENTS_CSV)
    customers = load_customers(cust_path)
    events = load_subscription_events(events_path)
    return customers, events, build_customer_panel(customers, events)


def test_zero_shock_recovers_phase3_point_estimate(panel):
    customers, _, p = panel
    point, _ = cac_payback_by_channel(p, customers, gross_margin=1.0)
    point = point.set_index("acquisition_channel")

    summary, dists = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=0.0, retention_shift_pts=0.0, n_sims=200
    )
    summary = summary.set_index("acquisition_channel")

    # Channel A: point estimate never reaches payback -- 0% of zero-shock sims should either.
    assert pd.isna(point.loc["A", "payback_month"])
    assert summary.loc["A", "pct_sims_reached_payback"] == pytest.approx(0.0)
    assert np.isnan(dists["A"]).all()

    # Channel B: point estimate reaches payback at exactly month 2 -- every zero-shock
    # sim is a deterministic replay of the same data, so 100% should land on exactly 2.
    assert point.loc["B", "payback_month"] == 2
    assert summary.loc["B", "pct_sims_reached_payback"] == pytest.approx(1.0)
    assert (dists["B"] == 2.0).all()
    assert summary.loc["B", "median_payback_month"] == pytest.approx(2.0)
    assert summary.loc["B", "p10_payback_month"] == pytest.approx(2.0)
    assert summary.loc["B", "p90_payback_month"] == pytest.approx(2.0)


def test_simulation_is_deterministic_given_a_seed(panel):
    """Convergence/reproducibility: same seed, same inputs -> byte-identical results."""
    customers, _, p = panel
    summary1, dists1 = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=15, retention_shift_pts=5, n_sims=300, seed=7
    )
    summary2, dists2 = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=15, retention_shift_pts=5, n_sims=300, seed=7
    )
    pd.testing.assert_frame_equal(summary1, summary2)
    for key in dists1:
        np.testing.assert_array_equal(dists1[key], dists2[key])


def test_different_seeds_still_converge_to_similar_summary_statistics(panel):
    """
    Not byte-identical (different random draws), but a large-n_sims run with a
    different seed should land close to the same median/pct_reached -- if it
    didn't, the simulation wouldn't be a stable estimate of anything.
    """
    customers, _, p = panel
    summary_a, _ = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=15, retention_shift_pts=5, n_sims=3000, seed=1
    )
    summary_b, _ = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=15, retention_shift_pts=5, n_sims=3000, seed=2
    )
    row_a = summary_a.set_index("acquisition_channel").loc["B"]
    row_b = summary_b.set_index("acquisition_channel").loc["B"]
    assert row_a["pct_sims_reached_payback"] == pytest.approx(row_b["pct_sims_reached_payback"], abs=0.05)


def test_low_confidence_propagates_from_thin_segments(panel):
    """
    Both channels in this fixture are far below MIN_CUSTOMERS_FOR_PARAMETRIC_FIT
    (n=3 and n=1), so both should carry fit_confidence="low_confidence" through
    into the simulation output -- a thin segment doesn't get to look more
    precise just because the output is now a distribution instead of a point.
    """
    customers, _, p = panel
    summary, _ = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=10, retention_shift_pts=3, n_sims=200
    )
    summary = summary.set_index("acquisition_channel")
    assert summary.loc["A", "fit_confidence"] == "low_confidence"
    assert summary.loc["B", "fit_confidence"] == "low_confidence"


def test_larger_shocks_widen_the_reached_percentage_spread():
    """
    Sanity check on the mechanism's directionality using a bigger, less
    degenerate fixture: a segment sitting close to breakeven should be more
    sensitive to shocks (pct_reached should move) than doing nothing (0-shock
    always gives the same deterministic answer).
    """
    import io
    from engine.loaders import build_customer_panel as bcp, load_customers as lc, load_subscription_events as lse

    # 20 customers, cac=100, MRR=40/mo with gross_margin=1.0 -> breakeven exactly at month 2.5,
    # i.e. right on the boundary between reached-at-3 and never-reached depending on shocks.
    rows = ["customer_id,signup_date,acquisition_channel,initial_plan,cac"]
    for i in range(20):
        rows.append(f"D{i},2024-01-01,X,Basic,100")
    customers_csv = "\n".join(rows) + "\n"

    event_rows = ["customer_id,month,mrr"]
    for i in range(20):
        event_rows.append(f"D{i},2024-01-01,40")
        event_rows.append(f"D{i},2024-02-01,40")
        event_rows.append(f"D{i},2024-03-01,40")
    events_csv = "\n".join(event_rows) + "\n"

    customers = lc(io.BytesIO(customers_csv.encode()))
    events = lse(io.BytesIO(events_csv.encode()))
    p = bcp(customers, events)

    summary_no_shock, _ = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=0, retention_shift_pts=0, n_sims=200
    )
    summary_shocked, _ = monte_carlo_payback_sensitivity(
        p, customers, gross_margin=1.0, cac_shock_pct=30, retention_shift_pts=10, n_sims=200
    )
    # zero-shock is deterministic: either always reaches or never does
    assert summary_no_shock.iloc[0]["pct_sims_reached_payback"] in (0.0, 1.0)
    # a real shock range should introduce genuine variance for a segment this close to the boundary
    assert 0.0 < summary_shocked.iloc[0]["pct_sims_reached_payback"] < 1.0
