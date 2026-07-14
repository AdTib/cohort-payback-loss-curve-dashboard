"""
Unit tests against hand-calculated examples.

Fixture: 4 customers, 2 channels, 2 signup cohorts, a churn event, and one upsell
event, chosen so every metric can be verified by hand (see comments inline and the
worked calculations in README.md "Methodology" -> "Worked example").

Customers:
  C1  channel A  cohort 2024-01  cac=250   MRR: 100, 100, 0    (churns at tenure 2)
  C2  channel A  cohort 2024-01  cac=250   MRR: 100, 0         (churns at tenure 1)
  C3  channel A  cohort 2024-02  cac=250   MRR: 100, 100       (right-censored: only 2 months observed)
  C4  channel B  cohort 2024-01  cac=500   MRR: 200, 200, 300  (upsells at tenure 2)

Global max observed month is 2024-03, so every customer's grid is filled out to that
month regardless of how many actual billing rows they have.
"""

import pandas as pd
import pytest

from engine.loaders import build_customer_panel, load_customers, load_subscription_events
from engine.metrics import (
    bootstrap_payback_ci,
    bootstrap_payback_curve_ci,
    cac_payback_by_channel,
    cohort_retention_table,
    blended_retention_curve,
    grr_nrr_table,
    ltv_by_channel,
    ltv_to_cac_by_channel,
    payback_ratio_bridge,
)

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


def test_cohort_retention_table(panel):
    _, _, p = panel
    table = cohort_retention_table(p)

    # cohort 2024-01 = {C1, C2, C4}, size 3
    assert table.loc["2024-01", 0] == pytest.approx(100.0)          # all 3 active at signup
    assert table.loc["2024-01", 1] == pytest.approx(200 / 3)        # C1, C4 active; C2 churned
    assert table.loc["2024-01", 2] == pytest.approx(100 / 3)        # only C4 active

    # cohort 2024-02 = {C3}, size 1
    assert table.loc["2024-02", 0] == pytest.approx(100.0)
    assert table.loc["2024-02", 1] == pytest.approx(100.0)
    assert pd.isna(table.loc["2024-02", 2])                          # never observed (would be 2024-04)


def test_cac_payback_by_channel(panel):
    customers, _, p = panel
    summary, curves = cac_payback_by_channel(p, customers, gross_margin=1.0)

    # Channel A: avg_cum_margin = [100, 500/3, 150] vs cac=250 -> ratio never reaches 1.0
    row_a = summary.set_index("acquisition_channel").loc["A"]
    assert row_a["avg_cac"] == pytest.approx(250.0)
    assert pd.isna(row_a["payback_month"])

    curve_a = curves["A"].set_index("tenure_month")
    assert curve_a.loc[0, "avg_cum_margin"] == pytest.approx(100.0)
    assert curve_a.loc[1, "avg_cum_margin"] == pytest.approx(500 / 3)
    assert curve_a.loc[2, "avg_cum_margin"] == pytest.approx(150.0)  # only C1, C2 observed at t=2

    # Channel B: cumulative margin = [200, 400, 700] vs cac=500 -> crosses 1.0 at t=2 (700/500=1.4)
    row_b = summary.set_index("acquisition_channel").loc["B"]
    assert row_b["avg_cac"] == pytest.approx(500.0)
    assert row_b["payback_month"] == 2


def test_payback_ratio_bridge_decomposes_exactly(panel):
    """
    Channel B (cac=500, margin(2)=700, ratio=1.4) beats channel A (cac=250,
    margin(2)=150, ratio=0.6) on the bottom line despite a HIGHER CAC -- a
    good real case for the bridge, because the two effects pull in opposite
    directions and a naive "which CAC is lower" glance would get the story
    backwards.

    Swap in B's (higher) CAC under A's margin: 150/500 = 0.3 -- WORSE than A's
    actual 0.6, so cac_effect is negative (B's CAC works against it). The
    remaining gap up to B's actual 1.4 is entirely a margin/retention effect,
    and must be large enough to both cover the CAC penalty and explain B's
    full lead -- this is exact arithmetic, not a fitted or approximated
    number, so every piece is hand-checkable:
      cac_effect    = 150/500 - 150/250 = 0.3 - 0.6 = -0.3
      margin_effect = 700/500 - 150/500 = 1.4 - 0.3 =  1.1
      total_gap     = 700/500 - 150/250 = 1.4 - 0.6 =  0.8
      cac_effect + margin_effect == total_gap  (must hold exactly, by construction)
    """
    customers, _, p = panel
    _, curves = cac_payback_by_channel(p, customers, gross_margin=1.0)

    bridge = payback_ratio_bridge(curves["B"], 500.0, curves["A"], 250.0)

    assert bridge["reference_month"] == 2
    assert bridge["ratio_a"] == pytest.approx(1.4)
    assert bridge["ratio_b"] == pytest.approx(0.6)
    assert bridge["cac_effect"] == pytest.approx(-0.3)
    assert bridge["margin_effect"] == pytest.approx(1.1)
    assert bridge["total_gap"] == pytest.approx(0.8)
    assert bridge["cac_effect"] + bridge["margin_effect"] == pytest.approx(bridge["total_gap"])


def test_bootstrap_payback_curve_ci_degenerate_single_customer_segment(panel):
    """
    Same degenerate-but-correct logic as the crossing-point CI test: channel B
    has one customer, so every bootstrap resample is identical, and the curve
    band should collapse to lo == mid == hi == the exact point-estimate ratio
    (700/500=1.4 at tenure 2, see test_cac_payback_by_channel) at every month.
    """
    customers, _, p = panel
    bands = bootstrap_payback_curve_ci(p, customers, gross_margin=1.0, n_boot=100, seed=1)

    band_b = bands["B"].set_index("tenure_month")
    assert band_b.loc[2, "ratio_lo"] == pytest.approx(1.4)
    assert band_b.loc[2, "ratio_mid"] == pytest.approx(1.4)
    assert band_b.loc[2, "ratio_hi"] == pytest.approx(1.4)

    # Channel A has real customer-level variation (n=3), so the band should
    # have genuine width, and lo <= mid <= hi must hold at every month.
    band_a = bands["A"]
    assert (band_a["ratio_lo"] <= band_a["ratio_mid"]).all()
    assert (band_a["ratio_mid"] <= band_a["ratio_hi"]).all()
    assert (band_a["ratio_hi"] - band_a["ratio_lo"]).max() > 0


def test_bootstrap_payback_ci_degenerate_single_customer_segment(panel):
    """
    Channel B has exactly one customer (C4), whose hand-verified payback month
    is 2 (see test_cac_payback_by_channel). Every bootstrap resample of a
    1-customer segment just redraws that same customer, so the "confidence
    interval" has to collapse to a single point at exactly that value with
    100% of replicates reaching payback -- this is the correct degenerate
    behavior, not an edge case the function should choke on or paper over.
    Channel A never reaches payback at all (see test_cac_payback_by_channel),
    so 0% of its bootstrap replicates should reach it either, and the CI
    should be reported as unavailable (NaN) rather than a fabricated range.
    """
    customers, _, p = panel
    ci = bootstrap_payback_ci(p, customers, gross_margin=1.0, n_boot=100, seed=1).set_index("acquisition_channel")

    assert ci.loc["B", "payback_ci_lo"] == pytest.approx(2.0)
    assert ci.loc["B", "payback_ci_hi"] == pytest.approx(2.0)
    assert ci.loc["B", "pct_bootstrap_reached_payback"] == pytest.approx(1.0)

    assert ci.loc["A", "pct_bootstrap_reached_payback"] == pytest.approx(0.0)
    assert pd.isna(ci.loc["A", "payback_ci_lo"])
    assert pd.isna(ci.loc["A", "payback_ci_hi"])


def test_blended_retention_curve(panel):
    _, _, p = panel
    channel_a = p[p["acquisition_channel"] == "A"]
    curve = blended_retention_curve(channel_a)

    curve_idx = curve.set_index("tenure_month")
    assert curve_idx.loc[0, "retention_pct"] == pytest.approx(100.0)
    assert curve_idx.loc[1, "retention_pct"] == pytest.approx(200 / 3)
    assert curve_idx.loc[2, "retention_pct"] == pytest.approx(0.0)   # both observed customers (C1, C2) churned


def test_ltv_and_ltv_to_cac(panel):
    """
    As of Phase 3, expected_active_months comes from a parametric survival fit
    (engine.survival), not a deterministic hand-calculable formula -- so this
    fixture (3 customers in channel A, 1 in channel B) is no longer about
    verifying an exact number by arithmetic. It's precisely the kind of thin
    sample fit_confidence exists for, so that's what's tested here: the fit
    still runs and produces a finite, sane, bounded LTV, and fit_confidence
    correctly flags it as low-confidence (too few customers for a stable MLE
    fit) rather than implying false precision. See tests/test_survival.py for
    the actual fit-correctness validation (recovering known parameters from
    simulated data) and for extrapolation_confidence, the separate signal
    this fixture is too small to meaningfully exercise on its own.
    """
    customers, _, p = panel
    ltv = ltv_by_channel(p, customers, gross_margin=1.0).set_index("acquisition_channel")

    assert ltv.loc["A", "arpa"] == pytest.approx(100.0)
    assert ltv.loc["A", "expected_active_months"] > 0
    assert ltv.loc["A", "ltv"] > 0
    assert ltv.loc["A", "fit_confidence"] == "low_confidence"  # n=3, far below the MLE sample-size threshold
    assert ltv.loc["B", "fit_confidence"] == "low_confidence"  # n=1
    assert ltv.loc["A", "extrapolation_confidence"] in ("ok", "low_confidence")

    ratio = ltv_to_cac_by_channel(p, customers, gross_margin=1.0).set_index("acquisition_channel")
    assert ratio.loc["A", "ltv_to_cac"] == pytest.approx(ltv.loc["A", "ltv"] / 250, abs=0.01)


def test_grr_nrr_cap_on_expansion(panel):
    _, _, p = panel
    table = grr_nrr_table(p, group_col="cohort_month").set_index(["cohort_month", "tenure_month"])

    # cohort 2024-01 base = C1(100) + C2(100) + C4(200) = 400
    assert table.loc[("2024-01", 0), "nrr_pct"] == pytest.approx(100.0)
    assert table.loc[("2024-01", 0), "grr_pct"] == pytest.approx(100.0)

    # t=1: mrr = C1:100, C2:0, C4:200 -> total 300/400=75%; no expansion yet so grr==nrr
    assert table.loc[("2024-01", 1), "nrr_pct"] == pytest.approx(75.0)
    assert table.loc[("2024-01", 1), "grr_pct"] == pytest.approx(75.0)

    # t=2: mrr = C1:0, C2:0, C4:300 (upsold from 200) -> nrr = 300/400 = 75%
    # grr caps C4's contribution at its own baseline (200) -> capped total = 200/400 = 50%
    assert table.loc[("2024-01", 2), "nrr_pct"] == pytest.approx(75.0)
    assert table.loc[("2024-01", 2), "grr_pct"] == pytest.approx(50.0)
    assert table.loc[("2024-01", 2), "grr_pct"] <= 100.0
