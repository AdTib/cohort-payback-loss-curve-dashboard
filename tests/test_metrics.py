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
    cac_payback_by_channel,
    cohort_retention_table,
    expected_active_months,
    blended_retention_curve,
    grr_nrr_table,
    ltv_by_channel,
    ltv_to_cac_by_channel,
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


def test_blended_retention_curve_and_expected_active_months(panel):
    _, _, p = panel
    channel_a = p[p["acquisition_channel"] == "A"]
    curve = blended_retention_curve(channel_a)

    curve_idx = curve.set_index("tenure_month")
    assert curve_idx.loc[0, "retention_pct"] == pytest.approx(100.0)
    assert curve_idx.loc[1, "retention_pct"] == pytest.approx(200 / 3)
    assert curve_idx.loc[2, "retention_pct"] == pytest.approx(0.0)   # both observed customers (C1, C2) churned

    # Survival hits 0% at the last observed point, so there's no tail to extrapolate --
    # expected active months is just the sum of the observed curve.
    exp_months = expected_active_months(curve)
    assert exp_months == pytest.approx(1.0 + 2 / 3 + 0.0)


def test_ltv_and_ltv_to_cac(panel):
    customers, _, p = panel
    ltv = ltv_by_channel(p, customers, gross_margin=1.0).set_index("acquisition_channel")

    # Channel A: expected_active_months = 5/3, ARPA = mean(100,100,100,100,100) = 100
    assert ltv.loc["A", "expected_active_months"] == pytest.approx(5 / 3, abs=0.01)
    assert ltv.loc["A", "arpa"] == pytest.approx(100.0)
    assert ltv.loc["A", "ltv"] == pytest.approx(5 / 3 * 100, abs=1.0)

    ratio = ltv_to_cac_by_channel(p, customers, gross_margin=1.0).set_index("acquisition_channel")
    assert ratio.loc["A", "ltv_to_cac"] == pytest.approx((5 / 3 * 100) / 250, abs=0.01)


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
