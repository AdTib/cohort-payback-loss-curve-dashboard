"""
Unit tests for Phase 2's segment_col generalization.

Fixture is built so channel and plan are deliberately cross-cutting partitions of the
same 4 customers, so a test that only checked "segmenting by acquisition_channel still
works" couldn't catch a bug where segment_col was silently ignored:

  E1  channel Z1  plan P1  cac=100   MRR: 100, 100  (stays active)
  E2  channel Z2  plan P1  cac=300   MRR: 100, 0    (churns at tenure 1)
  E3  channel Z1  plan P2  cac=100   MRR: 100, 0    (churns at tenure 1)
  E4  channel Z2  plan P2  cac=300   MRR: 100, 100  (stays active)

By channel: Z1={E1,E3} avg_cac=100, Z2={E2,E4} avg_cac=300 -- different.
By plan:    P1={E1,E2} avg_cac=200, P2={E3,E4} avg_cac=200 -- identical (by symmetric
            construction), which is itself a useful check: if segment_col were bugged
            and secretly still grouping by channel, this would come out (100, 300),
            not (200, 200).
"""

import pandas as pd
import pytest

from engine.loaders import build_customer_panel, load_customers, load_subscription_events
from engine.metrics import cac_payback_by_channel, ltv_to_cac_by_channel

CUSTOMERS_CSV = """customer_id,signup_date,acquisition_channel,initial_plan,cac
E1,2024-01-01,Z1,P1,100
E2,2024-01-01,Z2,P1,300
E3,2024-01-01,Z1,P2,100
E4,2024-01-01,Z2,P2,300
"""

EVENTS_CSV = """customer_id,month,mrr
E1,2024-01-01,100
E1,2024-02-01,100
E2,2024-01-01,100
E2,2024-02-01,0
E3,2024-01-01,100
E3,2024-02-01,0
E4,2024-01-01,100
E4,2024-02-01,100
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


def test_segment_col_changes_the_grouping(panel):
    customers, _, p = panel

    by_channel, _ = cac_payback_by_channel(p, customers, gross_margin=1.0, segment_col="acquisition_channel")
    by_plan, _ = cac_payback_by_channel(p, customers, gross_margin=1.0, segment_col="initial_plan")

    # Output column is named after segment_col, not hardcoded to "acquisition_channel"
    assert "acquisition_channel" in by_channel.columns
    assert "initial_plan" in by_plan.columns
    assert "initial_plan" not in by_channel.columns

    channel_cacs = by_channel.set_index("acquisition_channel")["avg_cac"]
    assert channel_cacs["Z1"] == pytest.approx(100.0)
    assert channel_cacs["Z2"] == pytest.approx(300.0)

    plan_cacs = by_plan.set_index("initial_plan")["avg_cac"]
    assert plan_cacs["P1"] == pytest.approx(200.0)
    assert plan_cacs["P2"] == pytest.approx(200.0)
    # If segment_col were silently ignored (still grouping by channel under the hood),
    # these would come out as (100, 300) instead of (200, 200).
    assert plan_cacs["P1"] != channel_cacs["Z1"]


def test_ltv_to_cac_respects_segment_col(panel):
    customers, _, p = panel
    by_plan = ltv_to_cac_by_channel(p, customers, gross_margin=1.0, segment_col="initial_plan")
    assert list(by_plan["initial_plan"].sort_values()) == ["P1", "P2"]
    assert "acquisition_channel" not in by_plan.columns
