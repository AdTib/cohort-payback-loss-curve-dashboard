"""
Unit tests for revenue_model-aware activity, added in Phase 1.

Fixture: 3 customers, window = DEFAULT_REACTIVATION_WINDOW_MONTHS (3 months).

  D1  signup 2024-01  one transaction at tenure 0, then silent forever
  D2  signup 2024-01  transacts at tenure 0, silent for 2 months, reactivates at tenure 3
  D3  signup 2024-07  exists purely to push the dataset's global observed window out to
                       July 2024, so D1/D2 have 7 months of panel to be silent across

Hand-calculated expectation for D1 under "blended" (trailing 3-month window, inclusive
of the current month):
  t0 active (transacted)
  t1 dormant (t0's transaction is still inside the window)
  t2 dormant (t0 is still inside the window: [t0,t1,t2])
  t3 churned (window is now [t1,t2,t3], t0 has aged out, nothing else transacted)

D2 reactivates right at tenure 3, so the window resets from there:
  t3 active (transacted again)
  t4 dormant, t5 dormant (t3 still inside the window)
  t6 churned (window [t4,t5,t6] no longer includes t3)

Under "subscription" mode on the same data, there's no window at all: D1 and D2 both go
straight to churned the month after their last transaction, since subscription mode
means MRR = 0 this exact month IS the churn event.
"""

import pandas as pd
import pytest

from engine.loaders import build_customer_panel, load_customers, load_subscription_events

CUSTOMERS_CSV = """customer_id,signup_date,acquisition_channel,initial_plan,cac
D1,2024-01-01,organic,Basic,50
D2,2024-01-01,organic,Basic,50
D3,2024-07-01,organic,Basic,50
"""

EVENTS_CSV = """customer_id,month,mrr
D1,2024-01-01,50
D2,2024-01-01,80
D2,2024-04-01,80
D3,2024-07-01,10
"""


@pytest.fixture
def raw(tmp_path):
    cust_path = tmp_path / "customers.csv"
    events_path = tmp_path / "subscription_events.csv"
    cust_path.write_text(CUSTOMERS_CSV)
    events_path.write_text(EVENTS_CSV)
    return load_customers(cust_path), load_subscription_events(events_path)


def test_blended_mode_dormant_then_churned(raw):
    customers, events = raw
    panel = build_customer_panel(customers, events, revenue_model="blended")
    d1 = panel[panel["customer_id"] == "D1"].set_index("tenure_month")

    assert d1.loc[0, "activity_state"] == "active"
    assert d1.loc[1, "activity_state"] == "dormant"
    assert d1.loc[2, "activity_state"] == "dormant"
    assert d1.loc[3, "activity_state"] == "churned"
    assert bool(d1.loc[2, "is_active"]) is True   # still counted as retained
    assert bool(d1.loc[3, "is_active"]) is False  # window exceeded


def test_blended_mode_reactivation_resets_the_window(raw):
    customers, events = raw
    panel = build_customer_panel(customers, events, revenue_model="blended")
    d2 = panel[panel["customer_id"] == "D2"].set_index("tenure_month")

    assert d2.loc[0, "activity_state"] == "active"
    assert d2.loc[1, "activity_state"] == "dormant"
    assert d2.loc[2, "activity_state"] == "dormant"
    assert d2.loc[3, "activity_state"] == "active"    # reactivated
    assert d2.loc[4, "activity_state"] == "dormant"   # still inside window of t3
    assert d2.loc[5, "activity_state"] == "dormant"
    assert d2.loc[6, "activity_state"] == "churned"   # t3 has now aged out


def test_subscription_mode_has_no_grace_window(raw):
    """Same underlying data, revenue_model='subscription' -- a $0 month is churn, immediately."""
    customers, events = raw
    panel = build_customer_panel(customers, events, revenue_model="subscription")
    d1 = panel[panel["customer_id"] == "D1"].set_index("tenure_month")

    assert d1.loc[0, "activity_state"] == "active"
    assert d1.loc[1, "activity_state"] == "churned"  # no dormant state in subscription mode
    assert bool(d1.loc[1, "is_active"]) is False


def test_unknown_revenue_model_raises(raw):
    customers, events = raw
    with pytest.raises(ValueError):
        build_customer_panel(customers, events, revenue_model="subscription_box_of_the_month")
