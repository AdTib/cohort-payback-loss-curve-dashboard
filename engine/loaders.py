"""
Data ingestion layer -- the only part of the engine that needs to change to point
this dashboard at a real company's data instead of the synthetic generator.

Expected input schema (two CSVs):

customers.csv
    customer_id           str    unique customer identifier
    signup_date            date   first day of the signup month is recommended but any
                                  date within the month works -- it is floored to month start
    acquisition_channel    str    e.g. "paid_social", "organic_referral"
    initial_plan            str    plan/tier at signup (informational; not required downstream)
    cac                     float  fully-loaded cost to acquire this customer

subscription_events.csv
    customer_id     str    foreign key into customers.customer_id
    month           date   calendar month of the billing event (any day-of-month; floored)
    mrr             float  monthly recurring revenue for that customer in that month,
                             0 if churned that month

Everything downstream (engine/metrics.py, app.py) consumes only the DataFrames this
module returns -- swap in a real export by pointing `load_dataset` at new file paths.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_CUSTOMER_COLS = {"customer_id", "signup_date", "acquisition_channel", "initial_plan", "cac"}
REQUIRED_EVENT_COLS = {"customer_id", "month", "mrr"}

REVENUE_MODELS = ("subscription", "blended")

# ASSUMPTION, not derived from data: how many trailing months of silence a
# blended/episodic-revenue customer gets before they're counted as churned
# rather than just between purchases. 90 days maps to 3 monthly periods at
# this engine's monthly granularity. There's no universally "correct" value
# here -- it's a business judgment call (a subscription box and a furniture
# retailer would reasonably use different windows), which is exactly why this
# is a named constant with this comment attached to it, not a number that
# quietly shows up inline somewhere.
DEFAULT_REACTIVATION_WINDOW_MONTHS = 3


def load_customers(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_CUSTOMER_COLS - set(df.columns)
    if missing:
        raise ValueError(f"customers file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["signup_date"] = pd.to_datetime(df["signup_date"]).values.astype("datetime64[M]")
    df["acquisition_channel"] = df["acquisition_channel"].astype(str).str.strip()
    df["initial_plan"] = df["initial_plan"].astype(str).str.strip()
    df["cac"] = pd.to_numeric(df["cac"], errors="raise")

    if df["customer_id"].duplicated().any():
        raise ValueError("customers file has duplicate customer_id values")
    if (df["cac"] < 0).any():
        raise ValueError("customers file has negative CAC values")

    df["cohort_month"] = df["signup_date"].dt.to_period("M")
    return df


def load_subscription_events(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_EVENT_COLS - set(df.columns)
    if missing:
        raise ValueError(f"subscription_events file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["month"] = pd.to_datetime(df["month"]).values.astype("datetime64[M]")
    df["mrr"] = pd.to_numeric(df["mrr"], errors="raise")

    # Negative mrr is legitimate, not just a data error: a pure subscription
    # business never has it, but a blended/episodic revenue business (e.g. a
    # retailer) can show net-negative revenue in a month where refunds exceed
    # purchases. Real UCI Online Retail II data does this. So we don't reject
    # it here -- schema validation shouldn't assume a subscription-shaped
    # revenue model. (Found during Phase 0 real-data validation.)
    if df.duplicated(subset=["customer_id", "month"]).any():
        raise ValueError("subscription_events file has duplicate (customer_id, month) rows")

    return df


def _compute_activity(panel: pd.DataFrame, revenue_model: str,
                       reactivation_window_months: int) -> pd.DataFrame:
    """
    Defines "is_active" -- the single boolean every retention-based metric in
    engine/metrics.py is built on -- and why it has to differ by revenue model.

    subscription: a customer either pays this month or they don't. MRR > 0 this
    exact month is the whole definition, and once it goes to 0 it stays 0 (see
    build_customer_panel's dense-grid fill). This is correct for recurring
    billing, where a $0 month IS the churn event.

    blended: episodic/transactional revenue (retail, marketplace, usage-based)
    doesn't work that way. A customer with no purchase this month hasn't
    necessarily churned, they might just not have bought anything *yet* this
    month. Real UCI Online Retail II data makes this concrete: 63% of its
    customers go quiet for a month or more and then transact again. Treating
    every $0 month as churn (the subscription definition) makes retention look
    like it collapses to ~20-30% by month 1 for every cohort, which is not a
    real signal, it's the wrong lens on the data. So for blended mode,
    "active" means "transacted at least once in the trailing
    reactivation_window_months window" -- silence inside the window is
    "dormant" (not churned), and only silence that outlasts the window flips a
    customer to "churned". See DEFAULT_REACTIVATION_WINDOW_MONTHS for why that
    window is a stated assumption, not a fitted number.

    Mechanical note: every cohort's retention will read at or near 100% for
    the first reactivation_window_months tenure months no matter what the
    underlying behavior is, because every customer's signup month is by
    definition their first transaction and the window hasn't had a chance to
    lapse yet. That's an expected property of this definition, not a bug --
    the real signal in the curve starts at tenure == window.

    Adds two columns: is_active (bool, what metrics.py consumes) and
    activity_state ("active" / "dormant" / "churned", informational --
    useful for diagnostics and dashboard display, not required by any metric).
    """
    if revenue_model not in REVENUE_MODELS:
        raise ValueError(f"revenue_model must be one of {REVENUE_MODELS}, got {revenue_model!r}")

    panel = panel.sort_values(["customer_id", "tenure_month"]).reset_index(drop=True)

    if revenue_model == "subscription":
        panel["is_active"] = panel["mrr"] > 0
        panel["activity_state"] = panel["is_active"].map({True: "active", False: "churned"})
        return panel

    # blended: rolling "transacted in the last N months, including this one"
    transacted = (panel["mrr"] != 0)
    within_window = (
        transacted.groupby(panel["customer_id"])
        .rolling(window=reactivation_window_months, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(bool)
    )
    panel["is_active"] = within_window
    panel["activity_state"] = "churned"
    panel.loc[within_window & transacted, "activity_state"] = "active"
    panel.loc[within_window & ~transacted, "activity_state"] = "dormant"
    return panel


def build_customer_panel(customers: pd.DataFrame, events: pd.DataFrame,
                          revenue_model: str = "subscription",
                          reactivation_window_months: int = DEFAULT_REACTIVATION_WINDOW_MONTHS) -> pd.DataFrame:
    """
    Merge customers + events into one row-per-customer-month panel, filled to a dense
    grid from each customer's signup month through the dataset's global max observed
    month. Missing billing rows within that window default to MRR=0 -- this is a
    documented assumption, see README "Methodology" for the rationale.

    Adds: cohort_month (Period[M]), tenure_month (int, 0 = signup month),
    is_active (bool), activity_state (str). How is_active is derived from MRR
    depends on revenue_model -- see _compute_activity.
    """
    max_observed_month = events["month"].max()

    grids = []
    for row in customers.itertuples(index=False):
        n_months = (max_observed_month.to_period("M") - row.cohort_month).n + 1
        if n_months <= 0:
            continue
        months = pd.date_range(row.signup_date, periods=n_months, freq="MS")
        grids.append(pd.DataFrame({
            "customer_id": row.customer_id,
            "month": months,
            "tenure_month": range(n_months),
        }))
    grid = pd.concat(grids, ignore_index=True)

    panel = grid.merge(events[["customer_id", "month", "mrr"]], on=["customer_id", "month"], how="left")
    panel["mrr"] = panel["mrr"].fillna(0.0)
    # Carry every customer-level column into the panel, not just the required
    # schema fields -- so a real column beyond the minimum schema (Telco's
    # Contract type, a real prospect's own segment field, etc.) is available
    # for per-segment metrics downstream without engine changes. See Phase 2.
    customer_cols = [c for c in customers.columns if c != "customer_id"]
    panel = panel.merge(customers[["customer_id"] + customer_cols], on="customer_id", how="left")
    panel = _compute_activity(panel, revenue_model, reactivation_window_months)
    return panel


def load_dataset(customers_path, events_path, revenue_model: str = "subscription",
                  reactivation_window_months: int = DEFAULT_REACTIVATION_WINDOW_MONTHS
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: returns (customers, events, panel)."""
    customers = load_customers(customers_path)
    events = load_subscription_events(events_path)
    panel = build_customer_panel(customers, events, revenue_model, reactivation_window_months)
    return customers, events, panel
