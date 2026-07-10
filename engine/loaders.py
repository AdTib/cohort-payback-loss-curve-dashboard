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

    if (df["mrr"] < 0).any():
        raise ValueError("subscription_events file has negative MRR values")
    if df.duplicated(subset=["customer_id", "month"]).any():
        raise ValueError("subscription_events file has duplicate (customer_id, month) rows")

    return df


def build_customer_panel(customers: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Merge customers + events into one row-per-customer-month panel, filled to a dense
    grid from each customer's signup month through the dataset's global max observed
    month. Missing billing rows within that window are treated as MRR=0 (churned/lapsed) --
    this is a documented assumption, see README "Methodology" for the rationale.

    Adds: cohort_month (Period[M]), tenure_month (int, 0 = signup month), is_active (bool).
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
    panel = panel.merge(
        customers[["customer_id", "cohort_month", "acquisition_channel", "initial_plan", "cac"]],
        on="customer_id", how="left",
    )
    panel["is_active"] = panel["mrr"] > 0
    return panel


def load_dataset(customers_path, events_path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: returns (customers, events, panel)."""
    customers = load_customers(customers_path)
    events = load_subscription_events(events_path)
    panel = build_customer_panel(customers, events)
    return customers, events, panel
