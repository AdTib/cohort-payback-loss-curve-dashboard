"""
Phase 0: reshape two real public datasets into the engine's customers.csv /
subscription_events.csv schema, so the metrics engine gets stress-tested against
real churn/tenure/revenue behavior instead of only the clean synthetic generator.

Two datasets, two different revenue shapes on purpose:

  1. IBM Telco Customer Churn -- a real subscription business snapshot (tenure in
     months, monthly charge, churn flag). Reshaped into a cohort panel under one
     explicit, documented assumption: monthly charge is treated as constant over a
     customer's tenure, because the dataset gives us a snapshot (current tenure +
     outcome), not a month-by-month billing history. This is the "subscription"
     revenue model validation target for Phase 1.

  2. UCI Online Retail II -- real e-commerce transactions with irregular,
     episodic repeat-purchase behavior. A customer having $0 revenue in a given
     month does NOT mean they churned the way it does for a subscription --
     they might buy again three months later. This is deliberately the
     "blended"/episodic revenue model validation target for Phase 1, and it's
     expected to stress (and likely break) the subscription-shaped "MRR > 0 =
     active" assumption that's baked into the engine today.

Neither dataset has acquisition channel or CAC. Both are real businesses, so
this data has no CAC to give us -- we assign synthetic acquisition channels and
CAC values on top, clearly and repeatedly labeled as ASSUMED. This is the one
part of both output files that is not real, and every consumer of this module
(README, docstrings, the diagnostics dict) says so explicitly. Never let this
render next to real revenue/churn numbers without the "assumed" label attached.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RETAIL_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
REAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real")

SEED = 7

# ---------------------------------------------------------------------------
# ASSUMED acquisition economics -- not present in either source dataset.
# Layered on so the reshaped data can exercise the engine's per-channel CAC
# payback / LTV:CAC logic. These numbers are fabricated for stress-testing and
# must never be presented as the real companies' actual CAC.
# ---------------------------------------------------------------------------
ASSUMED_CHANNEL_CAC = {
    "organic":     {"share": 0.30, "mean": 40,  "sd": 10, "clip": (15, 90)},
    "referral":    {"share": 0.25, "mean": 60,  "sd": 15, "clip": (20, 130)},
    "paid_search": {"share": 0.25, "mean": 140, "sd": 25, "clip": (70, 240)},
    "paid_social": {"share": 0.20, "mean": 220, "sd": 35, "clip": (110, 380)},
}


def _download(url: str, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        urllib.request.urlretrieve(url, dest_path)
    return dest_path


def _assign_channels_and_cac(customer_ids: pd.Index, seed: int = SEED) -> pd.DataFrame:
    """ASSUMED, not derived from either source dataset. See module docstring."""
    rng = np.random.default_rng(seed)
    names = list(ASSUMED_CHANNEL_CAC.keys())
    probs = np.array([ASSUMED_CHANNEL_CAC[c]["share"] for c in names])
    probs = probs / probs.sum()

    channels = rng.choice(names, size=len(customer_ids), p=probs)
    cac = np.empty(len(customer_ids))
    for name in names:
        mask = channels == name
        cfg = ASSUMED_CHANNEL_CAC[name]
        cac[mask] = np.clip(rng.normal(cfg["mean"], cfg["sd"], mask.sum()), *cfg["clip"])

    return pd.DataFrame({
        "customer_id": customer_ids,
        "acquisition_channel": channels,
        "cac": np.round(cac, 2),
    })


# ---------------------------------------------------------------------------
# Dataset 1: Telco Customer Churn -> subscription cohort panel
# ---------------------------------------------------------------------------

def build_telco_dataset(download: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_path = os.path.join(RAW_DIR, "telco_customer_churn.csv")
    if download:
        _download(TELCO_URL, raw_path)
    raw = pd.read_csv(raw_path)

    raw["TotalCharges_numeric"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    # Sanity check on the constant-MRR assumption: for customers with tenure > 0,
    # tenure * MonthlyCharges should track TotalCharges reasonably closely if
    # monthly charge really was ~constant over their history.
    checkable = raw[raw["tenure"] > 0].copy()
    checkable["implied_total"] = checkable["tenure"] * checkable["MonthlyCharges"]
    pct_error = ((checkable["implied_total"] - checkable["TotalCharges_numeric"]).abs()
                 / checkable["TotalCharges_numeric"].clip(lower=1)).median()

    # Anchor date is arbitrary -- only relative spacing between cohorts matters.
    anchor = pd.Timestamp("2024-01-01")
    raw["signup_date"] = raw["tenure"].apply(lambda t: anchor - pd.DateOffset(months=int(t)))

    # active_months has to distinguish censored from churned, or every censored
    # (Churn=No) customer ends up with a "phantom" extra grid row past their
    # last real event -- build_customer_panel fills any month with no event
    # row as inactive, and every customer's grid runs through the same global
    # anchor month regardless of churn status (see Phase 0). A Churn=No
    # customer is still active AS OF the anchor/snapshot month, so they need
    # an event row there too (tenure + 1 active months) or that final row
    # silently reads as churned. A Churn=Yes customer's real churn IS that
    # final unobserved month, so they correctly get exactly `tenure` months
    # of events and no more. Getting this wrong doesn't crash anything -- it
    # silently mislabels ~5,000 real "No" customers as churned, caught only
    # by cross-checking Kaplan-Meier's censoring rate against the raw Churn
    # column (0.16% vs the real 73.5%) during Phase 3 validation.
    raw["active_months"] = raw.apply(
        lambda r: int(r.tenure) + 1 if r.Churn == "No" else max(int(r.tenure), 1), axis=1
    )

    channel_cac = _assign_channels_and_cac(raw["customerID"], seed=SEED)

    customers = pd.DataFrame({
        "customer_id": raw["customerID"],
        "signup_date": raw["signup_date"].dt.strftime("%Y-%m-%d"),
        "initial_plan": raw["Contract"],           # real field: Month-to-month / One year / Two year
        "internet_service": raw["InternetService"],  # real field: DSL / Fiber optic / No
        "payment_method": raw["PaymentMethod"],       # real field: e.g. Electronic check, Mailed check
    }).merge(channel_cac, on="customer_id")

    event_rows = []
    for row in raw.itertuples(index=False):
        for t in range(int(row.active_months)):
            event_rows.append({
                "customer_id": row.customerID,
                "month": (row.signup_date + pd.DateOffset(months=t)).strftime("%Y-%m-%d"),
                "mrr": round(float(row.MonthlyCharges), 2),
            })
    events = pd.DataFrame(event_rows)

    diagnostics = {
        "source": "IBM Telco Customer Churn (github.com/IBM/telco-customer-churn-on-icp4d)",
        "revenue_model_target": "subscription",
        "n_customers": len(customers),
        "n_events": len(events),
        "churn_rate": float((raw["Churn"] == "Yes").mean()),
        "tenure_min_max": (int(raw["tenure"].min()), int(raw["tenure"].max())),
        "single_month_cohort_customers": int((raw["tenure"] == 0).sum()),
        "constant_mrr_assumption_median_pct_error": round(float(pct_error), 4),
        "cac_channel_and_amounts_are_assumed": True,
        "downloaded_from": TELCO_URL,
    }
    return customers, events, diagnostics


# ---------------------------------------------------------------------------
# Dataset 2: UCI Online Retail II -> episodic/blended revenue panel
# ---------------------------------------------------------------------------

def build_retail_dataset(download: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    zip_path = os.path.join(RAW_DIR, "online_retail_ii.zip")
    xlsx_path = os.path.join(RAW_DIR, "online_retail_ii.xlsx")
    if download:
        _download(RETAIL_URL, zip_path)
        if not os.path.exists(xlsx_path):
            with zipfile.ZipFile(zip_path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
                with zf.open(name) as src, open(xlsx_path, "wb") as dst:
                    dst.write(src.read())

    frames = []
    for sheet in ["Year 2009-2010", "Year 2010-2011"]:
        d = pd.read_excel(xlsx_path, sheet_name=sheet,
                           usecols=["Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"])
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)

    n_raw = len(raw)
    raw = raw.dropna(subset=["Customer ID"])
    n_dropped_no_customer = n_raw - len(raw)
    # Drop non-product / adjustment rows (postage, bank charges, manual entries,
    # etc. commonly show Price <= 0 in this dataset) -- keep returns (negative
    # quantity against a real positive price) since that's real repeat-purchase
    # / refund behavior, not a data artifact.
    raw = raw[raw["Price"] > 0].copy()

    raw["customer_id"] = "RETAIL-" + raw["Customer ID"].astype(int).astype(str)
    raw["revenue"] = raw["Quantity"] * raw["Price"]
    raw["month"] = raw["InvoiceDate"].values.astype("datetime64[M]")

    first_purchase = raw.groupby("customer_id")["month"].min().rename("signup_date").reset_index()
    country = raw.groupby("customer_id")["Country"].agg(lambda s: s.mode().iat[0]).rename("country").reset_index()

    channel_cac = _assign_channels_and_cac(first_purchase["customer_id"], seed=SEED)

    customers = pd.DataFrame({
        "customer_id": first_purchase["customer_id"],
        "signup_date": first_purchase["signup_date"].dt.strftime("%Y-%m-%d"),
        "initial_plan": "n/a",  # no plan concept in transactional retail data
    }).merge(channel_cac, on="customer_id").merge(country, on="customer_id")

    monthly = raw.groupby(["customer_id", "month"])["revenue"].sum().reset_index()
    events = pd.DataFrame({
        "customer_id": monthly["customer_id"],
        "month": monthly["month"].dt.strftime("%Y-%m-%d"),
        "mrr": monthly["revenue"].round(2),  # NOT recurring -- episodic revenue, see module docstring
    })

    active_months_per_customer = monthly.groupby("customer_id").size()
    diagnostics = {
        "source": "UCI Online Retail II (archive.ics.uci.edu/dataset/502)",
        "revenue_model_target": "blended/episodic",
        "n_customers": len(customers),
        "n_events": len(events),
        "n_raw_transaction_rows": n_raw,
        "n_rows_dropped_no_customer_id": int(n_dropped_no_customer),
        "median_active_months_per_customer": float(active_months_per_customer.median()),
        "pct_customers_with_gap_then_repeat": float(
            (active_months_per_customer < (raw["month"].max().to_period("M") - raw["month"].min().to_period("M")).n + 1
             ).mean()
        ),
        "cac_channel_and_amounts_are_assumed": True,
        "downloaded_from": RETAIL_URL,
    }
    return customers, events, diagnostics


def main():
    os.makedirs(REAL_DIR, exist_ok=True)

    print("Building Telco (subscription) dataset...")
    telco_customers, telco_events, telco_diag = build_telco_dataset()
    telco_dir = os.path.join(REAL_DIR, "telco")
    os.makedirs(telco_dir, exist_ok=True)
    telco_customers.to_csv(os.path.join(telco_dir, "customers.csv"), index=False)
    telco_events.to_csv(os.path.join(telco_dir, "subscription_events.csv"), index=False)
    print(telco_diag)

    print("\nBuilding Online Retail II (blended) dataset...")
    retail_customers, retail_events, retail_diag = build_retail_dataset()
    retail_dir = os.path.join(REAL_DIR, "retail")
    os.makedirs(retail_dir, exist_ok=True)
    retail_customers.to_csv(os.path.join(retail_dir, "customers.csv"), index=False)
    retail_events.to_csv(os.path.join(retail_dir, "subscription_events.csv"), index=False)
    print(retail_diag)


if __name__ == "__main__":
    main()
