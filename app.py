"""
Streamlit UI. This file only renders -- every number on the page comes from
engine/loaders.py (ingestion) and engine/metrics.py (calculations). Point the
sidebar uploader at a real customers.csv / subscription_events.csv export and the
same charts render against real data.
"""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.loaders import build_customer_panel, load_customers, load_subscription_events
from engine.metrics import (
    DEFAULT_GROSS_MARGIN,
    cac_payback_by_channel,
    cohort_retention_table,
    grr_nrr_table,
    ltv_to_cac_by_channel,
)

st.set_page_config(page_title="Cohort / Payback / Loss-Curve Dashboard", layout="wide")

DEMO_CUSTOMERS_PATH = "data/customers.csv"
DEMO_EVENTS_PATH = "data/subscription_events.csv"


@st.cache_data(show_spinner=False)
def _load(customers_bytes: bytes, events_bytes: bytes):
    customers = load_customers(io.BytesIO(customers_bytes))
    events = load_subscription_events(io.BytesIO(events_bytes))
    panel = build_customer_panel(customers, events)
    return customers, events, panel


def _read_bytes(path_or_upload) -> bytes:
    if hasattr(path_or_upload, "getvalue"):
        return path_or_upload.getvalue()
    with open(path_or_upload, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Sidebar: data source + filters
# ---------------------------------------------------------------------------

st.sidebar.title("Data source")
use_demo = st.sidebar.toggle("Use synthetic demo data", value=True)

if use_demo:
    customers_bytes = _read_bytes(DEMO_CUSTOMERS_PATH)
    events_bytes = _read_bytes(DEMO_EVENTS_PATH)
    st.sidebar.caption("2,000 synthetic PFM subscribers · 18 months of signups")
else:
    cust_upload = st.sidebar.file_uploader("customers.csv", type="csv")
    events_upload = st.sidebar.file_uploader("subscription_events.csv", type="csv")
    if cust_upload is None or events_upload is None:
        st.sidebar.info("Upload both files to replace the demo dataset — same schema as the generator (see README).")
        customers_bytes = _read_bytes(DEMO_CUSTOMERS_PATH)
        events_bytes = _read_bytes(DEMO_EVENTS_PATH)
    else:
        customers_bytes = _read_bytes(cust_upload)
        events_bytes = _read_bytes(events_upload)

try:
    customers, events, panel = _load(customers_bytes, events_bytes)
except ValueError as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.sidebar.divider()
st.sidebar.title("Filters")
all_channels = sorted(customers["acquisition_channel"].unique())
selected_channels = st.sidebar.multiselect("Acquisition channel", all_channels, default=all_channels)
gross_margin = st.sidebar.slider(
    "Gross margin assumption", min_value=0.50, max_value=0.95, value=DEFAULT_GROSS_MARGIN, step=0.01,
    help="Applied to MRR to get gross margin $, used in CAC payback and LTV. See README for rationale.",
)

if not selected_channels:
    st.warning("Select at least one acquisition channel.")
    st.stop()

customers_f = customers[customers["acquisition_channel"].isin(selected_channels)]
panel_f = panel[panel["acquisition_channel"].isin(selected_channels)]

# ---------------------------------------------------------------------------
# Header / KPIs
# ---------------------------------------------------------------------------

st.title("Cohort, Payback & Loss-Curve Dashboard")
st.caption("Synthetic fintech PFM subscription business — reusable engine, swap in real data via the sidebar.")

max_month = panel_f["month"].max()
n_cohorts = customers_f["cohort_month"].nunique()

kpi_cols = st.columns(4)
kpi_cols[0].metric("Customers analyzed", f"{customers_f['customer_id'].nunique():,}")
kpi_cols[1].metric("Signup cohorts", n_cohorts)
kpi_cols[2].metric("Blended avg CAC", f"${customers_f['cac'].mean():,.0f}")
kpi_cols[3].metric("Data through", max_month.strftime("%b %Y"))

st.divider()

# ---------------------------------------------------------------------------
# 1. Cohort retention heatmap
# ---------------------------------------------------------------------------

st.header("Cohort retention")
st.caption("% of each signup cohort with MRR > 0, by months since signup. Blank cells haven't happened yet.")

retention_table = cohort_retention_table(panel_f)
fig_retention = px.imshow(
    retention_table,
    color_continuous_scale="Blues",
    aspect="auto",
    labels=dict(x="Months since signup", y="Signup cohort", color="Retention %"),
    zmin=0, zmax=100,
)
fig_retention.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_retention, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 2. CAC payback curve
# ---------------------------------------------------------------------------

st.header("CAC payback by channel")
st.caption("Cumulative gross margin per customer ÷ CAC. Payback month = first crossing of 1.0.")

payback_summary, payback_curves = cac_payback_by_channel(panel_f, customers_f, gross_margin=gross_margin)

fig_payback = go.Figure()
for channel in selected_channels:
    curve = payback_curves[channel]
    fig_payback.add_trace(go.Scatter(
        x=curve["tenure_month"], y=curve["payback_ratio"],
        mode="lines+markers", name=channel,
    ))
fig_payback.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Breakeven (1.0x)")
fig_payback.update_layout(
    xaxis_title="Months since signup", yaxis_title="Cumulative margin ÷ CAC",
    height=450, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_payback, use_container_width=True)

payback_display = payback_summary.rename(columns={
    "acquisition_channel": "Channel", "avg_cac": "Avg CAC",
    "payback_month": "Payback (months)", "max_observed_tenure": "Max observed tenure",
})
payback_display["Payback (months)"] = payback_display["Payback (months)"].astype(object).where(
    payback_display["Payback (months)"].notna(), "not yet reached"
)
st.dataframe(payback_display, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 3. LTV:CAC by channel
# ---------------------------------------------------------------------------

st.header("LTV:CAC by channel")
st.caption("LTV modeled to convergence from each channel's retention curve (not a fixed window). See README for method.")

ltvcac = ltv_to_cac_by_channel(panel_f, customers_f, gross_margin=gross_margin)
ltvcac = ltvcac[ltvcac["acquisition_channel"].isin(selected_channels)].sort_values("ltv_to_cac", ascending=False)

fig_ltvcac = px.bar(
    ltvcac, x="acquisition_channel", y="ltv_to_cac", color="acquisition_channel",
    text_auto=".2f", labels={"acquisition_channel": "Channel", "ltv_to_cac": "LTV : CAC"},
)
fig_ltvcac.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Breakeven (1x)")
fig_ltvcac.add_hline(y=3.0, line_dash="dot", line_color="green", annotation_text="Healthy benchmark (3x)")
fig_ltvcac.update_layout(height=450, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_ltvcac, use_container_width=True)

ltv_display = ltvcac.rename(columns={
    "acquisition_channel": "Channel", "expected_active_months": "Expected active months",
    "arpa": "ARPA (active)", "ltv": "LTV", "avg_cac": "Avg CAC", "ltv_to_cac": "LTV:CAC",
})
st.dataframe(ltv_display, hide_index=True, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. GRR / NRR loss curves
# ---------------------------------------------------------------------------

st.header("Loss curves: GRR & NRR")
st.caption("Revenue retained from a cohort's starting MRR base. GRR excludes expansion (capped at 100%); NRR includes it.")

view_mode = st.radio("View", ["Blended (all selected channels)", "By cohort"], horizontal=True)

if view_mode == "Blended (all selected channels)":
    blended = panel_f.copy()
    blended["segment"] = "Blended"
    grr_table = grr_nrr_table(blended, group_col="segment")

    fig_grr = go.Figure()
    fig_grr.add_trace(go.Scatter(x=grr_table["tenure_month"], y=grr_table["grr_pct"], name="GRR", mode="lines+markers"))
    fig_grr.add_trace(go.Scatter(x=grr_table["tenure_month"], y=grr_table["nrr_pct"], name="NRR", mode="lines+markers"))
    fig_grr.add_hline(y=100, line_dash="dash", line_color="gray")
    fig_grr.update_layout(
        xaxis_title="Months since cohort start", yaxis_title="% of starting MRR retained",
        height=450, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_grr, use_container_width=True)
else:
    metric_choice = st.selectbox("Metric", ["GRR", "NRR"])
    metric_col = "grr_pct" if metric_choice == "GRR" else "nrr_pct"
    cohort_grr = grr_nrr_table(panel_f, group_col="cohort_month")

    fig_cohort_grr = px.line(
        cohort_grr, x="tenure_month", y=metric_col, color="cohort_month",
        labels={"tenure_month": "Months since cohort start", metric_col: f"{metric_choice} %", "cohort_month": "Cohort"},
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_cohort_grr.add_hline(y=100, line_dash="dash", line_color="gray")
    fig_cohort_grr.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cohort_grr, use_container_width=True)
