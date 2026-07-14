"""
Streamlit UI. This file only renders -- every number on the page comes from
engine/loaders.py (ingestion) and engine/metrics.py (calculations). Point the
sidebar uploader at a real customers.csv / subscription_events.csv export and the
same charts render against real data.

Layout is compute-then-render: every DataFrame the page needs is built once in
the "Compute" block, then the Key Insights panel and every chart section below
just read from those already-computed values -- nothing in the render path
calls back into engine/ a second time for the same number.
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.loaders import DEFAULT_REACTIVATION_WINDOW_MONTHS, build_customer_panel, load_customers, load_subscription_events
from engine.metrics import (
    DEFAULT_GROSS_MARGIN,
    bootstrap_payback_ci,
    bootstrap_payback_curve_ci,
    cac_payback_by_channel,
    cohort_retention_table,
    grr_nrr_table,
    ltv_to_cac_by_channel,
    monte_carlo_payback_sensitivity,
    payback_ratio_bridge,
)
from engine.survival import bootstrap_km_ci, extract_durations

st.set_page_config(page_title="Cohort / Payback / Loss-Curve Dashboard", layout="wide", page_icon="📒")

# ---------------------------------------------------------------------------
# Design tokens -- one committed dark theme (see .streamlit/config.toml for
# the Streamlit-level colors this extends). Categorical series colors are the
# dark-mode CVD-safe set from the project's dataviz reference palette, used
# in fixed order everywhere so a channel/segment never changes color between
# charts on the same page. Status colors (confidence flags) are a separate
# semantic set, never reused as a series color.
# ---------------------------------------------------------------------------

ACCENT = "#57B593"
SERIES_COLORS = ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"]
GOOD = "#3ecf5c"
WARNING = "#fab219"
HEATMAP_SCALE = [[0.0, "#15171a"], [0.15, "#173226"], [0.5, "#1f6b4d"], [1.0, ACCENT]]

BENCHMARK_PAYBACK_WINDOW_MONTHS = 20  # a segment slower than this (or never reached) is flagged "at risk" below

CUSTOM_CSS = f"""
<style>
html, body, [class*="css"], .stApp {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
  font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums;
}}
h1 {{ letter-spacing: -0.02em; font-weight: 700; }}
h2 {{
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding-bottom: 0.5rem; letter-spacing: -0.01em; font-weight: 650;
}}
h3 {{ letter-spacing: -0.005em; font-weight: 600; }}

.hero {{
  border-left: 3px solid {ACCENT};
  background: linear-gradient(90deg, rgba(87,181,147,0.10), rgba(87,181,147,0.0) 70%);
  padding: 1.1rem 1.4rem;
  border-radius: 6px;
  margin-bottom: 0.5rem;
}}
.hero p {{ margin: 0; font-size: 1.05rem; line-height: 1.5; }}
.hero .eyebrow {{
  font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
  color: {ACCENT}; font-weight: 700; margin-bottom: 0.35rem; display: block;
}}

.badge {{
  display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.01em; white-space: nowrap;
}}
.badge-ok {{ background: rgba(62,207,92,0.15); color: {GOOD}; }}
.badge-low {{ background: rgba(250,178,25,0.16); color: {WARNING}; }}

.flagship-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.1rem; }}
.flagship-card {{
  border: 1px solid rgba(87,181,147,0.32); background: rgba(87,181,147,0.05);
  border-radius: 8px; padding: 1rem 1.2rem;
}}
.flagship-card .label {{
  font-size: 0.7rem; letter-spacing: 0.07em; text-transform: uppercase;
  color: {ACCENT}; font-weight: 700; margin-bottom: 0.45rem; display: block;
}}
.flagship-card .stat {{ font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 1.35rem; font-weight: 700; }}
.flagship-card .detail {{ font-size: 0.87rem; color: #c3c2b7; margin-top: 0.35rem; line-height: 1.45; }}
@media (max-width: 900px) {{ .flagship-grid {{ grid-template-columns: 1fr; }} }}

.insight-row {{
  border-left: 3px solid rgba(255,255,255,0.15); padding: 0.55rem 0.9rem;
  margin-bottom: 0.5rem; font-size: 0.92rem; line-height: 1.5;
}}
.insight-row.at-risk {{ border-left-color: {WARNING}; background: rgba(250,178,25,0.05); }}
.insight-row.not-meaningful {{ border-left-color: #898781; background: rgba(137,135,129,0.06); }}

.term-def {{ font-size: 0.82rem; color: #9aa0a6; margin: -0.4rem 0 0.9rem 0; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def confidence_badge(flag: str) -> str:
    cls = "badge-ok" if flag == "ok" else "badge-low"
    text = "confident" if flag == "ok" else "low confidence"
    return f'<span class="badge {cls}">{text}</span>'


DATA_SOURCES = {
    "Synthetic demo (subscription)": {
        "customers": "data/customers.csv", "events": "data/subscription_events.csv",
        "revenue_model": "subscription", "assumed_cac": False, "default_segment": "acquisition_channel",
        "caption": "2,000 synthetic PFM subscribers · 18 months of signups",
    },
    "Telco Customer Churn (real, subscription)": {
        "customers": "data/real/telco/customers.csv", "events": "data/real/telco/subscription_events.csv",
        "revenue_model": "subscription", "assumed_cac": True, "default_segment": "initial_plan",
        "caption": "7,043 real customers, real tenure/churn (IBM Telco dataset). Channel + CAC are assumed, not real. "
                    "Segment by Contract (initial_plan), internet_service, or payment_method -- all real fields.",
        "survivorship_caveat": True,
    },
    "Online Retail II (real, blended)": {
        "customers": "data/real/retail/customers.csv", "events": "data/real/retail/subscription_events.csv",
        "revenue_model": "blended", "assumed_cac": True, "default_segment": "acquisition_channel",
        "caption": "5,939 real customers, real transactions (UCI Online Retail II). Channel + CAC are assumed, not real.",
    },
    "Upload your own": {"customers": None, "events": None, "revenue_model": "subscription",
                         "assumed_cac": False, "default_segment": "acquisition_channel", "caption": ""},
}

TERM_DEFINITIONS = {
    "Payback period": "How many months of gross margin it takes to earn back what it cost to acquire a customer.",
    "LTV": "Lifetime value -- the total gross margin a customer is expected to generate, bounded to a defensible horizon (see below).",
    "GRR": "Gross Revenue Retention -- % of a cohort's starting revenue still there later, ignoring upsells. Capped at 100%.",
    "NRR": "Net Revenue Retention -- % of a cohort's starting revenue still there later, including upsells. Can exceed 100%.",
    "Confidence interval (CI)": "A range that likely contains the true value, from resampling the actual customers (bootstrap) many times.",
    "Fit confidence": "Whether there's enough data (customers, non-censored) to trust a model fit at all.",
}


# ---------------------------------------------------------------------------
# Cached compute wrappers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load(customers_bytes: bytes, events_bytes: bytes, revenue_model: str, reactivation_window_months: int):
    customers = load_customers(io.BytesIO(customers_bytes))
    events = load_subscription_events(io.BytesIO(events_bytes))
    panel = build_customer_panel(customers, events, revenue_model, reactivation_window_months)
    return customers, events, panel


def _read_bytes(path_or_upload) -> bytes:
    if hasattr(path_or_upload, "getvalue"):
        return path_or_upload.getvalue()
    with open(path_or_upload, "rb") as f:
        return f.read()


@st.cache_data(show_spinner="Bootstrapping retention curve confidence interval...")
def _km_ci_cached(durations: np.ndarray, event_observed: np.ndarray, t_grid: np.ndarray):
    return bootstrap_km_ci(durations, event_observed, t_grid, n_boot=300, seed=42)


@st.cache_data(show_spinner="Bootstrapping payback period confidence interval...")
def _payback_ci_cached(panel_f: pd.DataFrame, customers_f: pd.DataFrame, segment_col: str, gross_margin: float):
    return bootstrap_payback_ci(panel_f, customers_f, segment_col=segment_col, gross_margin=gross_margin, n_boot=300, seed=42)


@st.cache_data(show_spinner="Bootstrapping payback curve confidence band...")
def _payback_curve_ci_cached(panel_f: pd.DataFrame, customers_f: pd.DataFrame, segment_col: str, gross_margin: float):
    return bootstrap_payback_curve_ci(panel_f, customers_f, segment_col=segment_col, gross_margin=gross_margin, n_boot=300, seed=42)


@st.cache_data(show_spinner="Running Monte Carlo sensitivity simulation...")
def _monte_carlo_cached(panel_f: pd.DataFrame, customers_f: pd.DataFrame, segment_col: str, gross_margin: float,
                         cac_shock_pct: float, retention_shift_pts: float, n_sims: int):
    return monte_carlo_payback_sensitivity(
        panel_f, customers_f, segment_col=segment_col, gross_margin=gross_margin,
        cac_shock_pct=cac_shock_pct, retention_shift_pts=retention_shift_pts, n_sims=n_sims, seed=42,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📒 Controls")

    with st.expander("Data source", expanded=True):
        source_name = st.selectbox("Dataset", list(DATA_SOURCES.keys()), index=0)
        source = DATA_SOURCES[source_name]

        if source_name == "Upload your own":
            cust_upload = st.file_uploader("customers.csv", type="csv")
            events_upload = st.file_uploader("subscription_events.csv", type="csv")
            if cust_upload is None or events_upload is None:
                st.info("Upload both files — same schema as the generator (see README).")
                customers_bytes = _read_bytes(DATA_SOURCES["Synthetic demo (subscription)"]["customers"])
                events_bytes = _read_bytes(DATA_SOURCES["Synthetic demo (subscription)"]["events"])
            else:
                customers_bytes = _read_bytes(cust_upload)
                events_bytes = _read_bytes(events_upload)
        else:
            customers_bytes = _read_bytes(source["customers"])
            events_bytes = _read_bytes(source["events"])
            st.caption(source["caption"])
            if source["assumed_cac"]:
                st.caption("⚠️ Churn/revenue behavior is real. Acquisition channel and CAC are assumed for stress-testing, not real for this company.")

        revenue_model = st.selectbox(
            "Revenue model", ["subscription", "blended"],
            index=["subscription", "blended"].index(source["revenue_model"]),
            help="subscription: MRR=0 this month means churned. blended: episodic/transactional revenue, "
                 "a customer stays 'active' if they transacted within a trailing reactivation window "
                 "(not just this exact month). See README methodology for why this matters.",
        )
        reactivation_window_months = DEFAULT_REACTIVATION_WINDOW_MONTHS
        if revenue_model == "blended":
            reactivation_window_months = st.slider(
                "Reactivation window (months)", min_value=1, max_value=6, value=DEFAULT_REACTIVATION_WINDOW_MONTHS,
                help="A customer counts as active if they transacted within this many trailing months. "
                     "This is a stated business assumption, not something fit from the data.",
            )

    try:
        customers, events, panel = _load(customers_bytes, events_bytes, revenue_model, reactivation_window_months)
    except ValueError as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    with st.expander("Segment & filters", expanded=True):
        # Any customer-level text column can be a segmentation dimension, not just
        # acquisition_channel -- this is what lets Telco segment by real fields
        # (Contract type, internet service, payment method) instead of a fabricated
        # channel. See README "Methodology" -> Phase 2.
        _excluded_cols = {"customer_id", "signup_date"}
        segment_candidates = [c for c in customers.columns if c not in _excluded_cols and customers[c].dtype == object]
        default_segment = source["default_segment"] if source["default_segment"] in segment_candidates else segment_candidates[0]
        segment_col = st.selectbox(
            "Segment by", segment_candidates, index=segment_candidates.index(default_segment),
            help="Which field to segment cohort retention, payback, LTV, and GRR/NRR by. Prefers a real field "
                 "when the dataset has one; falls back to the assumed acquisition_channel otherwise.",
        )
        if source.get("survivorship_caveat"):
            st.warning(
                "Telco's cohort retention heatmap and per-segment payback/LTV carry a known survivorship-bias "
                "caveat (see README Methodology) -- use the Kaplan-Meier survival curve below instead for "
                "Telco's real retention story; it's built directly from tenure + churn outcome, not fake cohorts.",
                icon="⚠️",
            )

        segment_values = sorted(customers[segment_col].dropna().unique().tolist())
        selected_segments = st.multiselect(f"{segment_col} filter", segment_values, default=segment_values)

    with st.expander("Assumptions", expanded=True):
        gross_margin = st.slider(
            "Gross margin assumption", min_value=0.50, max_value=0.95, value=DEFAULT_GROSS_MARGIN, step=0.01,
            help="Applied to MRR to get gross margin $, used in CAC payback and LTV. See README for rationale.",
        )
        st.markdown("**Monte Carlo sensitivity ranges**")
        cac_shock_pct = st.slider(
            "CAC shock range (±%)", min_value=0, max_value=50, value=20, step=5,
            help="Each simulated run draws a random CAC shock uniformly from this +/- range.",
        )
        retention_shift_pts = st.slider(
            "Retention shift range (±pts)", min_value=0, max_value=20, value=5, step=1,
            help="Each simulated run draws a random retention shift (percentage points) uniformly from this +/- range.",
        )
        n_sims = st.slider("Number of simulated runs", min_value=200, max_value=3000, value=1000, step=200)

if not selected_segments:
    st.warning(f"No {segment_col} values selected. Choose at least one in the sidebar to see the dashboard.")
    st.stop()

customers_f = customers[customers[segment_col].isin(selected_segments)]
panel_f = panel[panel[segment_col].isin(selected_segments)]

# Color assigned once per segment value, from the full (unfiltered) sorted list --
# not per-chart sort order -- so a segment is always the same color on every
# chart on the page, even when one chart happens to sort by ltv_to_cac and
# another by tenure_month. "Color follows the entity, never its rank."
SEGMENT_COLOR_MAP = {seg: SERIES_COLORS[i % len(SERIES_COLORS)] for i, seg in enumerate(segment_values)}

if len(customers_f) == 0 or panel_f["is_active"].sum() == 0:
    st.error(
        "No usable data for this combination of dataset, segment, and filters. Try selecting a different "
        "dataset or widening the segment filter in the sidebar."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Compute -- every number the page needs, built once, up front
# ---------------------------------------------------------------------------

payback_summary, payback_curves = cac_payback_by_channel(panel_f, customers_f, gross_margin=gross_margin, segment_col=segment_col)
payback_ci = _payback_ci_cached(panel_f, customers_f, segment_col, gross_margin)
payback_summary = payback_summary.merge(payback_ci, on=segment_col, how="left")
payback_curve_ci = _payback_curve_ci_cached(panel_f, customers_f, segment_col, gross_margin)

ltvcac = ltv_to_cac_by_channel(panel_f, customers_f, gross_margin=gross_margin, segment_col=segment_col)
ltvcac = ltvcac[ltvcac[segment_col].isin(selected_segments)].sort_values("ltv_to_cac", ascending=False)

mc_summary, mc_distributions = _monte_carlo_cached(
    panel_f, customers_f, segment_col, gross_margin, float(cac_shock_pct), float(retention_shift_pts), n_sims
)

durations_df = extract_durations(panel_f)
max_dur = int(durations_df["duration"].max())
km_t_grid = np.arange(0, max_dur + 1, max(1, max_dur // 40))
km_ci = _km_ci_cached(durations_df["duration"].to_numpy(), durations_df["event_observed"].to_numpy(), km_t_grid)

max_month = panel_f["month"].max()
n_cohorts = customers_f["cohort_month"].nunique()


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero"><span class="eyebrow">Cohort · Payback · Loss-curve engine</span>'
    "<p>This dashboard answers the four questions every subscription-business board asks: "
    "which signup cohorts are actually sticking around, how many months it takes to earn back "
    "what was spent acquiring a customer, whether that spend is worth it long-term, and how much "
    "revenue survives churn versus grows from upsells -- each one with its uncertainty shown, not hidden.</p></div>",
    unsafe_allow_html=True,
)
st.caption(f"Data source: {source_name} · Revenue model: {revenue_model} · Segmented by {segment_col}")

kpi_cols = st.columns(4)
kpi_cols[0].metric("Customers analyzed", f"{customers_f['customer_id'].nunique():,}")
kpi_cols[1].metric("Signup cohorts", n_cohorts)
kpi_cols[2].metric("Blended avg CAC", f"${customers_f['cac'].mean():,.0f}")
kpi_cols[3].metric("Data through", max_month.strftime("%b %Y"))

st.divider()

# ---------------------------------------------------------------------------
# Key Insights
# ---------------------------------------------------------------------------

st.header("Key insights")
st.caption(
    "Everything below is computed from the tables further down this page -- nothing here is generated commentary. "
    "Observational, not prescriptive: what the numbers show, not what to do about it."
)

ranked = payback_summary.copy()
ranked["_sort_key"] = ranked["payback_month"].astype("Float64").fillna(9999)
ranked = ranked.sort_values("_sort_key").reset_index(drop=True)
ranked.index = ranked.index + 1  # 1-indexed rank

# --- Flagship stat 2 computed first: Phase 4 Monte Carlo finding (dynamic --
# most resilient vs. most fragile under the currently configured shock ranges) ---
flagship_2_html = None
mc_most_fragile_segment = None
if len(mc_summary) >= 2 and mc_summary["pct_sims_reached_payback"].notna().any():
    mc_valid = mc_summary.dropna(subset=["pct_sims_reached_payback"])
    most_resilient = mc_valid.sort_values("pct_sims_reached_payback", ascending=False).iloc[0]
    most_fragile = mc_valid.sort_values("pct_sims_reached_payback", ascending=True).iloc[0]
    if most_resilient[segment_col] != most_fragile[segment_col]:
        mc_most_fragile_segment = most_fragile[segment_col]
        flagship_2_html = (
            f'<div class="flagship-card"><span class="label">Resilience under simulated shocks</span>'
            f'<div class="stat">{most_resilient[segment_col]}: {most_resilient["pct_sims_reached_payback"]*100:.0f}% '
            f'vs. {most_fragile[segment_col]}: {most_fragile["pct_sims_reached_payback"]*100:.0f}%</div>'
            f'<div class="detail">share of {n_sims:,} Monte Carlo runs (±{cac_shock_pct}% CAC, ±{retention_shift_pts}pt retention shocks) '
            f"that reached payback at all -- direct evidence of which segments are fragile vs. resilient to bad-case "
            f"assumptions, not just which has the better point estimate.</div></div>"
        )

# --- Flagship stat 1: Phase 3 bootstrap finding (dynamic -- prefers a segment
# with a low but NONZERO bootstrap reach rate, since "occasionally crosses"
# is the more informative, less obvious finding than "literally never does"
# -- and avoids re-featuring whichever segment flagship 2 already leads with) ---
never_reached = payback_summary[payback_summary["payback_month"].isna()].copy()
flagship_1_html = None
if len(never_reached):
    never_reached["_prefer_nonzero"] = (never_reached["pct_bootstrap_reached_payback"] == 0)
    if mc_most_fragile_segment is not None and (never_reached[segment_col] != mc_most_fragile_segment).any():
        never_reached = never_reached[never_reached[segment_col] != mc_most_fragile_segment]
    worst = never_reached.sort_values(["_prefer_nonzero", "pct_bootstrap_reached_payback"]).iloc[0]
    flagship_1_html = (
        f'<div class="flagship-card"><span class="label">Point estimate vs. bootstrap reality</span>'
        f'<div class="stat">{worst[segment_col]}: "not yet reached"</div>'
        f'<div class="detail">but only <strong>{worst["pct_bootstrap_reached_payback"]*100:.0f}%</strong> of 300 bootstrap '
        f"resamples of this segment's actual customers ever crossed breakeven within the observed window -- "
        f"the point estimate and the resampled reality agree on the direction, not just a rounding difference.</div></div>"
    )

if flagship_1_html or flagship_2_html:
    st.markdown(
        f'<div class="flagship-grid">{flagship_1_html or ""}{flagship_2_html or ""}</div>',
        unsafe_allow_html=True,
    )

# --- Segment ranking (custom HTML table so fit-confidence badges can render in-cell) ---
st.subheader("Segment ranking, by payback speed")
fit_conf_by_segment = ltvcac.set_index(segment_col)["fit_confidence"].to_dict() if "fit_confidence" in ltvcac.columns else {}

ROW_STYLE = 'style="border-bottom:1px solid rgba(255,255,255,0.06);"'
CELL_STYLE = 'style="padding:0.4rem 0.6rem;"'
body_rows = []
for _, row in ranked.iterrows():
    payback_str = "not yet reached" if pd.isna(row["payback_month"]) else f'{row["payback_month"]:.0f} mo'
    ci_str = "n/a" if pd.isna(row["payback_ci_lo"]) else f'{row["payback_ci_lo"]:.0f}–{row["payback_ci_hi"]:.0f} mo'
    pct_str = f'{row["pct_bootstrap_reached_payback"]*100:.0f}%'
    badge = confidence_badge(fit_conf_by_segment.get(row[segment_col], "ok"))
    body_rows.append(
        f"<tr {ROW_STYLE}><td {CELL_STYLE}>{row.name}</td><td {CELL_STYLE}><strong>{row[segment_col]}</strong></td>"
        f"<td {CELL_STYLE}>{payback_str}</td><td {CELL_STYLE}>{ci_str}</td><td {CELL_STYLE}>{pct_str}</td>"
        f"<td {CELL_STYLE}>{badge}</td></tr>"
    )

header_style = 'style="border-bottom:1px solid rgba(255,255,255,0.15); text-align:left; color:#9aa0a6; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.03em;"'
st.markdown(
    f'<table style="width:100%; border-collapse:collapse; font-size:0.92rem;">'
    f'<thead><tr {header_style}>'
    f'<th {CELL_STYLE}>Rank</th><th {CELL_STYLE}>{segment_col.replace("_"," ").title()}</th>'
    f'<th {CELL_STYLE}>Payback</th><th {CELL_STYLE}>95% CI</th>'
    f'<th {CELL_STYLE}>% bootstrap reached</th><th {CELL_STYLE}>Fit confidence</th></tr></thead>'
    f'<tbody>{"".join(body_rows)}</tbody>'
    f'</table>',
    unsafe_allow_html=True,
)

# --- Confidence-aware top-2 comparison ---
if len(ranked) >= 2:
    top, second = ranked.iloc[0], ranked.iloc[1]
    if pd.notna(top["payback_month"]) and pd.notna(second["payback_month"]) and pd.notna(top["payback_ci_lo"]) and pd.notna(second["payback_ci_lo"]):
        overlap = not (top["payback_ci_hi"] < second["payback_ci_lo"] or second["payback_ci_hi"] < top["payback_ci_lo"])
        if overlap:
            st.markdown(
                f'<div class="insight-row not-meaningful">'
                f'<strong>{top[segment_col]}</strong> ({top["payback_month"]:.0f}mo) edges out '
                f'<strong>{second[segment_col]}</strong> ({second["payback_month"]:.0f}mo) on point estimate, but their '
                f"95% bootstrap intervals overlap ({top['payback_ci_lo']:.0f}–{top['payback_ci_hi']:.0f} vs. "
                f"{second['payback_ci_lo']:.0f}–{second['payback_ci_hi']:.0f}) -- "
                f"<strong>not a statistically meaningful difference</strong> between the top two.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="insight-row">'
                f'<strong>{top[segment_col]}</strong> pays back faster than <strong>{second[segment_col]}</strong> '
                f"({top['payback_month']:.0f}mo vs. {second['payback_month']:.0f}mo) with non-overlapping 95% bootstrap "
                f"intervals ({top['payback_ci_lo']:.0f}–{top['payback_ci_hi']:.0f} vs. "
                f"{second['payback_ci_lo']:.0f}–{second['payback_ci_hi']:.0f}) -- outside the margin of error.</div>",
                unsafe_allow_html=True,
            )

# --- Driver decomposition + benchmark flagging, against the current leader ---
st.subheader(f"Why segments lag the leader (benchmark: payback within {BENCHMARK_PAYBACK_WINDOW_MONTHS} months)")
best_row = ranked.iloc[0]
best_segment = best_row[segment_col]
best_cac = best_row["avg_cac"]
at_risk = ranked[(ranked["_sort_key"] > BENCHMARK_PAYBACK_WINDOW_MONTHS) & (ranked[segment_col] != best_segment)]

if len(at_risk) == 0:
    st.markdown(
        f'<div class="insight-row">Every selected segment reaches payback within '
        f"{BENCHMARK_PAYBACK_WINDOW_MONTHS} months at the current gross margin assumption -- none flagged at-risk.</div>",
        unsafe_allow_html=True,
    )
else:
    for _, row in at_risk.iterrows():
        seg = row[segment_col]
        if seg not in payback_curves or best_segment not in payback_curves:
            continue
        try:
            bridge = payback_ratio_bridge(payback_curves[best_segment], best_cac, payback_curves[seg], row["avg_cac"])
        except KeyError:
            continue
        driver = "CAC" if abs(bridge["cac_effect"]) > abs(bridge["margin_effect"]) else "retention/margin"
        payback_str = "not yet reached" if pd.isna(row["payback_month"]) else f'{row["payback_month"]:.0f} months'
        st.markdown(
            f'<div class="insight-row at-risk">'
            f'<strong>{seg}</strong>: {payback_str} (vs. {best_segment} at {best_row["payback_month"]:.0f} months). '
            f"At month {bridge['reference_month']}, {best_segment}'s payback ratio leads {seg}'s by "
            f"{bridge['total_gap']:.2f}x -- of that, {bridge['cac_effect']:+.2f}x from the CAC difference and "
            f"{bridge['margin_effect']:+.2f}x from retention/margin performance. "
            f"Primarily driven by <strong>{driver}</strong>.</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# 1. Cohort retention heatmap
# ---------------------------------------------------------------------------

st.header("Cohort retention")
st.markdown(f'<p class="term-def"><strong>Payback period</strong>: {TERM_DEFINITIONS["Payback period"]}</p>', unsafe_allow_html=True)
if revenue_model == "blended":
    st.caption(f"% of each signup cohort transacted within the trailing {reactivation_window_months}-month window, "
               f"by months since signup. Blank cells haven't happened yet.")
else:
    st.caption("% of each signup cohort with MRR > 0, by months since signup. Blank cells haven't happened yet.")

retention_table = cohort_retention_table(panel_f)
if retention_table.empty:
    st.info("No cohort has enough coverage to display a retention heatmap for this selection.")
else:
    fig_retention = px.imshow(
        retention_table,
        color_continuous_scale=HEATMAP_SCALE,
        aspect="auto",
        labels=dict(x="Months since signup", y="Signup cohort", color="Retention %"),
        zmin=0, zmax=100,
    )
    fig_retention.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_retention, use_container_width=True)

st.subheader("Pooled survival curve (Kaplan-Meier, 95% bootstrap CI)")
st.markdown(f'<p class="term-def"><strong>Confidence interval (CI)</strong>: {TERM_DEFINITIONS["Confidence interval (CI)"]}</p>', unsafe_allow_html=True)
st.caption("Built directly from each customer's own tenure + churn outcome, not bucketed into signup cohorts -- "
           "the correct view when signup dates are reconstructed (see Telco) rather than real, and a useful "
           "cross-check even when they are. Shaded band = 95% bootstrap confidence interval.")

fig_km = go.Figure()
fig_km.add_trace(go.Scatter(
    x=np.concatenate([km_ci["t"], km_ci["t"][::-1]]),
    y=np.concatenate([km_ci["ci_hi"], km_ci["ci_lo"][::-1]]),
    fill="toself", fillcolor="rgba(87,181,147,0.18)", line=dict(color="rgba(0,0,0,0)"),
    name="95% CI", hoverinfo="skip",
))
fig_km.add_trace(go.Scatter(x=km_ci["t"], y=km_ci["survival_prob"], mode="lines", name="Survival", line=dict(color=ACCENT, width=2.5)))
fig_km.update_layout(
    xaxis_title="Months since signup", yaxis_title="% still retained",
    yaxis_tickformat=".0%", height=420, margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_km, use_container_width=True)
censoring_frac = 1 - durations_df["event_observed"].mean()
st.caption(f"n={len(durations_df):,} customers · {durations_df['event_observed'].sum():,} observed churn events "
           f"· {censoring_frac:.0%} still active / right-censored as of the last observed month.")

st.divider()

# ---------------------------------------------------------------------------
# 2. CAC payback curve
# ---------------------------------------------------------------------------

st.header(f"CAC payback by {segment_col}")
st.markdown(f'<p class="term-def"><strong>Fit confidence</strong>: {TERM_DEFINITIONS["Fit confidence"]}</p>', unsafe_allow_html=True)
st.caption("Cumulative gross margin per customer ÷ CAC. Payback month = first crossing of 1.0. "
           "Side-by-side across every selected segment value, never blended. Shaded bands = 95% bootstrap CI.")

fig_payback = go.Figure()
for segment_value in selected_segments:
    if segment_value not in payback_curves:
        continue
    color = SEGMENT_COLOR_MAP[segment_value]
    curve = payback_curves[segment_value]
    if segment_value in payback_curve_ci and len(payback_curve_ci[segment_value]):
        band = payback_curve_ci[segment_value]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig_payback.add_trace(go.Scatter(
            x=pd.concat([band["tenure_month"], band["tenure_month"][::-1]]),
            y=pd.concat([band["ratio_hi"], band["ratio_lo"][::-1]]),
            fill="toself", fillcolor=f"rgba({r},{g},{b},0.15)", line=dict(color="rgba(0,0,0,0)"),
            name=f"{segment_value} 95% CI", hoverinfo="skip", showlegend=False,
        ))
    fig_payback.add_trace(go.Scatter(
        x=curve["tenure_month"], y=curve["payback_ratio"],
        mode="lines+markers", name=str(segment_value), line=dict(color=color),
    ))
fig_payback.add_hline(y=1.0, line_dash="dash", line_color="#898781", annotation_text="Breakeven (1.0x)")
fig_payback.update_layout(
    xaxis_title="Months since signup", yaxis_title="Cumulative margin ÷ CAC",
    height=450, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_payback, use_container_width=True)

payback_display = payback_summary.rename(columns={
    segment_col: segment_col.replace("_", " ").title(), "avg_cac": "Avg CAC",
    "payback_month": "Payback (months)", "max_observed_tenure": "Max observed tenure",
    "payback_ci_lo": "95% CI low", "payback_ci_hi": "95% CI high",
    "pct_bootstrap_reached_payback": "% bootstrap reps that reached payback",
})
payback_display["Payback (months)"] = payback_display["Payback (months)"].apply(
    lambda v: "not yet reached" if pd.isna(v) else str(int(v))
)
for col in ["95% CI low", "95% CI high"]:
    payback_display[col] = payback_display[col].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0f}")
payback_display["% bootstrap reps that reached payback"] = (payback_display["% bootstrap reps that reached payback"] * 100).round(0)
st.dataframe(payback_display.drop(columns=["n_bootstrap"]), hide_index=True, use_container_width=True)
st.caption("95% CI from 300 bootstrap resamples of customers within each segment. \"% bootstrap reps that reached "
           "payback\" below 100% means some resamples never crossed breakeven within the observed window -- "
           "the CI is computed only from the replicates that did.")

st.divider()

# ---------------------------------------------------------------------------
# 3. LTV:CAC by channel
# ---------------------------------------------------------------------------

st.header(f"LTV:CAC by {segment_col}")
st.markdown(f'<p class="term-def"><strong>LTV</strong>: {TERM_DEFINITIONS["LTV"]}</p>', unsafe_allow_html=True)
st.caption("LTV modeled to convergence from each segment's retention curve (not a fixed window). See README for method.")

if ltvcac.empty:
    st.info("No segment has enough data to fit a survival model for LTV under this selection.")
else:
    fig_ltvcac = px.bar(
        ltvcac, x=segment_col, y="ltv_to_cac", color=segment_col,
        color_discrete_map=SEGMENT_COLOR_MAP,
        text_auto=".2f", labels={segment_col: segment_col.replace("_", " ").title(), "ltv_to_cac": "LTV : CAC"},
    )
    fig_ltvcac.add_hline(y=1.0, line_dash="dash", line_color="#898781", annotation_text="Breakeven (1x)")
    fig_ltvcac.add_hline(y=3.0, line_dash="dot", line_color=GOOD, annotation_text="Healthy benchmark (3x)")
    fig_ltvcac.update_layout(height=450, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ltvcac, use_container_width=True)

    ltv_display = ltvcac.rename(columns={
        segment_col: segment_col.replace("_", " ").title(), "expected_active_months": "Expected active months (bounded)",
        "extrapolation_horizon_months": "Defensible through (months)", "arpa": "ARPA (active)", "ltv": "LTV (bounded)",
        "avg_cac": "Avg CAC", "ltv_to_cac": "LTV:CAC", "survival_model": "Model",
        "fit_confidence": "Fit confidence", "extrapolation_confidence": "Beyond-horizon confidence",
    })
    st.dataframe(ltv_display, hide_index=True, use_container_width=True)
    st.caption("LTV and expected active months are bounded estimates, defensible through \"Defensible through (months)\" "
               "-- not a claim about lifetime value beyond that. \"Fit confidence\" is about whether there's enough data "
               "to trust the model at all; \"Beyond-horizon confidence\" is a separate signal for whether there's likely "
               "more value past the horizon that this number doesn't count. See README Methodology.")

st.divider()

# ---------------------------------------------------------------------------
# 4. GRR / NRR loss curves
# ---------------------------------------------------------------------------

st.header("Loss curves: GRR & NRR")
st.markdown(
    f'<p class="term-def"><strong>GRR</strong>: {TERM_DEFINITIONS["GRR"]} &nbsp;·&nbsp; '
    f'<strong>NRR</strong>: {TERM_DEFINITIONS["NRR"]}</p>',
    unsafe_allow_html=True,
)
st.caption("Revenue retained from a cohort's starting MRR base. GRR excludes expansion (capped at 100%); NRR includes it.")

view_mode = st.radio("View", ["Blended (all selected segments)", "By cohort", f"By {segment_col}"], horizontal=True)

if view_mode == "Blended (all selected segments)":
    blended = panel_f.copy()
    blended["_segment"] = "Blended"
    grr_table = grr_nrr_table(blended, group_col="_segment")

    fig_grr = go.Figure()
    fig_grr.add_trace(go.Scatter(x=grr_table["tenure_month"], y=grr_table["grr_pct"], name="GRR", mode="lines+markers", line=dict(color=SERIES_COLORS[0])))
    fig_grr.add_trace(go.Scatter(x=grr_table["tenure_month"], y=grr_table["nrr_pct"], name="NRR", mode="lines+markers", line=dict(color=SERIES_COLORS[1])))
    fig_grr.add_hline(y=100, line_dash="dash", line_color="#898781")
    fig_grr.update_layout(
        xaxis_title="Months since cohort start", yaxis_title="% of starting MRR retained",
        height=450, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_grr, use_container_width=True)
elif view_mode == "By cohort":
    metric_choice = st.selectbox("Metric", ["GRR", "NRR"])
    metric_col = "grr_pct" if metric_choice == "GRR" else "nrr_pct"
    cohort_grr = grr_nrr_table(panel_f, group_col="cohort_month")

    fig_cohort_grr = px.line(
        cohort_grr, x="tenure_month", y=metric_col, color="cohort_month",
        labels={"tenure_month": "Months since cohort start", metric_col: f"{metric_choice} %", "cohort_month": "Cohort"},
        color_discrete_sequence=px.colors.sequential.Emrld,
    )
    fig_cohort_grr.add_hline(y=100, line_dash="dash", line_color="#898781")
    fig_cohort_grr.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cohort_grr, use_container_width=True)
else:
    metric_choice = st.selectbox("Metric", ["GRR", "NRR"], key="segment_grr_metric")
    metric_col = "grr_pct" if metric_choice == "GRR" else "nrr_pct"
    segment_grr = grr_nrr_table(panel_f, group_col=segment_col)

    fig_segment_grr = px.line(
        segment_grr, x="tenure_month", y=metric_col, color=segment_col,
        color_discrete_map={str(k): v for k, v in SEGMENT_COLOR_MAP.items()},
        labels={"tenure_month": "Months since cohort start", metric_col: f"{metric_choice} %",
                segment_col: segment_col.replace("_", " ").title()},
    )
    fig_segment_grr.add_hline(y=100, line_dash="dash", line_color="#898781")
    fig_segment_grr.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_segment_grr, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. Monte Carlo sensitivity on the payback period
# ---------------------------------------------------------------------------

st.header(f"Monte Carlo payback sensitivity by {segment_col}")
st.caption(
    f"Simulated distribution of the payback month under a random CAC shock (±{cac_shock_pct}%) and a random "
    f"retention shift (±{retention_shift_pts}pts) drawn independently per run -- not two fixed scenarios. "
    "The same simulated draws are reused across every segment, so differences between segments below reflect "
    "how sensitive each one is to the *same* shock, not different randomness. Adjust the ranges in the sidebar."
)

hist_rows = []
for segment_value in selected_segments:
    if segment_value not in mc_distributions:
        continue
    values = mc_distributions[segment_value]
    reached = values[~np.isnan(values)]
    if len(reached) == 0:
        continue
    hist_rows.append(pd.DataFrame({segment_col: str(segment_value), "payback_month": reached}))

if hist_rows:
    hist_df = pd.concat(hist_rows, ignore_index=True)
    fig_mc = px.histogram(
        hist_df, x="payback_month", color=segment_col, barmode="overlay", opacity=0.65, nbins=30,
        color_discrete_map={str(k): v for k, v in SEGMENT_COLOR_MAP.items()},
        labels={"payback_month": "Simulated payback month", segment_col: segment_col.replace("_", " ").title()},
    )
    fig_mc.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Simulated runs", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_mc, use_container_width=True)
else:
    st.info("No simulated run reached payback for any selected segment within the observed window.")

mc_display = mc_summary.rename(columns={
    segment_col: segment_col.replace("_", " ").title(),
    "n_customers": "N customers",
    "pct_sims_reached_payback": "% runs that reached payback",
    "median_payback_month": "Median payback (months)",
    "p10_payback_month": "P10 (months)",
    "p90_payback_month": "P90 (months)",
    "fit_confidence": "Fit confidence",
})
mc_display["% runs that reached payback"] = (mc_display["% runs that reached payback"] * 100).round(0)
st.dataframe(mc_display, hide_index=True, use_container_width=True)
st.caption(
    "P10/P90 = the middle 80% range of simulated payback months, among runs that reached it at all. "
    "\"Fit confidence\" carries forward the same data-sufficiency check from the LTV/survival model (Phase 3) -- "
    "a segment already too thin or too censored to trust at the point-estimate level doesn't get to look more "
    "precise just because the output here is a distribution instead of a single number."
)

with st.expander("Glossary"):
    for term, definition in TERM_DEFINITIONS.items():
        st.markdown(f"**{term}** — {definition}")
