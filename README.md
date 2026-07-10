# Cohort / Payback / Loss-Curve Dashboard

View it at: https://cohort-payback-loss-curve.streamlit.app/

A live cohort retention, CAC payback, LTV:CAC, and GRR/NRR dashboard for a subscription
business, built on a synthetic fintech personal-finance-management (PFM) dataset and a
reusable calculation engine. Point the engine at a real `customers.csv` /
`subscription_events.csv` export and it produces the same four views against real data.

## Why this exists

Seed/Series A finance and BI teams get asked for cohort retention, CAC payback, and loss
curves by investors and boards, and most don't have this instrumented yet. This repo is
that artifact — a working dashboard plus the engine underneath it, structured so a real
company's billing/CRM export can be dropped in without touching the calculation logic.

## Repo structure

```
data_gen/       synthetic dataset generator (not used in the calculation path)
engine/
  loaders.py      data ingestion + validation -- the schema boundary. Swap real data in here.
  metrics.py       all four metrics: cohort retention, CAC payback, LTV:CAC, GRR/NRR
tests/
  test_metrics.py  unit tests against a hand-calculated 4-customer fixture
data/            generated CSVs (customers.csv, subscription_events.csv)
app.py           Streamlit UI -- renders only; all numbers come from engine/
```

## Running locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 -m data_gen.generate      # writes data/customers.csv, data/subscription_events.csv
pytest tests/                     # verify the engine against hand-calculated examples
streamlit run app.py
```

## Input schema (the reusable part)

**customers.csv**

| column | type | notes |
|---|---|---|
| `customer_id` | str | unique |
| `signup_date` | date | any day-of-month; floored to month start |
| `acquisition_channel` | str | e.g. `paid_social`, `organic_referral` |
| `initial_plan` | str | informational only, not used in calculations |
| `cac` | float | fully-loaded cost to acquire this customer |

**subscription_events.csv**

| column | type | notes |
|---|---|---|
| `customer_id` | str | foreign key into `customers.customer_id` |
| `month` | date | calendar month of the billing event |
| `mrr` | float | that customer's MRR that month; `0` if churned |

To point the dashboard at real data: toggle off "Use synthetic demo data" in the sidebar
and upload two CSVs in this shape. Everything downstream — cohort table, payback curves,
LTV:CAC, loss curves — recomputes automatically. No code changes required if the schema
matches; if a real export uses different column names, only `engine/loaders.py` needs to
change.

## Methodology

This is the part to read before answering "how did you calculate X" in a conversation.

### Cohort retention

`% of signup cohort with MRR > 0 at month N since signup`, computed as a dense
customer × calendar-month grid from each customer's signup month through the dataset's
global max observed month. **A missing billing row within that window is treated as
churned (MRR = 0)** — this is a deliberate assumption, since most billing systems stop
emitting rows once a subscription is canceled rather than emitting explicit `$0` rows
forever. If a real export behaves differently (e.g. keeps emitting `$0` rows), the
result is identical either way.

Cells for `(cohort, tenure)` combinations that haven't happened yet (e.g. a cohort that
signed up 3 months ago doesn't have a 12-month data point) are left blank/`NaN`, not
zero — this is why the heatmap is a triangle, not a full rectangle. Treating unobserved
future cells as 0% would understate recent cohorts and bias every downstream average.

### CAC payback (months, per channel)

`payback_month = min(t)` such that `avg_cumulative_gross_margin_per_customer(t) / avg_CAC(channel) >= 1.0`,
computed independently per channel (never blended across channels — a channel with
$300 CAC and a channel with $65 CAC have completely different payback economics, and
averaging them would produce a number nobody could act on).

`avg_cumulative_gross_margin_per_customer(t)` is the mean, across that channel's
customers, of `sum(MRR_i * gross_margin_pct for i in 0..t)`. Two details matter for
correctness:

1. **Gross margin, not revenue.** Payback is measured against the cash actually
   recovered per customer, not top-line MRR. Default assumption: 75% gross margin
   (typical range for a fintech PFM SaaS business after infra, support, and payment
   processing costs) — adjustable via the sidebar slider, since this is the single
   biggest lever on the reported payback month and should be swapped for the real
   company's actual gross margin the moment real data is used.
2. **Right-censoring.** A customer who signed up last month can't have 12 months of
   observed margin yet. Averaging in a phantom `$0` for months they haven't reached
   would bias `avg_cumulative_margin` down and make payback look worse than it is for
   channels with a lot of recent volume. The engine only averages over customers who
   have actually been observed for `t` months, and only reports a tenure month once at
   least 20% of the channel's customers have reached it (`MIN_COHORT_COVERAGE` in
   `engine/metrics.py`) — past that point the average is too small-sample to trust, so
   the curve simply stops rather than showing a noisy tail.

If the ratio never crosses 1.0 within the reliably-covered window, payback is reported
as **"not yet reached"** rather than guessing — a real, honest answer for immature
channels (this happens for `paid_search` and `paid_social` in the demo data), not a bug.

### LTV : CAC

`LTV(channel) = expected_active_months(channel) × ARPA(channel) × gross_margin_pct`

- **`expected_active_months`** is modeled from the channel's empirical retention curve,
  extrapolated to convergence rather than cut off at an arbitrary window (e.g. 24
  months). Rationale: a fixed window either truncates real value for channels with a
  long tail of retained customers, or is generous to channels that happen to have more
  history in the dataset. Modeling to convergence avoids both problems and is the more
  defensible approach for a board-facing number. Concretely: observed monthly survival
  (`retention_pct / 100`) is summed directly for tenures with sufficient cohort
  coverage; beyond the last reliably-observed point, a constant tail hazard (the
  average monthly churn rate implied by the last 3 observed points) is used to
  extrapolate survival geometrically until it's negligible (< 0.1%) or a 240-month cap
  is hit. This is the standard SaaS "retention-curve LTV" approach, not a novel method.
- **`ARPA(channel)`** is the average MRR among that channel's *active* customer-months
  only (churned $0 months excluded) — this reflects the upsell/downgrade mix without
  being diluted by the churn already captured in `expected_active_months`.
- **`CAC`** is the simple average CAC across the channel's customers ("blended" per
  channel, as specified) — not fully-loaded with sales/marketing overhead beyond what's
  in each customer's recorded `cac`.

A fixed-window LTV (e.g. 24-month cumulative margin, computed the same coverage-aware
way as CAC payback) is a reasonable alternative and is straightforward to add if a
specific investor wants that number instead — see `_avg_cumulative_margin_by_tenure` in
`engine/metrics.py`, which the payback calculation already uses and which could be
reused directly for a windowed LTV.

### GRR and NRR (loss curves)

Both are computed relative to a cohort's (or channel's, or the whole business's)
**tenure-0 MRR base** — the standard cohort-revenue-retention convention:

```
NRR(t) = sum(MRR_t across cohort) / sum(MRR_0 across cohort)                    -- can exceed 100%
GRR(t) = sum(min(MRR_t, MRR_0) per customer) / sum(MRR_0 across cohort)         -- capped at 100%
```

The `min(MRR_t, MRR_0)` per-customer cap is what makes GRR mathematically incapable of
exceeding 100%: an upsell (`MRR_t > MRR_0`) is clipped back down to that customer's own
starting MRR before summing, so expansion revenue never enters the GRR numerator, while
churn and downgrades (`MRR_t < MRR_0`) still pull it down. NRR uses uncapped `MRR_t`, so
expansion shows up directly and can push the ratio above 100%. This is a more rigorous
formulation than "NRR minus expansion revenue" (which requires separately classifying
each dollar of MRR change as expansion vs. contraction) — capping per customer gets the
same answer without that bookkeeping and is harder to get subtly wrong.

### Worked example

`tests/test_metrics.py` has a 4-customer, 2-channel, 2-cohort fixture with every number
computed by hand in comments — the fastest way to see the arithmetic end to end (or to
sanity-check a claim against real data later) is to read that file.

## Synthetic data generator

`data_gen/generate.py` produces ~2,000 customers across 18 monthly signup cohorts and 4
acquisition channels with deliberately different unit economics, so the dashboard tells
a real story rather than a flat blob:

| Channel | CAC | Retention story |
|---|---|---|
| `organic_referral` | low (~$65) | mediocre long-run retention |
| `content_seo` | medium (~$150) | best long-run retention — the "invest here" channel |
| `paid_search` | medium-high (~$200) | mediocre, payback not yet observed |
| `paid_social` | high (~$300) | decent early retention, steep early drop-off |

Mechanics:
- **Hazard-rate churn model**: each active customer has a monthly churn probability
  that starts high and decays geometrically toward a per-channel floor, producing the
  classic steep-then-flattening retention curve real subscription businesses show.
- **Cohort-over-cohort improvement**: churn hazard is scaled down for later signup
  cohorts (down to 75% of the original hazard by the most recent cohort), so there's a
  genuine "the business is getting better at retention" trend in the data, not just
  noise.
- **Upsell/downgrade noise**: active customers occasionally move a plan tier up or down
  each month, which is what makes NRR diverge from GRR.
- **Fixed calendar observation window** (24 months from the first signup cohort): later
  cohorts are naturally right-censored, exactly like a real billing export, which is
  why the engine has to handle partial cohort observation correctly rather than
  assuming every cohort has full history.

Reproducible via a fixed seed (`SEED = 42` in `data_gen/generate.py`).

## Assumptions & limitations

- **75% gross margin** is a placeholder for a fintech PFM SaaS business — swap it via
  the sidebar slider (demo) or `DEFAULT_GROSS_MARGIN` in `engine/metrics.py` (real
  deployment) for the actual company's cost structure.
- **No reactivation**: once a synthetic customer churns, they stay churned. Real
  businesses see some win-back; the engine itself doesn't assume this either way — it
  just reflects whatever `MRR` sequence is in the data.
- **CAC is taken as given per customer**, not derived from a marketing spend ÷
  customers-acquired calculation. A real integration would need that reconciliation
  done upstream (finance and marketing usually already have this).
- **Coverage threshold (20%)** on payback/LTV curves is a judgment call trading off
  "show more of the curve" against "don't show a noisy 2-customer average." Tune
  `MIN_COHORT_COVERAGE` in `engine/metrics.py` if a specific audience wants a different
  bar.
