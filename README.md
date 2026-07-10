# Cohort / Payback / Loss-Curve Dashboard

**Live app:** https://cohort-payback-loss-curve.streamlit.app/

## Why this exists

I'm finishing my degree in Statistics and Economics at UIUC this spring, and I've been looking at strategic finance and analytics roles at fintech companies. A pattern kept showing up: investors and boards ask Seed and Series A teams for cohort retention, CAC payback, and loss curves, and most of those teams don't actually have anyone building that yet. It's usually a rushed spreadsheet before a board meeting, not something standing and reusable.

So I built it. This is a working dashboard on a synthetic subscription business, framed as a fintech personal finance app since that's close to what a lot of the companies I'm talking to actually build, sitting on top of a calculation engine that isn't hardcoded to the demo data. Give it a real `customers.csv` and `subscription_events.csv` export and it produces the same four charts against real numbers.

If you're reading this because you're hiring, this repo is meant to answer two things before you even ask: can I actually build the thing, and do I understand the finance well enough to defend every number in it. I tried to make both true, not just claimed.

## What it actually does

- **Cohort retention** – what percent of each signup cohort is still paying N months later
- **CAC payback** – how many months it takes each acquisition channel to earn back what it cost to get a customer
- **LTV:CAC** – whether a channel's customers are worth more than what it costs to acquire them, modeled off that channel's own retention curve instead of some arbitrary cutoff window
- **GRR and NRR** – how much of a cohort's revenue sticks around over time, with and without counting upsells

All four are unit tested against a hand calculated example, not just eyeballed against the synthetic data until the chart looked right.

## Repo structure

```
data_gen/       synthetic dataset generator (not used in the calculation path)
engine/
  loaders.py      data ingestion + validation, the schema boundary. Swap real data in here.
  metrics.py      the four metrics: cohort retention, CAC payback, LTV:CAC, GRR/NRR
tests/
  test_metrics.py unit tests against a hand-calculated 4-customer fixture
data/           generated CSVs (customers.csv, subscription_events.csv)
app.py          Streamlit UI, renders only. Every number on screen comes from engine/
```

## Running it locally

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python3 -m data_gen.generate      # writes data/customers.csv, data/subscription_events.csv
pytest tests/                     # check the engine against hand-calculated examples
streamlit run app.py
```

## The input schema

This is the part that makes it reusable instead of a one-off script.

**customers.csv**

| column | type | notes |
|---|---|---|
| `customer_id` | str | unique |
| `signup_date` | date | any day-of-month works, it gets floored to month start |
| `acquisition_channel` | str | e.g. `paid_social`, `organic_referral` |
| `initial_plan` | str | informational only, not used in the math |
| `cac` | float | fully-loaded cost to acquire this customer |

**subscription_events.csv**

| column | type | notes |
|---|---|---|
| `customer_id` | str | foreign key into `customers.customer_id` |
| `month` | date | calendar month of the billing event |
| `mrr` | float | that customer's MRR that month, `0` if churned |

To point the dashboard at real data, toggle off "Use synthetic demo data" in the sidebar and upload two CSVs in this shape. Everything downstream (the cohort table, payback curves, LTV:CAC, loss curves) recomputes on its own. No code changes needed if the schema matches. If a real export uses different column names, `engine/loaders.py` is the only file that has to change.

## How I calculated everything

This is the part I'd want to read before someone asks me "how did you get this number" in an interview.

### Cohort retention

Percent of a signup cohort with MRR greater than 0 at month N since signup, computed as a dense customer by calendar-month grid running from each customer's signup month through the dataset's last observed month. A missing billing row inside that window counts as churned. Most billing systems stop emitting rows once someone cancels instead of emitting `$0` forever, so that's the safer assumption, and if a real export behaves differently the result comes out the same either way.

Cells for a cohort and tenure combination that hasn't happened yet (a cohort that signed up 3 months ago can't have a 12-month data point) are left blank, not zero. That's why the heatmap renders as a triangle instead of a full rectangle. Counting an unobserved future month as 0% would make recent cohorts look worse than they are and drag down every average that touches them.

### CAC payback, in months, per channel

`payback_month` is the first month where average cumulative gross margin per customer, divided by average CAC, crosses 1.0. Computed separately per channel, never blended, because a channel with $300 CAC and a channel with $65 CAC don't have anything useful to say to each other averaged together.

Two things matter here:

1. **Gross margin, not revenue.** Payback should measure cash actually recovered, not top-line MRR. Default assumption is 75% gross margin, which is roughly typical for a fintech SaaS business after infra, support, and payment processing costs. It's adjustable in the sidebar because it's the single biggest lever on the reported payback month, and it should get swapped for a real company's actual margin the moment this runs on real data.
2. **Right-censoring.** A customer who signed up last month can't have 12 months of margin yet. Averaging in a phantom $0 for months they haven't reached would understate the channel, especially one with a lot of recent volume. So the engine only averages customers who've actually been observed that long, and only reports a given month once at least 20% of the channel has reached it. Past that point the average is too small a sample to trust, so the curve just stops instead of showing a noisy tail.

If the ratio never crosses 1.0 inside the reliably covered window, payback shows up as "not yet reached" instead of a guess. That happens for `paid_search` and `paid_social` in the demo data. It's an honest answer for a channel that's still young, not a bug.

### LTV : CAC

`LTV(channel) = expected active months × ARPA(channel) × gross margin`

`expected active months` comes from that channel's own retention curve, extrapolated out to convergence instead of cut off at some fixed window like 24 months. A fixed window either shortchanges channels with a long retention tail or flatters channels that just haven't existed long enough in the dataset yet to be measured fairly. In practice: observed survival gets summed directly wherever there's enough coverage, and past that point a constant tail hazard, the average monthly churn rate implied by the last few observed points, extrapolates survival forward until it's basically zero or a 240-month cap kicks in. This is the standard retention-curve LTV approach used in SaaS finance, not something I invented for this project.

`ARPA(channel)` is the average MRR among that channel's active customer-months only, so churned $0 months don't drag it down (that's already captured separately in expected active months). `CAC` is the plain average across the channel's customers.

### GRR and NRR, the loss curves

Both measured against a cohort's starting MRR:

```
NRR(t) = sum(MRR at t) / sum(MRR at 0)                     -- can go above 100%
GRR(t) = sum(min(MRR at t, MRR at 0)) / sum(MRR at 0)       -- capped at 100%, by construction
```

The `min()` is the whole trick. It clips any customer's upsell back down to what they started at before summing, so expansion can never end up in GRR's numerator, while a downgrade or churn still pulls it down. NRR uses the uncapped number, so expansion shows up directly and can push it past 100%. This is more reliable than trying to separately classify every dollar of MRR change as expansion versus contraction, and it's mechanically incapable of exceeding 100%, which is the entire point of the metric.

## The synthetic data

`data_gen/generate.py` builds about 2,000 customers across 18 monthly signup cohorts and 4 acquisition channels, each with different economics on purpose, so the dashboard actually tells a story instead of showing a flat blob:

| Channel | CAC | What happens to it |
|---|---|---|
| `organic_referral` | low, ~$65 | mediocre long-run retention |
| `content_seo` | medium, ~$150 | best long-run retention, the "spend more here" channel |
| `paid_search` | medium-high, ~$200 | mediocre, payback not yet observed |
| `paid_social` | high, ~$300 | okay early retention, then a steep drop-off |

A few things make the story coherent instead of random:

- **Churn follows a hazard curve** that starts high and decays toward a per-channel floor, which is what produces the classic steep-then-flat shape of a real retention curve.
- **Later cohorts churn a bit less** than earlier ones, on purpose, so there's a genuine "we're getting better at retention" trend in the data instead of noise pretending to be one.
- **Customers occasionally move a plan tier up or down** each month, which is what makes NRR different from GRR.
- **The observation window is fixed by calendar month**, not per customer, so recent cohorts are naturally cut short, exactly like a real billing export. That forces the engine to handle partially observed cohorts correctly instead of assuming everyone has full history.

Reproducible with a fixed seed (`SEED = 42` in `data_gen/generate.py`).

## Where this is still a demo

Being upfront about this is part of the point.

- **75% gross margin is a placeholder,** not a researched number for any real company. It's adjustable specifically so the real figure can replace it immediately once this runs on real data.
- **No win-back modeling.** Once a synthetic customer churns, they stay churned. A real business would see some reactivation, and the engine would just reflect whatever the actual MRR sequence shows.
- **CAC is taken as a given per customer,** not derived from marketing spend divided by customers acquired. That reconciliation usually already lives somewhere in a real company's finance or marketing stack and would get fed in upstream.
- **The 20% coverage threshold is a judgment call,** trading off "show more of the curve" against "don't show a two-customer average like it means something." It's a constant in `engine/metrics.py` if a specific use case wants a different bar.

## If you want to talk about it

This was built end to end by me, generator, tested engine, dashboard, deployment. Happy to walk through any of it or point it at a real dataset. [linkedin.com/in/aadit-tibrewala](https://linkedin.com/in/aadit-tibrewala)
