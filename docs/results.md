# Results: rolling annual holdouts

> ### ⚠️ These results are from synthetic data
>
> The market used here is generated from a data-generating process this
> repository defines, so the right answer is known in advance. Nothing below is
> a claim about real equities.
>
> **What it does establish** is the thing you actually have to establish before
> any real result is worth reading: that the evaluation machinery finds an
> effect when one is present, and finds nothing when one is not. Two runs of the
> identical pipeline are reported side by side — one with a post-announcement
> drift planted in the data, one with no effect planted at all. If the second
> column produced a result, everything in the first column would be an artefact.
>
> To regenerate both, with no network access required:
>
> ```bash
> eee holdout --out reports/holdout                 # effect planted
> eee holdout --drift 0 --out reports/holdout_null  # null control
> ```
>
> To run the same procedure on real data, see [§7](#7-what-changes-on-real-data).

---

## 1. The design

For each year *Y* from 2019 to 2024:

1. take every announcement whose 20-day outcome was **fully resolved before 1
   January Y**, minus a 25-session embargo;
2. fit the model on those events only;
3. **freeze it** — no tuning, no peeking;
4. predict every announcement in year *Y*;
5. compare the predictions with what actually happened.

Then move to *Y+1* and refit on the wider window. Six independent out-of-sample
years, not one. The point of six is that a single good year is indistinguishable
from luck.

The prediction target is `car_market_model_1_20` — the cumulative abnormal
return from **one session after** the announcement to twenty sessions after.
Windows starting at day 0 contain the opening gap, which nobody can trade, and
are excluded from anything the model is scored on.

Each year is also scored against a **single-feature baseline** (time-series
earnings surprise on its own). A thirty-feature model that cannot beat one
column is not earning its complexity.

---

## 2. Effect planted — the six held-out years

| Year | Train n | Test n | IC | IC t | Predicted (bp) | Realised (bp) | Calib slope | Net return (%) | Sharpe | Sharpe t | Baseline IC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 2594 | 600 | 0.202 | 3.95 | 602 | 414 | 0.79 | 5.1 | 2.62 | 1.90 | 0.008 |
| 2020 | 3194 | 600 | 0.211 | 6.91 | 561 | 586 | 0.92 | 8.3 | 4.35 | 3.04 | 0.032 |
| 2021 | 3794 | 600 | 0.236 | 14.39 | 560 | 627 | 0.98 | 10.7 | 5.75 | 3.57 | −0.026 |
| 2022 | 4394 | 600 | 0.164 | 2.68 | 559 | 445 | 0.71 | 9.1 | 5.01 | 3.35 | 0.030 |
| 2023 | 4994 | 600 | 0.169 | 5.09 | 548 | 490 | 0.80 | 8.6 | 5.11 | 3.79 | 0.047 |
| 2024 | 5594 | 600 | 0.198 | 9.43 | 550 | 435 | 0.87 | 8.8 | 5.65 | 4.26 | 0.021 |

*Predicted / Realised* are the top-minus-bottom quintile spreads: what the model
said the gap between its best and worst quintile would be, and what that gap
turned out to be.

![Out-of-sample IC by held-out year](figures/holdout_ic.png)

![Predicted vs realised quintile spread](figures/predicted_vs_realised.png)

---

## 3. The null control — same pipeline, no effect planted

| Year | Train n | Test n | IC | IC t | Predicted (bp) | Realised (bp) | Calib slope | Net return (%) | Sharpe | Sharpe t | Baseline IC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2019 | 2594 | 600 | 0.047 | 1.51 | 295 | 177 | 0.26 | 3.1 | 1.88 | 1.44 | −0.026 |
| 2020 | 3194 | 600 | −0.043 | −1.31 | 248 | −73 | −0.28 | −3.9 | −2.17 | −1.49 | −0.017 |
| 2021 | 3794 | 600 | 0.080 | 3.67 | 205 | 257 | 1.23 | 7.0 | 3.79 | 3.10 | −0.057 |
| 2022 | 4394 | 600 | 0.049 | 1.77 | 231 | 73 | 0.22 | 0.5 | 0.28 | 0.25 | −0.030 |
| 2023 | 4994 | 600 | −0.027 | −0.51 | 208 | 5 | −0.13 | −2.4 | −1.47 | −1.10 | −0.005 |
| 2024 | 5594 | 600 | 0.029 | 0.85 | 195 | 96 | 0.35 | −2.2 | −1.30 | −0.95 | −0.043 |

Note what the null run still produces: **predicted spreads of 200 bp every
year.** The model always has an opinion. It is confident about 2020 and 2023 in
exactly the tone it uses for the years it gets right, and it is wrong. The
predicted column alone can never tell you whether a model works — which is the
entire reason the realised column has to sit next to it.

Note also 2021, where the null run posts an IC of 0.080 with a t of 3.67 and a
Sharpe of 3.79. **That is what a false positive looks like**, in data where we
know for certain there is nothing to find. Anyone reporting a single year would
have reported that one.

---

## 4. Side by side

| Statistic | Effect planted | Null control |
|---|---:|---:|
| Mean out-of-sample IC | **0.197** | 0.022 |
| Std of IC across years | 0.027 | 0.048 |
| t-statistic across the six years | **17.99** | 1.15 |
| Years with positive IC | **6 / 6** | 4 / 6 |
| Mean realised quintile spread | **50 bp** | 9 bp |
| Mean calibration slope | **0.85** | 0.27 |
| Net Sharpe, all years stitched | **4.82** | 0.37 |
| Newey–West t on daily net P&L | **8.27** | 0.70 |

The separation is the result. Every statistic that matters is an order of
magnitude apart, and the null column is indistinguishable from zero on the two
that carry weight — the across-year t and the HAC t on daily P&L.

---

## 5. Calibration: ranking and scale are different things

![Calibration](figures/calibration.png)

Predictions binned into twenties, plotted against what those bins realised. The
fitted slope is **0.84**, against 1.00 for perfect calibration.

This is the normal outcome and it is worth understanding rather than glossing.
A slope below 1 means the ordering is right — high predictions really do earn
more than low ones — but the *magnitudes* are overstated. Read the per-year
table again with that in mind: the model predicted a 602 bp spread for 2019 and
got 414 bp; it predicted 550 bp for 2024 and got 435 bp. The direction was right
six times out of six. The size was optimistic five times out of six.

That matters practically, because position sizing scaled to predicted magnitude
would be systematically over-levered. It is also the sharpest single contrast
with the null run, where the slope averages 0.27 and swings negative in two
years — a model with no signal cannot be calibrated, only lucky.

---

## 6. Economics, net of costs

![Equity curve](figures/equity_curve.png)

Sector-neutral long/short, entered one session after each announcement, held 20
sessions, overlapping vintages, net of a square-root market-impact cost model.
Across all six held-out years the book returns **8.8% annualised net**, Sharpe
**4.82**, worst drawdown **−1.1%**, with annual turnover around 14× — the cost
drag between the gross and net lines in the figure is what that turnover buys.

**Do not read that Sharpe as a forecast of anything.** The planted drift in the
synthetic process is roughly three times larger than any post-earnings drift
documented in real equities, and deliberately so: it has to clear the noise in a
short sample for the demonstration to mean anything. The number to take from
this section is not 4.82. It is that the machinery *converts* a signal of known
size into a net-of-cost result without losing it along the way, and converts an
absent signal into nothing.

### Coefficient stability

![Coefficient stability](figures/coefficient_stability.png)

Each dot is one annual refit. Coefficients that flip sign between refits mean
the model is chasing noise even when its headline numbers look fine — here the
leading features hold their sign and rough magnitude across all six, which is
what you want to see before believing any of the above.

---

## 7. What changes on real data

Everything above runs offline against a generated market. Pointing the same
commands at real data changes three things and nothing else:

```bash
pip install -e ".[dev,data]"
eee ingest --provider yahoo --tickers <list> --start 2014-01-01 --end 2024-12-31
eee holdout --provider yahoo --years 2019-2024 --out reports/holdout_real
```

1. **Expect much smaller numbers.** Post-earnings-announcement drift has been
   documented since Ball and Brown (1968) and the published evidence is that it
   decayed substantially after roughly 2004, as it became widely traded. A real
   IC around 0.02–0.04 with a net Sharpe under 1 would be a *good* result. A
   Sharpe of 4 on real data means a bug, and [`biases.md`](biases.md) lists the
   four places to look first.
2. **The universe file starts mattering.** The synthetic universe has no
   survivorship bias because nothing is ever deleted from it. A real run without
   a point-in-time membership file is survivorship-biased, and the code will
   refuse to build one silently — see [`conf/universe/README.md`](../conf/universe/README.md).
3. **Announcement timing becomes a real source of error.** Free data sources get
   BMO/AMC wrong often enough to matter, and a mislabelled flag moves the whole
   event window by a session. Events carry `timing_source`, so the honest
   robustness check is to re-run restricted to declared-only timings.

The falsification criteria were fixed in advance and are in
[`methodology.md` §9](methodology.md#9-what-would-falsify-the-result). They apply
unchanged to a real run.

---

## 8. What this evidence does not cover

Stated plainly, because a results document that only lists strengths is
marketing:

- **Six years is six observations.** The across-year t-statistic is reported for
  consistency, not power. The daily HAC t-statistic on P&L is the one doing real
  work.
- **One synthetic process.** The generator is linear in a single latent
  surprise, with Gaussian noise and a stable factor structure. Real markets have
  regime shifts, fat tails, and non-linear relationships between fundamentals
  and returns. That the model recovers a linear effect says nothing about
  whether it would recover a non-linear one.
- **Costs are assumed, not measured.** 3 bp half-spread and a 10 bp impact
  coefficient at 5% participation are plausible for liquid US large caps and are
  not measurements. The cost-sensitivity curve in each run's `report.md` is
  there so the conclusion can be stated as "survives up to *N* bp", which is a
  falsifiable claim, rather than as a single unverifiable number.
- **No capacity analysis.** The book is sector-neutral and weight-capped, but
  nothing here establishes how much money it would absorb before impact eats the
  edge.
- **The null control is one draw.** A single null run at seed 20260818 shows the
  pipeline does not manufacture a result on this particular sample of noise.
  Repeating it across many seeds would bound the false-positive rate properly;
  that is worth doing and has not been done.
