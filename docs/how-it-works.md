# How this project works, and what every number in it means

*A guide for readers without a quantitative finance background. It assumes you
know what a share is and nothing beyond that. Read it top to bottom and you will
be able to explain the whole project, including why the answer is not a simple
yes or no.*

---

## Part 1 — The question

### 1.1 What actually happens when a company reports earnings

Four times a year, every listed company publishes its results: revenue, profit,
cash flow, and management's account of how things are going. In the United
States this arrives as a press release attached to a regulatory form called an
**8-K**, filed with the Securities and Exchange Commission. The filing is
timestamped to the minute.

The share price reacts almost immediately — often within seconds, and usually
before any human has finished reading. That first jump is not what this project
is about, and the reason matters: **you cannot trade it.** By the time the
information is public, the price has already moved. Anyone who claims to profit
from that reaction is either mismeasuring or describing something else.

### 1.2 The thing that is actually interesting

In 1968 two accounting researchers, Ray Ball and Philip Brown, noticed something
strange. After the initial jump, prices kept *drifting* in the same direction —
for weeks. Companies that beat expectations carried on rising. Companies that
missed carried on falling.

This is called **post-earnings-announcement drift**, and it is a problem for the
standard theory of markets, which says today's price already reflects everything
known. If prices take six weeks to finish reacting to public information, the
theory is at best incomplete — and there is money lying on the floor.

It has been studied continuously since. The consensus is that it was strong in
the 1970s and 1980s, weaker after it was published, and largely gone from large,
heavily-traded companies by the mid-2000s.

### 1.3 The question this project asks

> Using only information that was genuinely public before the market opened, can
> we rank companies by how much they will out- or under-perform their sector
> over the following month?

Two words in that sentence are doing a great deal of work.

**"Genuinely public before"** — the single easiest way to produce a fake result
in this field is to accidentally use information that did not exist yet. Section
3 is entirely about how that is prevented.

**"Out- or under-perform their sector"** — see next.

### 1.4 Why "beat its sector" and not "went up"

Suppose a bank reports results and the shares rise 3% over the next month. Did
the earnings report tell you anything?

Not necessarily. If every bank rose 3% that month, the report told you nothing
about *this* bank — it told you something about interest rates, or the economy,
or the market's mood. The information you want is what the share did **relative
to what it would have done anyway**.

That quantity is called an **abnormal return**:

> abnormal return = what the share actually did − what it was expected to do

Everything in this project is measured in abnormal returns. It is the single
choice that removes most of the ways a study like this fools itself.

---

## Part 2 — The measurements

### 2.1 Abnormal return, three ways

"What it was expected to do" can be defined three ways, and the project computes
all three rather than picking the flattering one.

| Method | Expected return is… | Strength | Weakness |
|---|---|---|---|
| **Market-adjusted** | whatever the market index did | simple, hard to game | ignores that some shares move more than the market |
| **Market model** | the share's own historical sensitivity to the market | accounts for volatile vs. sleepy shares | needs a year of history to estimate |
| **Sector-neutral** | the average of the share's own sector, excluding itself | removes industry-wide news | needs a reliable sector classification |

The **market model** is the headline. It works by looking at roughly a year of
history *ending a month before* the announcement, and asking: when the market
moved 1%, how much did this share typically move? A share that historically
moves 1.5% for every 1% of market move has a **beta** of 1.5. If the market rose
2% and the share rose 2%, the share actually *underperformed*, because it was
expected to rise 3%.

The estimation window stops a month before the announcement on purpose, so that
the run-up to the event cannot contaminate the estimate of normal behaviour.

**Leave-one-out** in the sector method means a company is never compared against
a sector average that includes itself — otherwise a large company is partly
benchmarked against its own return, which mechanically shrinks any effect.

### 2.2 The windows: [1, 20] and why not [0, 20]

Returns are accumulated over a window of trading days measured from the
announcement. `[1, 20]` means "from the open of the *next* trading day, for the
following twenty trading days" — about a calendar month.

The `1` rather than `0` is one of the most important decisions in the project.

If the window starts on day 0, it includes the overnight jump — the price gap
between the close before the announcement and the open after it. That gap is
large, it is highly correlated with the news, and **it is not available to you**.
The announcement typically lands after the market has closed; by the time you
can place an order, the price has already moved.

Including it is the single largest source of overstated results in this
literature. In this project it was caught by a test: on data with *no* effect
deliberately planted in it, a window starting at day 0 produced a Sharpe ratio
above 2 — an excellent-looking strategy, from nothing. See section 5.2.

The project reports five windows — `[0,0]`, `[0,4]`, `[0,19]`, `[1,5]`, `[1,20]`
— so a reader can see the difference for themselves rather than take it on
trust.

---

## Part 3 — Not seeing the future

This is the part that separates a real study from a plausible-looking one, and
it is worth understanding properly because it is subtler than it sounds.

### 3.1 Look-ahead bias

**Look-ahead bias** is using information that was not available at the time you
pretend to have acted. It is rarely deliberate and almost always invisible in
the results — a strategy with look-ahead simply looks good.

The classic example: a company's annual accounts describe the year ending 31
December, but they are not *published* until late February. A study that uses
December's figures to make a January decision is using information that did not
exist. It will show excellent performance and be entirely worthless.

**How this project prevents it.** Every piece of data carries two dates, not
one:

- **`period_end`** — what period the information describes
- **`available_from`** — the moment it actually became public

Every event carries the timestamp of when a trade could first have been placed.
Then there is a single rule, enforced by a check that stops the run:

> `available_from` must be earlier than the moment the trade is placed.

That check is not a convention someone remembers to follow. It runs on every
assembled dataset and raises an error.

### 3.2 The subtler problem: point-in-time is not the same as *relevant*

Here is a distinction this project only discovered late, and it is the most
interesting thing in it.

The check above asks *"was this public before the trade?"* It cannot ask
*"is this about the event?"* — and those turn out to be very different
questions.

The earnings press release is filed within minutes of the announcement. But the
detailed financial statements for that same quarter — the structured, machine-
readable accounts — are not filed until the quarterly report weeks later. So at
the moment of any given announcement, the most recent structured accounts
available describe the **previous** quarter.

Measured across this sample: **the median piece of financial data is 83 days
old** at the announcement it is attached to.

Nothing leaks. The check is right to pass it. And yet almost none of it is
*about* the event being studied. A study can be perfectly disciplined about
timing and still be answering a different question from the one it asked.

This is why the project's conclusion is carefully worded: the fundamentals
predict returns, but they are last quarter's fundamentals, so the result is a
statement about quarterly accounts rather than about earnings announcements.

### 3.3 Survivorship bias

If you build a study on the companies in an index *today*, you have silently
excluded every company that collapsed, was taken over cheaply, or shrank out of
the index. Those are exactly the companies a bad earnings report predicts.

The result is a universe where everything worked out, which flatters any
strategy tested on it.

**How this project prevents it.** Index membership is stored as intervals — a
company is in the sample only for the years it was actually a member. Building a
universe from today's list raises an error unless you explicitly acknowledge the
bias, and that acknowledgement then appears in the output.

**Where it survives anyway.** Preventing it in the *definition* is not enough if
the price data cannot supply the companies. Free price sources drop most
delisted companies. In this sample **61% of the companies deleted from the index
have no price history available**, against 5% of the survivors. Silicon Valley
Bank and First Republic — both of which failed in 2023 — are among the missing.

The direction of that bias is knowable: the missing companies are
disproportionately the ones that collapsed, so their absence flatters the
result. It is reported rather than hidden, because a bias you have measured is a
limitation and a bias you have not is a mistake.

### 3.4 Training on the past, testing on a future it has never seen

The standard machine-learning approach, **cross-validation**, splits data into
random chunks and tests on each. For financial time series it is invalid, for
two reasons.

First, it trains on 2022 to predict 2016. That is not a prediction.

Second, and more subtly: a 20-day outcome observed on 1 March is not *resolved*
until the end of March. If the training set contains an event from mid-March,
its outcome overlaps the test event's outcome — the two share the same days of
market movement. The model can learn the answer without learning anything.

**What this project does instead — walk-forward with purging and embargo:**

1. Train on every announcement whose 20-day outcome had fully resolved before 1
   January 2019.
2. Freeze the model. It never sees 2019 again.
3. Predict every announcement in 2019.
4. Repeat for 2020, 2021, 2022, 2023, 2024.

**Purging** removes training events whose outcomes overlap the test period.
**Embargo** removes a further 25 trading days as a buffer. Both are standard
practice and both are frequently skipped.

Six frozen years is a small number of independent observations, which is exactly
why a pattern found across six annual figures should be distrusted — see section
6.1.

---

## Part 4 — The metrics, explained

This is the reference section. Each metric gets what it measures, how to read
it, and what this project found.

### 4.1 Information coefficient (IC) — *does the ranking work?*

**What it is.** The correlation between the model's predictions and what
actually happened, across all the companies reporting in a period. Specifically
a **rank** correlation: it asks whether the model put the companies in roughly
the right order, not whether it guessed the exact numbers.

**Scale.** −1 to +1. Zero is no skill. In this field:

| IC | Interpretation |
|---|---|
| 0.00 | no skill |
| 0.02–0.03 | weak but potentially useful |
| 0.05–0.10 | genuinely good |
| above 0.15 | be suspicious — usually a bug |

**Why so small?** Single-stock returns are overwhelmingly noise. A signal
explaining half a percent of the variance can still be valuable if applied
across hundreds of positions repeatedly. Diversification, not accuracy, is what
makes a small edge usable.

**This project: 0.067.** Real skill by the standards of the field.

### 4.2 The t-statistic — *is it distinguishable from luck?*

**What it is.** The size of a result divided by its uncertainty. Roughly: how
many standard errors is this away from zero?

**How to read it.** Above about 2 (in absolute value) is conventionally
"unlikely to be chance" — under standard assumptions, a t of 2 corresponds to
roughly a 1-in-20 chance of arising from noise. Below 2 means the result is
consistent with luck.

**The trap.** A t-statistic is only as good as the assumption of independent
observations behind it. Hundreds of companies report in the same week and share
whatever the market did that week. Treating them as independent overstates
significance badly — see 4.9.

**This project: t = 3.40** on the information coefficient across six years.

### 4.3 Calibration slope — *are the magnitudes right?*

**What it is.** Take the model's predicted returns, regress the actual returns
on them, and read the slope.

**How to read it.**

- **1.0** — the model's magnitudes are exactly right
- **0.5** — the model predicts twice the movement that occurs
- **0** — the predictions carry no information about magnitude
- **negative** — the magnitudes are backwards

**Why it matters separately from IC.** Ranking and sizing are different skills.
A model can put companies in the right order and be wildly wrong about how far
apart they are — and since position sizes are set from predicted magnitude,
a badly calibrated model bets too much on the wrong things.

**This project: 0.589**, and 2019 is the cautionary example — a positive
information coefficient, a calibration slope of 0.10, and the worst realised
loss of the six years.

### 4.4 Sharpe ratio — *return per unit of risk*

**What it is.** Annual return divided by annual volatility. It answers: how much
return did you earn for how much stomach-churning?

**How to read it.**

| Sharpe | Interpretation |
|---|---|
| below 0 | lost money |
| 0.5 | poor |
| 1.0 | respectable |
| 2.0 | excellent |
| above 3 | in a backtest, almost always a bug |

**Net vs gross.** "Net" means after trading costs. Always ask which is quoted;
strategies that trade often can look excellent gross and lose money net.

**This project: −0.30 net.** The strategy loses money. But see the next section
before concluding anything from that.

### 4.5 The permutation null — *what does "no signal" actually score?*

This is the metric most studies omit, and it changes how everything else reads.

**The problem.** It is tempting to compare a Sharpe ratio against zero. But a
portfolio is not a neutral instrument. This one holds twenty-day positions,
rebalances daily, and caps how much can sit in any single company. Run it on a
*worthless* signal and it does not return zero — it drifts slightly negative,
because of how the machinery works rather than what it is trading.

So "the Sharpe is −0.30" is meaningless until you know what the machinery does
to noise.

**The method.** Take the completed study. Shuffle the predictions at random
within each year, so that a prediction made for Company A is now attached to
Company B. Everything else is identical — same companies, same dates, same
spread of predicted values, same sectors. Only the *link* between a prediction
and the company it belongs to is destroyed. Run the whole portfolio again.
Repeat 200 times.

What comes back is the distribution of results this machinery produces from pure
noise.

**This project:**

| | |
|---|---:|
| Real result | −0.30 |
| Shuffled average | −0.70 |
| Where the real one sits | **85th percentile** |
| Probability of getting this from noise | **0.16** |

The real strategy beats shuffled predictions — but not by enough to be
distinguishable from luck. Crucially, the correct reading of −0.30 is *"better
than the −0.70 this machine produces from nothing, though not significantly"*,
**not** *"lost money, therefore the idea is wrong."*

### 4.6 Alpha and factor exposure — *is it clever, or ordinary in disguise?*

**The problem.** A strategy can look skilful while really just doing something
well known. Decades of research have identified a handful of characteristics
that have historically earned higher returns:

| Factor | Plain English |
|---|---|
| **Market** | exposure to the stock market generally |
| **Size (SMB)** | smaller companies |
| **Value (HML)** | cheap companies relative to their book value |
| **Profitability (RMW)** | companies with high, stable profit margins |
| **Investment (CMA)** | companies that grow their assets conservatively |
| **Momentum (MOM)** | companies whose shares have recently risen |

You can buy each of these for a few basis points through an index fund. So the
question is not "did the strategy make money" but "did it make money *beyond*
what these already explain?"

**The method.** Regress the strategy's daily returns on the factors' daily
returns. The **coefficients** say how much of each it was holding. The
**intercept** is **alpha** — the part left over.

**This project:**

| Factor | Loading | t |
|---|---:|---:|
| **Alpha** | **−0.91%/yr** | **−1.11** |
| Profitability (RMW) | +0.041 | **3.62** |
| Value (HML) | +0.021 | **2.55** |
| Investment (CMA) | +0.031 | **2.15** |
| Size (SMB) | +0.017 | **2.02** |
| Momentum (MOM) | −0.024 | **−3.25** |

No alpha. What the regression *does* show is that the strategy is a
quality-and-value portfolio in disguise — it systematically holds profitable,
cheap, conservative companies and bets against recent winners. That is not a
criticism of the model so much as an explanation of it: the inputs are
year-on-year changes in margins, growth and debt, which is close to a definition
of quality.

**R² = 0.18** means these ordinary factors explain 18% of the variation in the
strategy's returns.

### 4.7 Deflated Sharpe ratio — *how many things did you try?*

**The problem.** Test twenty strategies on the same data and one will look
excellent by luck alone. Report only that one and you have a publishable-looking
result and nothing else. This is **data-snooping**, and it is the reason most
published backtests do not survive contact with reality.

**The method.** The **deflated Sharpe ratio** (Bailey and López de Prado) adjusts
an observed Sharpe for the number of attempts behind it. Given N tried, what is
the highest Sharpe you would expect from luck alone? The deflated figure is the
probability the true Sharpe is positive *given* that search.

**How to read it.** It is a probability. Above 0.95 is the usual bar.

**The requirement it imposes.** It needs an honest N. This project keeps a log
of every specification evaluated, **including the abandoned ones** — twelve of
them, four abandoned. Under-reporting the count inflates the deflated Sharpe and
defeats the point of computing it.

**This project: 0.00.** It does not survive, which is what should happen to a
strategy whose raw Sharpe is negative.

### 4.8 Cost sensitivity — *how wrong would the assumptions have to be?*

Trading costs are modelled, not measured. Three components:

- **Half-spread** — the gap between buying and selling price
- **Commission** — the broker's fee
- **Market impact** — your own buying pushes the price against you, and it grows
  with the square root of how much of the day's volume you take

Rather than defend a single number, the project scales all three together and
reports the result:

| Cost multiple | One-way (bp) | Annual return | Net Sharpe |
|---:|---:|---:|---:|
| 0.5× | 2.9 | −0.19% | −0.08 |
| **1.0× (assumed)** | **5.7** | **−0.70%** | **−0.30** |
| 2.0× | 11.5 | −1.72% | −0.74 |

At **half** the assumed cost it still loses. Break-even is near 0.31× —
about 1.8bp one way, below the bid-offer spread of a typical S&P 500 company
before commission exists at all. **The conclusion does not depend on the cost
assumption being right.**

### 4.9 Clustering — *why the obvious statistics are wrong here*

**The problem.** Earnings announcements are not spread evenly through the year.
They cluster: hundreds land in the same fortnight, four times a year. Every one
of those shares whatever the market did that fortnight.

Standard statistical tests assume independent observations. With 13,675
announcements arriving in roughly 1,900 distinct clusters, treating them as
13,675 independent draws overstates precision by roughly the square root of the
cluster size — often two or three times over. That is frequently the difference
between "significant" and "not".

**The method.** Every test in this project resamples **whole dates** rather than
individual announcements. Draw 1,900 dates at random with replacement, take all
the announcements on each drawn date, compute the statistic, repeat 2,000 times.
The spread of those results is the honest uncertainty.

The project also uses two specialist event-study tests:

- **Patell** — corrects for the fact that an abnormal return estimated from a
  fitted model is more uncertain than one measured directly
- **Boehmer–Musumeci–Poulsen (BMP)** — corrects for volatility *rising* around
  announcements, which is exactly when this study measures things

### 4.10 Turnover and drawdown

**Turnover** — how much of the portfolio is replaced per year. This project:
**17×**, which is high, and it is why costs matter.

**Maximum drawdown** — the worst peak-to-trough loss. This project: **−7.4%**.

---

## Part 5 — Proving the machinery works

### 5.1 The problem with a negative result

A study that finds nothing has two possible explanations, and they look
identical from outside:

1. There is nothing to find.
2. The pipeline is broken.

You cannot distinguish them by staring at the code.

### 5.2 The solution: test it on data with a known answer

The project generates a **synthetic market** — invented companies, invented
prices, invented earnings — where the answer is known in advance because it was
put there.

**Run A: an effect is deliberately planted.** The pipeline finds it in 6 years
out of 6, at a Sharpe of 4.9. *The machinery can detect an effect when one
exists.*

**Run B: nothing is planted.** The pipeline returns an information coefficient
of 0.024 and a Sharpe of 0.48 — near enough nothing. *The machinery does not
invent effects.*

Both runs are necessary. Only the pair licenses a claim about the real data.

**This is also how the project's worst bug was found.** An early version scored a
Sharpe above 2 on Run B, where there was provably nothing to find. That is what
exposed the untradable overnight gap described in section 2.2. Without the null
control it would have gone unnoticed — and would have looked like a discovery.

**And the null control has a lesson of its own.** In one individual year, Run B
posts an information coefficient of 0.080 at t = 3.67 — a confident false
positive, in data where nothing exists. Anyone reporting a single good year from
a six-year study would have reported that one.

---

## Part 6 — What the project actually found

### 6.1 A trend across six points is not a trend

An earlier version of this study, on 114 companies, found that predictive skill
declined significantly year by year: −0.023 per year with a p-value of 0.033.

It fitted the literature perfectly. A published anomaly *should* decay as it
becomes widely traded. It was reported as the headline finding.

On 466 companies — four times the data — the same calculation gives **+0.006 per
year with a p-value of 0.61**. No trend at all. The good and bad years are
scattered, not ordered.

**The lesson, and the reason it is left visible in the write-up:** a regression
across six annual observations found a pattern that more data dissolved. The
direction of the error is the instructive part — it *agreed with the published
literature*, which is exactly when a small-sample result is least likely to be
questioned.

### 6.2 A null result can be a data-quality result

For one day this project reported a clean null: information coefficient 0.022,
t = 1.30, no skill.

Two bugs were suppressing it, both in the same place. Company accounts are filed
in a structured format called **XBRL**, and quarterly figures in it are
*cumulative*: the half-year filing reports six months, the nine-month filing
reports nine. To get a single quarter you subtract the previous cumulative
figure.

- A quarter that had been correctly derived that way was then **differenced a
  second time** further down the pipeline, because it carried a label saying
  "full year". 6,505 rows. Nothing complained — subtracting three quarters from
  one quarter still produces a plausible-looking number.
- About 4,100 genuine ninety-day quarters were **read as annual** because the
  company had labelled them "FY".

Fixing the data, with no change to the model, moved the information coefficient
from **0.022 (t = 1.30) to 0.067 (t = 3.40)**.

A null that is really a data-quality result is the most ordinary failure in
applied quantitative work. It is worth saying plainly that this project
published one for a day.

### 6.3 The language adds nothing

All 23,914 earnings press releases are scored for tone, uncertainty,
litigiousness, and how much the wording has changed since the company's previous
release — the last being the "Lazy Prices" idea that companies quietly rewrite
their boilerplate when something is wrong.

The text carries **41% of the model's total weight**, and change-in-tone is the
second-largest single input of any feature.

Adding the entire text block moves out-of-sample skill from **0.0676 to 0.0672**.

In-sample importance, no out-of-sample value. This is a clean answer to the half
of the original question about management language, and it is the kind of result
that only appears if you test on data the model has never seen.

### 6.4 A finding that turned out to be an artefact

Sorting all 13,675 announcements by earnings surprise and holding for a month
gives a spread of **−88bp with t = −3.65**. Companies that *beat* expectations
*underperformed*.

It had the right shape to be real: absent on the announcement day, absent after
a week, emerging only over twenty days — accumulation, which is what
distinguishes drift from an instant reaction.

Two things were wrong with it. It was monotonically strongest in the **most**
liquid companies, and every theory of why drift exists predicts the opposite.
And the diagnostic from section 3.2 explains it: the sorting variable was a
median 83 days old.

On the 1,982 announcements whose financial statements were filed *with* the
announcement — the subset where the question can actually be asked — the spread
is **−59bp at t = −1.03**. Nothing.

### 6.5 The conclusion, in two halves

**Earnings-related fundamentals do rank subsequent abnormal returns.**
Information coefficient 0.067 at t = 3.40, out of sample, across six frozen
years, against −0.047 for the naive alternative of sorting on the earnings
surprise alone.

**Nothing here is tradable.** Net Sharpe −0.30, sitting at the 85th percentile
of its own shuffled-prediction null. No alpha against standard factors. A
deflated Sharpe of 0.00 across twelve logged specifications. Still losing at
half the assumed trading costs.

**And one caveat over the first half:** the fundamentals are a median 83 days
old, so this is a statement about quarterly company accounts, not about earnings
announcements.

Those three statements are compatible, and holding all three at once is the
result. Two of the five falsification criteria written down *before* the study
ran are met.

---

## Part 7 — Questions you will be asked

**"So it doesn't work?"**
It ranks companies better than chance, significantly so. It does not make money.
Those are different claims and both are true.

**"Then why is it interesting?"**
Because the difficult part of this field is not building a model, it is knowing
whether to believe one. Most of the work here is machinery for telling a real
result from a flattering one — and it caught three flattering ones, two of them
in this project's own earlier conclusions.

**"Why not just use a neural network?"**
The constraint is not model capacity. Single-stock returns are almost entirely
noise, the sample is 13,675 events, and the binding problems are data quality,
timing discipline and multiple testing. A more flexible model applied to a
quarter-stale feature fitted on six years would overfit faster, not predict
better. The information coefficient tripled here because of two data fixes, not
a better estimator.

**"Couldn't you make it profitable with leverage?"**
No. Leverage scales returns *and* volatility, so the Sharpe ratio is unchanged.
A negative Sharpe levered up is a bigger negative number.

**"How do I know none of this is overfitted?"**
Every reported number comes from a year the model never saw during training.
Every specification tried is logged, including abandoned ones, and the headline
Sharpe is deflated for that count. The trading result is compared against 200
runs of the same machinery on shuffled predictions. And the whole pipeline is
run on data with a known planted answer, and on data with none, to show it
detects the first and not the second.

**"What would change the answer?"**
Three things, in order. A price source covering delisted companies — 61% of the
companies deleted from the index are missing, and they are disproportionately
the failures. Earnings figures taken from the announcement itself rather than
from accounts filed weeks later. And a universe beyond the S&P 500, since large,
heavily-traded companies are the least likely place for this effect to survive.

---

## Appendix — Glossary

| Term | Meaning |
|---|---|
| **8-K** | The SEC form a US company files to announce something material, including earnings |
| **10-Q / 10-K** | Quarterly and annual reports, filed weeks after the announcement |
| **Abnormal return** | Actual return minus expected return |
| **Alpha** | Return not explained by known risk factors |
| **Basis point (bp)** | One hundredth of a percent |
| **Beta** | How much a share moves for a given market move |
| **CAR** | Cumulative abnormal return over a window |
| **Deflated Sharpe** | A Sharpe ratio adjusted for how many strategies were tried |
| **Drawdown** | Peak-to-trough loss |
| **Embargo** | A gap between training and test data, beyond purging |
| **IC** | Information coefficient: correlation between prediction and outcome |
| **Long/short** | Buying expected winners and selling expected losers |
| **Look-ahead bias** | Using information that did not exist yet |
| **PEAD** | Post-earnings-announcement drift |
| **Point-in-time** | Data as it was known on a given date, not as later revised |
| **Purging** | Removing training events whose outcomes overlap the test period |
| **Quintile** | One fifth of a ranked list |
| **Sector-neutral** | Holding no net position in any industry |
| **Sharpe ratio** | Return per unit of volatility |
| **SUE** | Standardised unexpected earnings: the surprise, in units of its own past variability |
| **Survivorship bias** | Studying only the companies that made it |
| **t-statistic** | Size of a result relative to its uncertainty |
| **Turnover** | How much of the portfolio is replaced per year |
| **Walk-forward** | Train on the past, test on the future, roll forward, repeat |
| **XBRL** | The structured format SEC financial statements are filed in |
