# Predicting TikTok 30-Day View Counts

Forecast the cumulative view count a video reaches at **day 30**, using only the
**first five days** of engagement data. Evaluated with RMSE.

**Result: R² 93.4% on a held-out set of 3,001 videos — 52% lower RMSE than the
day-5 persistence baseline, 74% lower than predicting the mean.**

---

## Data

Three tables, ~6.5M rows:

| table | rows | contents |
|---|---|---|
| `videos` | 209,543 | metadata, topic, music, duration, emotion scores |
| `creator_daily` | 278,433 | daily follower / favourite counts per creator |
| `engagement_daily` | 6,068,955 | daily cumulative views, likes, shares, collects, downloads |

Only **73,940** videos have a day-30 observation, so those form the modelling
universe. 3,001 are held out as a test set and never touched during training.

Two columns were dropped immediately: `is_ads` has a single value, and
`enterprise_verified` is 100% null.

## The problem

The target is brutally skewed — **skew = 30**:

| p50 | p90 | p99 | max |
|---|---|---|---|
| 670 | 21,598 | 617,106 | 20,653,259 |

Because RMSE squares the error, **the top 1% of videos account for 97.7% of the
total squared error**. One miss on a 5M-view video outweighs thousands of misses
on 400-view videos. Every design decision below follows from that.

## Approach

**1. Predict the growth ratio, not the level.**
Views are cumulative and monotonic (99.4% of series never decrease), so
day-30 = day-5 × a multiplier. The multiplier has a median of 1.16 and almost no
skew, while the level spans seven orders of magnitude. Fitting
`log1p(y) − log1p(day5_views)` and adding the anchor back was worth **25,700 RMSE**
— the single biggest gain in the project, and it comes from understanding the
data rather than from a bigger model.

**2. Weight the fit by y².**
In log space an error `e` becomes a raw error of roughly `y·e`, so weighting each
row by `y²` makes the log-space fit approximate raw-scale least squares. Worth
another **6,100 RMSE**.

**3. Blend five models with NNLS weights.**
Ratio models at three weightings, plus raw-L2 and Tweedie. A paired bootstrap
confirms the blend beats every individual model with the 95% CI excluding zero.
Blending helps because the ratio models and the level models fail on *different*
videos.

**4. Features — 319, all from days 0–5 only.**
- *Trajectory*: levels, daily increments, acceleration, growth multipliers,
  log-slope and curvature for each of 7 metrics
- *Ratios*: shares/likes/comments per view at day 1 and day 5, and the change
  between them — a rising share rate is the classic pre-viral tell
- *Peer context*: video vs. its creator's / sound's / topic's usual performance,
  percentile ranks within creator and posting day
- *Creator*: follower counts as of the posting date via `merge_asof(direction="backward")`,
  which structurally prevents look-ahead leakage

## Results

| step | CV RMSE |
|---|---|
| Predict the mean | 324,033 |
| Day-5 views as-is | 161,605 |
| LightGBM, raw target | 152,860 |
| Log target | 157,999 |
| **Growth ratio vs day 5** | **127,159** |
| + weight by y² | 121,069 |
| + peer-context features | 119,823 |
| **5-model NNLS blend** | **114,748** |

Held-out set (3,001 videos): **RMSE 62,374 · R² 93.4% · Spearman 77.9%**.
On the top 1%, median error is **16.6%** and **87% land within 2×**.

## What did not work

| attempt | result | why |
|---|---|---|
| Creator history encoding | **much worse** (134k vs 120k) | virality is video-specific, not creator-specific — past growth does not transfer |
| Duan smearing on the log model | worse | the ratio parametrisation already removes most retransformation bias |
| Exact raw-L2 custom objective | diverged | the `exp(2z)` Hessian explodes on 7-figure targets |
| Tail-specialist model | worse on its own segment | training on 10% of rows loses more than specialisation gains |
| Capping sample weights | worse | capped videos are exactly the ones RMSE cares about |
| Segment-wise blend weights | inside noise | gain smaller than fold-to-fold variation |

## Honest limitations

**The score is dominated by luck.** Scoring the same model on 2,000 different
random 3,001-video test sets gives:

| p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|
| 48,346 | 67,111 | **85,637** | 120,464 | 223,322 |

Our holdout's 62,374 sits at the **19th percentile** — a favourable draw. The
honest expectation on a typical test set is ~85,600. The *relative* improvements
(52% / 74%) are the draw-robust numbers, since baseline and model are measured on
the same videos.

**13 videos out of 73,922 carry 37% of all squared error.** They were small at
day 5 and exploded later; nothing in days 0–5 distinguishes them, so that error
is largely irreducible.

**The model is deliberately bad on small videos** (median error ~150%) because
they contribute nothing to RMSE. Under a percentage-error metric the design would
be completely different — probably a plain log-target model with no tail weighting.

## Running it

```bash
python build_features.py    # raw tables -> features.pkl
python add_features.py      # + peer context -> features2.pkl
python final.py             # train, evaluate on holdout, write submission
```

Or open `tiktok_day30_views.ipynb`, which runs the whole pipeline start to finish.

| file | purpose |
|---|---|
| `build_features.py` | trajectory, video and creator features |
| `add_features.py` | peer-group context features |
| `final.py` | holdout split, ensemble, evaluation, submission |
| `exp_targets.py` | target-transform comparison |
| `exp_weight.py` | sample-weighting sweep |
| `ensemble.py` | base models + NNLS blending |
| `compare.py` | paired bootstrap significance tests |
| `diagnose.py` | error decomposition and calibration |

Requires: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `scipy`, `matplotlib`.
