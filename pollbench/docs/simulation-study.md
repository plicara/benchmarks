# Simulation study

This runs the method on tables where the true reliability is known.
It reports the model after protocol amendment 5 (NUTS sampler, location and scale fixed exactly).
The raw numbers are in `results/simulation-study.json`; regenerate with `make simstudy` or `python -m pollbench.simstudy --seeds 8`.

## Design

Each table has 120 models and 40 benchmarks drawn from the sigmoid-Beta model: newer models are stronger, newer benchmarks are harder, slopes and precisions vary by benchmark, and each model reports on the benchmarks available at its release.
Eight seeds per scenario, with five-fold cross-validation over cells for every method.

- `well_specified`: the data come from the fitted model.
- `second_factor`: each benchmark also loads on a second model trait (sd 0.8), so capability is not one number.
- `selective_reporting`: a score in the bottom quartile of its benchmark goes unreported with probability 0.6.
- `sparse`: half the coverage, about 460 cells.
- `chance_floor`: 40% of benchmarks have a chance floor of 0.25 or 0.5 (expected score f + (1 - f) sigmoid), which the fitted model does not have. Added after the independent audit (amendment 5).

## Results

| scenario | rank agreement of reliability with truth | 90% interval covers truth | mean bias | `hier_beta` best CRPS in | held-out 90% coverage of `hier_beta` | condition 4 statistic |
| --- | --- | --- | --- | --- | --- | --- |
| well specified | 0.90 | 0.90 | -0.01 | 8 of 8 seeds | 0.89 | 0.93 |
| selective reporting | 0.88 | 0.92 | -0.02 | 8 of 8 | 0.89 | 0.91 |
| sparse | 0.84 | 0.90 | -0.02 | 8 of 8 | 0.89 | 0.87 |
| chance floor | 0.89 | 0.86 | -0.00 | 8 of 8 | 0.89 | 0.95 |
| second factor | 0.49 | 0.46 | -0.22 | 8 of 8 | 0.89 | 0.93 |

Mean held-out CRPS (lower is better), well specified: `benchmark_mean` 0.124, `mean_rank` 0.070, `low_rank_1` 0.065, `eci_ls` 0.054, `hier_beta` 0.047.
The same ordering of `eci_ls` and `hier_beta` holds in every scenario.

## What it shows

1. The ranking of benchmarks by reliability is recovered well when capability is close to one number (0.84 to 0.90), and the hierarchical model forecasts better than the four simpler predictors in all 40 runs.
2. The 90% intervals on reliability cover the truth 90% to 92% of the time when the model is right, and the held-out 90% intervals cover 89%.
3. Chance floors cost little: rank agreement 0.89, no bias, interval coverage 0.86. This is consistent with the real-data result that a floor model does not remove the leftover residual correlation (results report).
4. A second capability factor is the failure mode that matters. Reliability is understated by 0.22 on average, rank agreement falls to 0.49, and the intervals cover the truth 46% of the time. Held-out forecast error and coverage barely move, and the condition 4 statistic stays at 0.93, so condition 4 cannot detect it. That is why conditions 3 and 4 are reported as consistency checks and why the second-factor check exists; its own type-I error and power are in `results/second-factor-calibration.json`.
5. Selective reporting and sparse coverage cost little here.

## Limits of this study

- The generator and the fitted model share their functional form, so the well-specified row is a best case and does not validate coverage on real data.
- The second factor and the chance floor are two shapes of misspecification. Real structure (era effects, agentic versus knowledge benchmarks, contamination, scaffolds) may be different, and there are no ceilings, heteroskedasticity or era effects here.
- Eight seeds per scenario give rough means; differences of a few hundredths between scenarios are within noise.
