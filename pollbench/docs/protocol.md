# Protocol v1

Status: registered before the first fit on real data, with five dated amendments below; amendments 2 to 5 were written after results had been seen.
The commit that adds this file is the registration; anything that differs in the report is listed there as a deviation.

## Question

How much of the score variation between models on a benchmark reflects a shared capability rather than noise, how well can a benchmark's scores be predicted out of sample from the models' other scores, and does a public audit verdict line up with either?

## Data (tier 1)

- Epoch AI benchmark data, `benchmark_data.zip` (CC BY 4.0, attribution to Epoch AI), as a long CSV of one score per model per benchmark. The loader records the SHA-256 of the file and of the filtered cell table.
- Scores outside [0, 1] and non-numeric scores are dropped and counted. Scores are clipped to [0.005, 0.995].
- A model scored more than once on a benchmark keeps the maximum (the choice Epoch's index makes for multiple setups); the number collapsed is reported.
- Benchmarks with fewer than 5 models and models with fewer than 4 benchmarks are dropped, repeatedly, until both hold.
- The `source` field is kept so cells reported by model developers can be separated from cells Epoch ran; the primary analysis uses all cells, a sensitivity analysis drops developer-reported cells.
- No data from any other source enters tier 1. Messier (task-level agent data) is a later tier and is not part of this protocol.

## Methods compared

Five predictors of a held-out score, each returning predictive samples:

1. `benchmark_mean`: moment-matched Beta of the benchmark's training scores.
2. `mean_rank`: the model's average within-benchmark z-score, mapped to the benchmark by simple regression, Gaussian residual.
3. `low_rank_1`: rank-one matrix completion, Gaussian residual.
4. `eci_ls`: score = sigmoid(slope x (capability - difficulty)) by ridge least squares (Epoch's index form), Gaussian residual per benchmark. The predictive spread is ours; the index itself publishes none.
5. `hier_beta`: the same mean with a Beta likelihood whose precision is estimated per benchmark, theta ~ N(0, s^2), difficulty ~ N(mu_d, (2 s)^2), log slope ~ N(0, 0.5), log s ~ N(log 1.5, 0.75) (amendment 1), log precision ~ N(mu_k, 1), weak priors on the three hyper-means, posterior by Hamiltonian Monte Carlo.

Hyper-scales above are fixed by this protocol. The sensitivity analysis refits with the precision scale at 0.5 and 2.

## Evaluation

- **Split A, five-fold over cells**: each cell is assigned to a random fold (seed 0); cells that would leave a model or benchmark with fewer than 3 training cells return to training. Primary outcome.
- **Split B, frontier**: the newest 25% of dated models; half of each such model's cells (random, seed 0) are held out, subject to the same minimum.
- **Score**: CRPS from 300 predictive draws per cell. Coverage of the central 50% and 90% intervals is reported alongside.
- **Comparisons**: mean CRPS per model, then a paired bootstrap over models (2,000 resamples, seed 0) for each pair of methods.
- The sampler for cross-validation folds uses 2 chains with 100 pilot, 100 warm-up and 150 kept iterations; the final rating fit uses 4 chains with 150 pilot, 300 warm-up and 600 kept iterations.

## Ratings

For each benchmark and each posterior draw:

    R = Var_m(p) / (Var_m(p) + E_m[p (1 - p) / (kappa + 1)])

where p is the model-implied expected score of each model scored on the benchmark and kappa is the benchmark's precision.
Reported: posterior mean and 90% interval of R; noise in score points (sd at p = 0.5); difficulty and slope; share of models at or above 0.90 and the same for the newest quarter of models; held-out skill, 1 - CRPS(hier_beta) / CRPS(benchmark_mean) on the benchmark's held-out cells with a bootstrap interval.

**Letter grades are a presentation convenience, fixed here**: A if P(R >= 0.85) >= 0.90, else B if P(R >= 0.70) >= 0.90, else C if P(R >= 0.50) >= 0.90, else D. A benchmark with fewer than 10 models, or whose worst split R-hat among difficulty, slope and precision exceeds 1.1, is ungraded. A benchmark is flagged **saturated**, separately from its grade, if at least half of the newest quarter of models score 0.90 or more (needing at least 3 such models).

## Conditions for reporting ratings

The ratings are reported as findings only if all four hold; if not, the report states which failed and presents the ratings as exploratory:

1. On Split A, `hier_beta` beats `mean_rank` on mean CRPS with a paired-bootstrap 95% interval that excludes zero.
2. Held-out 90% interval coverage of `hier_beta` lies in [0.80, 0.95] on Split A.
3. Reliability computed from a fit on models dated up to 12 months before the newest model correlates with reliability from the full fit at Spearman 0.70 or higher, over benchmarks with at least 10 models in both.
4. Across benchmarks, model-based reliability and held-out skill agree at Spearman 0.50 or higher.

If conditions 1 and 3 both fail, the report is a negative result and no meta-benchmark is built.

## Audit join (exploratory)

Epoch's Benchmark Reviews verdicts (Verified, Flawed, Not enough information) are transcribed by hand from the public review pages with the review date and link; they are attributed to Epoch and used for citation only.
For benchmarks present in both tables the report shows reliability and skill by verdict.
With roughly ten overlapping benchmarks no significance test is run or claimed.

## Robustness

- Drop developer-reported cells and refit.
- Precision scale 0.5 and 2.
- Five refits on random 70% subsets of models: report how often each benchmark's letter changes.
- Leave-one-model-out influence on each grade for the three most influential models.

## What this study does not claim

- That a grade is a measure of validity: reliability here is relative to the shared capability the model infers, which is itself shaped by the same benchmarks.
- That relative offsets are accuracy or bias: nothing in tier 1 anchors scores to an external outcome.
- Anything about benchmarks absent from tier 1 or about item-level quality.

## Validation of the method before real data

A simulation study (`pollbench.simstudy`) draws tables from the generating model with known truth, including a second capability factor, selective reporting and sparse coverage, and checks rank agreement and interval coverage of the reliability, and held-out CRPS and coverage of every method. Its output is `results/simulation-study.json` and its summary is in `simulation-study.md`.

## Amendment 1 (before any rating was inspected)

Two facts changed between writing the protocol and the first fit; neither depends on how any benchmark was rated.

**The file.** The archive Epoch publishes is not a single long CSV.
It holds one CSV per benchmark and `epoch_capabilities_index/processed_data_for_eci.csv`, the long table Epoch's index is fitted on: 2,898 scores, 274 models, 60 benchmarks, one score per model and benchmark, dates 2023-02-24 to 2026-09-29.
Tier 1 uses that table as it is, so the benchmarks are the ones Epoch put in its index (61 flagged `in_eci` in `benchmark_metadata.csv`, 60 with scores).
That is a selection: a benchmark that did not fit the index is less likely to be in it, so reliabilities here are conditional on inclusion and probably flatter the set.
The archive is `epoch-benchmark_data-2026-10-10.zip`, SHA-256 `8f663d418c96c478315d5f8684267e2059529b325305d685d52e55b189040b23`.
The minimum-count filters change nothing on this table; the loader reports that.

**Developer-reported cells** (for the robustness refit) are those whose `source` names a technical report, system card, model card, release announcement, or a developer's own paper: it contains one of `technical report`, `system card`, `model card`, `announcement`, `annoucement` (Epoch's spelling in one source), `tweet`, `report`, `herd of models`, `open foundation`, `open models`, `mixtral`, `llama`, `gemma`, `falcon`, `baichuan`, `yi:`, `textbooks are all you need`, `training compute-optimal`, or `papers`, case-insensitively.
Blank sources are cells Epoch ran itself.
Leaderboards, benchmark websites and third-party repositories are not developer-reported.

**The model's scale.** The first fit on the real table failed to converge (split R-hat up to 1.8, 19 of 60 benchmarks under 1.1).
The cause is identification: with theta fixed at sd 1 and the mean log slope free, the mode collapses theta's spread to 0.1 and sends slopes toward e^3, and the sampler cannot cross that ridge.
The model now pins the mean log slope at 0 and estimates the capability spread s, with difficulty measured in units of s (the table above).
That fit was run once, to read diagnostics; no reliability, grade or held-out score from it was looked at.
The simulation study was rerun on the amended model and `simulation-study.md` reports the amended model only.

**Definitions left open.** "Newest quarter of models" means the 25% of dated models with the latest date.
The temporal-stability check (condition 3) fits only models dated on or before 365 days before the newest model.
"The three most influential models" for leave-one-model-out are the three with the most scores in the table, a rule fixed here because influence cannot be measured before the fits.

## Amendment 2 (after one full run; the run is disclosed)

The full analysis was run once on the real table with the amendment 1 model.
Its robustness refits on random 70% subsets did not converge (split R-hat 4.8 to 70, effective sample size 1 to 3), so the "how often does a letter change" check was uninformative.
The cause was the sampler's starting point: the mode used to start and whiten HMC sits in a degenerate corner where a sparse benchmark is fitted perfectly and its precision runs away.
The starting point now comes from the conditional mode with the capability spread, the per-benchmark precisions and their hyper-mean held at their prior centres; the model and the priors are unchanged.
The first run's headline numbers were seen before this change (all four reporting conditions held; hierarchical Beta and the ECI form tied on Split A).
Nothing in the change was chosen from them, and the simulation study and the real analysis were rerun from scratch after it; only the rerun is reported.

## Amendment 3 (diagnostic added after the first real run)

The construction below was replaced by amendment 4 because it was anti-conservative.

The simulation study found that a second capability factor is the one misspecification the reporting conditions cannot detect.
A posterior predictive check was added for it, `pollbench.checks.second_factor_check`: the mean squared correlation of residuals over benchmark pairs sharing at least 12 models, compared with the same statistic on 100 tables simulated from the fitted model on the same cells.
On simulated tables it flags a 0.8 second factor and stays quiet without one (`tests/test_analysis.py`).
It was added after the first full run, so it is a diagnostic and not one of the four reporting conditions.
A p-value below 0.05 does not stop the ratings from being reported.
It changes what they are called: reliability with respect to the single shared factor, which understates the reliability of benchmarks that load mainly on something else.
The most correlated residual pairs are listed as exploratory.

## Amendment 4 (after an adversarial review of the first complete results)

An independent review and my own stress tests found the following. Each item is a change made after results were seen, and the results were regenerated.

- The second-factor check of amendment 3 was anti-conservative: observed residuals were taken against the posterior-mean fit and replicates against that same mean, which are not the same quantity, and its type-I error was about 11% at the 0.05 level.
  It is replaced by a within-draw posterior predictive check (observed and simulated statistics computed against the same draw's expected scores) with 500 draws.
  `pollbench.calibrate` measures it on 40 simulated tables per setting (200 models, 50 benchmarks, 12% of cells observed): with the amendment 5 sampler it rejects 1 of 40 with no second factor, 12 of 40 at a loading sd of 0.3, and 39 of 40 at 0.8 (`results/second-factor-calibration.json`; with the earlier sampler the counts were 2, 14 and 38).
  A non-rejection therefore does not rule out a modest second factor.
- Reliability over the scored models depends on who happened to be scored.
  On the first complete run the rank correlation between it and the reliability of the same benchmark with every model scored was 0.84 and 34 of 60 letters differed (current numbers are in the results report).
  A benchmark run only on frontier models has a narrow capability range and reads low however sharp it is.
  The protocol reliability stays the graded quantity.
  The all-models reliability and the spread of capability among the scored models (as a share of the spread among all models) are reported next to it, and the page says the D group is "not shown to clear 0.50", not "does not clear it".
- Seed noise flips a few letters near the thresholds (|change in R| up to 0.03).
  A repeat fit with a different seed is now part of the robustness arms and the number of changed letters is reported.
- Robustness counts are now taken over the benchmarks graded in both fits and report how many could not be compared (the earlier count treated a benchmark missing from the other fit as unchanged).
  The developer-reported arm uses minimums of 3 models per benchmark and 3 benchmarks per model, because the protocol minimums removed most of the benchmarks.
- The developer-reported marker list had two false positives (`tweet`, `papers`, which matched a benchmark author's tweet and PapersWithCode) and missed two developer sources (the Claude 3 model family post and the Claude Instant release).
  Cells with an empty source (1,270 of them) are treated as not developer-reported.
- Condition 1 is also reported with a crossed bootstrap that resamples models and benchmarks together (it counts benchmark-to-benchmark variation that the protocol's model bootstrap ignores).
  The protocol's interval still defines the condition, and both are shown.
  The CRPS estimator now divides the pairwise term by n(n-1), which removes a small positive bias that is the same for all methods.
- Condition 1 was also run against `eci_ls` (the nearest competitor) and a clip level of 0.02 was run as an arm, with letters compared as above.
- The digest now covers sources and dates as well as scores, the input file hash is recorded, a model's or benchmark's date is the most common date it appears with (ties to the earliest) and not the first one seen, the sampler starts chains from dispersed points, the effective sample size uses Geyer's initial positive sequence, and `rate_benchmarks` refuses Laplace-only fits.
- The wording "frozen" is narrowed accordingly: the protocol was frozen before the first fit and amended afterwards, and amendments 2, 3 and 4 came after results were seen.

## Amendment 5 (sampler, after an independent audit and the dispersed-start test)

Starting chains from dispersed points (amendment 4) showed that the hand-written HMC did not converge reliably: on the seed-repeat fit and on several robustness refits one or more chains sat at a vanishing capability spread with step sizes near 1e-6, and the worst R-hat ran from 1.2 to about 100.
An independent audit of the committed results found the same problem from the other side (a rerun of the committed seed gave worst R-hat 1.25 against the committed 1.09).
Changes:

- The sampler is NumPyro's NUTS (4 chains, 500 warmup, 1,000 kept draws each in the final fits; 2 chains with 150 and 150 in cross-validation folds), started from dispersed points around the conditional mode of amendment 2.
- Location and scale are fixed exactly instead of through priors: capability has mean zero, the log slopes have mean zero (amendment 1 pinned it only through the prior), and mu_d is the mean difficulty.
  Only a(theta - d) enters the likelihood, so this changes no prediction and no reliability; with only soft pins, chains drifted along the two ridges (difficulties had effective sample sizes near 4).
  Capability and difficulty are sampled non-centred.
- R-hat is the rank-normalised split R-hat, the larger of the bulk and folded versions (Vehtari et al. 2021), and ESS is bulk ESS with Geyer's initial positive sequence.
  A benchmark is graded only when the R-hat of its difficulty, slope and precision is at most 1.01 (was 1.1 on plain split R-hat).
- Conditions 3 and 4 are labelled consistency checks in every report: 3 compares the full fit with a fit on a subset of the same data, and 4 compares two variance-explained quantities from the same fit.
  The comparison with Epoch's own form (`eci_ls`) leads the report.
- The audit noted that the most correlated residual pairs sit among older multiple-choice benchmarks with chance floors, which the model lacks.
  A sensitivity fit with a per-benchmark floor (f_b + (1 - f_b) sigmoid, logit f_b ~ N(-3, 1.5^2)) is run, the second-factor check is repeated on it, and the simulation study gains a scenario in which 40% of benchmarks have a chance floor of 0.25 or 0.5.
- Ungraded benchmarks show no point reliability on the page, and the page leads with intervals.
- The results report is generated from the results JSON (`python -m pollbench.report`), so its numbers cannot drift from the run.
