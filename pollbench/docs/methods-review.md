# Methods review: how should we rate benchmarks?

Status: research pass 4, 2026-10-10 (pass 1 was 2026-10-09; additions in sections 6 to 8 supersede earlier text where they differ; section 8.6 states the current plan).
Reading depth matters here, so it is marked per source: **full** means I read the content; **abstract** means I only saw the abstract or a summary of the page; **memory** means I am recalling it and it must be checked before we cite it.
Nothing here is a result.
It is an attempt to make the method choice earn its place against alternatives instead of inheriting it from the Silver Bulletin framing.

## 1. Bottom line

1. The closest existing work is closer than I said earlier.
   Epoch's Capabilities Index already fits `performance = sigmoid(discriminability * (capability - difficulty))` and publishes a **difficulty and a discriminability (slope) per benchmark, with bootstrap intervals**, under an MIT licence ([methodology](https://epoch.ai/data/eci-documentation/methodology), [code](https://github.com/epoch-research/eci-public), [paper](https://arxiv.org/abs/2512.00193); methodology page and README read in full, paper abstract plus a summary of the full text).
   So "scoring benchmarks with an IRT-style model" is not new by itself.
2. What I could not find in this pass: a **benchmark rating** that combines (a) a statistical reliability and predictive-validity score with honest uncertainty, (b) an audit score for item quality, contamination, reward hacks and openness, and (c) a published protocol that is graded against held-out outcomes before the ratings are trusted.
   Pieces exist separately (ECI slope; BetterBench checklist; an IRT mislabel audit; a construct-validity review), but I found no one combining them.
   This is absence-of-evidence from about a dozen searches, not proof.
3. The Silver Bulletin analogy is a **source of ideas, not a method**.
   I found no independent evidence in this pass that pollster-style letter grades predict future accuracy; the only support I read was the rater's own methodology description.
   That does not make them wrong, but it means "Nate Silver does it" should carry no weight in our justification.
4. My recommendation changes from "adopt the pollster-rating scheme" to: **build several candidate rating methods, grade them against held-out data under a protocol written before any fit, and publish the one that wins, with the losers reported.**
   The pollster-style ideas become one candidate among several.

## 2. What the literature says, by question

### 2.1 Is there a common factor, and how strong?

- Epoch's benchmark-correlation analysis: median Spearman 0.73 across 15 benchmarks, 0.79 within category and 0.68 across, and 0.90 median against ECI, which is partly circular because ECI is built from them. Caveats the authors give: few shared models, selection bias, sparse coverage, and correlation inflated because newer models are better at everything ([data insight](https://epoch.ai/data-insights/benchmark-correlations); read in full; the page's own summary and body disagree on the benchmark count and median, so check the CSV before quoting).
- metabench distils six older benchmarks (ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande) to under 3% of items and recovers a single factor with r = 0.94 to the total score ([abstract](https://arxiv.org/abs/2407.12844); abstract).
- ECI's own paper assumes unidimensionality and says benchmark-specific strengths show up in residuals; average residual R² is 0.71 for benchmarks with more than 25 scores ([summary of paper](https://arxiv.org/pdf/2512.00193v1); full text via a summarizer).
- Implication: a single factor explains much but not all. A rating that equates "loads on the common factor" with "good" will penalise benchmarks that measure something real and different (agentic work, long context, safety). We must report loading and distinctness separately and not collapse them.

### 2.2 The scale and time confound

- Kearns (2026 thesis) argues latent-factor models ignore model scale so factors end up tracking size, while scaling laws ignore measurement error; a structured model that lets scale inform capability and capability inform scores fits better and predicts out-of-distribution benchmarks better on the Open LLM Leaderboard ([abstract](https://arxiv.org/abs/2602.15532); abstract only, single leaderboard, her own "first" claim unverified).
- Observational scaling laws predict agentic and post-training effects from low-dimensional capability measures built from about 100 public models ([abstract](https://arxiv.org/abs/2405.10938); abstract only; held-out procedure not read).
- Epoch's caveat (above) says the same thing from the other side: long-lived benchmarks correlate because time passes.
- Implication: any "predictive validity" grade must control for release date and scale, otherwise every benchmark that spans a long period looks excellent. This is the closest analogue of the pollster rating's adjustment for election type and timing, and it is more important here.

### 2.3 Item-level versus aggregate-level models

- ECI and our planned data are **aggregates** (one number per model per benchmark). The ECI paper itself notes that this differs from item response theory proper and that splitting one benchmark in two would double its weight (summary of full text).
- Item-level IRT work exists: PSN-IRT on 41,871 items from 11 benchmarks finds significant measurement-quality shortcomings and builds smaller benchmarks ([abstract](https://arxiv.org/abs/2505.15055)); an IRT indicator flags likely mislabeled items, with about 95% of the top 200 flagged items looking mislabeled across seven benchmarks and 114 models ([abstract](https://arxiv.org/abs/2605.30504)).
- Implication: item-level data is far more informative about benchmark quality but is only public for some benchmarks. v1 with public aggregates cannot do item-level audits. Item-level results should be a labelled second tier for benchmarks where it exists, not a requirement.

### 2.4 Uncertainty: what the polling literature actually contributes

- Shirani-Mehr, Rothschild, Goel and Gelman decompose poll error into variance plus squared bias; across 4,221 polls in 608 races (1998 to 2014), total error is about twice what reported sampling margins imply ([blog summary](https://statmodeling.stat.columbia.edu/2026/01/20/survey-statistics-total-margin-of-error/); read in full; the primary paper I have only seen listed).
- Implication, and it is the strongest transferable idea so far: a benchmark's reported error bar (binomial on item count) understates its true error, and **non-sampling error is the thing to estimate and rate**.
  ECI's bootstrap resamples each model's benchmark results, which captures fit instability but not this item-versus-construct bias.
- A caution from the same post: adding covariates (like pollster) can move error from variance into bias without reducing total error, so how we partition matters less than reporting total error out of sample.

### 2.5 Quality audits and construct validity

- BetterBench: 46 best practices over a benchmark lifecycle, 24 benchmarks assessed, wide quality variation, most benchmarks lack significance reporting and easy replication ([abstract](https://arxiv.org/abs/2411.12990); abstract; I did not confirm how items are scored or whether scores correlate with anything).
- A review of 445 LLM benchmarks by 29 reviewers finds recurring construct-validity weaknesses and gives eight recommendations ([abstract](https://arxiv.org/abs/2511.04703)).
- Saturation study of 60 benchmarks: nearly half show saturation, saturation rises with age, and expert curation matters more than keeping test data private ([abstract](https://arxiv.org/abs/2602.16763); the 14 properties and the definition are not in what I saw).
- Implication: audit criteria can be adopted from BetterBench instead of invented, and "does the audit score predict anything statistical?" is itself a testable and publishable question (do checklist-good benchmarks have higher out-of-sample predictive validity?).

### 2.6 Pollster-rating ideas

From Silver Bulletin's January 2026 update ([page](https://www.natesilver.net/p/pollster-ratings-silver-bulletin); read in full): plus-minus against expected error, house effect versus bias, shrinkage by poll count, a transparency bonus, a herding penalty, grades from predicted future accuracy.
Evidence for these as methods is largely the rater's own description; I did not find an independent evaluation in this pass.
Treat them as candidate features to test.

## 3. Decision points, options and what would decide them

| # | Decision | Options | Evidence so far | How we decide |
|---|---|---|---|---|
| 1 | Measurement model | A. ECI-style least squares sigmoid (reuse MIT code). B. Hierarchical Bayesian sigmoid with binomial likelihood and per-benchmark overdispersion. C. PCA first factor. D. Kearns-style structured model with scale or date covariate. E. Mean rank baseline. | ECI is published and reusable; B is the natural fix for its weak error model but is untested here; C is the observational-scaling-laws tradition; D had promising abstract-level evidence, but pass 2 found its out-of-fold gains over observational scaling laws were not statistically significant (section 6). | Leave-one-benchmark-out and leave-one-model-out prediction error on the same data, including a time-ordered split. Pick the simplest model within noise of the best. |
| 2 | Rating target | i. Common-factor loading. ii. Out-of-sample predictive validity on held-out benchmarks, controlling for date and scale. iii. Agreement with human preference. iv. Agreement with real-task outcomes. | (i) is internal only and inflated by the time confound; (ii) is the closest to "graded against the result"; (iii) and (iv) have no dataset chosen yet. | Report (i) and (ii) separately as v1. (iii) only if a clean public dataset exists. Do not collapse into one score before seeing whether they disagree. |
| 3 | Audit component | BetterBench items; item-level IRT flags where data exists; our own checks (contamination, reward hacks, openness). | Exists but unscored against anything. | Test whether audit score predicts statistical score. If not, report them as two axes, not one grade. |
| 4 | Grades | Letter grades from a composite versus continuous scores with posterior intervals and a probability of being above a bar. | Polling uses letters, but it is an editorial choice, not an evidence-based one. | Publish continuous scores and intervals as primary; derive letters only as a labelled convenience with rounding when intervals straddle a cutoff. |
| 5 | Treatment of missingness | Ignore; report sensitivity; model selection into reporting. | I found no ready method for LLM benchmark reporting selection; labs choose which scores to publish. | Prefer sources with uniform harnesses; report a sensitivity analysis dropping developer-reported cells. |
| 6 | Time | Static snapshot with dates versus a dynamic model of benchmark decay. | Saturation is common and age-related; a sigmoid already encodes ceilings. | v1: snapshot, date-stamped, sigmoid only. Dynamic decay later if the snapshot ratings are unstable across two cut-offs. |
| 7 | Data source | Epoch benchmarking data (check licence and the exact CSV), public leaderboards, papers. | Epoch's ECI input is a public CSV; the licence for the data was not stated on the pages I read. | Check licence first. Prefer one coherent source plus a clearly separate external-reports tier. |

## 4. What I would write down before fitting anything

1. The candidate rating methods in decision 1, with code stubs.
2. Two cut-offs (an earlier and a later date) so ratings can be re-estimated and compared (stability is itself a result).
3. The held-out protocol: which benchmarks and models are held out, how date and scale are controlled, and the error metric.
4. Kill criteria: if no model beats the mean-rank baseline out of sample, or if ratings flip between cut-offs, the honest output is "benchmark ratings from public aggregates are not reliable" and we publish that.
5. A data-use note covering licence and any redistribution limits.

## 5. Gaps remaining after pass 3

- Jackman (2005), the standard pooled-polls model, was not accessed directly (the publisher page failed); I read two re-implementations that cite it (7.3). The Shirani-Mehr et al. paper itself was read in full (7.2).
- The psychometric short-form benchmark paper ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S016028962500025X)) is blocked from this environment and unread.
- The scoringRules paper does not cover significance tests for score differences; the test for "method A beats method B out of sample" needs another source or our own bootstrap (7.4).
- Two searches found no independent quantitative evaluation of pollster ratings (6.5); this is a negative result about my search, not about the literature.
- LLM-side papers were mostly read through a summariser of the full text; numbers I plan to quote publicly still need a line-by-line check against the PDF.

## 6. Second pass additions (2026-10-10)

### 6.1 Data availability and licence

- Epoch's benchmarking data is **CC BY 4.0**, with attribution to Epoch AI; benchmark questions and answers belong to their creators and data from external projects keeps its original licence. Download: `benchmark_data.zip` (CSV, updated 2026-10-09) at https://epoch.ai/data/benchmark_data.zip, or the `epochai` Python client ([use this data](https://epoch.ai/benchmarks/use-this-data); read in full). The ECI documentation pages themselves state no licence, so the licence comes from this separate page.
- The ECI input lists **61 benchmarks**: 14 run by Epoch, 33 run by the benchmark's creator, 14 reported by model developers (mostly older ones such as MMLU, GSM8K, HellaSwag, WinoGrande) ([data page](https://epoch.ai/data/eci-documentation/data); read in full). Scores must lie in [0, 1], so Elo-style benchmarks are excluded, and a model needs at least 4 benchmark evaluations ([FAQ](https://epoch.ai/data/eci-documentation/faq); read in full).
- Consequence: the first data tier can be a reproducible, licensed, single-source snapshot. Rating benchmarks that are outside it (Arena-style, many agent benchmarks) needs another source.
- Messier (agent benchmarks, see 7.1) is a second data tier with task-level results, released with code and data at github.com/Andromede-AI/messier.

### 6.2 What the measurement-model papers actually tested

- ECI: no benchmark is held out as a final test. Validation is 10-fold cross-validation (sigmoid and clipped-linear fit alike), a 60/40 model split for the downstream time-horizon map (R² 0.85), and 100 random benchmark drops (median slope 0.54 against 0.55). Scores for developer-reported benchmarks may be cherry-picked and Epoch's FAQ says so; its own permutation test for "optimised-for" benchmarks found a 0.12 gap but p between 0.1 and 0.37, so no significant evidence, which with so few benchmarks is also weak evidence of absence. Where a model has several setups, the maximum is taken ([paper](https://arxiv.org/pdf/2512.00193v1), summary of full text).
- Observational scaling laws: PCA over eight benchmarks and 77 base models from 21 families, three components (about 97% of variance, the first about 80%), with scale-based and capability-based held-out splits, preprocessing fitted on training data only, and a **preregistered prediction** on about 20 later models. Stated limitations include that contamination is not accounted for and within-family heterogeneity is not modelled ([paper](https://arxiv.org/pdf/2405.10938), summary of full text). This is the best template for our own protocol: split by scale or date, fit preprocessing on training data only, preregister.
- Kearns: structured capabilities model fitted on 4,395 models from the Open LLM Leaderboard with leave-one-subtask-out tests. Out-of-fold MSE gains over observational scaling laws were **not statistically significant** (p = 0.776 for training, p = 0.637 for test), fit uses 19 BBH subtasks from one benchmark family, and the author warns the model population may not support standard inference ([thesis](https://arxiv.org/pdf/2602.15532), summary of the excerpt through section 5.3). So option D is a reasonable structural idea, not an evidenced winner.

### 6.3 Audit tier

- BetterBench scores each criterion 0, 5, 10 or 15 (5 for mentioning without fulfilling, 15 for fully meeting), averages with equal weights within four stages (design 14 criteria, implementation 11, documentation 19, maintenance 3, though the paper says 46 total and the figures list 47). Design and usability scores correlate (Pearson 0.73 across foundation-model benchmarks, p = 0.001). **It does not test whether quality scores correlate with citations, popularity or model performance**, and MMLU scored lowest (5.5) against GPQA (11.0) ([paper](https://arxiv.org/pdf/2411.12990), summary of full text).
- Consequence: our planned question "does an audit score predict statistical reliability or out-of-sample validity?" is untested in the paper that defines the audit, which makes it a legitimate contribution rather than a duplicate.
- Equal-weight averaging of criteria is itself a choice we should test (weights by contribution to out-of-sample validity) instead of copying.

### 6.4 Rank and score fragility

- Dropping 0.003% of human preferences changed the top-ranked model on Chatbot Arena, while MT-bench rankings were notably more robust ([abstract](https://arxiv.org/abs/2508.11847); abstract only). Rating by threshold (letter grades) inherits this fragility.
- Implication: every grade should come with an influence check (which models or cells drive it) and a probability of being above a cutoff, not just a point estimate. Leave-one-model-out and leave-one-benchmark-out refits are a cheap version.
- A framework for ranking chatbots from pairwise comparisons models ties and covariance between competitors and adds identifiability constraints ([abstract](https://arxiv.org/abs/2412.18407)); not directly applicable to aggregate scores but relevant if an Elo-style benchmark tier is ever added.

### 6.5 What the polling side says, after a second look

- The Economist's election model ([Heidemanns, Gelman and Morris 2020](https://hdsr.mitpress.mit.edu/pub/nw1dzd02), summary) does **not** use pre-computed pollster grades. It estimates house effects jointly from the current cycle's polls with priors centred at zero, adds population and mode effects, partisan nonresponse, and extra poll-level error beyond the stated margins, and reports that hierarchical error terms guarded against the overconfidence of simple averaging in 2016. That is a direct alternative to "rate then weight": **estimate offsets and extra noise jointly with the quantity of interest**. It maps onto option B in the decision table.
- Linzer (2013) borrows strength across states and time with random-walk priors; the abstract says nothing about house effects ([abstract](https://academicnewsletter.sufe.edu.cn/info/444968)).
- A model of bias and excess variance in polls reports that earlier estimates of presidential bias were about 10% too extreme and Senate bias about 25% too extreme, and that estimates of bias are sensitive to the time window chosen ([abstract](https://arxiv.org/abs/2206.14570); abstract only). Our analogue: benchmark "bias" and "difficulty" estimates may depend on which model-date window we use, so report them for two cut-offs.
- I still found no independent quantitative test that pollster grades predict future accuracy. The commentary I could read argues for averages over individual ratings and offers only anecdotes ([example](https://thedemocraticstrategist.org/2018/05/new-pollster-ratings-are-illuminating-but-better-rely-on-averages/)). Pollster ratings should remain an unvalidated analogy.

### 6.6 Updated recommendation

1. Keep the "grade candidate methods out of sample" plan and make **option B, the joint hierarchical model** the lead candidate, because the strongest polling analogue (Economist/Linzer family) estimates offsets and excess error jointly rather than grading first, and because ECI's weakness is its error model (bootstrap over model results, no explicit overdispersion).
2. Use ECI (MIT, CC BY data) as the baseline everyone must beat, not as a competitor to ignore.
3. Adopt the observational-scaling-laws validation recipe: split by scale or date, fit preprocessing on training only, preregister predictions, and add a later "fresh models" test.
4. Treat letter grades as presentation only; the primary outputs are intervals and exceedance probabilities with an influence check.
5. Start with the Epoch snapshot (licensed, 61 benchmarks, mostly consistent harness for the Epoch-run subset) as tier 1; record which cells are Epoch-run versus developer-reported and report results with and without the developer-reported cells.
6. Keep the audit tier separate and test whether it predicts the statistical tier before ever merging them.

## 7. Third pass additions (2026-10-10)

### 7.1 Messier is item-level, public and CC BY

- 30 agent benchmarks (programming, research and reasoning, enterprise workflows, GUI navigation, function calling), 745 agents, 957,611 records; code and data at github.com/Andromede-AI/messier; arXiv page lists CC BY 4.0 while the release "preserves upstream license and access conditions" and "does not relicense third-party benchmark materials", with detail in an appendix not read ([paper](https://arxiv.org/html/2607.25891v2); extract of the first 100,000 characters).
- Capability scores come from a **Rasch (1PL) item response model at task level**: success probability is 1 / (1 + exp(beta_task - theta_model)). A 2PL fit gave nearly identical rankings (Spearman 0.99 against 1PL). For each model-benchmark pair only the scaffold with the highest success fraction is kept, which is the same max-over-setups choice ECI makes.
- Against Epoch's ECI: Spearman 0.84 across 74 matched models (programming 0.85 on 56, mathematics 0.84 on 63), with a 1,000-resample bootstrap interval.
- Consequence: for agent benchmarks the item-level audit tier (7.5) is feasible, which the Epoch aggregates cannot support. Limitations the authors state: technical and English-heavy coverage, inherited upstream validity, and difficulty-prediction simplifications.

### 7.2 The Shirani-Mehr et al. paper, read in full

- Model: poll support y ~ Normal(p, sigma^2) with logit(p) = logit(final vote share) + alpha + beta * time, and sigma^2 = binomial variance + tau^2 (excess variance), with hierarchical priors across elections, fitted in Stan ([paper](https://5harad.com/papers/polling-errors.pdf)).
- Data: 4,221 polls over 608 state-level elections (1998 to 2014), final three weeks. Total RMSE about 3.5 points, roughly twice what reported margins imply; simple-random-sampling 95% intervals covered the outcome only 73% (Senate), 74% (governor) and 88% (president) of the time. Average absolute election-level bias is about 2 points (Senate 2.1, governor 2.3, president 1.2).
- **Pollster house effects are not estimated**; the authors say they are absorbed into the variance term, and bias and variance cannot be perfectly separated. So this paper supports "reported error bars understate total error" strongly, but it does not support pollster ratings.
- Direct transfer: a coverage check of our benchmarks' own binomial intervals against out-of-sample outcomes, and a per-benchmark excess-variance term tau^2 as a reliability measure.

### 7.3 How pooled-poll models identify house effects

- In the Jackman-style Stan implementation I could read, the latent series is pinned to **known election results at both ends** and each firm gets a zero-centred house effect (prior Normal(0, 7.5)); the author reports results close to Jackman's own, such as a 2.7-point house effect for one face-to-face pollster ([re-implementation](https://www.r-bloggers.com/2017/06/state-space-modelling-of-the-australian-2007-federal-election/); the identification reading is inferred from the code, the post does not state it).
- The Economist model instead estimates house effects within the current cycle with zero-centred priors (6.5).
- **The anchor is what makes offsets mean something.** Without an anchor, only offsets relative to the field are identified. ECI anchors two models (Claude 3.5 Sonnet at 130, GPT-5 at 150) to fix scale, but that fixes scale, not truth. For us, a benchmark's "bias" is relative to the consensus unless we bring an external outcome. This is the strongest reason to make the rating target (decision 2) an out-of-sample outcome and to say plainly that the consensus-relative offsets are not accuracy.

### 7.4 Scoring forecasts: proper scoring rules

- A rule is proper if the expected score is best when the forecaster reports the true distribution; strictly proper if that optimum is unique. The scoringRules package implements the log score and CRPS (plus multivariate energy and variogram scores); the log score is minimised at the forecast mode, CRPS at the median, the log score rises quickly in the tails and kernel-based versions are fragile with simulated samples, while CRPS is more symmetric ([paper](https://arxiv.org/pdf/1709.04743), read in full). It does not cover significance testing of score differences.
- Recommendation: for predictive-validity grades on held-out benchmark scores (continuous, bounded in [0, 1]) report CRPS as the primary score and the log score as a secondary check, using posterior predictive draws; compare methods by mean score over held-out cells with a paired bootstrap over models, not a point test from memory.

### 7.5 Reliability and noise at the item level

- A reliable-change framework on 2,000 MMLU-Pro items with 10 stochastic generations per item found high split-half reliability (r_xx 0.966 to 0.996); K = 3 samples gives about 0.90. But 52% to 67% of items were always right or always wrong, and the aggregate accuracy gain was "the net residual of opposing item-level movements" (34% of items improved and 28% deteriorated for one pair). Single-shot greedy evaluation missed 42% of reliably changed items. Limits: two 7-8B model pairs, one benchmark, homoscedastic SEM (the authors suggest a beta-binomial or hierarchical Bayesian model) ([paper](https://arxiv.org/html/2604.27405), summary of full text).
- Implication for ratings: a published benchmark score is usually a single run, so run-to-run noise is folded into our excess-variance term and cannot be separated from construct error unless a source reports repeats. Where item-level or repeated runs exist, report the share of floor and ceiling items per benchmark (a direct, cheap measure of how much of a benchmark still discriminates) and a beta-binomial reliability, rather than a Gaussian SEM.

### 7.6 Revised pre-fit checklist (replaces section 4 where it differs)

1. Candidates: ECI baseline; hierarchical Bayesian sigmoid with binomial plus per-benchmark excess variance (lead candidate); PCA first factor; scale-aware structured model; mean-rank baseline.
2. Outcome and score: held-out benchmark scores, graded by CRPS with a paired bootstrap over models, with a coverage check of the stated intervals.
3. Splits: by release date and by scale (observational-scaling-laws recipe), preprocessing fitted on training data only, preregistered predictions on later models.
4. Identification: state the anchor for every offset we report; call consensus-relative offsets "relative offsets", never "bias" or "accuracy".
5. Data: Epoch snapshot (CC BY 4.0) as tier 1 with Epoch-run cells flagged; Messier as tier 2 for agent benchmarks, with task-level data enabling Rasch/2PL audits and floor/ceiling shares.
6. Robustness: influence checks (leave-one-model-out, leave-one-benchmark-out), two cut-off dates, results with and without developer-reported cells.
7. Kill criteria: if no candidate beats mean-rank out of sample, or coverage of stated intervals is poor, or ratings flip between cut-offs, publish that as the result.

## 8. Fourth pass additions (2026-10-10)

Pass 4 targeted the prior art I had not yet looked for: existing ways of judging benchmarks themselves, error-bar practice for single evals, and rank-aggregation robustness.
It materially narrows what is new.

### 8.1 Epoch already audits benchmarks, with a published rubric

- Epoch's Benchmark Reviews rate external benchmarks Verified, Flawed, Not enough information, or By Epoch (its own benchmarks are not reviewed) ([overview](https://epoch.ai/data/benchmark-reviews-documentation), [methodology](https://epoch.ai/data/benchmark-reviews-documentation/methodology); both read in full).
- Rubric v1: **Reviewability** (Full, Partial, Inadequate; Inadequate ends the review as Not enough information); **Scoring** minimum standards where failing any is Flawed (task errors with a default threshold of at least 20% of the inspected sample or corruption at scale; version inconsistency; elicitation limits; evaluation-setup bias); and non-disqualifying **evaluation quality** items (contamination risk, scaffold fairness, human completability, floor and ceiling, statistical adequacy with at least 5 runs per model recommended, construct validity). Sampling: 50 tasks (100 if the observed error rate is 15 to 25%) or the full set if 50 or fewer.
- The included-benchmarks page lists 16 reviews with dates from 2026-08-10 to 2026-09-29: Verified (5) GraphWalks, ExploitBench v0.1, SimpleQA Verified, PostTrainBench v1.1, WeirdML v2; Flawed (8) BFCL v4, HealthBench Professional, DeepSWE v1.1, Terminal-Bench 4.0.0, SWE-bench Verified, SWE-Bench Pro, Humanity's Last Exam, Lech Mazur Writing, TextQuests; Not enough information (2) CritPt, FrontierCode ([list](https://epoch.ai/data/benchmark-reviews-documentation/included-benchmarks)). The page's own count says 15 elsewhere in press coverage; the list itself shows 16 rows, so use the list.
- A review can rest on external audits: the SWE-bench Verified review cites OpenAI's audit (27.6% of tasks audited, 59.4% of those with flawed tests, a floor of 16.4% broken tasks overall) and a reward hack, and Epoch says it stopped inspecting once the Flawed threshold was crossed ([review](https://epoch.ai/benchmarks/swe-bench-verified/review)).
- Consequence: **the audit tier should not be invented by us**. Epoch's verdicts are a public, rubric-based, dated label. I found no licence statement for the verdicts and no download; they would be transcribed by hand with attribution, and that needs a check with Epoch or at least a conservative citation-only use.
- Consequence for novelty: the question "does an audit verdict predict statistical behaviour?" becomes a **join of two Epoch products** (the ECI fit and the review verdicts) that, as far as I found, nobody has published. Overlap is small: several reviewed benchmarks (SWE-bench Verified, SimpleQA Verified, PostTrainBench, Terminal-Bench, DeepSWE, HLE, WeirdML, Lech Mazur Writing) also appear in the ECI input list, so the usable sample is roughly ten benchmarks. That is too few for a confirmatory test and fine for a descriptive one; we must say so.

### 8.2 Benchmark^2: statistical quality metrics for benchmarks, with limits

- Three metrics ([abstract](https://arxiv.org/abs/2601.03986), [full HTML](https://arxiv.org/html/2601.03986v1), summary of full text): **cross-benchmark ranking consistency** (mean Kendall tau against other benchmarks in the same domain; above 0.7 high, below 0.4 low), **discriminability** (coefficient of variation times the square root of the share of model pairs more than 0.02 apart), and **capability alignment deviation** (an exponential-decay transform, lambda = 12, of the rate at which a bigger model in a family fails an item a smaller sibling passes), plus a bootstrap stability score and a weighted quality score.
- Scale and limits: 15 benchmarks, 11 instruction-tuned models from 4 families, greedy decoding, held-out check on 3 base models; constants (lambda, the 0.02 gap) chosen empirically; the paper itself raises circularity of the consistency metric; the weights of the composite are in an appendix I did not read.
- Consequence: Benchmark^2 is the closest published recipe for a statistical benchmark rating. It supports our "consistency, discriminability, stability" columns and gives a natural baseline, but it uses small samples, hand-set constants and consistency against peers (the circularity problem). Our contribution against it is out-of-sample outcome-based grading, explicit uncertainty and the audit join.

### 8.3 Benchmark agreement testing is unstable unless done carefully

- Benchmark Agreement Testing practice (comparing a benchmark's model ranking to reference benchmarks) is sensitive to the reference choice (Alpaca v2 agreement ranged from about 0.57 against MT-Bench to 0.82 against Arena), to which models are included (Kendall 0.65 to 0.99 for Arena against MT-Bench depending on the model set), to granularity (lower agreement among closely ranked models), and to metric (Pearson about 0.2 above Kendall) ([paper](https://arxiv.org/html/2407.13696v2); summary of full text).
- Their remedies, which I will adopt: an **aggregate reference** (average over appropriate references; cut variance by more than 30%), a **data-driven threshold** (Z-score of the target's agreement against other benchmarks' agreement with the same reference; above -1 sigma counts as agreement), **at least 10 randomly sampled models**, and **multiple granularities** (5, 10, 20 contiguous models). Together they cut the standard deviation of results from 0.31 to 0.10.
- They caution that high agreement does not mean the same construct and low agreement can mean unreliability rather than invalidity.
- Consequence: any consistency number we publish must come with the model-set sensitivity and be computed against an aggregate reference that excludes the benchmark itself (leave-one-out reference), and the circularity warning from Benchmark^2 applies to the leave-one-out reference as well.

### 8.4 Error-bar practice we should reuse for item-level data

- Miller's recommendations: CLT standard errors instead of bootstrap, clustered standard errors when questions are grouped, resampling answers to cut variance, **paired question-level differences** for model comparisons, and power analysis; model scores are positively correlated across questions, so pairing is a "free" variance reduction (example correlations 0.50 MATH, 0.64 HumanEval, 0.37 MGSM; a correlation of 0.5 cuts the estimator variance by about a third) ([paper](https://arxiv.org/pdf/2411.00640); read in full).
- A random-effects extension handles K models at once (models fixed, questions random) and shows that some pairwise claims that look significant under naive comparison do not survive pairing and multiple-comparison correction ([abstract](https://arxiv.org/abs/2608.22659); abstract).
- Consequence: where we have item-level data (Messier tier), use paired, clustered intervals. Where we have only aggregates, state that the binomial interval is a lower bound on error (7.2), which is what the excess-variance term estimates.

### 8.5 Aggregation rule matters for gaming resistance

- Treating datasets as voters and models as candidates, mean win rate is the hardest rule to manipulate by training on benchmarks (on BBH with 4,507 models, the median number of tasks a developer must include is 22 of 24 for mean win rate against 13 for the arithmetic mean and 12 for median and pairwise majority) ([abstract](https://arxiv.org/abs/2605.23628); abstract). Vote'n'Rank gives the social-choice framing ([paper](https://arxiv.org/pdf/2210.05769); listing only).
- Consequence for the later grade-A meta-benchmark: choose the aggregation rule on robustness grounds and report at least two (latent score and mean win rate) so readers can see whether the ordering depends on the rule.

### 8.6 Current plan, after four passes

What exists: Epoch ECI (model scores plus per-benchmark difficulty and slope, MIT code, CC BY data), Epoch Benchmark Reviews (rubric verdicts), Benchmark^2 (statistical quality metrics, small), BetterBench (checklist), Messier (item-level agent data, Rasch), BAT/BenchBench (agreement testing guidance).

What I could not find, and will therefore build and test:

1. **Outcome-graded benchmark ratings.** Each benchmark's predictability is scored out of sample (leave-one-benchmark-out, date-ordered and scale-ordered splits) with a proper score (CRPS), and its excess variance and exceedance probabilities are reported with intervals.
2. **A calibrated error model.** A hierarchical model with a per-benchmark excess-variance term, checked by interval coverage on held-out cells, and compared with the ECI least-squares baseline and the mean-rank baseline.
3. **The audit join.** Epoch's Verified/Flawed verdicts (and Messier-derived item statistics where available) versus the statistical measures from steps 1 and 2, reported descriptively with the small-n caveat.
4. **A stability and fragility analysis.** Two cut-off dates, model-set subsampling (BAT-style), and leave-one-out influence.
5. **Only if 1 to 4 hold:** a versioned composite ("grade A" meta-benchmark) with mean win rate and latent-score versions side by side.

Kill criteria are in 7.6 and unchanged.

### 8.7 Practical blocker found in pass 4

The shell in this environment cannot reach `epoch.ai` (egress policy), so the Epoch data zip cannot be downloaded here. Fetching the CSV through the web-fetch tool returns a model-summarised, truncated transcription, which I will not treat as data. The loader and all models are therefore built and tested on synthetic data with known truth (simulation recovery) while waiting for the real file, and the first real fit will be run only on the original CSV. The CSV schema, as shown by the fetch tool for the first rows, is: `model_id, benchmark_id, performance, benchmark, benchmark_release_date, model, model_version, Model, date, source`.
