# pollbench

A rating for AI benchmarks, borrowed from the way pollsters are rated.

Every benchmark gets a **reliability** between 0 and 1: the share of the differences between AI models' scores that reflects real differences in capability rather than noise. It comes with a 90% credible interval and a letter grade from A to D. The first release rates the 60 benchmarks in Epoch AI's Capabilities Index table, using 2,898 scores from 274 AI models.

The rendered results are at [plicara.ai/benchmarks/pollbench](https://plicara.ai/benchmarks/pollbench/). This folder is the machine-checkable record: the code, the exact input table, every number, and the tests the method had to pass.

Status: first release, exploratory. The one-capability assumption behind the rating model is rejected by a check built for that purpose (see [Limitations](#limitations)).

## How it works

A single Bayesian rating model is fit to all scores at once. Each AI model has one capability; each benchmark turns capability into an expected score along an S-shaped curve with its own difficulty and slope, the same form as Epoch's Capabilities Index; real scores scatter around that curve with a benchmark-specific precision (a Beta likelihood).

A benchmark's reliability is the variance of expected scores across the AI models scored on it, divided by that variance plus the average noise variance. It is computed on 1,000 posterior draws, and their 5th and 95th percentiles give the interval. Grades need 90% posterior probability of clearing a threshold: A at 0.85, B at 0.70, C at 0.50, D otherwise. Benchmarks with fewer than 10 scored models are not graded.

The method, its tests and the reporting conditions were written down in [docs/protocol.md](docs/protocol.md) before the first fit. Every later change is a dated amendment saying what had been seen when it was made.

## Where things are

| Path | What it is |
| --- | --- |
| [docs/protocol.md](docs/protocol.md) | The pre-registration: data, methods, evaluation, grades, reporting conditions and five dated amendments |
| [docs/results-epoch-2026-10-10.md](docs/results-epoch-2026-10-10.md) | The results in full, generated from the results file |
| [docs/simulation-study.md](docs/simulation-study.md) | The method tested on tables with known reliabilities, including where it fails |
| [docs/methods-review.md](docs/methods-review.md) | The research review: prior art, design decisions and data licence |
| [results/epoch-2026-10-10.json](results/epoch-2026-10-10.json) | Every number: ratings, intervals, grades, held-out tests, diagnostics |
| [results/epoch-2026-10-10-curves.json](results/epoch-2026-10-10-curves.json) | Each AI model's fitted capability and each benchmark's curve, for the page's figures |
| [results/second-factor-calibration.json](results/second-factor-calibration.json) | Error rate and power of the second-factor check, from simulated tables |
| [results/simulation-study.json](results/simulation-study.json) | Raw output of the simulation study |
| [inputs/epoch-2026-10-10/](inputs/epoch-2026-10-10/) | The exact Epoch AI table used, with its hashes and licence |
| [pollbench/](pollbench/) | The package; [tests/](tests/) holds its unit tests |

## Reproducing the results

Requires [uv](https://docs.astral.sh/uv/). Everything runs offline; no model provider is called.

```
make check     # unit tests, a few minutes on 4 cores
make results   # refit and rerun every held-out test on the committed table
make curves    # refit the primary model and export the page's figure data; refuses to write if it does not reproduce the committed reliabilities
make report    # regenerate docs/results-epoch-2026-10-10.md
make page      # render the results page to site/benchmarks/pollbench/index.html
make simstudy  # the simulation study, slow
```

`make results` refits the rating model for every held-out fold and robustness check, so it is slow; `make curves` takes about 12 minutes on 4 cores. The results file records the input table's digest, and `make curves` refuses to run on a table that does not match it.

## Limitations

- **Capability is not exactly one number.** After the fit, leftover errors on pairs of benchmarks move together about twice as much as the rating model predicts (0.089 against 0.046 in simulated tables; p = 0.002). Reliability here is reliability for measuring general capability, and a benchmark that tracks a narrower skill can read as noisier than it is.
- **Reliability depends on which AI models were scored.** A benchmark run only on frontier models sees a narrow band of capability and reads low. The results file also gives each benchmark's reliability as if every model had been scored.
- **One source.** All scores come from one table. Epoch runs some benchmarks itself and collects the rest from benchmark creators and model developers, whose reported scores may be selected.

## Data and licence

The input is Epoch AI's benchmarking data, used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [inputs/epoch-2026-10-10/](inputs/epoch-2026-10-10/) for attribution and the terms of externally sourced scores.

The code is licensed under Apache-2.0 ([LICENSE](LICENSE)). The results and documentation are licensed under CC BY 4.0. To cite this work, see [CITATION.cff](CITATION.cff).
