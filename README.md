# benchmarks

Reproducible benchmark definitions, methodology, and audited results from
Plicara Labs. Each benchmark is self-contained so its published scores can be
replayed from the committed evidence without calling a model provider.

## Benchmarks

| Benchmark | What it measures | Latest release |
|---|---|---|
| [`adventurebench/`](adventurebench/) | Grounded action interpretation and calibrated refusal in text-adventure scenes | 12 models, 244 cases, 3 repetitions |

The rendered results are published at
[`plicara.ai/benchmarks`](https://plicara.ai/benchmarks/). The files in this
repository are the corresponding machine-checkable record.

## Working in this repository

Project metadata and research context live in [.plicara/README.md](.plicara/README.md); agent constraints live in [AGENTS.md](AGENTS.md). Use `make setup` and `make check` for the default local environment and verification. Expensive experiments, model downloads, and publication are separate explicit steps. Project status is authoritative in `.plicara/project.yaml`; no central board update is required.
