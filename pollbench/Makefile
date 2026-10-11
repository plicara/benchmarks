PYTHON ?= uv run --locked python
DATA ?= inputs/epoch-2026-10-10/processed_data_for_eci.csv
RESULTS ?= results/epoch-2026-10-10.json

.PHONY: check test simstudy results curves page report

# Offline: unit tests only. Fitting on real data and the simulation study are
# explicit steps (see README), not part of the routine check.
check: test
	uv lock --check

test:
	$(PYTHON) -m unittest discover -v

# Explicit and slow (about 15 minutes on 4 CPUs).
simstudy:
	$(PYTHON) -m pollbench.simstudy --seeds 8 --out results/simulation-study.json

# Explicit and slow: the full protocol, every held-out fold and robustness refit.
results:
	$(PYTHON) -m pollbench run $(DATA) --out $(RESULTS)

# Refits the primary model and refuses to write unless it reproduces $(RESULTS).
curves:
	$(PYTHON) -m pollbench.curves $(DATA) --results $(RESULTS) --out results/epoch-2026-10-10-curves.json

page:
	$(PYTHON) -m pollbench.page $(RESULTS) --curves results/epoch-2026-10-10-curves.json --out site/benchmarks/pollbench/index.html

report:
	$(PYTHON) -m pollbench.report $(RESULTS) --calibration results/second-factor-calibration.json --out docs/results-epoch-2026-10-10.md
