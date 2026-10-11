.PHONY: setup metadata check

setup:
	uv sync --locked --project adventurebench
	uv sync --locked --project pollbench

metadata:
	uv run --python 3.12 --locked --script .plicara/check.py

check: metadata
	uv lock --check --project adventurebench
	$(MAKE) -C adventurebench check
	uv lock --check --project pollbench
	$(MAKE) -C pollbench check
