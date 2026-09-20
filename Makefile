.PHONY: setup metadata check

setup:
	uv sync --locked --project adventurebench

metadata:
	uv run --python 3.12 --locked --script .plicara/check.py

check: metadata
	uv lock --check --project adventurebench
	$(MAKE) -C adventurebench check
