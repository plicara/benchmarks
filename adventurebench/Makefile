PYTHON ?= python3

.PHONY: check test validate audit rescore-check aggregate-check render-site site-check

check: validate audit test

test:
	$(PYTHON) -m unittest discover -v

validate:
	$(PYTHON) -m adventure_bench.validate

# Repository-wide artifact replay.  This is always read-only and never calls
# a model provider; an empty pre-release repository reports zero audit items.
audit:
	$(PYTHON) -m adventure_bench.audit

# Set RUN_ID when audited evidence is present. This remains offline and never
# starts collection or spends against a provider.
rescore-check:
	@test -n "$(RUN_ID)" || (echo "set RUN_ID=<run-id>"; exit 2)
	$(PYTHON) -m adventure_bench.rescore --run-id "$(RUN_ID)" --check

# Set RELEASE_ID and ordered RUN_IDS="run-a run-b ..." after rescoring every
# input run.  The command is offline and only checks for drift.
aggregate-check:
	@test -n "$(RELEASE_ID)" || (echo "set RELEASE_ID=<release-id>"; exit 2)
	@test -n "$(RUN_IDS)" || (echo "set RUN_IDS='run-a run-b ...'"; exit 2)
	$(PYTHON) -m adventure_bench.aggregate --release-id "$(RELEASE_ID)" $(foreach run,$(RUN_IDS),--run-id "$(run)") --check

# Rendering first replays every explicit release member and requires the
# canonical release artifact before it will write the standalone page.
render-site:
	@test -n "$(RELEASE_ID)" || (echo "set RELEASE_ID=<release-id>"; exit 2)
	$(PYTHON) scripts/render_site.py --release-id "$(RELEASE_ID)"

site-check:
	@test -n "$(RELEASE_ID)" || (echo "set RELEASE_ID=<release-id>"; exit 2)
	$(PYTHON) scripts/render_site.py --release-id "$(RELEASE_ID)" --check
