# Makefile — public-external review -> B0 build pipeline entry points.
#
# Implementation lives in tools/vision/fc_bga_yolo/review-loop.sh (works in git-bash
# with no extra installs). These targets delegate to it so there is a single source
# of truth. Override the interpreter with: make status PYTHON=/path/to/python
#
# Quick start:
#   make review      # refresh artifacts + show progress
#   make apply LABEL_DIR=~/exports   # apply exported YOLO labels into candidates.jsonl
#   make b0-check    # B0 gate checklist (exit 1 while blocked — expected)
#   make b0-publish  # materialize versions/public-external-v0.1/ when ready
#   make test        # run the toolkit test suite (override PYTHON if needed)

PYTHON ?= python
LOOP := bash tools/vision/fc_bga_yolo/review-loop.sh

.PHONY: review-artifacts review-progress review-status apply b0-check b0-publish review test

review-artifacts:
	$(LOOP) artifacts

review-progress:
	$(LOOP) progress

review-status:
	$(LOOP) status

## Apply annotation-tool export into the manifest. Pass the export dir:
##   make apply LABEL_DIR=~/exports
##   make apply LABEL_DIR=~/exports CLASS_MAP=~/map.json DRY=1
apply:
	$(LOOP) apply --label-dir $(LABEL_DIR) $(if $(CLASS_MAP),--class-map $(CLASS_MAP)) $(if $(DRY),--dry-run)

b0-check:
	$(LOOP) b0-check

b0-publish:
	$(LOOP) b0-publish

## Full refresh + check loop (does NOT publish):
review: review-artifacts review-status
	@echo "==> artifacts refreshed, progress shown; run 'make b0-check' for the gate checklist"

## Run the toolkit test suite. Uses the Makefile PYTHON (override if you use the
## managed interpreter, e.g. make test PYTHON=/path/to/python):
test:
	$(PYTHON) -m pytest tools/vision/fc_bga_yolo/tests -p no:cacheprovider -q
