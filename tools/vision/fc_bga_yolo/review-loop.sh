#!/usr/bin/env bash
# review-loop.sh — single entry point for the public-external review -> B0 build pipeline.
#
# Source of truth for the chained commands; the repo-root Makefile delegates here.
# Runs from the repo root (auto-detected from this script's location).
#
# Subcommands:
#   artifacts   regenerate git-ignored contact_sheet.html + candidates.enriched.json
#   progress    print the review tally + B0 gate (text)
#   status      same as progress but machine-readable JSON
#   apply       apply exported YOLO label files into candidates.jsonl
#               (args: --label-dir <dir> [--class-map map.json] [--dry-run])
#   quarantine  mark unreadable / defect-unclear candidates as quarantined
#               (args: --ids <sample_id>... [--reason DEFECT_UNCLEAR] [--dry-run])
#   b0-check    dry-run the B0 gate checklist (exit 1 while blocked — expected)
#   b0-publish  materialize versions/public-external-v0.1/ once the gate is ready
#   all         artifacts -> status -> b0-check  (the full refresh + check loop)
#
# Usage:
#   bash tools/vision/fc_bga_yolo/review-loop.sh all
#   PYTHON=/path/to/python bash tools/vision/fc_bga_yolo/review-loop.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
TOOL="tools/vision/fc_bga_yolo"

cmd="${1:-help}"
shift || true

run_artifacts() { "$PYTHON" "$TOOL/build_review_artifacts.py" "$@"; }
run_progress()  { "$PYTHON" "$TOOL/review_progress.py" "$@"; }
run_apply()     { "$PYTHON" "$TOOL/apply_review_labels.py" "$@"; }
run_quarantine() { "$PYTHON" "$TOOL/quarantine_candidates.py" "$@"; }
run_b0()        { "$PYTHON" "$TOOL/build_b0_version.py" "$@"; }

case "$cmd" in
  artifacts)
    run_artifacts
    ;;
  progress)
    run_progress
    ;;
  status)
    run_progress --json
    ;;
  apply)
    run_apply "$@"
    ;;
  quarantine)
    run_quarantine "$@"
    ;;
  b0-check)
    run_b0
    ;;
  b0-publish)
    run_b0 --publish
    ;;
  all)
    echo "## 1/3 regenerate review artifacts"
    run_artifacts
    echo
    echo "## 2/3 review progress + B0 gate (json)"
    run_progress --json
    echo
    echo "## 3/3 B0 build check"
    run_b0 || true   # blocked is the expected pre-annotation state, not a chain failure
    ;;
  help|--help|-h)
    sed -n '2,18p' "${BASH_SOURCE[0]}"
    ;;
  *)
    echo "unknown subcommand: $cmd" >&2
    echo "try: bash $0 help" >&2
    exit 2
    ;;
esac
