"""Quarantine candidate records that the human reviewer could not judge.

Companion to `apply_review_labels.py`: that script handles *accepted-with-label*,
this one handles the other branch of the review decision flow
(ANNOTATION_SPEC.md §3):

    image unreadable / defect unclear  ->  review_status = "quarantined"

Why a script instead of a manual JSONL edit: quarantining has state-machine rules
(annotation_status must be cleared, accepted_classes/label_path must be dropped,
quarantine_reason must be non-empty) that the B0 gate's `audit_candidates`
enforces. Hand edits get these wrong; this tool cannot.

Behaviour
---------
* `--ids` selects candidates by `sample_id` (repeatable / multiple values).
* `--reason` defaults to `DEFECT_UNCLEAR`; other values from the spec:
  `UNREADABLE`, `LICENSE_AMBIGUOUS`.
* Records already quarantined with the same reason are left untouched
  (idempotent). Accepted records are demoted: label_path -> null,
  annotation_status -> null, accepted_classes -> ().
* `candidates.jsonl` is backed up to `candidates.jsonl.bak` before any write.
* `--dry-run` prints the plan and writes nothing.
* Unknown `--ids` are reported; the run still applies the known ones, and the
  exit code is 1 if any id was unknown (so scripts can detect typos).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_external_manifest import (  # noqa: E402
    CandidateRecord,
    load_candidate_manifest,
    load_source_registry,
)

KNOWN_REASONS = ("DEFECT_UNCLEAR", "UNREADABLE", "LICENSE_AMBIGUOUS")


def quarantine_records(
    manifest: Path,
    sources: Path,
    ids: tuple[str, ...],
    reason: str,
    *,
    dry_run: bool,
) -> int:
    reason = reason.strip()
    if not reason:
        print("error: --reason must be non-empty", file=sys.stderr)
        return 2

    src_records = load_source_registry(sources)
    records = load_candidate_manifest(manifest, src_records)
    wanted = set(ids)
    present = {record.sample_id for record in records}
    unknown = sorted(wanted - present)

    updated = 0
    already = 0
    new_records: list[CandidateRecord] = []
    for record in records:
        if record.sample_id not in wanted:
            new_records.append(record)
            continue
        if record.review_status == "quarantined" and record.quarantine_reason == reason:
            already += 1
            new_records.append(record)
            continue
        new_records.append(
            CandidateRecord(
                sample_id=record.sample_id,
                source_group_id=record.source_group_id,
                source_id=record.source_id,
                original_filename=record.original_filename,
                image_path=record.image_path,
                image_sha256=record.image_sha256,
                label_path=None,
                review_status="quarantined",
                annotation_status=None,
                accepted_classes=(),
                quarantine_reason=reason,
            )
        )
        updated += 1

    _print_summary(updated, already, unknown, dry_run)

    if dry_run or updated == 0:
        return 1 if unknown else 0

    backup = manifest.with_suffix(manifest.suffix + ".bak")
    shutil.copy2(manifest, backup)
    lines = [json.dumps(asdict(r)) for r in new_records]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {manifest}  (backup -> {backup.name})")
    return 1 if unknown else 0


def _print_summary(updated: int, already: int, unknown: list[str], dry_run: bool) -> None:
    tag = "[dry-run] " if dry_run else ""
    print(f"{tag}quarantined: {updated}")
    print(f"{tag}already quarantined (same reason): {already}")
    if unknown:
        print(f"{tag}unknown sample ids ({len(unknown)}):")
        for sid in unknown:
            print(f"  - {sid}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine candidate records (unreadable / defect-unclear)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "data/external/fc_bga_public_external/review/candidates.jsonl",
        help="path to candidates.jsonl",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
        / "data/external/fc_bga_public_external/sources.json",
        help="path to sources.json (needed to validate source_ids)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="sample_id(s) to quarantine, e.g. --ids public-abcdef public-123456",
    )
    parser.add_argument(
        "--reason",
        default="DEFECT_UNCLEAR",
        help=f"quarantine reason (default DEFECT_UNCLEAR; known: {', '.join(KNOWN_REASONS)})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print planned changes, write nothing"
    )
    args = parser.parse_args()
    return quarantine_records(
        args.manifest,
        args.sources,
        tuple(args.ids),
        args.reason,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
