"""Tests for the public-external review -> B0 build pipeline tools.

Covers:
  - apply_review_labels.apply_labels  (dry-run safety, acceptance, class-map,
    invalid-box rejection, accepted records pass the B0 gate audit)
  - review_progress.summarize_review  (blocked vs ready gate accounting)
  - build_review_artifacts            (enrich + render)
  - build_b0_version.main             (end-to-end publish materialization)

The fixtures copy a few *real* reviewed candidate images into a temp tree so the
sha256 / image-existence checks in the manifest audit run for real. No network,
no GPU.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools.vision.fc_bga_yolo.apply_review_labels import apply_labels
from tools.vision.fc_bga_yolo.build_review_artifacts import (
    build_enriched,
    load_candidates,
    render_html,
)
from tools.vision.fc_bga_yolo.quarantine_candidates import quarantine_records
from tools.vision.fc_bga_yolo.review_progress import summarize_review

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_IMAGES = REPO_ROOT / "data/external/fc_bga_public_external/review/images"

VALID_SOURCE = {
    "source_id": "src-test",
    "name": "Test Source",
    "url": "https://example.com/source",
    "version": "1",
    "license_name": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "license_sha256": "0" * 64,
    "retrieved_at": "2026-08-18T00:00:00Z",
    "attribution": "test",
}


def _build_review_tree(tmp_path: Path, n: int, src_id: str = "src-test") -> tuple[Path, Path]:
    """Copy `n` real candidate images into a temp review tree + minimal sources."""
    review = tmp_path / "review"
    (review / "images").mkdir(parents=True)
    real = sorted(REAL_IMAGES.glob("*.jpg"))
    if len(real) < n:
        pytest.skip(f"not enough real review images ({len(real)} < {n})")
    from tools.vision.fc_bga_yolo.model_metadata import sha256_file

    recs: list[dict] = []
    for i, src in enumerate(real[:n]):
        sid = f"sample-{i:03d}"
        dst = review / "images" / f"{sid}.jpg"
        shutil.copy2(src, dst)
        recs.append(
            {
                "sample_id": sid,
                "source_group_id": f"{src_id}:g{i}",
                "source_id": src_id,
                "original_filename": src.name,
                "image_path": f"images/{sid}.jpg",
                "image_sha256": sha256_file(dst),
                "label_path": None,
                "review_status": "review_required",
                "annotation_status": None,
                "accepted_classes": [],
                "quarantine_reason": None,
            }
        )
    (review / "candidates.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8"
    )
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([VALID_SOURCE]), encoding="utf-8")
    return review, sources


def test_summarize_blocked_with_zero_accepted(tmp_path: Path) -> None:
    review, sources = _build_review_tree(tmp_path, 5)
    p = summarize_review(review / "candidates.jsonl", sources)
    assert p.total == 5
    assert p.accepted == 0
    assert p.b0_status == "blocked_data"
    assert p.images_to_b0 == 20
    assert p.classes_to_b0 == 2


def test_apply_labels_dry_run_has_no_side_effects(tmp_path: Path) -> None:
    review, sources = _build_review_tree(tmp_path, 5)
    export = tmp_path / "export"
    export.mkdir()
    (export / "sample-000.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")

    before = (review / "candidates.jsonl").read_bytes()
    rc = apply_labels(review / "candidates.jsonl", sources, export, None, dry_run=True)
    after = (review / "candidates.jsonl").read_bytes()

    assert before == after  # manifest untouched
    # dry-run must not write any label file or backup (an empty labels/ dir is
    # created by mkdir but holds nothing, and is git-ignored anyway)
    assert not any((review / "labels").glob("*.txt"))
    assert not (review / "candidates.jsonl.bak").exists()  # no backup
    assert rc == 0


def test_apply_labels_accepts_and_flips_b0_ready(tmp_path: Path) -> None:
    review, sources = _build_review_tree(tmp_path, 20)
    export = tmp_path / "export"
    export.mkdir()
    for i in range(20):
        cls = i % 2  # alternate class 0 / 1 -> >= 2 represented classes
        (export / f"sample-{i:03d}.txt").write_text(
            f"{cls} 0.5 0.5 0.9 0.9\n", encoding="utf-8"
        )
    rc = apply_labels(review / "candidates.jsonl", sources, export, None, dry_run=False)
    assert rc == 0

    p = summarize_review(review / "candidates.jsonl", sources)
    assert p.accepted == 20
    assert p.represented_class_count >= 2
    assert p.b0_status == "ready"  # gate flips once 20 accepted / >=2 classes
    assert (review / "labels" / "sample-000.txt").exists()
    assert (review / "candidates.jsonl.bak").exists()


def test_apply_labels_rejects_invalid_box(tmp_path: Path) -> None:
    review, sources = _build_review_tree(tmp_path, 3)
    export = tmp_path / "export"
    export.mkdir()
    # cx = 1.5 is outside [0, 1] -> must be rejected, record stays review_required
    (export / "sample-000.txt").write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
    rc = apply_labels(review / "candidates.jsonl", sources, export, None, dry_run=False)

    p = summarize_review(review / "candidates.jsonl", sources)
    assert p.accepted == 0
    assert rc == 1  # one failure reported


def test_apply_labels_class_map_translation(tmp_path: Path) -> None:
    review, sources = _build_review_tree(tmp_path, 2)
    export = tmp_path / "export"
    export.mkdir()
    cmap = tmp_path / "map.json"
    # tool used reversed order: index 1 means BALL_BRIDGE
    cmap.write_text(
        json.dumps({"0": "MISSING_BALL", "1": "BALL_BRIDGE"}), encoding="utf-8"
    )
    (export / "sample-000.txt").write_text("1 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    apply_labels(review / "candidates.jsonl", sources, export, cmap, dry_run=False)

    recs = [
        json.loads(l)
        for l in (review / "candidates.jsonl").open(encoding="utf-8")
        if l.strip()
    ]
    rec = next(r for r in recs if r["sample_id"] == "sample-000")
    assert rec["accepted_classes"] == ["BALL_BRIDGE"]


def test_quarantine_dry_run_has_no_side_effects(tmp_path: Path) -> None:
    review, sources = _build_review_tree(tmp_path, 3)
    before = (review / "candidates.jsonl").read_bytes()

    rc = quarantine_records(
        review / "candidates.jsonl", sources, ("sample-000",), "DEFECT_UNCLEAR",
        dry_run=True,
    )

    assert (review / "candidates.jsonl").read_bytes() == before  # manifest untouched
    assert not (review / "candidates.jsonl.bak").exists()  # no backup
    assert rc == 0


def test_quarantine_demotes_and_passes_audit(tmp_path: Path) -> None:
    from tools.vision.fc_bga_yolo import public_external_manifest as pem

    review, sources = _build_review_tree(tmp_path, 5)
    # First accept sample-000 so the demotion path (accepted -> quarantined)
    # is exercised, then quarantine it along with sample-001.
    export = tmp_path / "export"
    export.mkdir()
    (export / "sample-000.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")
    rc = apply_labels(review / "candidates.jsonl", sources, export, None, dry_run=False)
    assert rc == 0
    assert summarize_review(review / "candidates.jsonl", sources).accepted == 1

    rc = quarantine_records(
        review / "candidates.jsonl", sources,
        ("sample-000", "sample-001"), "UNREADABLE",
        dry_run=False,
    )
    assert rc == 0

    srcs = pem.load_source_registry(sources)
    recs = pem.load_candidate_manifest(review / "candidates.jsonl", srcs)
    by_id = {r.sample_id: r for r in recs}
    for sid in ("sample-000", "sample-001"):
        rec = by_id[sid]
        assert rec.review_status == "quarantined"
        assert rec.quarantine_reason == "UNREADABLE"
        assert rec.annotation_status is None
        assert rec.label_path is None
        assert rec.accepted_classes == ()
    # The audit must accept the quarantined state (reason set, annotation cleared).
    audit = pem.audit_candidates(recs, sources=srcs, manifest_root=review)
    assert audit.errors == ()


def test_quarantine_applies_valid_but_reports_unknown_ids(tmp_path: Path) -> None:
    review, sources = _build_review_tree(tmp_path, 3)

    rc = quarantine_records(
        review / "candidates.jsonl", sources, ("sample-000", "does-not-exist"),
        "DEFECT_UNCLEAR", dry_run=False,
    )

    assert rc == 1  # unknown id -> non-zero exit
    assert (review / "candidates.jsonl.bak").exists()  # valid id still applied
    recs = [
        json.loads(l)
        for l in (review / "candidates.jsonl").open(encoding="utf-8")
        if l.strip()
    ]
    by_id = {r["sample_id"]: r for r in recs}
    assert by_id["sample-000"]["review_status"] == "quarantined"
    assert by_id["sample-001"]["review_status"] == "review_required"  # untouched
    assert len(recs) == 3  # manifest shape preserved


def test_build_review_artifacts_enrich_and_render(tmp_path: Path) -> None:
    review, _ = _build_review_tree(tmp_path, 4)
    candidates = load_candidates(review / "candidates.jsonl")
    enriched = build_enriched(candidates, review)
    assert len(enriched) == 4
    assert enriched[0]["_w"] > 0 and enriched[0]["_h"] > 0

    html = render_html(enriched, {"src-test": "Test Source"}, review)
    assert "七类缺陷" in html  # legend present
    assert "sample-000" in html  # at least one tile rendered


def test_build_b0_version_publish_materializes(tmp_path: Path, monkeypatch) -> None:
    from tools.vision.fc_bga_yolo import build_b0_version

    review, sources = _build_review_tree(tmp_path, 20)
    export = tmp_path / "export"
    export.mkdir()
    for i in range(20):
        cls = i % 2
        (export / f"sample-{i:03d}.txt").write_text(
            f"{cls} 0.5 0.5 0.9 0.9\n", encoding="utf-8"
        )
    apply_labels(review / "candidates.jsonl", sources, export, None, dry_run=False)

    out_root = tmp_path / "versions"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_b0_version.py",
            "--manifest", str(review / "candidates.jsonl"),
            "--sources", str(sources),
            "--output-root", str(out_root),
            "--version", "public-external-v0.1",
            "--stage", "B0",
            "--seed", "42",
            "--publish",
        ],
    )
    rc = build_b0_version.main()
    assert rc == 0

    version_dir = out_root / "public-external-v0.1"
    assert (version_dir / "data.yaml").exists()
    assert (version_dir / "revision.json").exists()
    for split in ("train", "val", "test"):
        assert (version_dir / split / "images").exists()
        assert (version_dir / split / "labels").exists()
