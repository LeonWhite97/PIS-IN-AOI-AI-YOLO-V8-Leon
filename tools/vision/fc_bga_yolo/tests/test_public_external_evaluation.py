import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest

from tools.vision.fc_bga_yolo.contracts import DEFECT_NAMES
from tools.vision.fc_bga_yolo.public_external_evaluation import (
    EMPTY_CLASS_FOOTNOTE,
    ImageValidationStats,
    ValidationStatsCollector,
    build_observed_class_report,
    grouped_bootstrap_map,
    write_public_evaluation_report,
)

# ultralytics is an optional heavy dependency used *only* by grouped_bootstrap_map
# (imported lazily inside that function). The offline test suite must not fail when
# it is absent, so only the grouped-bootstrap cases are skipped — not the whole file.
try:
    import ultralytics  # noqa: F401

    _HAS_ULTRALYTICS = True
except ModuleNotFoundError:
    _HAS_ULTRALYTICS = False

_requires_ultralytics = pytest.mark.skipif(
    not _HAS_ULTRALYTICS,
    reason="ultralytics (optional heavy dep) not installed; grouped bootstrap skipped offline",
)


def test_empty_classes_are_null_and_excluded_from_observed_map() -> None:
    report = build_observed_class_report(
        names=DEFECT_NAMES,
        nt_per_class=np.array([10, 5, 0, 0, 0, 0, 0]),
        ap_class_index=np.array([0, 1]),
        class_results=((0.8, 0.7, 0.6, 0.5), (0.4, 0.3, 0.2, 0.1)),
        native_results=(0.6, 0.5, 0.4, 0.3),
    )

    assert report["observed_class_mAP50"] == pytest.approx(0.4)
    assert report["observed_class_mAP50_95"] == pytest.approx(0.3)
    assert report["native_ultralytics"] == {
        "precision": 0.6,
        "recall": 0.5,
        "mAP50": 0.4,
        "mAP50_95": 0.3,
    }
    assert report["classes"]["EXTRA_BALL"] == {
        "total_gt": 0,
        "status": "no_evidence",
        "metrics": None,
    }
    assert report["footnote"] == EMPTY_CLASS_FOOTNOTE


def _validation_stats(groups: Mapping[str, int]) -> tuple[ImageValidationStats, ...]:
    result = []
    for group_id, image_count in groups.items():
        for index in range(image_count):
            result.append(
                ImageValidationStats(
                    sample_id=f"{group_id}-{index}",
                    source_group_id=group_id,
                    tp=np.ones((1, 10), dtype=bool),
                    conf=np.array([0.9]),
                    pred_cls=np.array([0.0]),
                    target_cls=np.array([0.0]),
                )
            )
    return tuple(result)


@_requires_ultralytics
def test_grouped_bootstrap_moves_all_group_images_as_one_block() -> None:
    groups = {"g1": 5, "g2": 2, "g3": 1}
    groups.update({f"g{index}": 1 for index in range(4, 31)})
    stats = _validation_stats(groups)
    sampled: list[tuple[ImageValidationStats, ...]] = []

    intervals = grouped_bootstrap_map(stats, resamples=1, seed=42, observer=sampled.append)

    assert set(intervals) == {"mAP50", "mAP50_95"}
    for resample in sampled:
        for group_id in set(item.source_group_id for item in resample):
            original_count = sum(item.source_group_id == group_id for item in stats)
            sampled_count = sum(item.source_group_id == group_id for item in resample)
            assert sampled_count % original_count == 0


@_requires_ultralytics
def test_grouped_bootstrap_requires_thirty_independent_groups() -> None:
    with pytest.raises(ValueError, match="BOOTSTRAP_GROUPS_BELOW_30"):
        grouped_bootstrap_map(_validation_stats({"g1": 2, "g2": 1}))


def test_validation_collector_copies_each_image_stats_by_revision_mapping() -> None:
    collector = ValidationStatsCollector(
        {
            "a.png": ("sample-a", "group-1"),
            "b.png": ("sample-b", "group-1"),
        }
    )
    stats = {
        "tp": [np.ones((1, 10), dtype=bool), np.zeros((0, 10), dtype=bool)],
        "conf": [np.array([0.9]), np.array([])],
        "pred_cls": [np.array([0.0]), np.array([])],
        "target_cls": [np.array([0.0]), np.array([1.0])],
        "im_name": ["a.png", str(Path("nested") / "b.png")],
    }

    collector.on_val_batch_end(SimpleNamespace(metrics=SimpleNamespace(stats=stats)))
    stats["conf"][0][0] = 0.1
    records = collector.records()

    assert [record.sample_id for record in records] == ["sample-a", "sample-b"]
    assert records[0].source_group_id == records[1].source_group_id == "group-1"
    assert records[0].conf.tolist() == [0.9]


def test_validation_collector_supports_ultralytics_8_4_120_stats_layout() -> None:
    collector = ValidationStatsCollector(
        {
            "a.png": ("sample-a", "group-1"),
            "b.png": ("sample-b", "group-2"),
        }
    )
    stats = {
        "tp": [np.ones((1, 10), dtype=bool), np.zeros((0, 10), dtype=bool)],
        "conf": [np.array([0.9]), np.array([])],
        "pred_cls": [np.array([0.0]), np.array([])],
        "target_cls": [np.array([0.0]), np.array([1.0])],
        "target_img": [np.array([0.0]), np.array([1.0])],
    }
    box = SimpleNamespace(image_metrics={"a.png": {}, "b.png": {}})
    validator = SimpleNamespace(metrics=SimpleNamespace(stats=stats, box=box))

    collector.on_val_batch_end(validator)

    assert [record.sample_id for record in collector.records()] == ["sample-a", "sample-b"]


def test_public_evaluation_report_is_json_serializable_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "public_evaluation_report.json"

    written = write_public_evaluation_report(path, {"observed_class_mAP50": 0.4})

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8"))["observed_class_mAP50"] == 0.4
    assert not path.with_name(f"{path.name}.tmp").exists()
