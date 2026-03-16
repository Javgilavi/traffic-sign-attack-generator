from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .labels import label_for_class, shape_for_class
from .models import DatasetSample


CSV_ALIASES = {
    "path": ["path", "Path", "filepath", "image_path", "Filename", "filename"],
    "width": ["Width", "width"],
    "height": ["Height", "height"],
    "class_id": ["ClassId", "class_id", "label", "ClassID"],
    "roi_x1": ["Roi.X1", "RoiX1", "roi_x1", "x1"],
    "roi_y1": ["Roi.Y1", "RoiY1", "roi_y1", "y1"],
    "roi_x2": ["Roi.X2", "RoiX2", "roi_x2", "x2"],
    "roi_y2": ["Roi.Y2", "RoiY2", "roi_y2", "y2"],
}


def _pick_key(row: dict[str, str], logical_name: str) -> str | None:
    for key in CSV_ALIASES[logical_name]:
        if key in row:
            return key
    return None


def _require_int(row: dict[str, str], logical_name: str) -> int:
    key = _pick_key(row, logical_name)
    if key is None:
        raise KeyError(f"Missing required column for {logical_name}.")
    return int(float(row[key]))


def _resolve_path(dataset_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return dataset_root / candidate


def load_gtsrb_samples(dataset_root: Path, split: str) -> list[DatasetSample]:
    split = split.lower()
    if (dataset_root / "Train.csv").exists() or (dataset_root / "Test.csv").exists():
        return _load_flat_csv_layout(dataset_root, split)
    return _load_official_layout(dataset_root, split)


def _load_flat_csv_layout(dataset_root: Path, split: str) -> list[DatasetSample]:
    split_to_name = {"train": "Train.csv", "test": "Test.csv"}
    split_names = [split_to_name[split]] if split in split_to_name else list(split_to_name.values())
    samples: list[DatasetSample] = []

    for csv_name in split_names:
        csv_path = dataset_root / csv_name
        if not csv_path.exists():
            continue
        current_split = csv_name.replace(".csv", "").lower()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                path_key = _pick_key(row, "path")
                if path_key is None:
                    raise KeyError("Missing path column in flat CSV layout.")
                class_id = _require_int(row, "class_id")
                image_path = _resolve_path(dataset_root, row[path_key])
                width = _require_int(row, "width")
                height = _require_int(row, "height")
                sample = DatasetSample(
                    image_path=image_path,
                    split=current_split,
                    class_id=class_id,
                    label=label_for_class(class_id),
                    shape=shape_for_class(class_id),
                    width=width,
                    height=height,
                    roi_x1=_require_int(row, "roi_x1"),
                    roi_y1=_require_int(row, "roi_y1"),
                    roi_x2=_require_int(row, "roi_x2"),
                    roi_y2=_require_int(row, "roi_y2"),
                    sample_id=f"{current_split}_{index:06d}",
                )
                samples.append(sample)
    return samples


def _iter_official_train_csvs(dataset_root: Path) -> Iterable[tuple[Path, Path]]:
    train_dir = dataset_root / "Train"
    if train_dir.exists():
        for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
            csv_path = class_dir / f"GT-{class_dir.name}.csv"
            if csv_path.exists():
                yield class_dir, csv_path
        return

    final_train = dataset_root / "Final_Training" / "Images"
    if final_train.exists():
        for class_dir in sorted(path for path in final_train.iterdir() if path.is_dir()):
            csv_path = class_dir / f"GT-{class_dir.name}.csv"
            if csv_path.exists():
                yield class_dir, csv_path


def _load_official_layout(dataset_root: Path, split: str) -> list[DatasetSample]:
    samples: list[DatasetSample] = []

    if split in {"train", "all"}:
        for class_dir, csv_path in _iter_official_train_csvs(dataset_root):
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=";")
                for index, row in enumerate(reader):
                    class_id = int(class_dir.name)
                    filename_key = _pick_key(row, "path")
                    if filename_key is None:
                        raise KeyError(f"Missing filename column in {csv_path}.")
                    image_path = class_dir / row[filename_key]
                    sample = DatasetSample(
                        image_path=image_path,
                        split="train",
                        class_id=class_id,
                        label=label_for_class(class_id),
                        shape=shape_for_class(class_id),
                        width=_require_int(row, "width"),
                        height=_require_int(row, "height"),
                        roi_x1=_require_int(row, "roi_x1"),
                        roi_y1=_require_int(row, "roi_y1"),
                        roi_x2=_require_int(row, "roi_x2"),
                        roi_y2=_require_int(row, "roi_y2"),
                        sample_id=f"train_{class_id:02d}_{index:06d}",
                    )
                    samples.append(sample)

    if split in {"test", "all"}:
        test_csv_candidates = [
            dataset_root / "Test.csv",
            dataset_root / "GT-final_test.csv",
            dataset_root / "GT-final_test.test.csv",
        ]
        image_roots = [
            dataset_root,
            dataset_root / "Test",
            dataset_root / "Final_Test" / "Images",
        ]
        test_csv = next((path for path in test_csv_candidates if path.exists()), None)
        if test_csv is not None:
            with test_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                delimiter = ";" if test_csv.name.startswith("GT-") else ","
                reader = csv.DictReader(handle, delimiter=delimiter)
                for index, row in enumerate(reader):
                    path_key = _pick_key(row, "path")
                    if path_key is None:
                        raise KeyError("Missing path column in test CSV.")
                    image_path = None
                    for root in image_roots:
                        candidate = root / row[path_key]
                        if candidate.exists():
                            image_path = candidate
                            break
                    if image_path is None:
                        image_path = image_roots[0] / row[path_key]
                    class_id = _require_int(row, "class_id")
                    sample = DatasetSample(
                        image_path=image_path,
                        split="test",
                        class_id=class_id,
                        label=label_for_class(class_id),
                        shape=shape_for_class(class_id),
                        width=_require_int(row, "width"),
                        height=_require_int(row, "height"),
                        roi_x1=_require_int(row, "roi_x1"),
                        roi_y1=_require_int(row, "roi_y1"),
                        roi_x2=_require_int(row, "roi_x2"),
                        roi_y2=_require_int(row, "roi_y2"),
                        sample_id=f"test_{index:06d}",
                    )
                    samples.append(sample)
    return samples
