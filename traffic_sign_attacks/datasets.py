from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .labels import label_for_class, shape_for_class
from .models import DatasetSample


CSV_ALIASES = {
    "path": ["path", "Path", "filepath", "image_path", "Filename", "filename"],
    "width": ["Width", "width"],
    "height": ["Height", "height"],
    "class_id": ["ClassId", "class_id", "label", "ClassID", "label_id", "LabelId"],
    "split": ["split", "Split", "subset", "Subset"],
    "label": ["label", "Label", "class_name", "ClassName", "sign_label", "sign_name", "name"],
    "shape": ["shape", "Shape", "sign_shape", "SignShape", "geometry", "Geometry"],
    "sample_id": ["sample_id", "SampleId", "sample", "Sample", "id", "ID"],
    "roi_x1": ["Roi.X1", "RoiX1", "roi_x1", "x1"],
    "roi_y1": ["Roi.Y1", "RoiY1", "roi_y1", "y1"],
    "roi_x2": ["Roi.X2", "RoiX2", "roi_x2", "x2"],
    "roi_y2": ["Roi.Y2", "RoiY2", "roi_y2", "y2"],
}

SHAPE_ALIASES = {
    "box": "box",
    "circle": "circle",
    "circular": "circle",
    "diamond": "diamond",
    "octagon": "octagon",
    "rectangle": "box",
    "rhombus": "diamond",
    "square": "box",
    "triangle": "triangle",
    "triangle-down": "triangle_inverted",
    "triangle-up": "triangle",
    "triangle_inverted": "triangle_inverted",
    "triangle_inverted_down": "triangle_inverted",
    "triangleinverted": "triangle_inverted",
    "triangular": "triangle",
    "up_triangle": "triangle",
    "down_triangle": "triangle_inverted",
    "inverted_triangle": "triangle_inverted",
}


def _is_int_like(value: str | None) -> bool:
    if value is None:
        return False
    try:
        int(float(value))
    except (TypeError, ValueError):
        return False
    return True


def _pick_key(
    row: dict[str, str],
    logical_name: str,
    *,
    numeric: bool = False,
    exclude: set[str] | None = None,
) -> str | None:
    excluded = exclude or set()
    for key in CSV_ALIASES[logical_name]:
        if key not in row or key in excluded:
            continue
        value = row[key]
        if numeric and not _is_int_like(value):
            continue
        return key
    return None


def _extract_class_id(row: dict[str, str]) -> tuple[int, str]:
    key = _pick_key(row, "class_id", numeric=True)
    if key is None:
        raise KeyError("Missing required column for class_id.")
    return int(float(row[key])), key


def _require_int(row: dict[str, str], logical_name: str) -> int:
    value = _optional_int(row, logical_name)
    if value is None:
        raise KeyError(f"Missing required column for {logical_name}.")
    return value


def _optional_int(row: dict[str, str], logical_name: str) -> int | None:
    key = _pick_key(row, logical_name, numeric=True)
    if key is None:
        return None
    return int(float(row[key]))


def _optional_text(
    row: dict[str, str],
    logical_name: str,
    *,
    exclude: set[str] | None = None,
) -> str | None:
    key = _pick_key(row, logical_name, exclude=exclude)
    if key is None:
        return None
    value = row[key].strip()
    return value or None


def _resolve_path(
    dataset_root: Path,
    raw_path: str,
    base_path: Path | None = None,
    extra_roots: Iterable[Path] | None = None,
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    roots = [dataset_root]
    if base_path is not None:
        roots.append(base_path.parent)
    if extra_roots is not None:
        roots.extend(extra_roots)
    for root in roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return dataset_root / candidate


def _normalize_shape(raw_shape: str | None, default_shape: str = "box") -> str:
    if not raw_shape:
        return default_shape
    normalized = raw_shape.strip().lower().replace(" ", "_")
    return SHAPE_ALIASES.get(normalized, normalized or default_shape)


def _coerce_mapping(raw_mapping: Any) -> dict[int, str]:
    if not isinstance(raw_mapping, dict):
        return {}
    mapping: dict[int, str] = {}
    for key, value in raw_mapping.items():
        try:
            mapping[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _resolve_label(
    row: dict[str, str],
    class_id: int,
    class_id_key: str,
    label_map: dict[int, str],
) -> str:
    explicit_label = _optional_text(row, "label", exclude={class_id_key})
    if explicit_label is not None:
        return explicit_label
    if class_id in label_map:
        return label_map[class_id]
    return label_for_class(class_id)


def _resolve_shape(
    row: dict[str, str],
    class_id: int,
    shape_map: dict[int, str],
    default_shape: str,
) -> str:
    explicit_shape = _optional_text(row, "shape")
    if explicit_shape is not None:
        return _normalize_shape(explicit_shape, default_shape)
    if class_id in shape_map:
        return _normalize_shape(shape_map[class_id], default_shape)
    fallback_shape = shape_for_class(class_id)
    if fallback_shape != "box":
        return fallback_shape
    return default_shape


def _read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as handle:
        return handle.size


def _row_split(
    row: dict[str, str],
    requested_split: str,
    default_split_name: str,
) -> str:
    raw_split = _optional_text(row, "split")
    if raw_split is None:
        if requested_split != "all":
            return requested_split
        return default_split_name
    return raw_split.lower()


def _row_sample_id(
    row: dict[str, str],
    split_name: str,
    index: int,
) -> str:
    explicit_sample_id = _optional_text(row, "sample_id")
    if explicit_sample_id is not None:
        return explicit_sample_id
    return f"{split_name}_{index:06d}"


def load_dataset_samples(
    dataset_root: Path,
    split: str,
    config: dict[str, Any] | None = None,
) -> list[DatasetSample]:
    split = split.lower()
    config = config or {}
    annotation_file = config.get("annotation_file")
    if isinstance(annotation_file, str) and annotation_file.strip():
        csv_path = _resolve_path(dataset_root, annotation_file.strip())
        return _load_annotation_csv(dataset_root, csv_path, split, config)

    for candidate_name in ("annotations.csv", "samples.csv", "dataset.csv"):
        candidate_path = dataset_root / candidate_name
        if candidate_path.exists():
            return _load_annotation_csv(dataset_root, candidate_path, split, config)

    if _looks_like_gtsrb_layout(dataset_root):
        if (dataset_root / "Train.csv").exists() or (dataset_root / "Test.csv").exists():
            return _load_flat_csv_layout(dataset_root, split, config)
        return _load_official_layout(dataset_root, split, config)

    raise FileNotFoundError(
        "Could not find a supported dataset manifest. Provide 'annotation_file' in the YAML config "
        "or place an annotations.csv file inside dataset_root."
    )


def load_gtsrb_samples(dataset_root: Path, split: str) -> list[DatasetSample]:
    return load_dataset_samples(dataset_root, split, {})


def _looks_like_gtsrb_layout(dataset_root: Path) -> bool:
    if (dataset_root / "Train.csv").exists() or (dataset_root / "Test.csv").exists():
        return True
    if (dataset_root / "GT-final_test.csv").exists():
        return True
    train_dir = dataset_root / "Train"
    if train_dir.exists():
        return any((class_dir / f"GT-{class_dir.name}.csv").exists() for class_dir in train_dir.iterdir() if class_dir.is_dir())
    final_train_dir = dataset_root / "Final_Training" / "Images"
    if final_train_dir.exists():
        return any((class_dir / f"GT-{class_dir.name}.csv").exists() for class_dir in final_train_dir.iterdir() if class_dir.is_dir())
    return False


def _csv_reader(handle: Any) -> csv.DictReader:
    sample = handle.read(2048)
    handle.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        return csv.DictReader(handle, dialect=dialect)
    except csv.Error:
        return csv.DictReader(handle)


def _sample_from_row(
    row: dict[str, str],
    *,
    index: int,
    dataset_root: Path,
    requested_split: str,
    default_split_name: str,
    label_map: dict[int, str],
    shape_map: dict[int, str],
    default_shape: str,
    base_path: Path | None = None,
    extra_roots: Iterable[Path] | None = None,
) -> DatasetSample | None:
    path_key = _pick_key(row, "path")
    if path_key is None:
        raise KeyError("Missing path column in dataset manifest.")
    class_id, class_id_key = _extract_class_id(row)
    split_name = _row_split(row, requested_split, default_split_name)
    if requested_split != "all" and split_name != requested_split:
        return None

    image_path = _resolve_path(dataset_root, row[path_key], base_path=base_path, extra_roots=extra_roots)
    width = _optional_int(row, "width")
    height = _optional_int(row, "height")
    if width is None or height is None:
        width, height = _read_image_size(image_path)

    roi_x1 = _optional_int(row, "roi_x1")
    roi_y1 = _optional_int(row, "roi_y1")
    roi_x2 = _optional_int(row, "roi_x2")
    roi_y2 = _optional_int(row, "roi_y2")

    if roi_x1 is None and roi_y1 is None and roi_x2 is None and roi_y2 is None:
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, width, height
    else:
        roi_x1 = 0 if roi_x1 is None else roi_x1
        roi_y1 = 0 if roi_y1 is None else roi_y1
        roi_x2 = width if roi_x2 is None else roi_x2
        roi_y2 = height if roi_y2 is None else roi_y2

    return DatasetSample(
        image_path=image_path,
        split=split_name,
        class_id=class_id,
        label=_resolve_label(row, class_id, class_id_key, label_map),
        shape=_resolve_shape(row, class_id, shape_map, default_shape),
        width=width,
        height=height,
        roi_x1=roi_x1,
        roi_y1=roi_y1,
        roi_x2=roi_x2,
        roi_y2=roi_y2,
        sample_id=_row_sample_id(row, split_name, index),
    )


def _load_annotation_csv(
    dataset_root: Path,
    csv_path: Path,
    split: str,
    config: dict[str, Any],
) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    label_map = _coerce_mapping(config.get("label_map"))
    shape_map = _coerce_mapping(config.get("shape_map"))
    default_shape = _normalize_shape(str(config.get("default_shape", "box")))
    default_split_name = str(config.get("default_split_name", "dataset")).lower()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = _csv_reader(handle)
        for index, row in enumerate(reader):
            sample = _sample_from_row(
                row,
                index=index,
                dataset_root=dataset_root,
                requested_split=split,
                default_split_name=default_split_name,
                label_map=label_map,
                shape_map=shape_map,
                default_shape=default_shape,
                base_path=csv_path,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def _load_flat_csv_layout(
    dataset_root: Path,
    split: str,
    config: dict[str, Any],
) -> list[DatasetSample]:
    split_to_name = {"train": "Train.csv", "test": "Test.csv"}
    split_names = [split_to_name[split]] if split in split_to_name else list(split_to_name.values())
    samples: list[DatasetSample] = []
    label_map = _coerce_mapping(config.get("label_map"))
    shape_map = _coerce_mapping(config.get("shape_map"))
    default_shape = _normalize_shape(str(config.get("default_shape", "box")))

    for csv_name in split_names:
        csv_path = dataset_root / csv_name
        if not csv_path.exists():
            continue
        current_split = csv_name.replace(".csv", "").lower()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                sample = _sample_from_row(
                    row,
                    index=index,
                    dataset_root=dataset_root,
                    requested_split=current_split,
                    default_split_name=current_split,
                    label_map=label_map,
                    shape_map=shape_map,
                    default_shape=default_shape,
                )
                if sample is not None:
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


def _load_official_layout(
    dataset_root: Path,
    split: str,
    config: dict[str, Any],
) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    label_map = _coerce_mapping(config.get("label_map"))
    shape_map = _coerce_mapping(config.get("shape_map"))
    default_shape = _normalize_shape(str(config.get("default_shape", "box")))

    if split in {"train", "all"}:
        for class_dir, csv_path in _iter_official_train_csvs(dataset_root):
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=";")
                for index, row in enumerate(reader):
                    sample = _sample_from_row(
                        row,
                        index=index,
                        dataset_root=class_dir,
                        requested_split="train",
                        default_split_name="train",
                        label_map=label_map,
                        shape_map=shape_map,
                        default_shape=default_shape,
                        base_path=csv_path,
                    )
                    if sample is not None:
                        samples.append(sample)

    if split in {"test", "all"}:
        test_csv_candidates = [
            dataset_root / "Test.csv",
            dataset_root / "GT-final_test.csv",
            dataset_root / "GT-final_test.test.csv",
        ]
        image_roots = [
            dataset_root / "Test",
            dataset_root / "Final_Test" / "Images",
        ]
        test_csv = next((path for path in test_csv_candidates if path.exists()), None)
        if test_csv is not None:
            with test_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                delimiter = ";" if test_csv.name.startswith("GT-") else ","
                reader = csv.DictReader(handle, delimiter=delimiter)
                for index, row in enumerate(reader):
                    sample = _sample_from_row(
                        row,
                        index=index,
                        dataset_root=dataset_root,
                        requested_split="test",
                        default_split_name="test",
                        label_map=label_map,
                        shape_map=shape_map,
                        default_shape=default_shape,
                        base_path=test_csv,
                        extra_roots=image_roots,
                    )
                    if sample is not None:
                        samples.append(sample)
    return samples
