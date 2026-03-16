from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .datasets import load_dataset_samples
from .occlusion import generate_occlusion_attack
from .shadow import generate_shadow_attack
from .utils import ensure_dir, maybe_apply_physical_transform, relative_to, to_serializable


def _enabled_attacks(config: dict[str, Any]) -> list[str]:
    attacks = []
    for attack_name in ("shadow", "occlusion"):
        if config.get("attacks", {}).get(attack_name, {}).get("enabled", False):
            attacks.append(attack_name)
    return attacks


def _attack_seed_offset(attack_name: str) -> int:
    offsets = {"shadow": 1, "occlusion": 2}
    return offsets[attack_name]


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    dataset_root = Path(config["dataset_root"]).expanduser().resolve()
    output_root = Path(config["output_root"]).expanduser().resolve()
    split = config.get("split", "test").lower()
    seed = int(config.get("seed", 7))
    limit = config.get("limit")
    attacks_config = config.get("attacks", {})
    enabled_attacks = _enabled_attacks(config)
    if not enabled_attacks:
        raise ValueError("No enabled attacks found in the config.")

    ensure_dir(output_root)
    manifest_path = output_root / "manifest.csv"
    samples = load_dataset_samples(dataset_root, split, config)
    if limit is not None:
        samples = samples[: int(limit)]

    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        with Image.open(sample.image_path) as handle:
            image = handle.convert("RGB")
            for attack_name in enabled_attacks:
                attack_config = attacks_config[attack_name]
                variants = int(attack_config.get("variants_per_image", 1))
                for variant_index in range(variants):
                    variant_seed = (
                        seed
                        + sample_index * 1000
                        + variant_index * 10
                        + _attack_seed_offset(attack_name)
                    )
                    variant_rng = random.Random(variant_seed)
                    if attack_name == "shadow":
                        attacked, attack_metadata = generate_shadow_attack(image, sample, variant_rng, attack_config)
                    else:
                        attacked, attack_metadata = generate_occlusion_attack(image, sample, variant_rng, attack_config)

                    transformed, transform_metadata = maybe_apply_physical_transform(
                        attacked,
                        variant_rng,
                        config.get("physical_transform", {}),
                    )

                    image_name = sample.image_path.stem
                    extension = config.get("output_extension", ".png")
                    output_path = output_root / sample.split / attack_name / f"{sample.class_id:02d}"
                    ensure_dir(output_path)
                    file_path = output_path / f"{image_name}_{attack_name}_{variant_index:02d}{extension}"
                    transformed.save(file_path)

                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "split": sample.split,
                            "class_id": sample.class_id,
                            "label": sample.label,
                            "shape": sample.shape,
                            "attack": attack_name,
                            "variant_index": variant_index,
                            "original_path": relative_to(sample.image_path, dataset_root),
                            "output_path": relative_to(file_path, output_root),
                            "roi_x1": sample.roi_x1,
                            "roi_y1": sample.roi_y1,
                            "roi_x2": sample.roi_x2,
                            "roi_y2": sample.roi_y2,
                            "attack_parameters": to_serializable(attack_metadata),
                            "transform_parameters": to_serializable(transform_metadata),
                        }
                    )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "sample_id",
            "split",
            "class_id",
            "label",
            "shape",
            "attack",
            "variant_index",
            "original_path",
            "output_path",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "attack_parameters",
            "transform_parameters",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "split": split,
        "num_input_samples": len(samples),
        "num_generated_samples": len(rows),
        "manifest_path": str(manifest_path),
        "enabled_attacks": enabled_attacks,
    }
