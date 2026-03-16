from __future__ import annotations

import random
from typing import Any

from PIL import Image

from .color import lab_to_rgb, rgb_to_lab
from .masks import blur_mask, combine_masks, mask_overlap_ratio, polygon_mask, sign_mask
from .models import DatasetSample
from .utils import clamp, random_in_range


def _sample_triangle(roi: tuple[int, int, int, int], rng: random.Random) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = roi
    width = x2 - x1
    height = y2 - y1
    margin_x = width * 0.30
    margin_y = height * 0.30
    side = rng.choice(["top", "bottom", "left", "right"])

    if side == "top":
        base_y = y1 - rng.uniform(0, margin_y)
        return [
            (x1 - rng.uniform(0, margin_x), base_y),
            (x2 + rng.uniform(0, margin_x), base_y + rng.uniform(-margin_y * 0.2, margin_y * 0.2)),
            (rng.uniform(x1 - margin_x, x2 + margin_x), y2 + rng.uniform(0, margin_y)),
        ]
    if side == "bottom":
        base_y = y2 + rng.uniform(0, margin_y)
        return [
            (x1 - rng.uniform(0, margin_x), base_y),
            (x2 + rng.uniform(0, margin_x), base_y + rng.uniform(-margin_y * 0.2, margin_y * 0.2)),
            (rng.uniform(x1 - margin_x, x2 + margin_x), y1 - rng.uniform(0, margin_y)),
        ]
    if side == "left":
        base_x = x1 - rng.uniform(0, margin_x)
        return [
            (base_x, y1 - rng.uniform(0, margin_y)),
            (base_x + rng.uniform(-margin_x * 0.2, margin_x * 0.2), y2 + rng.uniform(0, margin_y)),
            (x2 + rng.uniform(0, margin_x), rng.uniform(y1 - margin_y, y2 + margin_y)),
        ]
    base_x = x2 + rng.uniform(0, margin_x)
    return [
        (base_x, y1 - rng.uniform(0, margin_y)),
        (base_x + rng.uniform(-margin_x * 0.2, margin_x * 0.2), y2 + rng.uniform(0, margin_y)),
        (x1 - rng.uniform(0, margin_x), rng.uniform(y1 - margin_y, y2 + margin_y)),
    ]


def _apply_shadow_to_image(image: Image.Image, shadow_mask: Image.Image, coefficient: float) -> Image.Image:
    attacked = image.copy()
    pixels = list(attacked.getdata())
    mask_values = list(shadow_mask.getdata())
    updated_pixels: list[tuple[int, int, int]] = []

    for pixel, mask_value in zip(pixels, mask_values):
        if mask_value == 0:
            updated_pixels.append(pixel)
            continue
        l_channel, a_channel, b_channel = rgb_to_lab(pixel)
        strength = (mask_value / 255.0)
        adjusted_l = clamp(l_channel * (1 - ((1 - coefficient) * strength)), 0.0, 100.0)
        updated_pixels.append(lab_to_rgb((adjusted_l, a_channel, b_channel)))

    attacked.putdata(updated_pixels)
    return attacked


def generate_shadow_attack(
    image: Image.Image,
    sample: DatasetSample,
    rng: random.Random,
    config: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    roi_mask = sign_mask(image.size, sample.roi, sample.shape)
    coverage_range = config.get("coverage_ratio", [0.15, 0.55])
    max_attempts = int(config.get("max_sampling_attempts", 24))

    best_mask = None
    best_fraction = 0.0
    best_polygon = None
    for _ in range(max_attempts):
        triangle = _sample_triangle(sample.roi, rng)
        candidate = combine_masks(roi_mask, polygon_mask(image.size, triangle))
        fraction = mask_overlap_ratio(candidate, roi_mask)
        if coverage_range[0] <= fraction <= coverage_range[1]:
            best_mask = candidate
            best_polygon = triangle
            best_fraction = fraction
            break
        if fraction > best_fraction:
            best_mask = candidate
            best_polygon = triangle
            best_fraction = fraction

    if best_mask is None or best_polygon is None:
        raise RuntimeError("Failed to sample a shadow mask.")

    blur_radius = random_in_range(rng, config.get("edge_blur_radius", [1.5, 3.0]))
    softened_mask = blur_mask(best_mask, blur_radius)
    coefficient = random_in_range(rng, config.get("coefficient_range", [0.35, 0.55]))
    attacked = _apply_shadow_to_image(image, softened_mask, coefficient)

    metadata = {
        "attack": "shadow",
        "shape": "triangle",
        "shadow_coefficient": round(coefficient, 4),
        "mask_fraction": round(best_fraction, 4),
        "edge_blur_radius": round(blur_radius, 4),
        "polygon": [[round(x, 2), round(y, 2)] for x, y in best_polygon],
        "literature_reference": "ShadowAttack-style LAB L-channel triangular shadow",
    }
    return attacked, metadata
