from __future__ import annotations

import math
import random
from typing import Any

from PIL import Image, ImageColor, ImageDraw

from .masks import combine_masks, edge_ring_mask, mask_overlap_ratio, sign_mask
from .models import DatasetSample
from .utils import clamp, random_in_range


DEFAULT_OCCLUSION_PALETTE = [
    "#1A1A1A",
    "#2F2F2F",
    "#5D4037",
    "#6B7280",
    "#8C7A5B",
]


def _apply_overlay(image: Image.Image, mask: Image.Image, color: tuple[int, int, int], opacity: float) -> Image.Image:
    overlay = Image.new("RGB", image.size, color)
    alpha = mask.point(lambda value: int(value * clamp(opacity, 0.0, 1.0)))
    return Image.composite(overlay, image, alpha)


def _sample_center_patch_mask(image_size: tuple[int, int], sample: DatasetSample, rng: random.Random, area_ratio: float) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = sample.roi
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    base_area = width * height * area_ratio
    patch_width = max(4, int(math.sqrt(base_area) * rng.uniform(0.9, 1.1)))
    patch_height = max(4, int(base_area / patch_width))
    center_x = (x1 + x2) // 2 + int(rng.uniform(-0.08, 0.08) * width)
    center_y = (y1 + y2) // 2 + int(rng.uniform(-0.08, 0.08) * height)
    box = (
        center_x - patch_width // 2,
        center_y - patch_height // 2,
        center_x + patch_width // 2,
        center_y + patch_height // 2,
    )
    if rng.random() < 0.5:
        draw.rounded_rectangle(box, radius=max(2, patch_width // 6), fill=255)
    else:
        draw.ellipse(box, fill=255)
    return mask


def _sample_diagonal_band_mask(image_size: tuple[int, int], sample: DatasetSample, rng: random.Random, width_ratio: float) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = sample.roi
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    band_half_width = max(2.0, min(width, height) * width_ratio / 2)
    angle = math.radians(rng.uniform(25, 65) * rng.choice([-1, 1]))
    dx = math.cos(angle)
    dy = math.sin(angle)
    px = -dy
    py = dx
    length = math.hypot(width, height) * 0.9
    points = [
        (center_x - dx * length - px * band_half_width, center_y - dy * length - py * band_half_width),
        (center_x - dx * length + px * band_half_width, center_y - dy * length + py * band_half_width),
        (center_x + dx * length + px * band_half_width, center_y + dy * length + py * band_half_width),
        (center_x + dx * length - px * band_half_width, center_y + dy * length - py * band_half_width),
    ]
    draw.polygon(points, fill=255)
    return combine_masks(mask, sign_mask(image_size, sample.roi, sample.shape))


def generate_occlusion_attack(
    image: Image.Image,
    sample: DatasetSample,
    rng: random.Random,
    config: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    sign = sign_mask(image.size, sample.roi, sample.shape)
    styles = config.get("styles", ["edge_patch", "center_sticker", "diagonal_band"])
    style = rng.choice(styles)
    opacity = random_in_range(rng, config.get("opacity_range", [0.8, 1.0]))
    color = ImageColor.getrgb(rng.choice(config.get("palette", DEFAULT_OCCLUSION_PALETTE)))

    if style == "edge_patch":
        width_ratio = random_in_range(rng, config.get("edge_band_width_ratio", [0.03, 0.06]))
        width_pixels = max(2, int(min(sample.roi_x2 - sample.roi_x1, sample.roi_y2 - sample.roi_y1) * width_ratio))
        expand_pixels = max(0, int(width_pixels * 0.2))
        mask = edge_ring_mask(sign, width_pixels=width_pixels, expand_pixels=expand_pixels)
        metadata = {
            "attack": "occlusion",
            "style": style,
            "width_ratio": round(width_ratio, 4),
            "expand_pixels": expand_pixels,
            "literature_reference": "Edge-aligned annular patch inspired by TSEP-Attack",
        }
    elif style == "center_sticker":
        area_ratio = random_in_range(rng, config.get("center_area_ratio", [0.08, 0.18]))
        mask = combine_masks(_sample_center_patch_mask(image.size, sample, rng, area_ratio), sign)
        metadata = {
            "attack": "occlusion",
            "style": style,
            "area_ratio": round(area_ratio, 4),
            "literature_reference": "Sticker-like localized occlusion inspired by RP2-style patch attacks",
        }
    else:
        width_ratio = random_in_range(rng, config.get("diagonal_band_width_ratio", [0.10, 0.18]))
        mask = _sample_diagonal_band_mask(image.size, sample, rng, width_ratio)
        metadata = {
            "attack": "occlusion",
            "style": style,
            "width_ratio": round(width_ratio, 4),
            "literature_reference": "Band occlusion inspired by physically placed tape/sticker attacks",
        }

    attacked = _apply_overlay(image, mask, color, opacity)
    metadata["opacity"] = round(opacity, 4)
    metadata["color_rgb"] = list(color)
    metadata["mask_fraction"] = round(mask_overlap_ratio(mask, sign), 4)
    return attacked, metadata
