from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageEnhance, ImageFilter

RESAMPLING_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def to_serializable(data: dict) -> str:
    return json.dumps(data, sort_keys=True)


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def random_in_range(rng: random.Random, values: Iterable[float]) -> float:
    lower, upper = list(values)
    return rng.uniform(float(lower), float(upper))


def maybe_apply_physical_transform(image: Image.Image, rng: random.Random, config: dict) -> tuple[Image.Image, dict]:
    if not config.get("enabled", False):
        return image, {"enabled": False}

    params: dict[str, float | bool] = {"enabled": True}
    transformed = image

    rotation_range = config.get("rotation_degrees", [0.0, 0.0])
    rotation = random_in_range(rng, rotation_range)
    params["rotation_degrees"] = rotation
    if abs(rotation) > 1e-6:
        transformed = transformed.rotate(rotation, resample=RESAMPLING_BICUBIC)

    brightness_range = config.get("brightness", [1.0, 1.0])
    brightness = random_in_range(rng, brightness_range)
    params["brightness"] = brightness
    if abs(brightness - 1.0) > 1e-6:
        transformed = ImageEnhance.Brightness(transformed).enhance(brightness)

    contrast_range = config.get("contrast", [1.0, 1.0])
    contrast = random_in_range(rng, contrast_range)
    params["contrast"] = contrast
    if abs(contrast - 1.0) > 1e-6:
        transformed = ImageEnhance.Contrast(transformed).enhance(contrast)

    blur_range = config.get("gaussian_blur_radius", [0.0, 0.0])
    blur_radius = random_in_range(rng, blur_range)
    params["gaussian_blur_radius"] = blur_radius
    if blur_radius > 0:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return transformed, params
