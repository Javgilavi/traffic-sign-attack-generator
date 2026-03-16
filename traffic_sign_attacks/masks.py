from __future__ import annotations

import math
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFilter


def sign_mask(image_size: tuple[int, int], roi: tuple[int, int, int, int], shape: str) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = roi
    inset_x = max(1, round((x2 - x1) * 0.03))
    inset_y = max(1, round((y2 - y1) * 0.03))
    box = (x1 + inset_x, y1 + inset_y, x2 - inset_x, y2 - inset_y)
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    w = max(1, bx2 - bx1)
    h = max(1, by2 - by1)

    if shape == "circle":
        draw.ellipse(box, fill=255)
    elif shape == "triangle":
        draw.polygon(
            [
                (cx, by1),
                (bx1, by2),
                (bx2, by2),
            ],
            fill=255,
        )
    elif shape == "triangle_inverted":
        draw.polygon(
            [
                (bx1, by1),
                (bx2, by1),
                (cx, by2),
            ],
            fill=255,
        )
    elif shape == "diamond":
        draw.polygon(
            [
                (cx, by1),
                (bx2, cy),
                (cx, by2),
                (bx1, cy),
            ],
            fill=255,
        )
    elif shape == "octagon":
        radius = min(w, h) / 2
        points = []
        for index in range(8):
            angle = math.radians(22.5 + index * 45)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(points, fill=255)
    else:
        draw.rectangle(box, fill=255)
    return mask


def polygon_mask(image_size: tuple[int, int], polygon: Iterable[tuple[float, float]]) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(list(polygon), fill=255)
    return mask


def combine_masks(mask_a: Image.Image, mask_b: Image.Image) -> Image.Image:
    return ImageChops.multiply(mask_a, mask_b)


def subtract_masks(mask_a: Image.Image, mask_b: Image.Image) -> Image.Image:
    return ImageChops.subtract(mask_a, mask_b)


def blur_mask(mask: Image.Image, radius: float) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def erode_mask(mask: Image.Image, pixels: int) -> Image.Image:
    if pixels <= 0:
        return mask.copy()
    size = max(3, pixels * 2 + 1)
    return mask.filter(ImageFilter.MinFilter(size=size))


def dilate_mask(mask: Image.Image, pixels: int) -> Image.Image:
    if pixels <= 0:
        return mask.copy()
    size = max(3, pixels * 2 + 1)
    return mask.filter(ImageFilter.MaxFilter(size=size))


def edge_ring_mask(mask: Image.Image, width_pixels: int, expand_pixels: int = 0) -> Image.Image:
    expanded = dilate_mask(mask, expand_pixels)
    inner = erode_mask(mask, width_pixels)
    return subtract_masks(expanded, inner)


def mask_pixel_fraction(mask: Image.Image) -> float:
    pixels = mask.getdata()
    active = sum(1 for value in pixels if value > 0)
    return active / float(mask.width * mask.height)


def mask_overlap_ratio(mask: Image.Image, reference_mask: Image.Image) -> float:
    mask_pixels = list(mask.getdata())
    reference_pixels = list(reference_mask.getdata())
    reference_active = sum(1 for value in reference_pixels if value > 0)
    if reference_active == 0:
        return 0.0
    overlap = sum(
        1
        for value, reference in zip(mask_pixels, reference_pixels)
        if value > 0 and reference > 0
    )
    return overlap / float(reference_active)
