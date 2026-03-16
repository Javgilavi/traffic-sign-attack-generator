from __future__ import annotations

from .utils import clamp

EPSILON = 216 / 24389
KAPPA = 24389 / 27
REF_X = 0.95047
REF_Y = 1.0
REF_Z = 1.08883


def _srgb_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(channel: float) -> float:
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * (channel ** (1 / 2.4)) - 0.055


def _lab_f(value: float) -> float:
    if value > EPSILON:
        return value ** (1 / 3)
    return (KAPPA * value + 16) / 116


def _lab_f_inverse(value: float) -> float:
    cubic = value ** 3
    if cubic > EPSILON:
        return cubic
    return (116 * value - 16) / KAPPA


def rgb_to_lab(pixel: tuple[int, int, int]) -> tuple[float, float, float]:
    r = _srgb_to_linear(pixel[0] / 255.0)
    g = _srgb_to_linear(pixel[1] / 255.0)
    b = _srgb_to_linear(pixel[2] / 255.0)

    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    fx = _lab_f(x / REF_X)
    fy = _lab_f(y / REF_Y)
    fz = _lab_f(z / REF_Z)

    l = 116 * fy - 16
    a = 500 * (fx - fy)
    lab_b = 200 * (fy - fz)
    return l, a, lab_b


def lab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    l, a, lab_b = lab
    fy = (l + 16) / 116
    fx = fy + a / 500
    fz = fy - lab_b / 200

    x = REF_X * _lab_f_inverse(fx)
    y = REF_Y * _lab_f_inverse(fy)
    z = REF_Z * _lab_f_inverse(fz)

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    r = clamp(_linear_to_srgb(clamp(r_lin, 0.0, 1.0)), 0.0, 1.0)
    g = clamp(_linear_to_srgb(clamp(g_lin, 0.0, 1.0)), 0.0, 1.0)
    b = clamp(_linear_to_srgb(clamp(b_lin, 0.0, 1.0)), 0.0, 1.0)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
