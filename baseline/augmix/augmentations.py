from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR = Image.BILINEAR


def int_parameter(level: float, max_value: int | float) -> int:
    return int(level * max_value / 10)


def float_parameter(level: float, max_value: float) -> float:
    return float(level) * max_value / 10.0


def sample_level(severity: int) -> float:
    return float(np.random.uniform(low=0.1, high=severity))


def random_signed(value: float) -> float:
    return -value if np.random.uniform() > 0.5 else value


def autocontrast(image: Image.Image, severity: int) -> Image.Image:
    return ImageOps.autocontrast(image)


def equalize(image: Image.Image, severity: int) -> Image.Image:
    return ImageOps.equalize(image)


def posterize(image: Image.Image, severity: int) -> Image.Image:
    level = int_parameter(sample_level(severity), 4)
    return ImageOps.posterize(image, max(1, 4 - level))


def rotate(image: Image.Image, severity: int) -> Image.Image:
    degrees = random_signed(int_parameter(sample_level(severity), 30))
    return image.rotate(degrees, resample=BILINEAR)


def solarize(image: Image.Image, severity: int) -> Image.Image:
    level = int_parameter(sample_level(severity), 256)
    return ImageOps.solarize(image, 256 - level)


def shear_x(image: Image.Image, severity: int) -> Image.Image:
    level = random_signed(float_parameter(sample_level(severity), 0.3))
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=BILINEAR,
    )


def shear_y(image: Image.Image, severity: int) -> Image.Image:
    level = random_signed(float_parameter(sample_level(severity), 0.3))
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=BILINEAR,
    )


def translate_x(image: Image.Image, severity: int) -> Image.Image:
    pixels = random_signed(int_parameter(sample_level(severity), image.size[0] / 3))
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, pixels, 0, 1, 0),
        resample=BILINEAR,
    )


def translate_y(image: Image.Image, severity: int) -> Image.Image:
    pixels = random_signed(int_parameter(sample_level(severity), image.size[1] / 3))
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, 0, 0, 1, pixels),
        resample=BILINEAR,
    )


def color(image: Image.Image, severity: int) -> Image.Image:
    level = float_parameter(sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Color(image).enhance(level)


def contrast(image: Image.Image, severity: int) -> Image.Image:
    level = float_parameter(sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Contrast(image).enhance(level)


def brightness(image: Image.Image, severity: int) -> Image.Image:
    level = float_parameter(sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Brightness(image).enhance(level)


def sharpness(image: Image.Image, severity: int) -> Image.Image:
    level = float_parameter(sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Sharpness(image).enhance(level)


AUGMIX_OPS = (
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
)

AUGMIX_OPS_ALL = (
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
)


def augment_and_mix(
    image: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    all_ops: bool = True,
) -> Image.Image:
    """Apply AugMix image-space augmentation to one PIL image."""
    if alpha <= 0:
        raise ValueError("AugMix alpha must be positive.")
    if width <= 0:
        raise ValueError("AugMix width must be positive.")

    image = image.convert("RGB")
    image_array = np.asarray(image).astype(np.float32)
    mix = np.zeros_like(image_array)
    weights = np.float32(np.random.dirichlet([alpha] * width))
    mixture_weight = np.float32(np.random.beta(alpha, alpha))
    ops = AUGMIX_OPS_ALL if all_ops else AUGMIX_OPS

    for weight in weights:
        image_aug = image.copy()
        chain_depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(chain_depth):
            op = np.random.choice(ops)
            image_aug = op(image_aug, severity)
        mix += weight * np.asarray(image_aug).astype(np.float32)

    mixed = (1.0 - mixture_weight) * image_array + mixture_weight * mix
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed)
