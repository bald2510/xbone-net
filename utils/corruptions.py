"""
Image Corruption Transforms for OOD Robustness Testing
======================================================
Clinical-relevant image corruptions for testing OOD detector calibration.
Each corruption has 3 severity levels.

Usage:
    from utils.corruptions import apply_corruption, CORRUPTION_TYPES
    
    corrupted = apply_corruption(image, "gaussian_noise", severity=2)
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io


# Corruption severity configs: (severity_1, severity_2, severity_3)
SEVERITY_CONFIGS = {
    "gaussian_noise": {
        "sigma": [0.01, 0.05, 0.1],
    },
    "brightness_shift": {
        "factor": [0.7, 0.5, 1.5],  # <1 = darker, >1 = brighter
    },
    "contrast_reduction": {
        "factor": [0.75, 0.5, 0.25],  # lower = less contrast
    },
    "jpeg_compression": {
        "quality": [50, 20, 10],  # lower = worse quality
    },
}

CORRUPTION_TYPES = list(SEVERITY_CONFIGS.keys())


def _gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Add Gaussian noise to image. Simulates sensor noise."""
    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def _brightness_shift(image: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness. Simulates exposure variation."""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _contrast_reduction(image: Image.Image, factor: float) -> Image.Image:
    """Reduce contrast. Simulates low-quality scans."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def _jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """JPEG compression artifacts. Simulates storage/transmission artifacts."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


_CORRUPTION_FNS = {
    "gaussian_noise": _gaussian_noise,
    "brightness_shift": _brightness_shift,
    "contrast_reduction": _contrast_reduction,
    "jpeg_compression": _jpeg_compression,
}


def apply_corruption(
    image: Image.Image,
    corruption_type: str,
    severity: int = 1,
) -> Image.Image:
    """
    Apply a clinical-relevant image corruption.

    Args:
        image: PIL Image (RGB)
        corruption_type: one of CORRUPTION_TYPES
        severity: 1 (mild), 2 (moderate), or 3 (severe)

    Returns:
        Corrupted PIL Image
    """
    if corruption_type not in _CORRUPTION_FNS:
        raise ValueError(
            f"Unknown corruption: {corruption_type}. "
            f"Choose from: {CORRUPTION_TYPES}"
        )
    if severity not in (1, 2, 3):
        raise ValueError(f"Severity must be 1, 2, or 3. Got: {severity}")

    fn = _CORRUPTION_FNS[corruption_type]
    config = SEVERITY_CONFIGS[corruption_type]

    # Get the parameter value for this severity level
    param_name = list(config.keys())[0]
    param_value = config[param_name][severity - 1]

    return fn(image, param_value)


def get_corrupted_dataset_transform(corruption_type: str, severity: int):
    """
    Returns a callable transform that can be composed with torchvision transforms.

    Usage:
        transform = get_corrupted_dataset_transform("gaussian_noise", 2)
        corrupted_image = transform(pil_image)
    """

    def transform(image: Image.Image) -> Image.Image:
        return apply_corruption(image, corruption_type, severity)

    return transform
