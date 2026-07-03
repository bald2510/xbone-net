"""
Clinically-Relevant Image Corruptions for Robustness Evaluation.
===============================================================================
Simulates real-world X-ray image degradation in XBone-Net evaluation:
  - Equipment variation (Gaussian noise, brightness shift, contrast reduction)
  - Transmission artefacts (JPEG compression)
  - Calibrated severity levels (mild / moderate / severe)

These transforms are applied after data loading and before standard preprocessing
pipelines to evaluate how XBone-Net's classification performance degrades under
distribution shift.
"""

import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ============================================================
# Corruption Configurations & Constants
# ============================================================

# Parameter grids for each corruption type at three severity levels.
# Index 0 = mild (severity 1), index 1 = moderate (severity 2),
# index 2 = severe (severity 3).
SEVERITY_CONFIGS = {
    "gaussian_noise": {
        "sigma": [0.01, 0.05, 0.1],       # Std dev of additive noise
    },
    "brightness_shift": {
        "factor": [0.7, 0.5, 1.5],         # PIL Brightness enhancement factor
    },
    "contrast_reduction": {
        "factor": [0.75, 0.5, 0.25],       # PIL Contrast enhancement factor
    },
    "jpeg_compression": {
        "quality": [50, 20, 10],            # JPEG quality (lower = worse)
    },
}

CORRUPTION_TYPES = list(SEVERITY_CONFIGS.keys())


# ============================================================
# Internal Corruption Functions
# ============================================================

def _gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """Add zero-mean Gaussian noise to simulate sensor noise.

    Pixel values are normalized to [0, 1], perturbed with N(0, sigma) noise,
    clipped to [0, 1], and converted back to uint8 image.

    Args:
        image: Input PIL image.
        sigma: Standard deviation of the Gaussian noise.

    Returns:
        Noisy PIL image.
    """
    # --- Convert image to float array and add Gaussian noise ---
    arr = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def _brightness_shift(image: Image.Image, factor: float) -> Image.Image:
    """Adjust image brightness to simulate exposure variation.

    A factor < 1 darkens the image; a factor > 1 brightens it.

    Args:
        image: Input PIL image.
        factor: Brightness multiplier (0 = black, 1 = original).

    Returns:
        Brightness-adjusted PIL image.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _contrast_reduction(image: Image.Image, factor: float) -> Image.Image:
    """Reduce image contrast to simulate sub-optimal imaging conditions.

    A factor of 0 produces a uniform grey image; 1 preserves original contrast.

    Args:
        image: Input PIL image.
        factor: Contrast multiplier (0 = grey, 1 = original).

    Returns:
        Contrast-adjusted PIL image.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def _jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode the image with lossy JPEG compression.

    Simulates artefacts introduced by low-bandwidth PACS transmission or
    aggressive digital storage compression.

    Args:
        image: Input PIL image.
        quality: JPEG quality level (1-95). Lower values produce heavier
            blocking artefacts.

    Returns:
        JPEG-compressed PIL image (RGB).
    """
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


# ============================================================
# Public API Functions
# ============================================================

def apply_corruption(
    image: Image.Image,
    corruption_type: str,
    severity: int = 1,
) -> Image.Image:
    """Apply a single corruption to an image at the specified severity.

    Args:
        image: Input PIL image.
        corruption_type: One of 'gaussian_noise', 'brightness_shift',
            'contrast_reduction', or 'jpeg_compression'.
        severity: Corruption intensity level - 1 (mild), 2 (moderate),
            or 3 (severe).

    Returns:
        Corrupted PIL image.

    Raises:
        ValueError: If corruption_type is unknown or severity is not in {1, 2, 3}.

    Example:
        from PIL import Image
        img = Image.open("xray.png")
        noisy = apply_corruption(img, "gaussian_noise", severity=2)
    """
    # --- Validate arguments ---
    if corruption_type not in _CORRUPTION_FNS:
        raise ValueError(
            f"Unknown corruption: {corruption_type}. "
            f"Choose from: {CORRUPTION_TYPES}"
        )
    if severity not in (1, 2, 3):
        raise ValueError(f"Severity must be 1, 2, or 3. Got: {severity}")

    # --- Fetch configuration and apply transformation ---
    fn = _CORRUPTION_FNS[corruption_type]
    config = SEVERITY_CONFIGS[corruption_type]

    param_name = list(config.keys())[0]
    param_value = config[param_name][severity - 1]

    return fn(image, param_value)


def get_corrupted_dataset_transform(corruption_type: str, severity: int):
    """Create a reusable transform that applies a fixed corruption.

    Returns a closure suitable for use as a dataset transform argument,
    applying the chosen corruption to every image on the fly.

    Args:
        corruption_type: Corruption name (see CORRUPTION_TYPES).
        severity: Intensity level (1, 2, or 3).

    Returns:
        Callable returning a corrupted PIL Image given an input PIL Image.

    Example:
        transform = get_corrupted_dataset_transform("jpeg_compression", 3)
        dataset = MyDataset(transform=transform)
    """

    # --- Closure for image transformation ---
    def transform(image: Image.Image) -> Image.Image:
        return apply_corruption(image, corruption_type, severity)

    return transform


