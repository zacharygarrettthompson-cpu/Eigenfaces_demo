import numpy as np
from skimage.metrics import structural_similarity


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error between two images."""
    return float(np.mean((original - reconstructed) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    err = mse(original, reconstructed)
    if err == 0:
        return float("inf")
    max_pixel = max(original.max(), 1.0)
    return float(10 * np.log10(max_pixel ** 2 / err))


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural Similarity Index."""
    data_range = max(original.max() - original.min(), reconstructed.max() - reconstructed.min())
    if data_range == 0:
        data_range = 1.0
    return float(structural_similarity(original, reconstructed, data_range=data_range))


def compression_ratio(original_shape: tuple, n_components: int) -> float:
    """Ratio of original pixel count to number of coefficients stored."""
    original_size = original_shape[0] * original_shape[1]
    if n_components == 0:
        return float("inf")
    return original_size / n_components
