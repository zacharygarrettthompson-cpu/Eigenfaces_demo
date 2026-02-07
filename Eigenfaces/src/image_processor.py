import numpy as np
from skimage import io, color, transform


class ImageProcessor:
    """Handles image loading, resizing, and normalization for the eigenface pipeline."""

    def __init__(self, target_shape=(62, 47)):
        self.target_shape = target_shape

    def load_image(self, path: str) -> np.ndarray:
        """Load an image as grayscale."""
        img = io.imread(path)
        if img.ndim == 3:
            img = color.rgb2gray(img)
        return img

    def load_color_image(self, path: str) -> np.ndarray:
        """Load an image preserving color channels."""
        return io.imread(path)

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize an image to target_shape."""
        return transform.resize(image, self.target_shape, anti_aliasing=True)

    def prepare(self, path: str) -> np.ndarray:
        """Load, convert to grayscale, resize, and normalize — one call."""
        img = self.load_image(path)
        img = self.resize(img)
        img = self.normalize(img)
        return img

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """Scale pixel values to [0, 1]."""
        imin, imax = image.min(), image.max()
        if imax - imin == 0:
            return np.zeros_like(image, dtype=np.float64)
        return (image - imin) / (imax - imin)
