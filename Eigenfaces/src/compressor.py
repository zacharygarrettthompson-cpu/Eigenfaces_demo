import numpy as np
from sklearn.datasets import fetch_lfw_people


class EigenfaceCompressor:
    """Eigenface-based image compressor using SVD."""

    def __init__(self, n_components=100):
        self._n_components = n_components
        self._mean = None
        self._U = None
        self._sigma = None
        self._image_shape = None

    @property
    def is_fitted(self) -> bool:
        return self._U is not None

    @property
    def image_shape(self) -> tuple:
        return self._image_shape

    @property
    def n_components(self) -> int:
        return self._n_components

    @n_components.setter
    def n_components(self, value: int):
        if self.is_fitted:
            value = min(value, self._U.shape[1])
        self._n_components = max(1, value)

    @property
    def max_components(self) -> int:
        if not self.is_fitted:
            return 0
        return self._U.shape[1]

    @property
    def sigma(self) -> np.ndarray:
        return self._sigma

    def fit(self, images: list[np.ndarray]) -> None:
        """Train the compressor on a list of grayscale images (all same shape)."""
        self._image_shape = images[0].shape
        X = np.stack([img.flatten() for img in images], axis=1)
        self._mean = np.mean(X, axis=1, keepdims=True)
        self._U, self._sigma, _ = np.linalg.svd(X - self._mean, full_matrices=False)
        self._n_components = min(self._n_components, self._U.shape[1])

    def fit_from_lfw(self, n_images=1000, callback=None) -> None:
        """Convenience: fetch LFW dataset and train on unique faces."""
        if callback:
            callback("Downloading LFW dataset...")
        faces = fetch_lfw_people()
        self._image_shape = faces.images[0].shape

        if callback:
            callback("Selecting unique faces...")
        unique_images = []
        seen_names = set()
        for i in range(len(faces.target)):
            name = faces.target_names[faces.target[i]]
            if name not in seen_names:
                unique_images.append(faces.images[i])
                seen_names.add(name)
            if len(unique_images) >= n_images:
                break

        if callback:
            callback(f"Computing SVD on {len(unique_images)} faces...")
        self.fit(unique_images)
        if callback:
            callback(f"Done! Trained on {len(unique_images)} faces.")

    def compress(self, image: np.ndarray, n_components: int = None) -> np.ndarray:
        """Project an image into the eigenface space, returning coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Compressor is not fitted. Call fit() first.")
        n = n_components or self._n_components
        n = min(n, self._U.shape[1])
        U_n = self._U[:, :n]
        return U_n.T @ (image.flatten() - self._mean.squeeze())

    def reconstruct(self, coefficients: np.ndarray, n_components: int = None) -> np.ndarray:
        """Rebuild an image from its eigenface coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Compressor is not fitted. Call fit() first.")
        n = n_components or len(coefficients)
        n = min(n, self._U.shape[1], len(coefficients))
        U_n = self._U[:, :n]
        reconstructed = U_n @ coefficients[:n] + self._mean.squeeze()
        return reconstructed.reshape(self._image_shape)

    def compress_and_reconstruct(self, image: np.ndarray, n_components: int = None) -> np.ndarray:
        """Compress then reconstruct in one step."""
        coefficients = self.compress(image, n_components)
        return self.reconstruct(coefficients, n_components)

    def progressive_reconstruction(self, image: np.ndarray, steps: list[int] = None) -> list[tuple[int, np.ndarray]]:
        """Reconstruct at multiple n_component values."""
        if steps is None:
            steps = [10, 25, 50, 100, 200, 500, 1000]
        max_n = self._U.shape[1]
        steps = [s for s in steps if s <= max_n]

        # Compress once at the max step value to reuse coefficients
        max_step = max(steps)
        coefficients = self.compress(image, max_step)

        results = []
        for n in steps:
            reconstructed = self.reconstruct(coefficients, n)
            results.append((n, reconstructed))
        return results

    def get_eigenfaces(self, n: int = None) -> list[np.ndarray]:
        """Return the top n eigenfaces as images."""
        if not self.is_fitted:
            raise RuntimeError("Compressor is not fitted. Call fit() first.")
        n = n or self._n_components
        n = min(n, self._U.shape[1])
        return [self._U[:, i].reshape(self._image_shape) for i in range(n)]

    def compression_ratio(self, n_components: int = None) -> float:
        """Calculate the compression ratio (original_size / compressed_size)."""
        if not self.is_fitted:
            return 0.0
        n = n_components or self._n_components
        original_size = self._image_shape[0] * self._image_shape[1]
        compressed_size = n  # just the coefficients
        return original_size / compressed_size

    def save_model(self, path: str) -> None:
        """Save the fitted model to a .npz file."""
        if not self.is_fitted:
            raise RuntimeError("Compressor is not fitted. Nothing to save.")
        np.savez_compressed(
            path,
            mean=self._mean,
            U=self._U,
            sigma=self._sigma,
            image_shape=np.array(self._image_shape),
            n_components=np.array(self._n_components),
        )

    def load_model(self, path: str) -> None:
        """Load a fitted model from a .npz file."""
        data = np.load(path)
        self._mean = data["mean"]
        self._U = data["U"]
        self._sigma = data["sigma"]
        self._image_shape = tuple(data["image_shape"])
        self._n_components = int(data["n_components"])
