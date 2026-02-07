import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure


def plot_eigenfaces(eigenfaces: list[np.ndarray], n_cols: int = 8) -> Figure:
    """Grid of eigenface images. Returns a Figure (never calls plt.show)."""
    n = len(eigenfaces)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig = Figure(figsize=(n_cols * 1.5, n_rows * 1.8))
    for i, ef in enumerate(eigenfaces):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.imshow(ef, cmap="gray")
        ax.set_title(f"#{i+1}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_singular_values(sigma: np.ndarray) -> Figure:
    """Log-scale plot of singular values."""
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(sigma, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_title("Singular Values (log scale)")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructions: list[tuple[int, np.ndarray]],
    metrics_fn=None,
) -> Figure:
    """Original + progressive reconstructions side by side."""
    n_cols = 1 + len(reconstructions)
    fig = Figure(figsize=(n_cols * 2, 3))

    ax = fig.add_subplot(1, n_cols, 1)
    ax.imshow(original, cmap="gray")
    ax.set_title("Original", fontsize=9)
    ax.axis("off")

    for j, (n, recon) in enumerate(reconstructions):
        ax = fig.add_subplot(1, n_cols, 2 + j)
        ax.imshow(recon, cmap="gray")
        title = f"n={n}"
        if metrics_fn:
            m = metrics_fn(original, recon)
            title += f"\nPSNR={m:.1f}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_side_by_side(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "",
    metrics: dict = None,
) -> Figure:
    """Original vs reconstructed with optional metrics annotation."""
    fig = Figure(figsize=(7, 3.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(reconstructed, cmap="gray")
    recon_title = "Reconstructed"
    if metrics:
        parts = [f"{k}: {v:.3f}" for k, v in metrics.items()]
        recon_title += "\n" + "  ".join(parts)
    ax2.set_title(recon_title, fontsize=9)
    ax2.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
