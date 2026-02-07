"""Eigenfaces Image Compression Demo — entry point.

Usage:
    python main.py          # Launch GUI (default)
    python main.py --cli    # Run classic matplotlib demo
"""
import argparse
import sys
import os

# Ensure the Eigenfaces package root is importable
sys.path.insert(0, os.path.dirname(__file__))


def run_cli_demo():
    """Classic matplotlib-based demo (preserves original script functionality)."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from src.compressor import EigenfaceCompressor
    from src.image_processor import ImageProcessor
    from src import metrics, visualization

    compressor = EigenfaceCompressor(n_components=100)
    processor = ImageProcessor()

    print("Fetching LFW faces and training model...")
    compressor.fit_from_lfw(n_images=1000, callback=print)

    # Show eigenfaces
    eigenfaces = compressor.get_eigenfaces(40)
    fig = visualization.plot_eigenfaces(eigenfaces)
    fig.suptitle("Top 40 Eigenfaces", fontsize=14)
    fig.show()

    # Singular value plot
    fig2 = visualization.plot_singular_values(compressor.sigma)
    fig2.show()

    # Progressive reconstruction on a sample asset
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    sample_images = []
    if os.path.isdir(assets_dir):
        exts = {".jpg", ".jpeg", ".png"}
        for f in sorted(os.listdir(assets_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                sample_images.append(os.path.join(assets_dir, f))
            if len(sample_images) >= 2:
                break

    if sample_images:
        for img_path in sample_images:
            img = processor.prepare(img_path)
            steps = [10, 40, 100, 200, 500, 1000]
            recons = compressor.progressive_reconstruction(img, steps)
            fig3 = visualization.plot_reconstruction_comparison(
                img, recons, metrics_fn=metrics.psnr
            )
            fig3.suptitle(os.path.basename(img_path), fontsize=12)
            fig3.show()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Eigenfaces Image Compression Demo")
    parser.add_argument("--gui", action="store_true", default=True,
                        help="Launch GUI (default)")
    parser.add_argument("--cli", action="store_true",
                        help="Run classic matplotlib demo instead of GUI")
    args = parser.parse_args()

    if args.cli:
        run_cli_demo()
    else:
        from src.gui import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
