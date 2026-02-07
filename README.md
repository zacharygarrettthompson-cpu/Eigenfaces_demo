# Eigenfaces Image Compression Demo

An interactive application demonstrating image compression using **eigenfaces** and **Singular Value Decomposition (SVD)**. Train on real faces from the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset, then compress and reconstruct any image — faces or otherwise — while exploring how the number of eigenface components affects reconstruction quality.

## Features

- **Tkinter GUI** with real-time slider for adjusting compression level
- **Progressive reconstruction** strip showing quality at different component counts
- **Quality metrics**: PSNR, SSIM, MSE, and compression ratio displayed live
- **Eigenface gallery** showing the learned basis images
- **Save / Load models** as portable `.npz` files (no re-training needed)
- **CLI mode** for classic matplotlib-based exploration
- **Sample images** bundled for quick experimentation

## Quick Start

```bash
git clone https://github.com/<your-username>/Eigenfaces_demo.git
cd Eigenfaces_demo

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the GUI
cd Eigenfaces
python main.py
```

## Usage

### GUI Mode (default)

```bash
python main.py
```

1. Click **Train on LFW Faces** to download the dataset and compute eigenfaces (one-time, ~30s).
2. Load an image from the **sample dropdown** or click **Load Image...** to browse.
3. Drag the **Components slider** to see reconstruction quality change in real time.
4. Check the **Quality Metrics** panel on the right for PSNR, SSIM, MSE, and compression ratio.
5. Use **Save Model** / **Load Model** to persist your trained model.

### CLI Mode

```bash
python main.py --cli
```

Runs the classic matplotlib workflow: trains the model, displays eigenfaces, singular value plot, and progressive reconstruction comparisons in separate figure windows.

### Python API

```python
from src import EigenfaceCompressor, ImageProcessor, metrics

compressor = EigenfaceCompressor(n_components=100)
compressor.fit_from_lfw(n_images=1000)

processor = ImageProcessor()
img = processor.prepare("assets/Cat.jpg")

reconstructed = compressor.compress_and_reconstruct(img, n_components=50)
print(f"PSNR: {metrics.psnr(img, reconstructed):.2f} dB")
print(f"SSIM: {metrics.ssim(img, reconstructed):.4f}")

compressor.save_model("my_model.npz")
```

## How It Works

1. **Training**: Collect grayscale face images and flatten each into a vector. Stack them into a data matrix and subtract the mean face.
2. **SVD**: Decompose the mean-centered matrix as **X = U S V^T**. The columns of **U** are the *eigenfaces* — an orthonormal basis ordered by variance explained.
3. **Compression**: Project any image onto the first *n* eigenfaces to get a small coefficient vector (e.g., 100 numbers instead of 2,914 pixels).
4. **Reconstruction**: Multiply coefficients by eigenfaces and add the mean face back to approximate the original image.

Fewer components = higher compression but lower quality. The singular value drop-off (visible in the log plot) shows that most facial structure is captured by relatively few components.

## Project Structure

```
Eigenfaces_demo/
├── Eigenfaces/
│   ├── assets/              # Sample images
│   ├── src/
│   │   ├── __init__.py      # Package exports
│   │   ├── compressor.py    # EigenfaceCompressor class
│   │   ├── image_processor.py  # Image loading & preprocessing
│   │   ├── metrics.py       # PSNR, SSIM, MSE, compression ratio
│   │   ├── visualization.py # Reusable matplotlib figure factories
│   │   └── gui.py           # Tkinter GUI application
│   └── main.py              # Entry point (--gui / --cli)
├── requirements.txt
├── README.md
└── LICENSE
```

## Requirements

- Python 3.10+
- numpy
- matplotlib
- scikit-learn
- scikit-image
- tkinter (included with Python)

## License

See [LICENSE](LICENSE) for details.
