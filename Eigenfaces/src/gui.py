import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .compressor import EigenfaceCompressor
from .image_processor import ImageProcessor
from . import metrics, visualization


ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


class EigenfaceApp(tk.Tk):
    """Tkinter GUI for Eigenface image compression."""

    def __init__(self):
        super().__init__()
        self.title("Eigenfaces Image Compression Demo")
        self.geometry("1200x750")
        self.minsize(900, 600)

        self.compressor = EigenfaceCompressor(n_components=100)
        self.processor = ImageProcessor()
        self._current_image = None       # prepared grayscale, resized
        self._current_original = None    # for display / metrics
        self._slider_after_id = None     # debounce timer

        self._build_ui()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        # Main paned layout: left controls | center display | right metrics
        self._main_pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self._main_pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        left = ttk.Frame(self._main_pw, width=220)
        self._main_pw.add(left, weight=0)

        # ── Train section ──
        ttk.Label(left, text="Model", font=("", 11, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        ttk.Button(left, text="Train on LFW Faces", command=self._on_train).pack(fill=tk.X, padx=8, pady=2)
        self._status_var = tk.StringVar(value="Status: No model loaded")
        ttk.Label(left, textvariable=self._status_var, wraplength=200).pack(anchor="w", padx=8, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # ── Load image section ──
        ttk.Label(left, text="Image", font=("", 11, "bold")).pack(anchor="w", padx=8, pady=(4, 2))
        ttk.Button(left, text="Load Image...", command=self._on_load_image).pack(fill=tk.X, padx=8, pady=2)

        # Sample images dropdown
        sample_files = self._get_sample_files()
        self._sample_var = tk.StringVar(value="-- sample images --")
        if sample_files:
            combo = ttk.Combobox(left, textvariable=self._sample_var, values=sample_files, state="readonly")
            combo.pack(fill=tk.X, padx=8, pady=2)
            combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # ── Slider ──
        ttk.Label(left, text="Components", font=("", 11, "bold")).pack(anchor="w", padx=8, pady=(4, 2))
        self._slider_var = tk.IntVar(value=100)
        self._slider = ttk.Scale(left, from_=1, to=1000, variable=self._slider_var,
                                 orient=tk.HORIZONTAL, command=self._on_slider_changed)
        self._slider.pack(fill=tk.X, padx=8, pady=2)
        self._slider_label = ttk.Label(left, text="n = 100")
        self._slider_label.pack(anchor="w", padx=8)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # ── Save / Load model ──
        ttk.Label(left, text="Persistence", font=("", 11, "bold")).pack(anchor="w", padx=8, pady=(4, 2))
        ttk.Button(left, text="Save Model...", command=self._on_save_model).pack(fill=tk.X, padx=8, pady=2)
        ttk.Button(left, text="Load Model...", command=self._on_load_model).pack(fill=tk.X, padx=8, pady=2)

    def _build_center_panel(self):
        center = ttk.Frame(self._main_pw)
        self._main_pw.add(center, weight=1)

        # Top area: side-by-side original / reconstructed
        self._top_fig = Figure(figsize=(7, 3.5), dpi=90)
        self._top_canvas = FigureCanvasTkAgg(self._top_fig, master=center)
        self._top_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 2))

        # Bottom area: progressive reconstruction strip
        self._bot_fig = Figure(figsize=(7, 2.2), dpi=90)
        self._bot_canvas = FigureCanvasTkAgg(self._bot_fig, master=center)
        self._bot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False, padx=4, pady=(2, 4))

    def _build_right_panel(self):
        right = ttk.Frame(self._main_pw, width=220)
        self._main_pw.add(right, weight=0)

        # ── Metrics panel ──
        ttk.Label(right, text="Quality Metrics", font=("", 11, "bold")).pack(anchor="w", padx=8, pady=(8, 4))
        self._metrics_text = tk.Text(right, height=7, width=24, state="disabled",
                                     font=("Consolas", 10), relief=tk.FLAT, bg=self.cget("bg"))
        self._metrics_text.pack(anchor="w", padx=8)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # ── Eigenface gallery ──
        ttk.Label(right, text="Eigenfaces", font=("", 11, "bold")).pack(anchor="w", padx=8, pady=(4, 2))
        self._eigen_fig = Figure(figsize=(3, 4), dpi=80)
        self._eigen_canvas = FigureCanvasTkAgg(self._eigen_fig, master=right)
        self._eigen_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ── Event handlers ──────────────────────────────────────────────

    def _on_train(self):
        self._status_var.set("Status: Training...")
        self._disable_buttons()

        def _train():
            try:
                self.compressor.fit_from_lfw(
                    n_images=1000,
                    callback=lambda msg: self.after(0, self._status_var.set, f"Status: {msg}"),
                )
                self.after(0, self._post_train)
            except Exception as e:
                self.after(0, self._status_var.set, f"Error: {e}")
                self.after(0, self._enable_buttons)

        threading.Thread(target=_train, daemon=True).start()

    def _post_train(self):
        max_n = self.compressor.max_components
        self._slider.configure(to=max_n)
        self._slider_var.set(min(100, max_n))
        self._status_var.set(f"Status: Model ready ({max_n} max components)")
        self._enable_buttons()
        self._update_eigenface_gallery()
        if self._current_image is not None:
            self._update_reconstruction()

    def _on_load_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")],
        )
        if path:
            self._load_and_show(path)

    def _on_sample_selected(self, event):
        name = self._sample_var.get()
        path = os.path.join(ASSETS_DIR, name)
        if os.path.isfile(path):
            self._load_and_show(path)

    def _on_slider_changed(self, value):
        n = int(float(value))
        self._slider_label.configure(text=f"n = {n}")
        # Debounce: wait 50ms after last slider move
        if self._slider_after_id is not None:
            self.after_cancel(self._slider_after_id)
        self._slider_after_id = self.after(50, self._on_slider_apply)

    def _on_slider_apply(self):
        self._slider_after_id = None
        self.compressor.n_components = self._slider_var.get()
        if self._current_image is not None and self.compressor.is_fitted:
            self._update_reconstruction()

    def _on_save_model(self):
        if not self.compressor.is_fitted:
            messagebox.showwarning("Save Model", "No model to save. Train a model first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz")],
        )
        if path:
            self.compressor.save_model(path)
            self._status_var.set(f"Status: Model saved to {os.path.basename(path)}")

    def _on_load_model(self):
        path = filedialog.askopenfilename(
            title="Load model",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
        )
        if path:
            try:
                self.compressor.load_model(path)
                self._post_train()
                self._status_var.set(f"Status: Model loaded from {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Load Model", f"Failed to load model:\n{e}")

    # ── Core display logic ──────────────────────────────────────────

    def _load_and_show(self, path: str):
        try:
            self._current_image = self.processor.prepare(path)
            self._current_original = self._current_image.copy()
            if self.compressor.is_fitted:
                self._update_reconstruction()
            else:
                # Show original only
                self._top_fig.clear()
                ax = self._top_fig.add_subplot(1, 1, 1)
                ax.imshow(self._current_original, cmap="gray")
                ax.set_title("Original (train a model to see reconstruction)")
                ax.axis("off")
                self._top_fig.tight_layout()
                self._top_canvas.draw()
        except Exception as e:
            messagebox.showerror("Load Image", f"Failed to load image:\n{e}")

    def _update_reconstruction(self):
        n = self._slider_var.get()
        original = self._current_original
        reconstructed = self.compressor.compress_and_reconstruct(original, n)

        # Compute metrics
        m = {
            "PSNR": metrics.psnr(original, reconstructed),
            "SSIM": metrics.ssim(original, reconstructed),
            "MSE": metrics.mse(original, reconstructed),
            "Ratio": metrics.compression_ratio(original.shape, n),
        }
        self._show_metrics(m)

        # Top figure: side-by-side
        self._top_fig.clear()
        ax1 = self._top_fig.add_subplot(1, 2, 1)
        ax1.imshow(original, cmap="gray")
        ax1.set_title("Original", fontsize=10)
        ax1.axis("off")
        ax2 = self._top_fig.add_subplot(1, 2, 2)
        ax2.imshow(reconstructed, cmap="gray")
        ax2.set_title(f"Reconstructed (n={n})", fontsize=10)
        ax2.axis("off")
        self._top_fig.tight_layout()
        self._top_canvas.draw()

        # Bottom figure: progressive strip
        steps = [s for s in [10, 25, 50, 100, 200, 500, 1000]
                 if s <= self.compressor.max_components]
        recons = self.compressor.progressive_reconstruction(original, steps)
        n_cols = 1 + len(recons)
        self._bot_fig.clear()
        ax = self._bot_fig.add_subplot(1, n_cols, 1)
        ax.imshow(original, cmap="gray")
        ax.set_title("Original", fontsize=8)
        ax.axis("off")
        for j, (rn, rimg) in enumerate(recons):
            ax = self._bot_fig.add_subplot(1, n_cols, 2 + j)
            ax.imshow(rimg, cmap="gray")
            p = metrics.psnr(original, rimg)
            ax.set_title(f"n={rn}\n{p:.1f}dB", fontsize=7)
            ax.axis("off")
        self._bot_fig.tight_layout()
        self._bot_canvas.draw()

    def _show_metrics(self, m: dict):
        self._metrics_text.configure(state="normal")
        self._metrics_text.delete("1.0", tk.END)
        for key, val in m.items():
            if val == float("inf"):
                self._metrics_text.insert(tk.END, f"  {key:>6s}:    inf\n")
            else:
                self._metrics_text.insert(tk.END, f"  {key:>6s}: {val:>8.3f}\n")
        self._metrics_text.configure(state="disabled")

    def _update_eigenface_gallery(self):
        if not self.compressor.is_fitted:
            return
        n_show = min(16, self.compressor.max_components)
        eigenfaces = self.compressor.get_eigenfaces(n_show)
        n_cols = 4
        n_rows = max(1, (n_show + n_cols - 1) // n_cols)
        self._eigen_fig.clear()
        for i, ef in enumerate(eigenfaces):
            ax = self._eigen_fig.add_subplot(n_rows, n_cols, i + 1)
            ax.imshow(ef, cmap="gray")
            ax.axis("off")
        self._eigen_fig.tight_layout()
        self._eigen_canvas.draw()

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_sample_files(self) -> list[str]:
        if not os.path.isdir(ASSETS_DIR):
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return sorted(f for f in os.listdir(ASSETS_DIR) if os.path.splitext(f)[1].lower() in exts)

    def _disable_buttons(self):
        for w in self.winfo_children():
            self._set_state(w, "disabled")

    def _enable_buttons(self):
        for w in self.winfo_children():
            self._set_state(w, "!disabled")

    @staticmethod
    def _set_state(widget, state):
        try:
            widget.state([state])
        except (AttributeError, tk.TclError):
            pass
        for child in widget.winfo_children():
            EigenfaceApp._set_state(child, state)


def launch_gui():
    app = EigenfaceApp()
    app.mainloop()
