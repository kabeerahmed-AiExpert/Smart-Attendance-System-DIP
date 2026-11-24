"""
================================================================================
  DIP Experiments  -  generates figures and a results table for the report
================================================================================
  Damages an image in four ways (poor lighting, Gaussian noise, salt & pepper
  noise, blur), repairs it with the DIP operations from dip_operations.py and
  measures the result with PSNR (higher is better), contrast and entropy.

  Usage
      python dip_experiments.py                       # uses assets/bg.jpg
      python dip_experiments.py --image my_face.jpg   # use your own photo

  Output (folder output/dip_results/)
      experiment_<scenario>.png   before / after images + histograms
      dip_results.csv             all numbers
      dip_results.md              the same table, ready to paste in the report
  No TensorFlow is needed.
================================================================================
"""

import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import dip_operations as dip  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output", "dip_results")


def scenarios(clean):
    """name -> (degraded image, {method name: repair function})"""
    return {
        "poor_lighting": (
            dip.darken(clean, 0.3),
            {
                "Gamma 0.5": lambda i: dip.gamma_correction(i, 0.5),
                "Histogram equalization": dip.histogram_equalization,
                "CLAHE (clip 3, 8x8)": lambda i: dip.clahe(i, 3.0, 8),
                "Contrast stretching": dip.contrast_stretch,
            },
        ),
        "gaussian_noise": (
            dip.add_gaussian_noise(clean, 25, seed=1),
            {
                "Gaussian filter 5x5": lambda i: dip.denoise(i, "Gaussian", 5),
                "Median filter 5x5": lambda i: dip.denoise(i, "Median", 5),
                "Bilateral filter 5": lambda i: dip.denoise(i, "Bilateral", 5),
                "Box filter 5x5": lambda i: dip.denoise(i, "Box (mean)", 5),
            },
        ),
        "salt_pepper_noise": (
            dip.add_salt_pepper_noise(clean, 0.05, seed=1),
            {
                "Median filter 3x3": lambda i: dip.denoise(i, "Median", 3),
                "Median filter 5x5": lambda i: dip.denoise(i, "Median", 5),
                "Gaussian filter 5x5": lambda i: dip.denoise(i, "Gaussian", 5),
                "Bilateral filter 5": lambda i: dip.denoise(i, "Bilateral", 5),
            },
        ),
        "blur": (
            dip.blur(clean, 2.0),
            {
                "Unsharp mask (a=1, r=2)": lambda i: dip.sharpen_unsharp(i, 1.0, 2.0),
                "Unsharp mask (a=2, r=2)": lambda i: dip.sharpen_unsharp(i, 2.0, 2.0),
                "Laplacian sharpen (0.5)": lambda i: dip.sharpen_laplacian(i, 0.5),
            },
        ),
    }


def save_figure(name, clean, degraded, results):
    """Row 1: images (clean, damaged, best/others). Row 2: histograms."""
    items = [("Original", clean), ("Damaged", degraded)] + list(results.items())
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.2))
    for col, (title, img) in enumerate(items):
        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis("off")
        hist = dip.compute_histogram(dip.to_gray(img))
        axes[1, col].bar(np.arange(256), hist, width=1.0, color="#333333")
        axes[1, col].set_xlim(0, 255)
        axes[1, col].set_yticks([])
        axes[1, col].set_xlabel("intensity", fontsize=8)
    fig.suptitle(f"Scenario: {name.replace('_', ' ')}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"experiment_{name}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="DIP experiments for the report")
    parser.add_argument("--image", default=os.path.join(BASE_DIR, "assets", "bg.jpg"),
                        help="path of the test image (a face photo is recommended)")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    clean = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if clean.shape[1] > 900:  # keep it fast
        s = 900.0 / clean.shape[1]
        clean = cv2.resize(clean, (900, int(clean.shape[0] * s)), interpolation=cv2.INTER_AREA)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name, (degraded, methods) in scenarios(clean).items():
        results = {m: fn(degraded) for m, fn in methods.items()}
        save_figure(name, clean, degraded, results)

        d_stats = dip.image_stats(degraded)
        rows.append({"Scenario": name, "Method": "(damaged image, no repair)",
                     "PSNR vs original (dB)": round(dip.psnr(clean, degraded), 2),
                     "Brightness": round(d_stats["brightness"], 1),
                     "Contrast": round(d_stats["contrast"], 1),
                     "Entropy": round(d_stats["entropy"], 2)})
        for method, out in results.items():
            st = dip.image_stats(out)
            rows.append({"Scenario": name, "Method": method,
                         "PSNR vs original (dB)": round(dip.psnr(clean, out), 2),
                         "Brightness": round(st["brightness"], 1),
                         "Contrast": round(st["contrast"], 1),
                         "Entropy": round(st["entropy"], 2)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "dip_results.csv"), index=False)
    with open(os.path.join(OUT_DIR, "dip_results.md"), "w", encoding="utf-8") as f:
        f.write(f"Test image: {os.path.basename(args.image)}\n\n")
        header = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
        f.write(header)
        for _, r in df.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    print(df.to_string(index=False))
    print(f"\nSaved figures + tables in: {OUT_DIR}")


if __name__ == "__main__":
    main()
