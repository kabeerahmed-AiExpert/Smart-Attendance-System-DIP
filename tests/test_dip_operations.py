"""
Unit tests for dip_operations.py   (run with:  python -m pytest tests -q)
No TensorFlow or Streamlit is needed for these tests.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dip_operations as dip  # noqa: E402


@pytest.fixture(scope="module")
def img():
    """Deterministic test image: smooth gradient + shapes (no file needed)."""
    h, w = 120, 160
    y, x = np.mgrid[0:h, 0:w]
    base = np.stack([x * 255 // w, y * 255 // h, (x + y) * 255 // (w + h)], axis=-1).astype(np.uint8)
    cv2.circle(base, (80, 60), 25, (255, 255, 255), -1)
    cv2.rectangle(base, (10, 10), (50, 40), (0, 0, 0), 2)
    return base


def test_manual_equalization_matches_opencv(img):
    gray = dip.to_gray(img)
    assert np.array_equal(dip.equalize_channel_manual(gray), cv2.equalizeHist(gray))


def test_equalization_handles_flat_image():
    flat = np.full((20, 20), 90, dtype=np.uint8)
    assert np.array_equal(dip.equalize_channel_manual(flat), flat)


@pytest.mark.parametrize("fn", [
    lambda i: dip.histogram_equalization(i),
    lambda i: dip.clahe(i, 3.0, 8),
    lambda i: dip.gamma_correction(i, 0.5),
    lambda i: dip.contrast_stretch(i),
    lambda i: dip.adjust_brightness_contrast(i, 10, 1.1),
    lambda i: dip.denoise(i, "Gaussian", 5),
    lambda i: dip.denoise(i, "Median", 4),      # even size is fixed to odd
    lambda i: dip.denoise(i, "Bilateral", 5),
    lambda i: dip.denoise(i, "Box (mean)", 3),
    lambda i: dip.sharpen_unsharp(i, 1.0, 2.0),
    lambda i: dip.sharpen_laplacian(i, 0.5),
    lambda i: dip.apply_kernel(i, dip.KERNEL_PRESETS["Emboss"]),
    lambda i: dip.edge_detect(i, "Canny", 50, 150),
    lambda i: dip.edge_detect(i, "Sobel"),
    lambda i: dip.edge_detect(i, "Laplacian"),
])
def test_operations_keep_shape_and_dtype(img, fn):
    out = fn(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_gamma_direction(img):
    assert dip.image_stats(dip.gamma_correction(img, 0.5))["brightness"] > dip.image_stats(img)["brightness"]
    assert dip.image_stats(dip.gamma_correction(img, 2.0))["brightness"] < dip.image_stats(img)["brightness"]
    assert np.array_equal(dip.gamma_correction(img, 1.0), img)


def test_identity_kernel(img):
    assert np.array_equal(dip.apply_kernel(img, dip.KERNEL_PRESETS["Identity"]), img)


def test_clahe_and_he_raise_contrast_of_dark_photo():
    """
    Uses the real photo shipped with the project. (On the smooth synthetic
    gradient above, CLAHE's clip limit can slightly LOWER the contrast, so a
    real photograph is the correct test input for this behaviour.)
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "bg.jpg")
    if not os.path.exists(path):
        pytest.skip("assets/bg.jpg not found")
    photo = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    dark = dip.darken(photo, 0.3)
    base = dip.image_stats(dark)
    clahe_stats = dip.image_stats(dip.clahe(dark, 3.0, 8))
    he_stats = dip.image_stats(dip.histogram_equalization(dark))
    assert clahe_stats["contrast"] > base["contrast"]
    assert he_stats["contrast"] > base["contrast"]
    assert clahe_stats["brightness"] > base["brightness"]


def test_equalization_raises_contrast_of_low_contrast_gradient(img):
    low = (img.astype(np.float32) * 0.3 + 60).astype(np.uint8)
    assert dip.image_stats(dip.histogram_equalization(low))["contrast"] > dip.image_stats(low)["contrast"]


def test_denoising_improves_psnr(img):
    noisy = dip.add_gaussian_noise(img, 25, seed=1)
    assert dip.psnr(img, dip.denoise(noisy, "Bilateral", 5)) > dip.psnr(img, noisy)
    sp = dip.add_salt_pepper_noise(img, 0.05, seed=1)
    assert dip.psnr(img, dip.denoise(sp, "Median", 3)) > dip.psnr(img, sp)


def test_noise_is_reproducible(img):
    assert np.array_equal(dip.add_gaussian_noise(img, 10, seed=3), dip.add_gaussian_noise(img, 10, seed=3))


def test_metrics(img):
    assert dip.mse(img, img) == 0
    assert dip.psnr(img, img) == float("inf")
    stats = dip.image_stats(img)
    assert 0 <= stats["brightness"] <= 255 and 0 <= stats["entropy"] <= 8


def test_pipeline_runs_steps_in_order(img):
    dark = dip.darken(img, 0.3)
    out = dip.apply_pipeline(dark, [("gamma", {"gamma": 0.6}),
                                    ("clahe", {"clip_limit": 2.0, "tile_grid": 8}),
                                    ("denoise", {"method": "Gaussian", "ksize": 3})])
    assert out.shape == dark.shape
    assert dip.image_stats(out)["brightness"] > dip.image_stats(dark)["brightness"]
    with pytest.raises(ValueError):
        dip.apply_pipeline(dark, [("does_not_exist", {})])
