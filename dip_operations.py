"""
================================================================================
  DIP Operations Module  (Digital Image Processing)
================================================================================
  Classical image-processing operations used by the "DIP Lab" page and by the
  optional preprocessing step before face recognition.

  All functions take and return RGB uint8 images of shape (H, W, 3) unless
  stated otherwise. No TensorFlow is needed here, so this file can be tested
  and demonstrated on its own.

  Contents
    1. Histogram tools      : histogram, manual histogram equalization, CLAHE
    2. Intensity transforms : gamma (power-law), contrast stretching, brightness
    3. Spatial filtering    : denoising (box / gaussian / median / bilateral),
                              sharpening (unsharp mask / Laplacian),
                              custom convolution kernels
    4. Edge detection       : Sobel, Laplacian, Canny
    5. Degradation models   : Gaussian noise, salt & pepper noise, darkening,
                              blur (used to test the other operations)
    6. Quality metrics      : MSE, PSNR, brightness, contrast, entropy
    7. Pipeline helper      : apply several operations in order
================================================================================
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def to_gray(img_rgb):
    """RGB uint8 (H, W, 3) -> grayscale uint8 (H, W)."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def _odd(k):
    """Force a kernel size to be an odd integer >= 1."""
    k = int(k)
    if k < 1:
        k = 1
    return k if k % 2 == 1 else k + 1


def _luminance_apply(img_rgb, channel_fn):
    """
    Apply `channel_fn` (uint8 2-D -> uint8 2-D) only to the luminance channel
    (Y of YCrCb) so that colours are not distorted.
    """
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = channel_fn(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 1. HISTOGRAM TOOLS
# ─────────────────────────────────────────────────────────────────────────────


def compute_histogram(gray):
    """256-bin histogram of a grayscale uint8 image (manual, via bincount)."""
    return np.bincount(gray.ravel(), minlength=256)


def equalize_channel_manual(gray):
    """
    Histogram equalization written from scratch (no cv2.equalizeHist).

        s_k = round( (L - 1) * (CDF(r_k) - CDF_min) / (N - CDF_min) )

    where CDF is the cumulative histogram, N the number of pixels and
    L = 256 grey levels.
    """
    hist = compute_histogram(gray)
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return gray.copy()
    cdf_min = nonzero[0]
    n = gray.size
    if n == cdf_min:  # flat image
        return gray.copy()
    lut = np.round((cdf - cdf_min) / float(n - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[gray]


def histogram_equalization(img_rgb):
    """Global histogram equalization on the luminance channel."""
    return _luminance_apply(img_rgb, equalize_channel_manual)


def clahe(img_rgb, clip_limit=2.0, tile_grid=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    L channel of LAB colour space.
    """
    tile_grid = max(1, int(tile_grid))
    engine = cv2.createCLAHE(clipLimit=float(clip_limit),
                             tileGridSize=(tile_grid, tile_grid))
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = engine.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTENSITY TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────


def gamma_correction(img_rgb, gamma=1.0):
    """
    Power-law transform  s = 255 * (r / 255) ** gamma  using a lookup table.

        gamma < 1  -> brighter image (good for dark photos)
        gamma > 1  -> darker image   (good for over-exposed photos)
    """
    gamma = max(float(gamma), 1e-3)
    lut = (np.arange(256) / 255.0) ** gamma * 255.0
    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    return lut[img_rgb]


def contrast_stretch(img_rgb, low_percentile=1.0, high_percentile=99.0):
    """
    Linear contrast stretching: map [p_low, p_high] of the luminance to
    [0, 255] (percentiles make it robust against outlier pixels).
    """
    gray = to_gray(img_rgb)
    lo, hi = np.percentile(gray, [low_percentile, high_percentile])
    if hi - lo < 1e-6:
        return img_rgb.copy()
    out = (img_rgb.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(img_rgb, brightness=0, contrast=1.0):
    """g(x) = contrast * f(x) + brightness  (clipped to 0..255)."""
    out = img_rgb.astype(np.float32) * float(contrast) + float(brightness)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SPATIAL FILTERING
# ─────────────────────────────────────────────────────────────────────────────

DENOISE_METHODS = ["Gaussian", "Median", "Bilateral", "Box (mean)"]


def denoise(img_rgb, method="Gaussian", ksize=5):
    """
    Noise removal.

        Gaussian  : good for Gaussian noise (blurs edges a little)
        Median    : best for salt & pepper noise
        Bilateral : smooths flat areas but keeps edges
        Box       : simple mean filter
    """
    k = _odd(ksize)
    if method == "Gaussian":
        return cv2.GaussianBlur(img_rgb, (k, k), 0)
    if method == "Median":
        return cv2.medianBlur(img_rgb, k)
    if method == "Bilateral":
        # sigma values grow with kernel size so the slider has a visible effect
        return cv2.bilateralFilter(img_rgb, d=k, sigmaColor=15 * k, sigmaSpace=k * 3)
    if method == "Box (mean)":
        return cv2.blur(img_rgb, (k, k))
    raise ValueError(f"Unknown denoise method: {method}")


def sharpen_unsharp(img_rgb, amount=1.0, radius=2.0):
    """
    Unsharp masking:  out = original + amount * (original - blurred)
    `radius` is the sigma of the Gaussian blur.
    """
    radius = max(float(radius), 0.1)
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), radius)
    out = cv2.addWeighted(img_rgb, 1.0 + float(amount), blurred, -float(amount), 0)
    return out


def sharpen_laplacian(img_rgb, strength=1.0):
    """Sharpening with the Laplacian: out = original - strength * Laplacian."""
    lap = cv2.Laplacian(img_rgb, cv2.CV_32F, ksize=3)
    out = img_rgb.astype(np.float32) - float(strength) * lap
    return np.clip(out, 0, 255).astype(np.uint8)


# Preset convolution kernels (for the "kernel visualization" part of the lab)
KERNEL_PRESETS = {
    "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
    "Box blur 3x3": np.ones((3, 3), dtype=np.float32) / 9.0,
    "Gaussian blur 3x3": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    "Edge (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "Sobel X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Sobel Y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
}


def apply_kernel(img_rgb, kernel):
    """Convolve the image with any 2-D kernel (cv2.filter2D)."""
    kernel = np.asarray(kernel, dtype=np.float32)
    out = cv2.filter2D(img_rgb.astype(np.float32), -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 4. EDGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

EDGE_METHODS = ["Canny", "Sobel", "Laplacian"]


def edge_detect(img_rgb, method="Canny", threshold1=80, threshold2=160, ksize=3):
    """
    Edge detection. Returns an RGB image (white edges on black) so that it can
    be shown next to the colour images.

        Canny     : uses two hysteresis thresholds
        Sobel     : gradient magnitude, thresholds ignored
        Laplacian : second derivative magnitude
    """
    gray = to_gray(img_rgb)
    k = _odd(ksize)
    if method == "Canny":
        edges = cv2.Canny(gray, int(threshold1), int(threshold2))
    elif method == "Sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
        mag = cv2.magnitude(gx, gy)
        edges = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif method == "Laplacian":
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=k))
        edges = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        raise ValueError(f"Unknown edge method: {method}")
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEGRADATION MODELS (to simulate bad classroom conditions)
# ─────────────────────────────────────────────────────────────────────────────


def add_gaussian_noise(img_rgb, sigma=20.0, seed=0):
    """Add zero-mean Gaussian noise with standard deviation `sigma`."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), img_rgb.shape)
    return np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img_rgb, amount=0.03, seed=0):
    """Set a fraction `amount` of pixels to pure black or pure white."""
    rng = np.random.default_rng(seed)
    out = img_rgb.copy()
    mask = rng.random(img_rgb.shape[:2])
    out[mask < amount / 2.0] = 0
    out[(mask >= amount / 2.0) & (mask < amount)] = 255
    return out


def darken(img_rgb, factor=0.4):
    """Simulate poor lighting: multiply intensities by `factor` (0..1)."""
    return np.clip(img_rgb.astype(np.float32) * float(factor), 0, 255).astype(np.uint8)


def blur(img_rgb, sigma=2.0):
    """Simulate an out-of-focus camera with Gaussian blur."""
    return cv2.GaussianBlur(img_rgb, (0, 0), max(float(sigma), 0.1))


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────


def mse(img_a, img_b):
    """Mean squared error between two images of the same shape."""
    diff = img_a.astype(np.float64) - img_b.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(img_a, img_b):
    """
    Peak signal-to-noise ratio in dB (higher = more similar).
    Returns +inf when the images are identical.
    """
    err = mse(img_a, img_b)
    if err == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / err))


def image_stats(img_rgb):
    """
    Simple statistics of the grayscale version of an image:
        brightness : mean intensity (0..255)
        contrast   : standard deviation of intensity
        entropy    : Shannon entropy in bits (max 8)
    """
    gray = to_gray(img_rgb)
    hist = compute_histogram(gray).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "entropy": entropy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def apply_step(img_rgb, name, params):
    """Run one named operation with a dict of parameters."""
    if name == "histogram_equalization":
        return histogram_equalization(img_rgb)
    if name == "clahe":
        return clahe(img_rgb, **params)
    if name == "gamma":
        return gamma_correction(img_rgb, **params)
    if name == "contrast_stretch":
        return contrast_stretch(img_rgb, **params)
    if name == "brightness_contrast":
        return adjust_brightness_contrast(img_rgb, **params)
    if name == "denoise":
        return denoise(img_rgb, **params)
    if name == "sharpen_unsharp":
        return sharpen_unsharp(img_rgb, **params)
    if name == "sharpen_laplacian":
        return sharpen_laplacian(img_rgb, **params)
    if name == "kernel":
        return apply_kernel(img_rgb, **params)
    if name == "edge":
        return edge_detect(img_rgb, **params)
    raise ValueError(f"Unknown step: {name}")


def apply_pipeline(img_rgb, steps):
    """
    Apply a list of (name, params) steps in order.

        apply_pipeline(img, [("gamma", {"gamma": 0.6}),
                             ("clahe", {"clip_limit": 2.0, "tile_grid": 8})])
    """
    out = img_rgb
    for name, params in steps:
        out = apply_step(out, name, params)
    return out
