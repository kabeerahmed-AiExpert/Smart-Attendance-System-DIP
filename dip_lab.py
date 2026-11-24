"""
================================================================================
  DIP Lab  -  Interactive Digital Image Processing page (Streamlit)
================================================================================
  Requirements of the course project covered by this page:

    * image upload (file / camera / built-in sample)
    * more than two customized DIP operations (denoise, gamma, contrast
      stretching, histogram equalization, CLAHE, sharpening, custom kernels,
      edge detection)
    * adjustable parameters (sliders / dropdowns for every operation)
    * before-and-after comparison (images + histograms + quality metrics)
    * save the processed image (PNG download)
    * link with the attendance system: run face recognition on the original
      and on the processed image and compare the confidence.

  The image-processing code itself lives in dip_operations.py.
================================================================================
"""

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")  # no GUI backend needed on a server
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

import dip_operations as dip  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "assets", "bg.jpg")
MAX_WIDTH = 900  # large photos are shrunk so the sliders stay responsive


# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def show_image(img, caption=None):
    """
    st.image stretched to the column width. Tries the newest Streamlit API first
    (width="stretch") and falls back to the older arguments, so the app works on
    old and new Streamlit versions without deprecation errors.
    """
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}):
        try:
            st.image(img, caption=caption, **kwargs)
            return
        except Exception:
            continue
    st.image(img, caption=caption)


def show_dataframe(df):
    """st.dataframe stretched to full width (same compatibility idea as show_image)."""
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}):
        try:
            st.dataframe(df, **kwargs)
            return
        except Exception:
            continue
    st.dataframe(df)


def _to_rgb(pil_image):
    """PIL image -> RGB uint8 numpy array, shrunk to MAX_WIDTH if needed."""
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / float(w)
        rgb = cv2.resize(rgb, (MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def _png_bytes(img_rgb):
    """RGB array -> PNG file bytes (for the download buttons)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


def _histogram_figure(before, after, title_after):
    """Grayscale histograms of the two images drawn side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for ax, img, title, color in [
        (axes[0], before, "Histogram - input", "#555555"),
        (axes[1], after, f"Histogram - {title_after}", "#111111"),
    ]:
        hist = dip.compute_histogram(dip.to_gray(img))
        ax.bar(np.arange(256), hist, width=1.0, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity (0-255)")
    axes[0].set_ylabel("Pixels")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE SOURCE + OPTIONAL DEGRADATION
# ─────────────────────────────────────────────────────────────────────────────


def _get_input_image():
    """Let the user choose the image source. Returns RGB array or None."""
    source = st.radio(
        "Image source",
        ["Upload image", "Use camera", "Built-in sample"],
        horizontal=True,
        key="dip_source",
    )

    if source == "Upload image":
        up = st.file_uploader("Upload a JPG / PNG / WEBP image",
                              type=["jpg", "jpeg", "png", "webp"], key="dip_upload")
        if up is None:
            st.info("Upload an image (a face photo works best) or pick the built-in sample.")
            return None
        try:
            return _to_rgb(Image.open(up))
        except Exception as exc:  # corrupted / unsupported file
            st.error(f"Could not read this image: {exc}")
            return None

    if source == "Use camera":
        shot = st.camera_input("Take a picture", key="dip_camera")
        if shot is None:
            return None
        return _to_rgb(Image.open(shot))

    if not os.path.exists(SAMPLE_PATH):
        st.error("Sample image not found (assets/bg.jpg).")
        return None
    return _to_rgb(Image.open(SAMPLE_PATH))


def _degradation_controls(clean):
    """
    Optional 'bad conditions' simulator. Returns (degraded_image, list_of_names).
    Used to test how well the DIP operations repair a damaged image.
    """
    degraded = clean
    applied = []
    with st.expander("1. Simulate bad conditions (optional)"):
        st.caption("Damage the image on purpose, then try to repair it with the "
                   "operations below. PSNR against the clean image shows how well it worked.")
        if st.checkbox("Poor lighting (darken)", key="dip_deg_dark"):
            f = st.slider("Brightness factor", 0.1, 1.0, 0.4, 0.05, key="dip_deg_dark_f")
            degraded = dip.darken(degraded, f)
            applied.append(f"darken x{f}")
        if st.checkbox("Gaussian noise", key="dip_deg_gauss"):
            s = st.slider("Noise sigma", 1, 80, 25, key="dip_deg_gauss_s")
            degraded = dip.add_gaussian_noise(degraded, s)
            applied.append(f"gaussian noise s={s}")
        if st.checkbox("Salt & pepper noise", key="dip_deg_sp"):
            a = st.slider("Amount (fraction of pixels)", 0.005, 0.20, 0.03, 0.005, key="dip_deg_sp_a")
            degraded = dip.add_salt_pepper_noise(degraded, a)
            applied.append(f"salt&pepper {a}")
        if st.checkbox("Out-of-focus blur", key="dip_deg_blur"):
            b = st.slider("Blur sigma", 0.5, 8.0, 2.0, 0.5, key="dip_deg_blur_s")
            degraded = dip.blur(degraded, b)
            applied.append(f"blur s={b}")
    return degraded, applied


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION CONTROLS
# ─────────────────────────────────────────────────────────────────────────────


def _operation_controls():
    """
    Draw all operation widgets and return (steps, labels).
    steps  : list of (name, params) for dip.apply_pipeline
    labels : human readable names of the enabled steps
    Order used: denoise -> gamma -> stretch -> equalization -> CLAHE ->
    sharpen -> kernel -> edge (denoise first so noise is not amplified).
    """
    steps, labels = [], []

    st.markdown("#### 2. Choose operations (applied in the order listed)")

    with st.expander("Noise removal (spatial filtering)"):
        if st.checkbox("Enable noise removal", key="dip_en_denoise"):
            method = st.selectbox("Filter", dip.DENOISE_METHODS, key="dip_dn_method")
            k = st.slider("Kernel size", 3, 15, 5, 2, key="dip_dn_k")
            steps.append(("denoise", {"method": method, "ksize": k}))
            labels.append(f"Denoise: {method}, k={k}")

    with st.expander("Gamma correction (power-law)"):
        if st.checkbox("Enable gamma correction", key="dip_en_gamma"):
            g = st.slider("Gamma  (<1 brighter, >1 darker)", 0.2, 3.0, 0.6, 0.05, key="dip_gamma")
            steps.append(("gamma", {"gamma": g}))
            labels.append(f"Gamma {g}")

    with st.expander("Contrast stretching"):
        if st.checkbox("Enable contrast stretching", key="dip_en_stretch"):
            lo = st.slider("Low percentile", 0.0, 20.0, 1.0, 0.5, key="dip_st_lo")
            hi = st.slider("High percentile", 80.0, 100.0, 99.0, 0.5, key="dip_st_hi")
            steps.append(("contrast_stretch", {"low_percentile": lo, "high_percentile": hi}))
            labels.append(f"Contrast stretch {lo}-{hi}%")

    with st.expander("Histogram equalization (global)"):
        if st.checkbox("Enable histogram equalization", key="dip_en_he"):
            steps.append(("histogram_equalization", {}))
            labels.append("Histogram equalization")

    with st.expander("CLAHE (adaptive histogram equalization)"):
        if st.checkbox("Enable CLAHE", value=True, key="dip_en_clahe"):
            clip = st.slider("Clip limit", 1.0, 10.0, 2.0, 0.5, key="dip_clahe_clip")
            tile = st.slider("Tile grid size", 2, 16, 8, key="dip_clahe_tile")
            steps.append(("clahe", {"clip_limit": clip, "tile_grid": tile}))
            labels.append(f"CLAHE clip={clip}, tiles={tile}")

    with st.expander("Sharpening"):
        if st.checkbox("Enable sharpening", key="dip_en_sharp"):
            method = st.selectbox("Method", ["Unsharp mask", "Laplacian"], key="dip_sh_method")
            if method == "Unsharp mask":
                amount = st.slider("Amount", 0.1, 3.0, 1.0, 0.1, key="dip_sh_amount")
                radius = st.slider("Radius (sigma)", 0.5, 10.0, 2.0, 0.5, key="dip_sh_radius")
                steps.append(("sharpen_unsharp", {"amount": amount, "radius": radius}))
                labels.append(f"Unsharp mask a={amount}, r={radius}")
            else:
                strength = st.slider("Strength", 0.1, 2.0, 0.5, 0.1, key="dip_sh_strength")
                steps.append(("sharpen_laplacian", {"strength": strength}))
                labels.append(f"Laplacian sharpen s={strength}")

    with st.expander("Custom convolution kernel (kernel visualization)"):
        if st.checkbox("Enable kernel filter", key="dip_en_kernel"):
            names = list(dip.KERNEL_PRESETS.keys()) + ["Custom (edit 3x3 below)"]
            choice = st.selectbox("Kernel", names, key="dip_kernel_choice")
            if choice.startswith("Custom"):
                base = pd.DataFrame(dip.KERNEL_PRESETS["Identity"])
                edited = st.data_editor(base, key="dip_kernel_editor", hide_index=True)
                kernel = edited.to_numpy(dtype=np.float32)
            else:
                kernel = dip.KERNEL_PRESETS[choice]
                st.dataframe(pd.DataFrame(kernel).round(3), hide_index=True)
            steps.append(("kernel", {"kernel": kernel}))
            labels.append(f"Kernel: {choice}")

    with st.expander("Edge detection (output becomes an edge map)"):
        if st.checkbox("Enable edge detection", key="dip_en_edge"):
            method = st.selectbox("Method", dip.EDGE_METHODS, key="dip_edge_method")
            params = {"method": method}
            if method == "Canny":
                params["threshold1"] = st.slider("Lower threshold", 0, 255, 80, key="dip_edge_t1")
                params["threshold2"] = st.slider("Upper threshold", 0, 255, 160, key="dip_edge_t2")
            else:
                params["ksize"] = st.slider("Kernel size", 1, 7, 3, 2, key="dip_edge_k")
            steps.append(("edge", params))
            labels.append(f"Edge: {method}")

    return steps, labels


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────


def _metrics_table(before, after):
    sb, sa = dip.image_stats(before), dip.image_stats(after)
    return pd.DataFrame(
        {
            "Measure": ["Brightness (mean)", "Contrast (std dev)", "Entropy (bits)"],
            "Input": [round(sb["brightness"], 1), round(sb["contrast"], 1), round(sb["entropy"], 2)],
            "Processed": [round(sa["brightness"], 1), round(sa["contrast"], 1), round(sa["entropy"], 2)],
        }
    )


def _recognition_test(recognize_fn, before, after):
    """Run the attendance recognizer on both images and show the result."""
    rows = []
    for label, img in [("Input", before), ("Processed", after)]:
        try:
            results = recognize_fn(img)
        except Exception as exc:
            rows.append({"Image": label, "Faces found": 0, "Result": f"error: {exc}", "Confidence": "-"})
            continue
        if not results:
            rows.append({"Image": label, "Faces found": 0, "Result": "No face detected", "Confidence": "-"})
            continue
        for (_x, _y, _w, _h, name, conf) in results:
            rows.append({"Image": label, "Faces found": len(results), "Result": name,
                         "Confidence": round(float(conf), 4)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("This test does NOT mark attendance. A higher confidence on the processed "
               "image means the preprocessing helped recognition.")


def dip_lab_mode(recognize_fn=None):
    """
    Render the DIP Lab page.

    Args:
        recognize_fn : optional callable(img_rgb) -> list of
                       (x, y, w, h, name, confidence). When given, a
                       'recognition test' button is shown.
    """
    st.markdown("### 🧪 DIP Lab: Interactive Image Processing")
    st.caption("Upload an image, pick operations, adjust the parameters, compare before/after "
               "and download the result.")

    clean = _get_input_image()
    if clean is None:
        return

    degraded, degr_names = _degradation_controls(clean)
    steps, labels = _operation_controls()

    if steps:
        try:
            processed = dip.apply_pipeline(degraded, steps)
        except Exception as exc:
            st.error(f"Processing failed: {exc}")
            return
    else:
        processed = degraded
        st.info("No operation is enabled: the processed image equals the input.")

    st.markdown("#### 3. Before and after")
    col_a, col_b = st.columns(2)
    with col_a:
        show_image(degraded, "Input" + (f" ({', '.join(degr_names)})" if degr_names else ""))
    with col_b:
        show_image(processed, "Processed" + (f" ({len(labels)} step(s))" if labels else ""))
    if labels:
        st.markdown("**Pipeline:** " + "  →  ".join(labels))

    # Histograms
    fig = _histogram_figure(degraded, processed, "processed")
    st.pyplot(fig)
    plt.close(fig)

    # Metrics
    st.markdown("**Image statistics**")
    st.dataframe(_metrics_table(degraded, processed), hide_index=True)

    if degr_names:
        p_deg = dip.psnr(clean, degraded)
        p_proc = dip.psnr(clean, processed) if processed.shape == clean.shape else float("nan")
        m1, m2 = st.columns(2)
        m1.metric("PSNR: clean vs damaged input", f"{p_deg:.2f} dB")
        m2.metric("PSNR: clean vs processed", f"{p_proc:.2f} dB",
                  delta=f"{p_proc - p_deg:+.2f} dB")
        st.caption("PSNR is only meaningful for restoration operations (denoise, gamma, "
                   "equalization, sharpening). Edge maps will naturally score low.")
    else:
        st.caption(f"PSNR between input and processed: {dip.psnr(degraded, processed):.2f} dB "
                   "(how strongly the image was changed).")

    # Save
    st.markdown("#### 4. Save")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download processed image (PNG)", _png_bytes(processed),
                       file_name="processed_image.png", mime="image/png", key="dip_dl_proc")
    d2.download_button("⬇️ Download before | after (PNG)",
                       _png_bytes(np.hstack([degraded, processed])),
                       file_name="before_after.png", mime="image/png", key="dip_dl_both")

    # Recognition link
    if recognize_fn is not None:
        st.markdown("#### 5. Does it help face recognition?")
        if st.button("Run face recognition on input and processed image", key="dip_run_rec"):
            with st.spinner("Recognizing faces..."):
                _recognition_test(recognize_fn, degraded, processed)
