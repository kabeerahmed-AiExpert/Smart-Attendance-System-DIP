# Smart Attendance System with Interactive DIP Lab

**Course:** Digital Image Processing (semester project) | **University:** Aror University
**Stack:** Python, OpenCV, NumPy, Matplotlib, Streamlit, MTCNN, FaceNet

A face-recognition attendance system (MTCNN detection + FaceNet embeddings) extended with
an **interactive Digital Image Processing lab**. The lab lets the user upload an image, apply
classical DIP operations with adjustable parameters, compare before/after, save the result,
and check whether the preprocessing improves face recognition.

---

## 1. Features

### Attendance system
- Face detection with **MTCNN**, 512-d **FaceNet** embeddings, cosine-similarity matching
  with an "Unknown" threshold (0.70).
- Camera mode and image-upload mode (multiple faces per image).
- SQLite attendance log with duplicate prevention (one "Present" per student per day).
- Password-protected admin panel (today / all-time records).

### DIP Lab (new, `🧪 DIP Lab` page)
| # | Operation | Adjustable parameters |
|---|---|---|
| 1 | Noise removal: Gaussian, Median, Bilateral, Box filter | method, kernel size |
| 2 | Gamma correction (power-law transform) | gamma 0.2 to 3.0 |
| 3 | Contrast stretching (percentile based) | low / high percentile |
| 4 | Histogram equalization (written from scratch with the CDF formula) | none |
| 5 | CLAHE (adaptive histogram equalization) | clip limit, tile grid size |
| 6 | Sharpening: Unsharp mask, Laplacian | amount, radius, strength |
| 7 | Custom convolution kernels (box, Gaussian, sharpen, Laplacian, Sobel X/Y, emboss, editable 3x3) | kernel matrix |
| 8 | Edge detection: Canny, Sobel, Laplacian | thresholds, kernel size |

Page workflow:
1. Choose the image: upload, camera, or built-in sample.
2. *(Optional)* simulate bad conditions: darkening, Gaussian noise, salt-and-pepper noise, blur.
3. Enable any operations; they run in this order: denoise, gamma, contrast stretch,
   equalization, CLAHE, sharpen, kernel, edge.
4. See **before / after** images, **before / after histograms** (Matplotlib) and statistics
   (brightness, contrast, entropy, **PSNR**).
5. **Download** the processed image or a before | after comparison (PNG).
6. Press the recognition button to compare face-recognition confidence on the original and
   the processed image (this test does **not** mark attendance).

### DIP preprocessing for recognition
The sidebar block "DIP preprocessing for recognition" applies denoise, gamma and CLAHE to
the Camera and Upload images before face detection. Useful for dark or noisy classroom photos.

---

## 2. Project structure

```
Smart-Attendance-System/
├── app.py                 Streamlit app (pages, sidebar, attendance logic)
├── dip_operations.py      All DIP operations + metrics (no TensorFlow needed)
├── dip_lab.py             DIP Lab page (UI)
├── dip_experiments.py     Generates report figures and PSNR tables
├── face_detection.py      MTCNN detection, crop, resize 160x160, normalize
├── embedding.py           FaceNet embedding + database loading
├── recognition.py         Cosine-similarity matching, "Unknown" threshold
├── database.py            SQLite attendance database
├── build_embeddings.py    Builds output/face_embeddings.pkl from dataset/
├── evaluate_system.py     Accuracy / FAR / FRR evaluation (hold-out)
├── config.py              Shared constants for the two scripts above
├── tests/                 Unit tests for the DIP operations (pytest)
├── assets/                Logo and background
├── output/                face_embeddings.pkl, evaluation_results.txt, dip_results/
├── requirements.txt        Full requirements (Python 3.10/3.11, with TensorFlow)
└── requirements-dip.txt    Minimal requirements (DIP Lab only, no TensorFlow)
```

---

## 3. Installation and run

There are two ways to run the project:

**A) DIP Lab only (any Python version, no TensorFlow)**: enough to demo all DIP features.

```bash
pip install -r requirements-dip.txt
python -m streamlit run app.py
```
The Camera / Upload recognition pages show a message that recognition is unavailable; the
DIP Lab and the attendance records page work normally.

**B) Full system with face recognition (Python 3.10 or 3.11 only)**, because TensorFlow 2.15
does not install on Python 3.12+. On Windows:

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

The pre-built `output/face_embeddings.pkl` is included, so the app runs immediately.

### Admin password
The admin panel password comes from `st.secrets["ADMIN_PASSWORD"]` or the environment
variable `ADMIN_PASSWORD`. If neither is set the demo password `1234` is used, so **set
your own before deploying**:

```bash
export ADMIN_PASSWORD="your-password"        # Windows: set ADMIN_PASSWORD=your-password
```

### Add or update students
Put photos in `dataset/<StudentID>_<Name>/` (for example `dataset/F24ARI129_Kabeer/`), then run

```bash
python build_embeddings.py
```

---

## 4. Experiments for the report

```bash
python dip_experiments.py --image path/to/your_face_photo.jpg
```

For four damage scenarios (poor lighting, Gaussian noise, salt-and-pepper noise, blur) it
saves before/after figures with histograms and a PSNR / brightness / contrast / entropy
table in `output/dip_results/` (`.png`, `.csv`, `.md`). The files already in that folder were
produced on the sample image `assets/bg.jpg`; re-run the script with a face photo for the
final report.

Recognition performance (existing evaluation of the FaceNet system, 59 held-out samples):
accuracy 93.22 %, FAR 0.00 %, FRR 6.78 % (see `output/evaluation_results.txt`).

## 5. Tests

```bash
pip install pytest
python -m pytest tests -q
```

## 6. Mapping to the DIP project requirements

| Requirement (course document) | Where |
|---|---|
| Python, OpenCV, NumPy, Matplotlib | whole project; Matplotlib for histograms |
| GUI (Streamlit allowed) | `app.py`, `dip_lab.py` |
| Image upload | DIP Lab page, Upload page, Camera page |
| At least two customized DIP operations | `dip_operations.py` (8 operation groups, manual histogram equalization) |
| Before-and-after comparison | DIP Lab: images, histograms, statistics, PSNR |
| Adjustable parameters | sliders and dropdowns for every operation |
| Save the processed image | PNG download buttons in the DIP Lab |

Related topics from the project list: *Image Histogram Analysis and Equalization Tool*,
*Image Contrast Enhancement and Gamma Correction Tool*, *Image Noise Removal and
Restoration Tool*, *Interactive Image Filtering and Kernel Visualization Tool*,
*Interactive Digital Image Processing Lab Toolkit*.

## 7. Limitations
- Recognition needs frontal, reasonably lit faces; very small or heavily rotated faces may be missed.
- The camera page uses a snapshot (`st.camera_input`), not a continuous video stream.
- Evaluation numbers come from a small dataset (59 test samples), so they are indicative only.
- No liveness detection: a printed photo of a student could be accepted.
