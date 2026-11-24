"""
================================================================================
  Smart Attendance System — Streamlit Web Application
================================================================================
  - Black & White Professional Theme
  - WebRTC Camera with Fixed Multi-Face Logic
  - Password Protected Dashboard
- DIP Lab page (interactive Digital Image Processing) and optional DIP
  preprocessing before face recognition
================================================================================
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("KERAS_HOME", os.path.join(BASE_DIR, ".keras"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".matplotlib"))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
import base64

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL MODULE IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow-based modules are optional: without them (e.g. Python 3.12+ where
# TensorFlow 2.15 cannot be installed) the DIP Lab and the attendance records still work,
# only face recognition (Camera / Upload pages) is disabled.
try:
    from face_detection import get_detector, detect_faces_in_frame
    from embedding import get_facenet_model, generate_embedding, load_embedding_database
    RECOGNITION_AVAILABLE = True
    RECOGNITION_IMPORT_ERROR = ""
except ImportError as _exc:
    RECOGNITION_AVAILABLE = False
    RECOGNITION_IMPORT_ERROR = str(_exc)

    def load_embedding_database(path=None):
        """Read the embeddings pickle without needing TensorFlow."""
        import pickle
        path = path or os.path.join(BASE_DIR, "output", "face_embeddings.pkl")
        if not os.path.exists(path):
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)
from recognition import recognize_face, parse_student_display_name, COSINE_SIMILARITY_THRESHOLD
from dip_lab import dip_lab_mode, show_image, show_dataframe
import dip_operations as dip

from database import (
    init_database,
    insert_attendance,
    get_today_attendance,
    get_all_attendance,
    get_attendance_summary,
    clear_today_attendance,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System | Aror University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHING
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    """Load MTCNN detector and FaceNet model (reloaded per thread to avoid TF crashes)."""
    return get_detector(), get_facenet_model()

@st.cache_resource(show_spinner=False)
def load_database():
    """Load the prebuilt face embedding database (cached)."""
    return load_embedding_database()

# Models are loaded inside main() now

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def get_base64_image(image_path):
    """Read an image and convert to base64 for CSS use."""
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMG_PATH = os.path.join(BASE_DIR, "assets", "bg.jpg")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
if not os.path.exists(LOGO_PATH):  # the repository ships assets/logo.jpg
    LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.jpg")
LOGO_MIME = "image/png" if LOGO_PATH.endswith(".png") else "image/jpeg"

bg_base64 = get_base64_image(BG_IMG_PATH)
logo_base64 = get_base64_image(LOGO_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS / BRANDING (Black & White Theme)
# ─────────────────────────────────────────────────────────────────────────────
css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Full Page Background with Dark Overlay */
    .stApp {{
        background: linear-gradient(rgba(10, 10, 10, 0.9), rgba(15, 15, 15, 0.95)), 
                    url(data:image/jpeg;base64,{bg_base64});
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: rgba(15, 15, 15, 0.85) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Headers & Text */
    h1, h2, h3, p, span, div {{
        color: #e0e0e0;
    }}

    /* Header Container */
    .aror-header {{
        display: flex;
        align-items: center;
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }}
    .aror-header img {{
        margin-right: 20px;
        border-radius: 8px;
        background: rgba(255,255,255,0.1); /* Subtle background for contrast */
        padding: 5px;
    }}
    .aror-header h1 {{
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #ffffff;
    }}
    .aror-header p {{
        margin: 0;
        color: #aaaaaa;
        font-size: 1rem;
    }}

    /* Glass Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, background 0.3s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-3px);
        background: rgba(255, 255, 255, 0.05);
    }}
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #ffffff;
    }}
    .metric-label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #888888;
    }}

    /* Buttons */
    .stButton > button {{
        background: rgba(40, 40, 40, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        background: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        border-color: #ffffff !important;
        transform: translateY(-1px) !important;
    }}

    /* Results */
    .result-item {{
        background: rgba(30, 30, 30, 0.6);
        border-left: 4px solid #cccccc;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }}
    .result-item.unknown {{
        border-left-color: #555555;
    }}

    /* Hide elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DIP PREPROCESSING (optional, applied before face recognition)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_for_recognition(img_rgb):
    """
    Apply the DIP preprocessing chosen in the sidebar (denoise -> gamma -> CLAHE).
    Returns the image unchanged when preprocessing is disabled. The image size
    never changes, so face boxes still match the original photo.
    """
    if not st.session_state.get("pre_enable", False):
        return img_rgb

    steps = []
    k = int(st.session_state.get("pre_denoise_k", 1))
    if k > 1:
        steps.append(("denoise", {"method": "Gaussian", "ksize": k}))
    gamma = float(st.session_state.get("pre_gamma", 1.0))
    if abs(gamma - 1.0) > 1e-6:
        steps.append(("gamma", {"gamma": gamma}))
    clip = float(st.session_state.get("pre_clahe", 0.0))
    if clip > 0:
        steps.append(("clahe", {"clip_limit": clip, "tile_grid": 8}))
    return dip.apply_pipeline(img_rgb, steps) if steps else img_rgb


def get_admin_password():
    """Admin password from Streamlit secrets / environment (demo fallback: 1234)."""
    try:
        if "ADMIN_PASSWORD" in st.secrets:
            return str(st.secrets["ADMIN_PASSWORD"])
    except Exception:
        pass  # no secrets file present
    return os.environ.get("ADMIN_PASSWORD", "1234")


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def draw_results_on_frame(frame_bgr, face_results):
    for (x, y, w, h, name, confidence) in face_results:
        is_known = name != "Unknown"
        # B&W aesthetic bounding boxes
        color = (255, 255, 255) if is_known else (100, 100, 100) 

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

        _, display_name = parse_student_display_name(name) if is_known else ("", "Unknown")
        label = f"{display_name} ({confidence:.2f})" if is_known else "Unknown"

        # Safely position label: if face is too close to top edge, draw below the box
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y - 10
        if label_y - th < 0:
            label_y = y + h + th + 10

        cv2.rectangle(frame_bgr, (x, label_y - th - 5), (x + tw + 6, label_y + 5), color, -1)
        cv2.putText(frame_bgr, label, (x + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if is_known else (255, 255, 255), 2)
    return frame_bgr

def process_faces(rgb_frame, detector_model, facenet_model, db, is_upload=False):
    # Phase 7: Debugging Support - Logs
    def debug_log(msg):
        print(f"[DEBUG] {msg}")
        if is_upload:
            st.toast(f"🐛 {msg}")

    debug_log("Frame received for processing.")
    
    faces = detect_faces_in_frame(rgb_frame, detector_model)
    if faces:
        debug_log(f"Face detected: {len(faces)} found.")
    else:
        debug_log("No face detected in frame.")
        
    results = []
    for face in faces:
        x, y, w, h = face["box"]
        try:
            # Enforce dimensions strictly before generating embedding
            face_arr = face["face_array"]
            if face_arr.shape != (160, 160, 3):
                debug_log(f"Shape error: expected (160,160,3), got {face_arr.shape}")
                continue
                
            emb = generate_embedding(face_arr, facenet_model)
            debug_log("Embedding generated successfully.")
            
            name, confidence = recognize_face(emb, db)
            debug_log(f"Prediction completed: {name} ({confidence:.2f})")
            
            results.append((x, y, w, h, name, confidence))
        except Exception as e:
            debug_log(f"Error processing face: {e}")
            continue
    return results



# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    # Logo Handling
    logo_html = ""
    if logo_base64:
        # Properly displaying the logo if it exists
        logo_html = f'<img src="data:{LOGO_MIME};base64,{logo_base64}" width="80" alt="Logo">'
    else:
        logo_html = '<div style="font-size: 3rem; margin-right: 20px;">🏛️</div>'

    st.markdown(f"""
    <div class="aror-header">
        {logo_html}
        <div>
            <h1>Smart Attendance System</h1>
            <p>Face Recognition Based Attendance • Aror University</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        if logo_base64:
            st.markdown(f'<div style="text-align: center;"><img src="data:{LOGO_MIME};base64,{logo_base64}" width="120"></div>', unsafe_allow_html=True)
        else:
            st.markdown("## 🏛️ Aror Univ")
        
        st.markdown("---")
        mode = st.radio(
            "Navigation",
            ["📷 Camera Mode", "🖼️ Upload Image", "🧪 DIP Lab", "📊 View Attendance"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        
        with st.expander("🛠️ DIP preprocessing for recognition"):
            st.caption("Applied to Camera and Upload images before face detection "
                       "(helps with dark or noisy photos).")
            st.checkbox("Enable preprocessing", key="pre_enable")
            st.slider("Denoise kernel (1 = off)", 1, 9, 1, 2, key="pre_denoise_k")
            st.slider("Gamma (1.0 = off, <1 brighter)", 0.4, 2.0, 1.0, 0.05, key="pre_gamma")
            st.slider("CLAHE clip limit (0 = off)", 0.0, 6.0, 0.0, 0.5, key="pre_clahe")
        st.markdown("---")

        if st.button("🔄 Reload Database / Embeddings"):
            load_database.clear()
            st.success("✅ Database reloaded successfully!")
            
        st.markdown("### 📋 System Info")
        db = load_database()
        st.markdown(f"**Students Enrolled:** `{len(db)}`")
        st.markdown(f"**Threshold:** `{COSINE_SIMILARITY_THRESHOLD}`")
        
        return mode


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def render_metrics():
    summary = get_attendance_summary()
    total = summary['total_present'] + summary['total_unknown']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Present Today</div>
            <div class="metric-value" style="color: #ffffff;">{summary['total_present']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Unknown Access</div>
            <div class="metric-value" style="color: #aaaaaa;">{summary['total_unknown']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value" style="color: #dddddd;">{total}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────
def camera_mode(detector, facenet, database):
    st.markdown("### 📷 Camera Recognition")
    st.info("💡 Click 'Take Photo' to mark your attendance. This works on all networks!")
    
    # Built-in Streamlit camera
    picture = st.camera_input("Take a picture")
    
    if picture:
        st.info("⏳ Processing Image...")
        try:
            image = Image.open(picture)
            img_rgb = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error processing camera image: {e}")
            return
            
        with st.spinner("Recognizing faces..."):
            # Resize image to optimize speed
            h, w = img_rgb.shape[:2]
            scale = 1.0
            if w > 800:
                scale = 800.0 / w
                img_rgb = cv2.resize(img_rgb, (800, int(h * scale)))
                
            results = process_faces(preprocess_for_recognition(img_rgb), detector, facenet, database, is_upload=True)
            
            # Scale boxes back
            if scale != 1.0:
                scaled_results = []
                for (x, y, bw, bh, name, conf) in results:
                    scaled_results.append((
                        int(x / scale), int(y / scale),
                        int(bw / scale), int(bh / scale),
                        name, conf
                    ))
                results = scaled_results

        if not results:
            st.warning("⚠️ No face detected. Please try again.")
            return
            
        st.success("✅ Image processed successfully.")
        
        # Draw bounding boxes
        annotated = draw_results_on_frame(img_bgr.copy(), results)
        show_image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        for i, (x, y, w, h, name, conf) in enumerate(results):
            if name != "Unknown":
                sid, display = parse_student_display_name(name)
                # Check if already inserted or insert new
                if insert_attendance(name, "Present", conf):
                    st.markdown(f"""
                    <div class="result-item">
                        <b>✅ {display}</b> (ID: {sid}) — Confidence: {conf:.4f}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-item">
                        <b>⚠️ {display}</b> (ID: {sid}) — Already marked present today.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                insert_attendance("Unknown", "Unknown", conf)
                st.markdown(f"""
                <div class="result-item unknown">
                    <b>❌ Unknown Face Detected</b> — Confidence: {conf:.4f}
                </div>
                """, unsafe_allow_html=True)

def upload_mode(detector, facenet, database):
    # Phase 2: Use st.file_uploader()
    st.markdown("### 🖼️ Batch Image Recognition")
    uploaded = st.file_uploader("Upload an image file (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        st.info("⏳ Processing Image...")
        
        try:
            # Phase 2: Convert uploaded file correctly using PIL -> numpy -> BGR
            image = Image.open(uploaded)
            img_rgb = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Phase 6: Invalid upload -> show error
            st.error(f"Invalid file format: {e}")
            return
        
        with st.spinner("Processing faces..."):
            # Phase 8: Resize frames before detection (Performance Optimization)
            h, w = img_rgb.shape[:2]
            scale = 1.0
            if w > 800:
                scale = 800.0 / w
                img_rgb_resized = cv2.resize(img_rgb, (800, int(h * scale)))
            else:
                img_rgb_resized = img_rgb

            # Phase 3: Unified preprocessing (process_faces applies same pipeline)
            results = process_faces(preprocess_for_recognition(img_rgb_resized), detector, facenet, database, is_upload=True)
            
            # Scale boxes back
            if scale != 1.0:
                scaled_results = []
                for (x, y, bw, bh, name, conf) in results:
                    scaled_results.append((
                        int(x / scale), int(y / scale),
                        int(bw / scale), int(bh / scale),
                        name, conf
                    ))
                results = scaled_results
        
        # Phase 6: No face detected -> show warning
        if not results:
            st.warning("⚠️ No face detected in the uploaded image.")
            show_image(img_rgb)
            return
            
        if len(results) > 1:
            st.info(f"👥 Multiple faces detected: {len(results)}")
            
        st.success("✅ Image processed successfully.")
        
        annotated = draw_results_on_frame(img_bgr.copy(), results)
        show_image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        for i, (x, y, w, h, name, conf) in enumerate(results):
            if name != "Unknown":
                sid, display = parse_student_display_name(name)
                insert_attendance(name, "Present", conf)
                st.markdown(f"""
                <div class="result-item">
                    <b>✅ {display}</b> (ID: {sid}) — Confidence: {conf:.4f}
                </div>
                """, unsafe_allow_html=True)
            else:
                insert_attendance("Unknown", "Unknown", conf)
                st.markdown(f"""
                <div class="result-item unknown">
                    <b>❌ Unknown Face Detected</b> — Confidence: {conf:.4f}
                </div>
                """, unsafe_allow_html=True)

def attendance_view():
    st.markdown("### 📊 Admin Panel: Attendance Records")
    
    # Simple Session State Authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        st.warning("🔒 This section is protected. Please enter the admin password.")
        pwd = st.text_input("Admin Password", type="password")
        if st.button("Login"):
            # Password comes from st.secrets / ADMIN_PASSWORD (demo fallback: 1234)
            if pwd == get_admin_password():
                st.session_state['authenticated'] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    # Authenticated Area
    if st.button("🔒 Logout"):
        st.session_state['authenticated'] = False
        st.rerun()

    col1, col2 = st.columns([3, 1])
    with col1:
        view = st.selectbox("Filter", ["Today", "All Time"])
    with col2:
        if st.button("🗑️ Clear Today's Log"):
            clear_today_attendance()
            st.rerun()

    records = get_today_attendance() if view == "Today" else get_all_attendance()
    
    if records:
        df = pd.DataFrame(records)
        # Prevent drop error if DataFrame is magically empty but records list isn't
        if not df.empty and "id" in df.columns:
            df = df.drop(columns=["id"])

        if not df.empty and "name" in df.columns:
            # Map names
            df["Display Name"] = df["name"].apply(lambda n: parse_student_display_name(n)[1] if n != "Unknown" else "Unknown")
            # Rename columns nicely
            df.rename(columns={"name": "Folder_ID", "status": "Status", "confidence": "Confidence", "date": "Date", "time": "Time"}, inplace=True)
            
            # Reorder
            df = df[["Display Name", "Folder_ID", "Status", "Confidence", "Date", "Time"]]

            def color_status(val):
                return 'color: #ffffff; font-weight: bold;' if val == 'Present' else 'color: #888888;'
            
            show_dataframe(df.style.map(color_status, subset=["Status"]))
    else:
        st.info("No records found.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def recognize_for_lab(img_rgb):
    """
    Face recognition used by the DIP Lab test button (does NOT mark attendance).
    Models are loaded here, only when the button is pressed, so moving a slider
    in the lab never reloads MTCNN / FaceNet.
    """
    if not RECOGNITION_AVAILABLE:
        raise RuntimeError("TensorFlow / MTCNN / FaceNet are not installed in this Python "
                           "environment, so face recognition is unavailable.")
    database = load_database()
    if not database:
        raise RuntimeError("Face database missing. Run build_embeddings.py first.")
    detector, facenet = load_models()
    return process_faces(img_rgb, detector, facenet, database)


def main():
    init_database()
    render_header()

    mode = render_sidebar()
    render_metrics()
    st.markdown("<br>", unsafe_allow_html=True)

    # DIP Lab works without the face database / models (they load on demand)
    if mode == "🧪 DIP Lab":
        dip_lab_mode(recognize_fn=recognize_for_lab)
        return

    # Attendance records need no models
    if mode == "📊 View Attendance":
        attendance_view()
        return

    if not RECOGNITION_AVAILABLE:
        st.error("Face recognition is unavailable: TensorFlow / MTCNN / FaceNet could not be "
                 "imported in this Python environment.")
        st.info("Use Python 3.10 or 3.11 and run `pip install -r requirements.txt` to enable "
                "this page. The **🧪 DIP Lab** page works without them.")
        with st.expander("Technical details"):
            st.code(RECOGNITION_IMPORT_ERROR)
        return

    # Load Models & DB inside main
    detector, facenet = load_models()
    database = load_database()

    if not database:
        st.error("Database missing! Run `build_embeddings.py` first.")
        if st.button("🔄 Try Reloading Database"):
            load_database.clear()
            st.rerun()
        return

    if mode == "📷 Camera Mode":
        camera_mode(detector, facenet, database)
    elif mode == "🖼️ Upload Image":
        upload_mode(detector, facenet, database)


if __name__ == "__main__":
    main()
