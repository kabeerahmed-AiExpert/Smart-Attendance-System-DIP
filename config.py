"""
================================================================================
  Project Configuration
================================================================================
  Shared constants used by build_embeddings.py and evaluate_system.py.
  (The web app itself keeps its own constants in face_detection.py and
  recognition.py.)

  Dataset layout expected by the scripts:

      dataset/
      ├── F24ARI129_Kabeer/    <- one folder per student: <StudentID>_<Name>
      │   ├── img1.jpg
      │   └── img2.jpg
      └── F24ARI131_SaifUllah/
          └── ...
================================================================================
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "face_embeddings.pkl")

# Face detection / FaceNet input
FACENET_INPUT_SIZE = (160, 160)
EMBEDDING_DIM = 512  # keras-facenet returns 512-d embeddings
MTCNN_CONFIDENCE_THRESHOLD = 0.90
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Recognition thresholds
PRIMARY_METRIC = "cosine"  # "cosine" or "euclidean"
COSINE_SIMILARITY_THRESHOLD = 0.70  # same value as recognition.py
EUCLIDEAN_DISTANCE_THRESHOLD = 1.0

# Reproducibility
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
