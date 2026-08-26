"""
Central configuration for the Face Recognition pipeline.
Edit these paths / thresholds to match your environment and dataset.
"""
import os

# ---------------------------------------------------------------------------
# Paths (all relative to the project root by default -> fully portable,
# no hardcoded machine-specific paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(PROJECT_ROOT, "data", "train")      # data/train/<person_name>/*.jpg
EMBEDDINGS_DB_PATH = os.path.join(PROJECT_ROOT, "data", "embeddings_db.pkl")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")           # data/test/<person_name>/*.jpg (+ optional "unknown/")

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "Facenet512"
DETECTOR_BACKEND = "mtcnn"       # opencv | mtcnn | retinaface | ssd | mediapipe ...
ENFORCE_DETECTION = False

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# Stage 1 - appearance (embedding) distance. Lower = more similar.
# DeepFace/Facenet512 uses cosine distance; empirically < ~0.30 is a strong match.
EMBEDDING_MATCH_THRESHOLD = 0.30

# Anything between MATCH and UNCERTAIN thresholds is borderline and gets
# routed through the landmark (Stage 2) verification instead of being
# accepted or rejected outright.
EMBEDDING_UNCERTAIN_THRESHOLD = 0.45

# Stage 2 - geometric landmark distance (normalized Euclidean distance on
# MediaPipe FaceMesh-derived facial ratios). At or below this -> geometry
# agrees with the appearance match -> confirm identity.
LANDMARK_MATCH_THRESHOLD = 0.15
