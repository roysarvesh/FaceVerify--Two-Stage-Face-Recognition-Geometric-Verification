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

# ---------------------------------------------------------------------------
# Learned Stage 2 (optional) - see train_classifier.py
# ---------------------------------------------------------------------------
# If True and the model file below exists, FaceVerifier uses a trained
# logistic-regression classifier over [embedding_distance, landmark_distance]
# instead of the nested EMBEDDING_MATCH_THRESHOLD/LANDMARK_MATCH_THRESHOLD
# rule above. Run `python train_classifier.py` first to produce the model
# file and an honest cross-validated ROC/EER report justifying it - see
# data/evaluation_report.md after running it.
#
# On this dataset the evaluation showed the hand-tuned rule above sitting
# well off the achievable ROC curve (27% FPR / 23% FNR vs. ~1% FPR at
# ~93% TPR available from the same two features, properly calibrated) -
# so the learned classifier is enabled by default. See
# data/evaluation_report.md for the honest headline finding: geometry adds
# almost nothing beyond the embedding distance alone on this dataset
# (AUC 0.992 vs 0.993) - the fitted classifier's own weights reflect that
# (embedding weighted ~14x more heavily than landmark). The real
# improvement here isn't "two stages beat one" - it's "a data-calibrated
# threshold beats a hand-picked one."
USE_LEARNED_STAGE2 = True
LEARNED_STAGE2_MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "stage2_classifier.json")

# Probabilities within this margin of the learned model's decision
# threshold are reported as "uncertain" rather than a hard match/no_match -
# mirrors the spirit of the original three-way rule-based outcome.
LEARNED_STAGE2_UNCERTAIN_MARGIN = 0.10
