"""
landmark_engine.py
--------------------
Wraps MediaPipe FaceMesh to turn a face image into a compact, scale- and
translation-invariant "facial geometry signature": a vector of distances
between anatomically stable landmarks, each normalized by the inter-ocular
distance.

This signature is the second, independent verification signal used by the
pipeline ("Stage 2") to catch cases where two different people happen to
produce close FaceNet512 embeddings - look-alikes, low-quality crops,
lighting-driven embedding drift, makeup, etc. Appearance says "maybe";
geometry has to agree before the pipeline commits to an identity.
"""

import os
import urllib.request

import numpy as np
import cv2

# NOTE: mediapipe itself is imported lazily inside LandmarkEngine.__init__,
# not here at module level - see the comment there for why.

# MediaPipe's newer releases (>=0.10.x) ship face landmark detection through
# the Tasks API rather than the older `mp.solutions.face_mesh` module. The
# Tasks API needs a small model file, which is cached locally on first use.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model():
    """Downloads the FaceLandmarker model file once and caches it locally."""
    if os.path.exists(_MODEL_PATH):
        return _MODEL_PATH

    os.makedirs(_MODEL_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not download the MediaPipe FaceLandmarker model "
            f"({_MODEL_URL}). Check your internet connection, or manually "
            f"download the file and place it at {_MODEL_PATH}. "
            f"Original error: {e}"
        )
    return _MODEL_PATH


# Anatomically stable landmark indices (unified 468/478-point face topology
# used by both the legacy FaceMesh solution and the current FaceLandmarker
# task, so these indices are valid either way)
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
NOSE_BRIDGE = 6
CHIN = 152
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_EYEBROW = 105
RIGHT_EYEBROW = 334
UPPER_LIP = 13
LOWER_LIP = 14


class LandmarkExtractionError(Exception):
    """Raised when no face / usable landmarks can be extracted from an image."""
    pass


class LandmarkEngine:
    """Extracts a geometric signature from a BGR image using MediaPipe's
    FaceLandmarker (Tasks API)."""

    def __init__(self, max_num_faces=1, min_detection_confidence=0.5):
        # Imported lazily, not at module level: mediapipe's import (and the
        # native library it loads) takes real time, and this class is only
        # ever instantiated when the two-stage pipeline is actually about to
        # run a verification - not at app startup. Deferring the import
        # until here means the app's UI renders immediately on load instead
        # of blocking on mediapipe's import chain nobody has asked for yet.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        self._mp = mp
        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def _get_landmarks(self, image_bgr):
        if image_bgr is None:
            raise LandmarkExtractionError("Empty image.")

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise LandmarkExtractionError("No face landmarks detected.")

        landmarks = result.face_landmarks[0]
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)
        return points

    @staticmethod
    def _dist(p, a, b):
        return float(np.linalg.norm(p[a] - p[b]))

    def signature(self, image_bgr):
        """
        Returns (vector, named_features) - a normalized geometric feature
        vector describing facial proportions, plus a dict of the same values
        with human-readable keys (handy for debugging/inspection).
        """
        p = self._get_landmarks(image_bgr)

        interocular = self._dist(p, LEFT_EYE_OUTER, RIGHT_EYE_OUTER)
        if interocular < 1e-6:
            raise LandmarkExtractionError("Degenerate landmark geometry (interocular distance ~ 0).")

        features = {
            "eye_inner_gap":        self._dist(p, LEFT_EYE_INNER, RIGHT_EYE_INNER) / interocular,
            "left_eye_to_nose":     self._dist(p, LEFT_EYE_OUTER, NOSE_TIP) / interocular,
            "right_eye_to_nose":    self._dist(p, RIGHT_EYE_OUTER, NOSE_TIP) / interocular,
            "nose_to_chin":         self._dist(p, NOSE_TIP, CHIN) / interocular,
            "mouth_width":          self._dist(p, MOUTH_LEFT, MOUTH_RIGHT) / interocular,
            "lip_gap":              self._dist(p, UPPER_LIP, LOWER_LIP) / interocular,
            "cheek_width":          self._dist(p, LEFT_CHEEK, RIGHT_CHEEK) / interocular,
            "left_brow_to_eye":     self._dist(p, LEFT_EYEBROW, LEFT_EYE_OUTER) / interocular,
            "right_brow_to_eye":    self._dist(p, RIGHT_EYEBROW, RIGHT_EYE_OUTER) / interocular,
            "nose_bridge_to_tip":   self._dist(p, NOSE_BRIDGE, NOSE_TIP) / interocular,
            "chin_to_mouth_left":   self._dist(p, CHIN, MOUTH_LEFT) / interocular,
            "left_jaw_to_nose":     self._dist(p, LEFT_CHEEK, NOSE_TIP) / interocular,
            "right_jaw_to_nose":    self._dist(p, RIGHT_CHEEK, NOSE_TIP) / interocular,
        }

        vector = np.array(list(features.values()), dtype=np.float64)
        return vector, features

    @staticmethod
    def distance(sig_a, sig_b):
        """Normalized Euclidean distance between two geometry signatures."""
        sig_a = np.asarray(sig_a, dtype=np.float64)
        sig_b = np.asarray(sig_b, dtype=np.float64)
        return float(np.linalg.norm(sig_a - sig_b) / np.sqrt(len(sig_a)))

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
