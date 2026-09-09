"""
embedding_engine.py
--------------------
Thin wrapper around DeepFace that computes FaceNet512 appearance embeddings
and compares them. This is "Stage 1" of the pipeline: fast, appearance-based
nearest-neighbor matching against the reference database.
"""

import numpy as np


class EmbeddingEngine:
    def __init__(self, model_name="Facenet512", detector_backend="opencv",
                 enforce_detection=False):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection

    def embed(self, image_path):
        """Returns the FaceNet512 embedding (np.ndarray) for the primary face
        found in the image."""
        # Imported lazily, not at module level: `deepface` pulls in
        # TensorFlow, which takes several seconds to import. Deferring this
        # until the first actual embedding request means the app's UI
        # (sidebar, tabs, database stats) renders immediately on load
        # instead of blocking on a TensorFlow import nobody asked for yet.
        from deepface import DeepFace

        reps = DeepFace.represent(
            img_path=image_path,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
        )
        if not reps:
            raise ValueError(f"No face found in {image_path}")
        return np.array(reps[0]["embedding"], dtype=np.float64)

    @staticmethod
    def cosine_distance(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return float(1.0 - np.dot(a, b) / denom)
