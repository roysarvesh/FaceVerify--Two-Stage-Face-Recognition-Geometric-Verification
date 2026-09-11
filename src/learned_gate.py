"""
learned_gate.py
-----------------
Lightweight inference-time wrapper around a Stage-2 classifier trained by
train_classifier.py. Deliberately dependency-free at inference: the
deployed app only needs three numbers and a sigmoid, not sklearn - so
using a learned classifier instead of hand-tuned thresholds adds zero
runtime dependencies.
"""

import json
import math


class LearnedGate:
    def __init__(self, model_path: str):
        with open(model_path) as f:
            m = json.load(f)
        self.w_embedding = m["w_embedding_distance"]
        self.w_landmark = m["w_landmark_distance"]
        self.bias = m["bias"]
        self.threshold = m["decision_threshold_probability"]

    def probability(self, embedding_distance: float, landmark_distance: float) -> float:
        z = self.w_embedding * embedding_distance + self.w_landmark * landmark_distance + self.bias
        return 1.0 / (1.0 + math.exp(-z))

    def is_match(self, embedding_distance: float, landmark_distance: float):
        p = self.probability(embedding_distance, landmark_distance)
        return p >= self.threshold, p
