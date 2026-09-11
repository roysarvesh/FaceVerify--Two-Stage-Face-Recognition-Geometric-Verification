"""
verifier.py
------------
Two-stage face verification pipeline:

  Stage 1 (appearance) - FaceNet512 embedding distance against every
                          reference image in the database -> best candidate.

  Stage 2 (geometry)   - MediaPipe facial-landmark signature distance
                          between the probe and the best candidate's
                          reference image(s). This stage exists purely to
                          VETO appearance-only matches whose facial geometry
                          disagrees - i.e. it filters out the false
                          positives that a Stage-1-only pipeline would
                          accept (look-alikes, poor-quality crops, makeup,
                          lighting-driven embedding drift, etc.).

The result of verify() is one of: "match", "uncertain", "no_match".
"""

import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .embedding_engine import EmbeddingEngine
from .landmark_engine import LandmarkEngine, LandmarkExtractionError
from .learned_gate import LearnedGate


@dataclass
class VerificationResult:
    status: str                                  # "match" | "uncertain" | "no_match"
    identity: Optional[str]
    embedding_distance: Optional[float]
    landmark_distance: Optional[float] = None
    reason: str = ""
    all_candidates: List[Tuple[str, float]] = field(default_factory=list)  # top matches, appearance-only
    match_probability: Optional[float] = None    # only set when the learned Stage-2 gate is used


class FaceVerifier:
    def __init__(self, database, cfg):
        self.database = database
        self.cfg = cfg
        self.embedder = EmbeddingEngine(
            model_name=cfg.EMBEDDING_MODEL,
            detector_backend=cfg.DETECTOR_BACKEND,
            enforce_detection=cfg.ENFORCE_DETECTION,
        )
        self.landmarker = LandmarkEngine()

        # Optional: a trained logistic-regression Stage-2 gate (see
        # train_classifier.py), used instead of the nested-threshold rule
        # when enabled. Loading is best-effort - if the model file doesn't
        # exist (e.g. train_classifier.py hasn't been run yet), we silently
        # fall back to the rule-based thresholds rather than erroring, so
        # this is a strict opt-in that never breaks an existing deployment.
        self.learned_gate = None
        if getattr(cfg, "USE_LEARNED_STAGE2", False):
            model_path = getattr(cfg, "LEARNED_STAGE2_MODEL_PATH", None)
            if model_path:
                try:
                    self.learned_gate = LearnedGate(model_path)
                except Exception:
                    self.learned_gate = None

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------ #
    # Stage 1 - appearance
    # ------------------------------------------------------------------ #
    def _best_appearance_match(self, probe_embedding):
        best_person, best_distance = None, float("inf")
        ranked = []

        for person, entries in self.database.items():
            person_best = min(
                EmbeddingEngine.cosine_distance(probe_embedding, e["embedding"])
                for e in entries
            )
            ranked.append((person, person_best))
            if person_best < best_distance:
                best_distance, best_person = person_best, person

        ranked.sort(key=lambda x: x[1])
        return best_person, best_distance, ranked[:5]

    # ------------------------------------------------------------------ #
    # Stage 2 - geometry
    # ------------------------------------------------------------------ #
    def _geometry_distance(self, probe_image_bgr, candidate_person):
        try:
            probe_sig, _ = self.landmarker.signature(probe_image_bgr)
        except LandmarkExtractionError:
            # Can't run the geometry veto -> caller falls back to
            # appearance-only, but the result records that this happened.
            return None, "probe_landmarks_unavailable"

        entries = self.database[candidate_person]
        distances = [
            LandmarkEngine.distance(probe_sig, e["geometry"])
            for e in entries if e["geometry"] is not None
        ]
        if not distances:
            return None, "reference_landmarks_unavailable"

        return min(distances), "ok"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def verify(self, probe_image_path, embedding_match_threshold=None,
               embedding_uncertain_threshold=None, landmark_match_threshold=None):
        """
        Runs the two-stage pipeline on a probe image.

        The three threshold arguments are optional per-call overrides for
        `self.cfg.EMBEDDING_MATCH_THRESHOLD` / `EMBEDDING_UNCERTAIN_THRESHOLD`
        / `LANDMARK_MATCH_THRESHOLD`. Passing them here (rather than mutating
        `self.cfg` in place) keeps threshold tweaks local to a single call -
        important in a multi-user context like a Streamlit app, where `cfg`
        is a shared, cached object and mutating it would leak one user's
        slider settings into every other concurrent session.
        """
        match_thr = embedding_match_threshold if embedding_match_threshold is not None \
            else self.cfg.EMBEDDING_MATCH_THRESHOLD
        uncertain_thr = embedding_uncertain_threshold if embedding_uncertain_threshold is not None \
            else self.cfg.EMBEDDING_UNCERTAIN_THRESHOLD
        landmark_thr = landmark_match_threshold if landmark_match_threshold is not None \
            else self.cfg.LANDMARK_MATCH_THRESHOLD

        probe_embedding = self.embedder.embed(probe_image_path)
        best_person, best_distance, ranked = self._best_appearance_match(probe_embedding)

        if best_person is None or best_distance > uncertain_thr:
            return VerificationResult(
                status="no_match", identity=None, embedding_distance=best_distance,
                reason="No candidate within the embedding distance threshold.",
                all_candidates=ranked,
            )

        probe_image = cv2.imread(probe_image_path)
        landmark_distance, note = self._geometry_distance(probe_image, best_person)

        # --- Learned Stage-2 gate (opt-in) ------------------------------
        # Only used once Stage 1 has already narrowed things down to a
        # plausible candidate (best_distance <= uncertain_thr, checked
        # above) - the classifier was trained on realistic "nearest wrong
        # candidate" distances, not arbitrary out-of-range ones, so we
        # don't hand it degenerate inputs it never saw during training.
        if self.learned_gate is not None and landmark_distance is not None:
            is_match, probability = self.learned_gate.is_match(best_distance, landmark_distance)
            margin = getattr(self.cfg, "LEARNED_STAGE2_UNCERTAIN_MARGIN", 0.1)
            if abs(probability - self.learned_gate.threshold) <= margin:
                status = "uncertain"
                reason = (f"Learned classifier probability ({probability:.2f}) is close to its "
                          f"decision threshold ({self.learned_gate.threshold:.2f}) - too close to call.")
            elif is_match:
                status, reason = "match", f"Learned classifier: probability={probability:.2f} >= threshold."
            else:
                status, reason = "no_match", f"Learned classifier: probability={probability:.2f} < threshold."
            return VerificationResult(
                status=status,
                identity=best_person if status != "no_match" else None,
                embedding_distance=best_distance, landmark_distance=landmark_distance,
                reason=reason, all_candidates=ranked, match_probability=probability,
            )

        # --- Confident appearance match (hand-tuned rule) ---------------
        if best_distance <= match_thr:
            if landmark_distance is None:
                return VerificationResult(
                    status="match", identity=best_person, embedding_distance=best_distance,
                    landmark_distance=None,
                    reason=f"Strong appearance match; geometry check skipped ({note}).",
                    all_candidates=ranked,
                )
            if landmark_distance <= landmark_thr:
                return VerificationResult(
                    status="match", identity=best_person, embedding_distance=best_distance,
                    landmark_distance=landmark_distance,
                    reason="Appearance and facial geometry both agree.",
                    all_candidates=ranked,
                )
            # Embeddings looked close, but facial proportions disagree -
            # reject instead of returning a false positive.
            return VerificationResult(
                status="no_match", identity=None, embedding_distance=best_distance,
                landmark_distance=landmark_distance,
                reason=(f"Embedding suggested '{best_person}', but facial geometry "
                        f"disagreed (distance {landmark_distance:.3f} > "
                        f"{landmark_thr}) - rejected as a likely false positive."),
                all_candidates=ranked,
            )

        # --- Borderline appearance match -> let geometry decide ---------
        if landmark_distance is not None and landmark_distance <= landmark_thr:
            return VerificationResult(
                status="match", identity=best_person, embedding_distance=best_distance,
                landmark_distance=landmark_distance,
                reason="Borderline appearance match confirmed by facial geometry.",
                all_candidates=ranked,
            )

        return VerificationResult(
            status="uncertain", identity=best_person, embedding_distance=best_distance,
            landmark_distance=landmark_distance,
            reason="Appearance match was borderline and geometry could not confirm it.",
            all_candidates=ranked,
        )
