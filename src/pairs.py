"""
pairs.py
---------
Builds labeled (genuine / impostor) verification pairs from held-out test
images against the training reference database, for training and honestly
evaluating a learned Stage-2 decision rule.

Methodology (mirrors LFW-style face verification benchmarking, adapted to
this project's two-stage pipeline):

For every test image with known identity `y_true`, we embed it once and
compare it against every person in the training database - exactly what
`FaceVerifier._best_appearance_match` does at inference time. From that we
derive two samples:

  - GENUINE:  candidate = y_true.       label = 1
  - IMPOSTOR: candidate = nearest WRONG person by embedding distance.
              label = 0

The impostor is deliberately the *hardest* wrong candidate (the one Stage 1
would actually propose if it were fooled), not a random unrelated person -
a random impostor would be trivially easy to reject and would make Stage 2
look better than it really is. This choice ties pair generation directly
to what the live pipeline actually has to get right.

No test image is ever used to build the training database (see
config.DATABASE_DIR vs config.TEST_DIR) - there is no leakage between the
reference embeddings and the evaluation pairs.
"""

import os

import numpy as np
from tqdm import tqdm

from .embedding_engine import EmbeddingEngine
from .landmark_engine import LandmarkEngine, LandmarkExtractionError

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _best_distance_and_landmark(person_entries, probe_embedding, probe_geometry):
    """Given one person's reference entries, returns (best embedding
    distance, landmark distance to that same best-matching reference)."""
    best_idx, best_dist = None, float("inf")
    for i, entry in enumerate(person_entries):
        d = EmbeddingEngine.cosine_distance(probe_embedding, entry["embedding"])
        if d < best_dist:
            best_dist, best_idx = d, i

    landmark_dist = None
    if probe_geometry is not None and person_entries[best_idx]["geometry"] is not None:
        landmark_dist = LandmarkEngine.distance(probe_geometry, person_entries[best_idx]["geometry"])

    return best_dist, landmark_dist


def build_pairs(train_database, test_dir, embedding_model="Facenet512",
                 detector_backend="mtcnn"):
    """
    Returns a list of dicts, one per test image, each containing BOTH the
    genuine and impostor sample derived from that image:
        {
            "image_path": str,
            "true_person": str,
            "genuine_embedding_distance": float,
            "genuine_landmark_distance": float | None,
            "impostor_person": str,
            "impostor_embedding_distance": float,
            "impostor_landmark_distance": float | None,
        }
    Flatten with `flatten_pairs()` below to get one row per sample for
    training/evaluation.
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    embedder = EmbeddingEngine(model_name=embedding_model, detector_backend=detector_backend)
    results = []

    with LandmarkEngine() as landmarker:
        people = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))

        for true_person in tqdm(people, desc="Mining pairs"):
            if true_person not in train_database:
                continue  # can't form a genuine pair without training references
            person_dir = os.path.join(test_dir, true_person)

            for fname in os.listdir(person_dir):
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                fpath = os.path.join(person_dir, fname)

                try:
                    probe_embedding = embedder.embed(fpath)
                except Exception:
                    continue

                probe_geometry = None
                try:
                    import cv2
                    image = cv2.imread(fpath)
                    probe_geometry, _ = landmarker.signature(image)
                except LandmarkExtractionError:
                    pass
                except Exception:
                    pass

                # Genuine: distance to the probe's own true identity.
                gen_emb_dist, gen_land_dist = _best_distance_and_landmark(
                    train_database[true_person], probe_embedding, probe_geometry
                )

                # Impostor: nearest WRONG identity - the realistic adversarial case.
                impostor_person, impostor_emb_dist, impostor_land_dist = None, float("inf"), None
                for other_person, entries in train_database.items():
                    if other_person == true_person:
                        continue
                    d, land_d = _best_distance_and_landmark(entries, probe_embedding, probe_geometry)
                    if d < impostor_emb_dist:
                        impostor_emb_dist, impostor_land_dist, impostor_person = d, land_d, other_person

                results.append({
                    "image_path": fpath,
                    "true_person": true_person,
                    "genuine_embedding_distance": gen_emb_dist,
                    "genuine_landmark_distance": gen_land_dist,
                    "impostor_person": impostor_person,
                    "impostor_embedding_distance": impostor_emb_dist,
                    "impostor_landmark_distance": impostor_land_dist,
                })

    return results


def flatten_pairs(pairs, require_landmark=True):
    """
    Converts build_pairs() output into (X, y) arrays ready for sklearn:
    X columns = [embedding_distance, landmark_distance], y = 1 (genuine) / 0 (impostor).

    If require_landmark, samples where the landmark distance couldn't be
    computed (e.g. no face landmarks detected) are dropped - a learned
    2-feature classifier needs both features present for every row.
    """
    X, y, meta = [], [], []
    for p in pairs:
        if p["genuine_landmark_distance"] is not None or not require_landmark:
            X.append([p["genuine_embedding_distance"], p["genuine_landmark_distance"] or 0.0])
            y.append(1)
            meta.append({"image_path": p["image_path"], "person": p["true_person"], "kind": "genuine"})

        if p["impostor_landmark_distance"] is not None or not require_landmark:
            X.append([p["impostor_embedding_distance"], p["impostor_landmark_distance"] or 0.0])
            y.append(0)
            meta.append({"image_path": p["image_path"], "person": p["impostor_person"], "kind": "impostor"})

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64), meta
