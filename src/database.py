"""
database.py
------------
Builds and loads the face database. For every person folder under
<database_dir>, precomputes and stores:
  - a FaceNet512 embedding for each reference image      (appearance)
  - a MediaPipe geometric signature for each reference image (geometry)

so that recognition time only needs a single pass over the probe image
instead of recomputing every reference embedding/signature each run.
"""

import os
import pickle
import cv2
from tqdm import tqdm

from .embedding_engine import EmbeddingEngine
from .landmark_engine import LandmarkEngine, LandmarkExtractionError

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_database(database_dir, output_path, embedding_model="Facenet512",
                    detector_backend="opencv"):
    if not os.path.isdir(database_dir):
        raise FileNotFoundError(
            f"Database directory not found: {database_dir}\n"
            f"Expected layout: {database_dir}/<person_name>/<image files>"
        )

    embedder = EmbeddingEngine(model_name=embedding_model, detector_backend=detector_backend)
    db = {}  # person -> list of {"path": str, "embedding": np.ndarray, "geometry": np.ndarray | None}

    with LandmarkEngine() as landmarker:
        people = sorted(
            d for d in os.listdir(database_dir)
            if os.path.isdir(os.path.join(database_dir, d))
        )

        if not people:
            raise FileNotFoundError(
                f"No person sub-folders found under {database_dir}. "
                f"Organize images as {database_dir}/<person_name>/*.jpg"
            )

        for person in tqdm(people, desc="Building database"):
            person_dir = os.path.join(database_dir, person)
            entries = []

            for fname in os.listdir(person_dir):
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                fpath = os.path.join(person_dir, fname)

                try:
                    embedding = embedder.embed(fpath)
                except Exception as e:
                    print(f"  [skip] {fpath}: embedding failed ({e})")
                    continue

                geometry = None
                try:
                    image = cv2.imread(fpath)
                    geometry, _ = landmarker.signature(image)
                except LandmarkExtractionError as e:
                    print(f"  [warn] {fpath}: no landmarks ({e}) "
                          f"- geometry check will be skipped for this reference image")

                entries.append({"path": fpath, "embedding": embedding, "geometry": geometry})

            if entries:
                db[person] = entries
            else:
                print(f"  [warn] no usable images found for '{person}' - skipped")

    if not db:
        raise RuntimeError("No identities were successfully processed; database is empty.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(db, f)

    n_refs = sum(len(v) for v in db.values())
    print(f"\nSaved database with {len(db)} identities ({n_refs} reference images) -> {output_path}")
    return db


def load_database(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Database not found at {path}. Run build_database.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
