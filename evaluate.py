"""
evaluate.py
------------
Benchmarks the two-stage pipeline against an appearance-only (Stage-1-only)
baseline on a labeled test set, and reports the false-positive rate for
each - this is how you reproduce a number like "reduced false positives by
40% vs. the baseline recognition pipeline" for your own dataset.

Expected layout for --test-dir:
    data/test/
        Ben Afflek/            <- images of Ben Afflek (ground-truth identity)
            1.jpg
            2.jpg
        Elton John/
            ...
        unknown/                <- OPTIONAL: images of people NOT in the database.
            1.jpg                  Any prediction here is a false positive.

Definitions used below:
    False positive  = pipeline confidently returns the WRONG identity
                       (a different in-DB person, or any identity at all
                       for an "unknown" probe).
    False negative   = pipeline returns "no match"/"uncertain" for a probe
                       that IS in the database.

Usage:
    python build_database.py            # once, to create the .pkl database
    python evaluate.py --test-dir data/test
"""

import os
import argparse

import config
from src.database import load_database
from src.verifier import FaceVerifier
from src.embedding_engine import EmbeddingEngine

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def baseline_predict(embedder, database, probe_path, threshold):
    """Stage-1-only baseline: pure FaceNet512 nearest-neighbor match."""
    probe_embedding = embedder.embed(probe_path)
    best_person, best_distance = None, float("inf")
    for person, entries in database.items():
        d = min(EmbeddingEngine.cosine_distance(probe_embedding, e["embedding"]) for e in entries)
        if d < best_distance:
            best_distance, best_person = d, person
    return best_person if best_person is not None and best_distance <= threshold else None


def _score(bucket, pred, true_identity, is_unknown):
    if is_unknown:
        bucket["tn" if pred is None else "fp"] += 1
    else:
        if pred == true_identity:
            bucket["tp"] += 1
        elif pred is None:
            bucket["fn"] += 1
        else:
            bucket["fp"] += 1


def run_evaluation(test_dir, db_path):
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            f"Expected layout: {test_dir}/<person_name>/<image files> (+ optional 'unknown/')"
        )

    database = load_database(db_path)
    embedder = EmbeddingEngine(model_name=config.EMBEDDING_MODEL,
                                detector_backend=config.DETECTOR_BACKEND,
                                enforce_detection=config.ENFORCE_DETECTION)

    stats = {
        "baseline": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "pipeline": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    }

    people = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
    if not people:
        raise FileNotFoundError(f"No person sub-folders found under {test_dir}.")

    with FaceVerifier(database, config) as verifier:
        for true_identity in people:
            person_dir = os.path.join(test_dir, true_identity)
            is_unknown = true_identity.lower() == "unknown"

            for fname in os.listdir(person_dir):
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                fpath = os.path.join(person_dir, fname)

                try:
                    pred_baseline = baseline_predict(embedder, database, fpath, config.EMBEDDING_MATCH_THRESHOLD)
                except Exception as e:
                    print(f"  [warn] baseline failed on {fpath}: {e}")
                    pred_baseline = None
                _score(stats["baseline"], pred_baseline, true_identity, is_unknown)

                try:
                    result = verifier.verify(fpath)
                    pred_pipeline = result.identity if result.status == "match" else None
                except Exception as e:
                    print(f"  [warn] pipeline failed on {fpath}: {e}")
                    pred_pipeline = None
                _score(stats["pipeline"], pred_pipeline, true_identity, is_unknown)

    return stats


def print_report(stats):
    print("\n=== Evaluation Report ===")
    for name in ("baseline", "pipeline"):
        s = stats[name]
        total = sum(s.values())
        fpr = s["fp"] / total if total else 0.0
        print(f"\n[{name}] TP={s['tp']}  FP={s['fp']}  TN={s['tn']}  FN={s['fn']}  "
              f"(false-positive rate: {fpr:.2%})")

    b_fp, p_fp = stats["baseline"]["fp"], stats["pipeline"]["fp"]
    if b_fp:
        reduction = (b_fp - p_fp) / b_fp
        print(f"\nFalse positives reduced by {reduction:.1%} "
              f"({b_fp} -> {p_fp}) vs. the appearance-only baseline.")
    else:
        print("\nBaseline produced 0 false positives on this test set - "
              "no reduction to report (try a harder/larger test set, "
              "e.g. add look-alike or 'unknown' probes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline vs. two-stage pipeline.")
    parser.add_argument("--test-dir", default=config.TEST_DIR)
    parser.add_argument("--db", default=config.EMBEDDINGS_DB_PATH)
    args = parser.parse_args()

    results = run_evaluation(args.test_dir, args.db)
    print_report(results)
