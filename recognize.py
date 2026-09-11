"""
recognize.py
-------------
CLI entry point: identify the person in a probe image using the two-stage
(FaceNet512 embedding + MediaPipe landmark) verification pipeline, print a
report, and save an annotated copy of the image.

Usage:
    python recognize.py --image path/to/photo.jpg
    python recognize.py --image path/to/photo.jpg --db data/embeddings_db.pkl
"""

import os
import argparse
import cv2

import config
from src.database import load_database
from src.verifier import FaceVerifier


def annotate(image_path, result, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    if result.status == "match":
        label = f"{result.identity} (emb={result.embedding_distance:.3f}"
        if result.landmark_distance is not None:
            label += f", geo={result.landmark_distance:.3f}"
        if result.match_probability is not None:
            label += f", p={result.match_probability:.2f}"
        label += ")"
        color = (0, 200, 0)
    elif result.status == "uncertain":
        label = f"Uncertain: {result.identity}?"
        color = (0, 165, 255)
    else:
        label = "No match"
        color = (0, 0, 255)

    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage face recognition (FaceNet512 embeddings + MediaPipe landmarks)."
    )
    parser.add_argument("--image", required=True, help="Path to the probe image.")
    parser.add_argument("--db", default=config.EMBEDDINGS_DB_PATH,
                         help="Path to the prebuilt embeddings database (.pkl).")
    parser.add_argument("--output", default=None, help="Where to save the annotated image.")
    args = parser.parse_args()

    database = load_database(args.db)

    with FaceVerifier(database, config) as verifier:
        result = verifier.verify(args.image)

    print("\n=== Recognition Result ===")
    print(f"Status:             {result.status}")
    print(f"Identity:           {result.identity}")
    print(f"Embedding distance: {result.embedding_distance}")
    print(f"Landmark distance:  {result.landmark_distance}")
    if result.match_probability is not None:
        print(f"Match probability:  {result.match_probability:.3f}  (learned Stage-2 classifier)")
    print(f"Reason:             {result.reason}")
    print("\nTop candidates (appearance only):")
    for person, dist in result.all_candidates:
        print(f"  {person:20s} {dist:.4f}")

    output_path = args.output or os.path.join(
        os.path.dirname(args.image) or ".", "recognized_" + os.path.basename(args.image)
    )
    saved = annotate(args.image, result, output_path)
    print(f"\nAnnotated image saved to: {saved}")


if __name__ == "__main__":
    main()
