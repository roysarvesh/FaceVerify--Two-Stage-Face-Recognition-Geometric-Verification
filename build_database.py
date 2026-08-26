import argparse
import config
from src.database import build_database


def main():
    parser = argparse.ArgumentParser(description="Build the face database (embeddings + landmarks).")
    parser.add_argument("--database-dir", default=config.DATABASE_DIR,
                         help="Folder containing one sub-folder of images per person.")
    parser.add_argument("--output", default=config.EMBEDDINGS_DB_PATH,
                         help="Where to save the resulting .pkl database.")
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--detector", default=config.DETECTOR_BACKEND)
    args = parser.parse_args()

    build_database(
        database_dir=args.database_dir,
        output_path=args.output,
        embedding_model=args.model,
        detector_backend=args.detector,
    )


if __name__ == "__main__":
    main()
