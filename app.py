"""
app.py
-------
Streamlit front-end for the two-stage face recognition pipeline
(FaceNet512 appearance embeddings + MediaPipe facial-landmark geometry).

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub (include data/embeddings_db.pkl and
       models/face_landmarker.task - both are small and should be committed
       so the app doesn't have to rebuild the database on every boot).
    2. On streamlit.io/cloud, point a new app at this repo, main file app.py.
    3. Make sure packages.txt (system deps) and requirements.txt are present
       at the repo root - Streamlit Cloud reads both automatically.
"""

import os
import io
import tempfile

import streamlit as st
from PIL import Image
import cv2
import numpy as np

import config
from src.database import load_database, build_database
from src.verifier import FaceVerifier

st.set_page_config(
    page_title="Face Recognition — FaceNet512 + Landmark Verification",
    page_icon="🧑‍🤝‍🧑",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Cached resources - loaded once per session, not on every rerun/interaction
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading face database...")
def get_database(db_path):
    return load_database(db_path)


@st.cache_resource(show_spinner="Warming up FaceNet512 + MediaPipe models...")
def get_verifier(_database):
    # Streamlit's cache_resource hashes arguments by default; the database
    # dict can be large, so we prefix with "_" to tell Streamlit to skip
    # hashing it and just cache on first call per session.
    return FaceVerifier(_database, config)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_temp_image(pil_image: Image.Image) -> str:
    """DeepFace/verifier expect a file path, not an in-memory array, so
    probe images from the uploader are written to a short-lived temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    pil_image.convert("RGB").save(tmp.name, format="JPEG", quality=95)
    return tmp.name


def status_badge(status: str) -> str:
    return {
        "match": "🟢 **MATCH**",
        "uncertain": "🟠 **UNCERTAIN**",
        "no_match": "🔴 **NO MATCH**",
    }.get(status, status)


# --------------------------------------------------------------------------- #
# Sidebar - database info + threshold controls
# --------------------------------------------------------------------------- #

st.sidebar.title("⚙️ Pipeline settings")

db_exists = os.path.exists(config.EMBEDDINGS_DB_PATH)

if not db_exists:
    st.sidebar.error(
        f"No database found at `{config.EMBEDDINGS_DB_PATH}`.\n\n"
        "Build one from the **Manage database** tab before recognizing faces."
    )
else:
    database = get_database(config.EMBEDDINGS_DB_PATH)
    n_people = len(database)
    n_refs = sum(len(v) for v in database.values())
    st.sidebar.success(f"Database loaded: **{n_people}** identities, **{n_refs}** reference images")

st.sidebar.markdown("---")
st.sidebar.markdown("**Detection thresholds**")
st.sidebar.caption(
    "These override `config.py` for this session only — tune them here to "
    "see the effect live, then bake the final values into config.py."
)

emb_match = st.sidebar.slider(
    "Embedding match threshold", 0.05, 0.60, float(config.EMBEDDING_MATCH_THRESHOLD), 0.01,
    help="Cosine distance below which Stage 1 (appearance) is confident.",
)
emb_uncertain = st.sidebar.slider(
    "Embedding uncertain threshold", emb_match, 0.80, float(config.EMBEDDING_UNCERTAIN_THRESHOLD), 0.01,
    help="Above this, reject outright without running Stage 2.",
)
landmark_match = st.sidebar.slider(
    "Landmark match threshold", 0.02, 0.40, float(config.LANDMARK_MATCH_THRESHOLD), 0.01,
    help="Normalized geometry distance Stage 2 requires to agree.",
)

# NOTE: these are passed as per-call overrides to verifier.verify() below,
# NOT written into the shared `config` module. `config` is imported once per
# server process and `get_verifier`'s cache is shared across every visitor -
# mutating it here would leak one user's slider settings into everyone
# else's concurrent session. See src/verifier.py's verify() signature.

st.sidebar.markdown("---")
st.sidebar.caption(
    "Stage 1: FaceNet512 appearance embedding match.\n\n"
    "Stage 2: MediaPipe facial-landmark geometry veto — rejects confident "
    "appearance matches whose facial proportions disagree, and can confirm "
    "borderline appearance matches whose geometry agrees."
)

# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #

tab_recognize, tab_manage, tab_about = st.tabs(["🔍 Recognize", "🗂️ Manage database", "ℹ️ About"])

# --- Recognize tab --------------------------------------------------------- #
with tab_recognize:
    st.header("Identify a face")

    if not db_exists:
        st.warning("Build or upload a database first (see the **Manage database** tab).")
    else:
        source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)

        pil_image = None
        if source == "Upload a photo":
            uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                pil_image = Image.open(uploaded)
        else:
            camera_file = st.camera_input("Take a photo")
            if camera_file is not None:
                pil_image = Image.open(camera_file)

        if pil_image is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(pil_image, caption="Probe image", use_container_width=True)

            with st.spinner("Running two-stage verification..."):
                temp_path = save_temp_image(pil_image)
                try:
                    verifier = get_verifier(database)
                    result = verifier.verify(
                        temp_path,
                        embedding_match_threshold=emb_match,
                        embedding_uncertain_threshold=emb_uncertain,
                        landmark_match_threshold=landmark_match,
                    )
                finally:
                    os.unlink(temp_path)

            with col2:
                st.markdown(f"### {status_badge(result.status)}")
                if result.identity:
                    st.markdown(f"**Identity:** {result.identity}")
                st.markdown(f"**Embedding distance:** `{result.embedding_distance:.4f}`"
                             if result.embedding_distance is not None else "**Embedding distance:** n/a")
                st.markdown(f"**Landmark distance:** `{result.landmark_distance:.4f}`"
                             if result.landmark_distance is not None else "**Landmark distance:** n/a")
                st.caption(result.reason)

            st.markdown("#### Top candidates (appearance only)")
            if result.all_candidates:
                st.table(
                    {
                        "Person": [p for p, _ in result.all_candidates],
                        "Embedding distance": [f"{d:.4f}" for _, d in result.all_candidates],
                    }
                )

# --- Manage database tab ---------------------------------------------------- #
with tab_manage:
    st.header("Database management")

    st.markdown(
        "The database (`data/embeddings_db.pkl`) stores precomputed FaceNet512 "
        "embeddings and MediaPipe geometry signatures for every reference image, "
        "so recognition doesn't have to recompute them on every request."
    )

    if db_exists:
        database = get_database(config.EMBEDDINGS_DB_PATH)
        with st.expander(f"View current identities ({len(database)})"):
            for person, entries in sorted(database.items()):
                st.write(f"- **{person}** — {len(entries)} reference image(s)")

    st.markdown("---")
    st.subheader("Add a new person")
    st.caption(
        "Upload a few clear, front-facing photos of one person. They'll be "
        "added to `data/train/<name>/` and the database will be rebuilt for "
        "just this person (existing identities are left untouched)."
    )
    st.info(
        ":warning: On Streamlit Community Cloud, storage is **ephemeral** - "
        "anything added here is written to the running container's disk and "
        "is lost on redeploy/restart. It's fine for a live demo, but for a "
        "permanent addition, add the photos to `data/train/` locally, rerun "
        "`python build_database.py`, and commit the updated "
        "`data/embeddings_db.pkl`.",
        icon=":material/info:",
    )

    new_name = st.text_input("Person's name")
    new_images = st.file_uploader(
        "Reference photos (2-10 recommended)", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button("Add person to database", type="primary", disabled=not (new_name and new_images)):
        person_dir = os.path.join(config.DATABASE_DIR, new_name.strip())
        os.makedirs(person_dir, exist_ok=True)

        for i, f in enumerate(new_images):
            img = Image.open(f).convert("RGB")
            img.save(os.path.join(person_dir, f"{new_name.strip()}_{i}.jpg"), format="JPEG", quality=95)

        try:
            with st.spinner(f"Computing embeddings + landmarks for {new_name}..."):
                build_database(
                    database_dir=config.DATABASE_DIR,
                    output_path=config.EMBEDDINGS_DB_PATH,
                    embedding_model=config.EMBEDDING_MODEL,
                    detector_backend=config.DETECTOR_BACKEND,
                )
        except Exception as e:
            st.error(f"Couldn't rebuild the database: {e}")
        else:
            st.cache_resource.clear()
            st.success(f"Added **{new_name}** and rebuilt the database.")
            st.rerun()

    st.markdown("---")
    st.subheader("Rebuild entire database")
    st.caption(
        "Re-scans every folder in `data/train/` from scratch. Use this after "
        "manually adding/removing files outside the app. Requires "
        "`data/train/` to exist with at least one person's photos - it "
        "won't exist on a fresh deploy since raw images aren't committed "
        "to the repo (see `.gitignore`)."
    )
    if st.button("Rebuild from data/train/"):
        if not os.path.isdir(config.DATABASE_DIR):
            st.error(
                f"`{config.DATABASE_DIR}` doesn't exist. Add at least one "
                "person via the form above first, or add photos locally "
                "and redeploy."
            )
        else:
            try:
                with st.spinner("Rebuilding full database - this can take a while..."):
                    build_database(
                        database_dir=config.DATABASE_DIR,
                        output_path=config.EMBEDDINGS_DB_PATH,
                        embedding_model=config.EMBEDDING_MODEL,
                        detector_backend=config.DETECTOR_BACKEND,
                    )
            except Exception as e:
                st.error(f"Rebuild failed: {e}")
            else:
                st.cache_resource.clear()
                st.success("Database rebuilt.")
                st.rerun()

# --- About tab --------------------------------------------------------------- #
with tab_about:
    st.header("About this pipeline")
    st.markdown(
        """
**Stage 1 — Appearance (FaceNet512):** DeepFace computes a 512-d embedding
for the probe image and finds the nearest reference image by cosine distance.

**Stage 2 — Geometry (MediaPipe landmarks):** For the best appearance
candidate, MediaPipe's FaceLandmarker extracts a normalized facial geometry
signature (ratios like eye spacing, jaw width, nose-to-chin distance) and
compares it against the candidate's reference images. Confident appearance
matches whose geometry disagrees are rejected as likely false positives;
borderline appearance matches whose geometry agrees get promoted to a
confirmed match.

This two-stage design is what reduces false positives compared to an
appearance-only baseline — see `evaluate.py` in the repo for a script that
measures this on your own test set.
        """
    )
    st.markdown("---")
    st.caption(
        "Adjust thresholds in the sidebar to explore the trade-off between "
        "false positives and false negatives for your dataset."
    )
