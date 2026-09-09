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
    3. Make sure packages.txt (system deps), requirements.txt, and
       runtime.txt (Python version) are present at the repo root - Streamlit
       Cloud reads all three automatically. See README.md for the specific
       platform issues these files work around.
"""

import hashlib
import os
import tempfile
import traceback

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import config
from src.database import build_database, load_database
from src.verifier import FaceVerifier

st.set_page_config(
    page_title="FaceVerify — Two-Stage Face Recognition",
    page_icon=":bust_in_silhouette:",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Styling - a premium dark theme layered on top of Streamlit's native
# components (which already pick up the dark base + primaryColor from
# .streamlit/config.toml). Real containers/columns/metrics still do the
# structural work; CSS adds card depth, glow accents, and refines
# typography/spacing on top of that.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .fv-hero {
        display: flex; align-items: center; gap: 14px;
        margin-bottom: 0.25rem;
    }
    .fv-hero-icon {
        font-size: 2.1rem; line-height: 1;
        filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.45));
    }
    .fv-hero h1 {
        font-size: 1.95rem; font-weight: 750; margin: 0; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #A5B4FC 0%, #E5E7EB 60%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .fv-subtitle {
        color: #94A3B8; font-size: 0.98rem; margin-top: 0.15rem; margin-bottom: 1.6rem;
    }

    .fv-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 16px; border-radius: 999px;
        font-weight: 650; font-size: 0.95rem;
        border: 1px solid transparent;
    }
    .fv-pill-match {
        background: rgba(34, 197, 94, 0.12); color: #4ADE80;
        border-color: rgba(74, 222, 128, 0.35);
        box-shadow: 0 0 16px rgba(34, 197, 94, 0.15);
    }
    .fv-pill-uncertain {
        background: rgba(245, 158, 11, 0.12); color: #FBBF24;
        border-color: rgba(251, 191, 36, 0.35);
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.12);
    }
    .fv-pill-nomatch {
        background: rgba(239, 68, 68, 0.12); color: #F87171;
        border-color: rgba(248, 113, 113, 0.35);
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.12);
    }

    .fv-stage-card {
        border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px;
        padding: 18px 20px; background: #131826; height: 100%;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }
    .fv-stage-card h4 { margin: 0 0 8px 0; font-size: 1.02rem; color: #E5E7EB; }
    .fv-stage-card p { margin: 0; color: #94A3B8; font-size: 0.9rem; line-height: 1.55; }

    .fv-footnote { color: #64748B; font-size: 0.82rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #131826; border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 14px 18px;
    }
    div[data-testid="stMetricValue"] { color: #E0E7FF; }

    /* Bordered st.container(border=True) cards */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #131826; border-radius: 14px;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Sliders - glow the thumb/track with the accent color for a
       premium feel; Streamlit already colors these from primaryColor,
       this just adds depth. */
    div[data-testid="stSlider"] div[role="slider"] {
        box-shadow: 0 0 0 5px rgba(99, 102, 241, 0.18);
    }
    div[data-testid="stSlider"] { padding-bottom: 6px; }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; }

    hr { border-color: rgba(255, 255, 255, 0.08) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Cached resources - loaded once per server process, not on every rerun
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading face database...")
def get_database(db_path):
    return load_database(db_path)


@st.cache_resource(show_spinner="Warming up FaceNet512 + MediaPipe models...")
def get_verifier(_database):
    # Streamlit's cache_resource hashes arguments by default; the database
    # dict can be large, so we prefix with "_" to tell Streamlit to skip
    # hashing it and just cache on first successful call per process.
    return FaceVerifier(_database, config)


@st.cache_data(show_spinner="Running two-stage verification...")
def run_verification(image_bytes: bytes, emb_match: float, emb_uncertain: float, landmark_match: float):
    """
    Cached by (image bytes, thresholds). This matters for more than just
    speed: Streamlit reruns every tab's code on every interaction regardless
    of which tab is visible, and a file_uploader's value persists across
    reruns. Without this cache, switching tabs after uploading a photo would
    silently re-run verification (and re-trigger any transient model-loading
    error) on every single rerun, not just when the photo changes.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        database = get_database(config.EMBEDDINGS_DB_PATH)
        verifier = get_verifier(database)
        return verifier.verify(
            tmp_path,
            embedding_match_threshold=emb_match,
            embedding_uncertain_threshold=emb_uncertain,
            landmark_match_threshold=landmark_match,
        ), None
    except Exception as e:
        return None, "".join(traceback.format_exception(type(e), e, e.__traceback__))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def status_pill(status: str, identity: str | None) -> str:
    if status == "match":
        return f'<span class="fv-pill fv-pill-match">✅ MATCH — {identity}</span>'
    if status == "uncertain":
        return f'<span class="fv-pill fv-pill-uncertain">⚠️ UNCERTAIN — possibly {identity}</span>'
    return '<span class="fv-pill fv-pill-nomatch">❌ NO MATCH</span>'


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="fv-hero">
        <div class="fv-hero-icon">🧑‍🤝‍🧑</div>
        <h1>FaceVerify</h1>
    </div>
    <div class="fv-subtitle">
        Two-stage face recognition — FaceNet512 appearance embeddings, verified
        against MediaPipe facial-landmark geometry.
    </div>
    """,
    unsafe_allow_html=True,
)

db_exists = os.path.exists(config.EMBEDDINGS_DB_PATH)
database = get_database(config.EMBEDDINGS_DB_PATH) if db_exists else None

# --------------------------------------------------------------------------- #
# Sidebar - database status + threshold controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Pipeline settings")

    if not db_exists:
        st.error(
            f"No database found at `{config.EMBEDDINGS_DB_PATH}`.\n\n"
            "Build one from the **Manage database** tab first."
        )
    else:
        n_people = len(database)
        n_refs = sum(len(v) for v in database.values())
        with st.container(border=True):
            st.markdown(f"**Database loaded:** {n_people} identities,  \n{n_refs} reference images")

        # Models (FaceNet512/TensorFlow + MediaPipe) load lazily on first
        # use, not at app startup, so the UI above appears instantly. This
        # button lets a user pay that one-time cost proactively - handy
        # right after a cold start, so the *first* actual recognition
        # isn't the one that eats the load time.
        if st.button("⚡ Warm up models", use_container_width=True,
                      help="Pre-loads FaceNet512 + MediaPipe now instead of on your first photo."):
            with st.spinner("Loading FaceNet512 + MediaPipe..."):
                get_verifier(database)
            st.toast("Models ready.", icon="⚡")

    st.divider()
    st.markdown("**Detection thresholds**")
    st.caption(
        "These override `config.py` for this session only — tune them here "
        "to see the effect live, then bake the final values into config.py."
    )

    emb_match = st.slider(
        "Embedding match threshold", 0.05, 0.60, float(config.EMBEDDING_MATCH_THRESHOLD), 0.01,
        help="Cosine distance below which Stage 1 (appearance) is confident.",
    )
    emb_uncertain = st.slider(
        "Embedding uncertain threshold", emb_match, 0.80,
        float(max(config.EMBEDDING_UNCERTAIN_THRESHOLD, emb_match)), 0.01,
        help="Above this, reject outright without running Stage 2.",
    )
    landmark_match = st.slider(
        "Landmark match threshold", 0.02, 0.40, float(config.LANDMARK_MATCH_THRESHOLD), 0.01,
        help="Normalized geometry distance Stage 2 requires to agree.",
    )
    # NOTE: passed as per-call overrides to verifier.verify(), never written
    # into the shared config module - config is a single object cached
    # across every visitor, and mutating it here would leak one user's
    # slider settings into everyone else's concurrent session.

    st.divider()
    st.caption(
        "**Stage 1:** FaceNet512 appearance embedding match.\n\n"
        "**Stage 2:** MediaPipe facial-landmark geometry veto — rejects "
        "confident appearance matches whose facial proportions disagree, "
        "and can confirm borderline matches whose geometry agrees."
    )

# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
tab_recognize, tab_manage, tab_about = st.tabs(["🔍  Recognize", "🗂️  Manage database", "ℹ️  About"])

# --- Recognize tab --------------------------------------------------------- #
with tab_recognize:
    if not db_exists:
        st.warning("Build or upload a database first (see the **Manage database** tab).")
    else:
        left, right = st.columns([1, 1], gap="large")

        with left:
            source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True, label_visibility="collapsed")
            pil_image = None
            if source == "Upload a photo":
                uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                if uploaded is not None:
                    pil_image = Image.open(uploaded)
            else:
                camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
                if camera_file is not None:
                    pil_image = Image.open(camera_file)

            if pil_image is not None:
                st.image(pil_image, caption="Probe image", use_container_width=True)

        with right:
            if pil_image is None:
                st.markdown(
                    '<div class="fv-stage-card" style="text-align:center; color:#64748B; padding:60px 20px;">'
                    "Upload or capture a photo to run recognition."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                buf = tempfile.SpooledTemporaryFile()
                pil_image.convert("RGB").save(buf, format="JPEG", quality=95)
                buf.seek(0)
                image_bytes = buf.read()

                result, error = run_verification(image_bytes, emb_match, emb_uncertain, landmark_match)

                if error is not None:
                    st.error(
                        "Recognition failed. This is usually a model-loading issue on the "
                        "server (see the README's troubleshooting section) rather than "
                        "something wrong with your photo."
                    )
                    with st.expander("Show technical details"):
                        st.code(error)
                else:
                    st.markdown(status_pill(result.status, result.identity), unsafe_allow_html=True)
                    st.write("")

                    m1, m2 = st.columns(2)
                    m1.metric(
                        "Embedding distance",
                        f"{result.embedding_distance:.3f}" if result.embedding_distance is not None else "—",
                    )
                    m2.metric(
                        "Landmark distance",
                        f"{result.landmark_distance:.3f}" if result.landmark_distance is not None else "—",
                    )
                    st.caption(result.reason)

                    if result.all_candidates:
                        st.markdown("**Top candidates (appearance only)**")
                        st.dataframe(
                            {
                                "Person": [p for p, _ in result.all_candidates],
                                "Embedding distance": [round(d, 4) for _, d in result.all_candidates],
                            },
                            use_container_width=True,
                            hide_index=True,
                        )

# --- Manage database tab ---------------------------------------------------- #
with tab_manage:
    st.markdown(
        "The database (`data/embeddings_db.pkl`) stores precomputed FaceNet512 "
        "embeddings and MediaPipe geometry signatures for every reference "
        "image, so recognition doesn't have to recompute them on every request."
    )

    if db_exists:
        with st.expander(f"View current identities ({len(database)})"):
            cols = st.columns(3)
            for i, (person, entries) in enumerate(sorted(database.items())):
                cols[i % 3].markdown(f"**{person}**  \n{len(entries)} reference image(s)")

    st.divider()

    col_add, col_rebuild = st.columns(2, gap="large")

    with col_add:
        with st.container(border=True):
            st.markdown("#### ➕ Add a new person")
            st.caption(
                "Upload a few clear, front-facing photos. They'll be added to "
                "`data/train/<name>/` and the database rebuilt for just this "
                "person - existing identities are left untouched."
            )
            st.info(
                "On Streamlit Community Cloud, storage is **ephemeral** - "
                "anything added here is lost on redeploy/restart. Fine for a "
                "live demo; for a permanent addition, add photos to "
                "`data/train/` locally, rerun `build_database.py`, and commit "
                "the updated `.pkl`.",
                icon=":material/info:",
            )

            new_name = st.text_input("Person's name")
            new_images = st.file_uploader(
                "Reference photos (2-10 recommended)", type=["jpg", "jpeg", "png"],
                accept_multiple_files=True, key="add_person_uploader",
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
                    st.cache_data.clear()
                    st.success(f"Added **{new_name}** and rebuilt the database.")
                    st.rerun()

    with col_rebuild:
        with st.container(border=True):
            st.markdown("#### 🔄 Rebuild entire database")
            st.caption(
                "Re-scans every folder in `data/train/` from scratch. Use "
                "after manually adding/removing files outside the app. "
                "Requires `data/train/` to exist - it won't on a fresh "
                "deploy, since raw images aren't committed (see `.gitignore`)."
            )
            st.write("")
            if st.button("Rebuild from data/train/"):
                if not os.path.isdir(config.DATABASE_DIR):
                    st.error(
                        f"`{config.DATABASE_DIR}` doesn't exist. Add at least "
                        "one person via the form first, or add photos locally "
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
                        st.cache_data.clear()
                        st.success("Database rebuilt.")
                        st.rerun()

# --- About tab --------------------------------------------------------------- #
with tab_about:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="fv-stage-card">
                <h4>Stage 1 — Appearance</h4>
                <p>DeepFace computes a 512-d FaceNet512 embedding for the probe
                image and finds the nearest reference image by cosine
                distance across every identity in the database.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="fv-stage-card">
                <h4>Stage 2 — Geometry</h4>
                <p>For the best appearance candidate, MediaPipe's
                FaceLandmarker extracts a normalized facial geometry
                signature (eye spacing, jaw width, nose-to-chin distance,
                etc.) and compares it against that candidate's reference
                images.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        "Confident appearance matches whose geometry **disagrees** are "
        "rejected as likely false positives; borderline appearance matches "
        "whose geometry **agrees** get promoted to a confirmed match. This "
        "two-stage design is what reduces false positives compared to an "
        "appearance-only baseline - see `evaluate.py` in the repo for a "
        "script that measures this on your own test set."
    )
    st.divider()
    st.caption(
        "Adjust thresholds in the sidebar to explore the trade-off between "
        "false positives and false negatives for your dataset."
    )
