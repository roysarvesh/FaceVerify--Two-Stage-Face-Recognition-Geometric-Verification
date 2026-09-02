# Face Recognition — Appearance + Geometry Two-Stage Verification

A face recognition pipeline that combines **DeepFace / FaceNet512** appearance
embeddings with **MediaPipe** facial-landmark geometry to reduce false-positive
identifications versus an appearance-only baseline.

## Why two stages?

A pure embedding-based recognizer (FaceNet512 nearest-neighbor) sometimes
returns a confident-looking match for the *wrong* person — look-alikes,
siblings, poor lighting, heavy makeup, or low-quality crops can all pull two
different people's embeddings close together in appearance space.

This project adds a second, independent check that appearance-only systems
don't have: **facial geometry**. MediaPipe's 468-point FaceMesh is used to
build a scale/translation-invariant signature of each face (ratios like
eye-to-nose distance, mouth width, jaw width, all normalized by inter-ocular
distance). Two people can look similar in a crop yet have measurably
different facial proportions — and vice versa.

```
                 ┌────────────────────┐
   probe image → │  Stage 1: FaceNet512│ → best appearance candidate + distance
                 └─────────┬──────────┘
                           │ distance ≤ MATCH_THRESHOLD          → confident?
                           │ MATCH_THRESHOLD < d ≤ UNCERTAIN_THR → borderline
                           │ d > UNCERTAIN_THRESHOLD             → reject (no_match)
                           ▼
                 ┌────────────────────┐
                 │ Stage 2: MediaPipe  │ → geometry distance vs. candidate's
                 │ FaceMesh landmarks  │   reference image(s)
                 └─────────┬──────────┘
                           ▼
        geometry agrees  → MATCH
        geometry disagrees (was confident) → NO_MATCH  (false positive filtered)
        geometry disagrees (was borderline) → UNCERTAIN
```

Confident appearance matches whose geometry **disagrees** are rejected
instead of returned — this is the mechanism that cuts false positives
compared to a Stage-1-only baseline. Borderline appearance matches whose
geometry **agrees** get promoted to a confirmed match, rather than being
lost entirely.

## Project structure

```
face_recognition_project/
├── config.py               # paths + thresholds (portable, no hardcoded machine paths)
├── requirements.txt
├── build_database.py       # CLI: precompute embeddings + landmarks for data/train/
├── recognize.py            # CLI: identify one probe image, save annotated output
├── evaluate.py             # CLI: baseline vs. pipeline false-positive comparison
├── src/
│   ├── embedding_engine.py # DeepFace/FaceNet512 wrapper (Stage 1)
│   ├── landmark_engine.py  # MediaPipe FaceMesh geometry signature (Stage 2)
│   ├── database.py         # builds/loads the precomputed reference database
│   └── verifier.py         # combines both stages into one verify() call
├── data/
│   ├── train/<person_name>/*.jpg   # <- put your reference images here
│   └── test/<person_name>/*.jpg    # <- optional, for evaluate.py
└── legacy_lbph_baseline/   # your original files, kept for reference:
    ├── original_deepface_only_script.py  # the single-stage script you uploaded
    ├── haar_face.xml, face_trained.yml, features.npy, labels.npy  # OpenCV LBPH model
```

Your uploaded `face_recognition.py` was a single-stage, appearance-only
DeepFace/Facenet512 script with a hardcoded Windows path — it's preserved in
`legacy_lbph_baseline/original_deepface_only_script.py` as the "before"
picture. The `haar_face.xml` / `face_trained.yml` / `features.npy` /
`labels.npy` files are from an even earlier OpenCV LBPH classifier and are
kept only for reference — they aren't used by the new pipeline.

## Setup

```bash
pip install -r requirements.txt
```

Organize your reference dataset (any people, any number of images each) as:

```
data/train/
    Ben Afflek/
        1.jpg
        2.jpg
    Elton John/
        1.jpg
        2.jpg
```

## Usage

**1. Build the database** (embeddings + landmark signatures, done once):

```bash
python build_database.py
```

**2. Recognize a face:**

```bash
python recognize.py --image path/to/photo.jpg
```

Prints the decision (`match` / `uncertain` / `no_match`), both distances, the
reasoning, and the top-5 appearance candidates for transparency — then saves
an annotated copy of the image next to the input.

**3. Measure the false-positive reduction on your own data:**

```bash
python evaluate.py --test-dir data/test
```

`data/test/` should mirror `data/train/` (known identities) and can also
include a `data/test/unknown/` folder of people who are *not* in the
database — any prediction there counts as a false positive. `evaluate.py`
runs both the Stage-1-only baseline and the full two-stage pipeline over the
same test images and prints something like:

```
[baseline] TP=42  FP=9   TN=6  FN=3   (false-positive rate: 15.00%)
[pipeline] TP=41  FP=5   TN=9  FN=5   (false-positive rate: 8.33%)

False positives reduced by 44.4% (9 -> 5) vs. the appearance-only baseline.
```

This is how you reproduce (and can cite) a concrete false-positive reduction
number for your own dataset — actual numbers depend on your images, so run
it on your data before quoting a specific percentage.

## Streamlit app

`app.py` is a Streamlit front-end over the same pipeline - upload a photo or
use your webcam, see the match/uncertain/no-match decision live, tune
thresholds with sliders, and manage the database (add a person, rebuild)
from the browser instead of the command line.

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

**Deploy on Streamlit Community Cloud:**
1. Make sure `data/embeddings_db.pkl` and `models/face_landmarker.task` are
   committed to your repo (both are small - just numbers/weights, not raw
   images). `.gitignore` already excludes `data/train/` and `data/test/`
   so the ~2,000 training photos never get pushed.
2. Push to GitHub.
3. Go to [share.streamlit.io](https://share.streamlit.io) -> **New app** ->
   point it at your repo, branch, and `app.py`.
4. Streamlit Cloud automatically installs `packages.txt` (system libs
   `opencv-python` needs on Linux) and `requirements.txt` (Python deps) -
   nothing else to configure.

**Storage is ephemeral on Streamlit Cloud** - the "Add a new person"
feature in the app writes to the running container's disk, which is wiped
on every redeploy/restart. Fine for a live demo; for a permanent addition,
add photos to `data/train/` locally, rerun `build_database.py`, and commit
the updated `.pkl`.

**Why `opencv-python-headless`, not `opencv-python`:** `deepface` and
`mediapipe` both pull in a full/GUI OpenCV build as a transitive dependency
(`opencv-python` and `opencv-contrib-python` respectively), even though this
app never opens a GUI window. Requesting `opencv-python-headless` explicitly
(and listing it first in `requirements.txt`) is the standard mitigation -
though as the note below explains, it doesn't always "win," so
`packages.txt` still needs to cover the GUI build's needs as a fallback.

**Note on `packages.txt` formatting:** unlike a normal apt sources file,
Streamlit Cloud's parser does **not** strip `#` comments - it passes every
whitespace-separated token straight to `apt-get install`, including comment
text, which fails with `Unable to locate package <word>` for each word in a
comment. Keep `packages.txt` to bare package names only, one per line, no
comments.

**If you still see a `cv2` import error after this fix:** see the
`libgthread-2.0.so.0` note further down - `packages.txt` needs the
correctly-named GLib package (`libglib2.0-0t64` on Debian trixie, not the
older `libglib2.0-0`) since more than one OpenCV variant tends to get
installed regardless of what `requirements.txt` prefers.

**Python version — Streamlit Cloud currently defaults to Python 3.14,
which breaks this app.** TensorFlow (a `deepface` dependency) doesn't ship
wheels for Python 3.14 at any version yet, so `pip install` fails with "No
matching distribution found for tensorflow-cpu". `runtime.txt` (pinning
`python-3.11`) is included to request an older, TensorFlow-compatible
Python version - but Python version **cannot be changed on an
already-deployed app**, and `runtime.txt` is unreliable for changing it
after the fact (a currently-known Streamlit Cloud issue). If your app was
first deployed before `runtime.txt` was added, do this once:
1. On Streamlit Cloud, delete the existing app (this doesn't touch your
   GitHub repo, just the deployment).
2. Redeploy: point it at the same repo/branch/`app.py`, but before
   clicking Deploy, open **Advanced settings** and explicitly select
   **Python 3.11** (or 3.12) from the dropdown.
3. Future redeploys of that same app will keep using 3.11 automatically.

**`ImportError: libgthread-2.0.so.0`, even with `packages.txt` present:**
`deepface` and `mediapipe` both hard-pin their own full/GUI OpenCV variant
(`opencv-python` and `opencv-contrib-python` respectively) regardless of
this project's explicit `opencv-python-headless` requirement, so more than
one OpenCV package can end up installed side by side; whichever's files
land in site-packages last "wins" the shared `cv2` folder - and it isn't
always the headless one. If that happens, the winning build needs
`libgthread-2.0.so.0`, which comes from GLib. On Debian **trixie**
(Streamlit Cloud's current base image), the correct package for that is
`libglib2.0-0t64` - **not** the older `libglib2.0-0`, which is broken on
this image (see the `packages.txt` note above). `packages.txt` already
includes `libglib2.0-0t64` for this reason.

## Tuning

All thresholds live in `config.py`:

| Setting | Meaning |
|---|---|
| `EMBEDDING_MATCH_THRESHOLD` | Cosine distance below which Stage 1 is "confident" |
| `EMBEDDING_UNCERTAIN_THRESHOLD` | Above this, reject outright without running Stage 2 |
| `LANDMARK_MATCH_THRESHOLD` | Normalized geometry distance Stage 2 requires to agree |

Lower `LANDMARK_MATCH_THRESHOLD` → stricter geometry check → fewer false
positives, but may also reject some genuine matches (e.g. extreme
expressions/angles change landmark ratios slightly). Tune both thresholds
against your own `evaluate.py` results.

## Notes / limitations

- `DeepFace.represent` downloads pretrained model weights on first run —
  make sure the machine running this has internet access the first time.
- MediaPipe FaceMesh needs a reasonably frontal, unobstructed face to
  extract landmarks; if it fails on a reference image, that image is just
  excluded from the geometry check (recorded as `geometry: None`) rather
  than breaking the whole pipeline.
- The geometry signature is a hand-picked set of 13 ratios chosen for
  robustness to scale/translation; it is not the same feature space as the
  embedding, by design — it's meant to be an independent check, not a
  duplicate of Stage 1.
