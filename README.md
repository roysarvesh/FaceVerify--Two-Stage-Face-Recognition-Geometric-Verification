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
├── requirements.txt        # deployed app - kept lean, no sklearn/matplotlib
├── requirements-dev.txt    # extra deps for train_classifier.py only
├── app.py                  # Streamlit UI
├── build_database.py       # CLI: precompute embeddings + landmarks for data/train/
├── recognize.py            # CLI: identify one probe image, save annotated output
├── evaluate.py             # CLI: baseline vs. pipeline false-positive comparison
├── train_classifier.py     # CLI: trains + evaluates the learned Stage 2 (ROC/EER)
├── src/
│   ├── embedding_engine.py # DeepFace/FaceNet512 wrapper (Stage 1)
│   ├── landmark_engine.py  # MediaPipe FaceLandmarker geometry signature (Stage 2)
│   ├── database.py         # builds/loads the precomputed reference database
│   ├── pairs.py            # genuine/impostor pair mining for train_classifier.py
│   ├── learned_gate.py     # dependency-free inference wrapper for the trained classifier
│   ├── person_info.py      # Wikipedia lookup for "who is this" after a match
│   ├── image_lab.py        # rotation/morphology/edges/contours/diagnostics playground
│   └── verifier.py         # combines both stages into one verify() call
├── data/
│   ├── train/<person_name>/*.jpg   # <- put your reference images here
│   ├── test/<person_name>/*.jpg    # <- held out, for evaluate.py and train_classifier.py
│   ├── stage2_classifier.json      # trained Stage-2 model (generated)
│   ├── roc_curve.png               # ROC comparison plot (generated)
│   └── evaluation_report.md        # full evaluation writeup (generated)
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

**`OSError` inside `ctypes.CDLL(...)` when loading MediaPipe's native
library:** MediaPipe's compiled C++ bindings (`libmediapipe.so`) need
several system libraries beyond what OpenCV requires. `packages.txt`
already includes the full set, confirmed by directly inspecting the
library's dependencies with `ldd` rather than guessing one error at a time:
`libstdc++6` (C++ standard library), `libgomp1` (OpenMP runtime), `libegl1`
(`libEGL.so.1`), and `libgles2` (`libGLESv2.so.2`) - on top of
`libgl1`/`libglib2.0-0t64` already needed by OpenCV.

**If a *different* missing-library error shows up later** (from `cv2`,
`mediapipe`, or anything else), the fastest way to find every dependency at
once - rather than fixing one `ImportError` per redeploy - is to inspect
the actual `.so` file directly instead of guessing:
```bash
# find the native library (path varies by package/version)
python -c "import cv2; print(cv2.__file__)"              # for OpenCV
python -c "import mediapipe; print(mediapipe.__file__)"  # for MediaPipe

# then, in the same environment (a local Docker container matching
# Debian trixie is closest to Streamlit Cloud's image):
ldd /path/to/the/actual_native_module.so
```
Anything in the output resolving to a path inside the package's own
bundled `.libs` folder is fine and needs nothing extra; anything resolving
to a system path (or showing `=> not found`) is a real system dependency.
Search [packages.debian.org](https://packages.debian.org) for each missing
filename to find the exact package name - and check for a `t64`-suffixed
rename on trixie before assuming the "obvious" package name is correct
(see the `libglib2.0-0` note above).

**One failed recognition shouldn't crash the whole app:** Streamlit
re-executes the code inside *every* `st.tabs()` block on every rerun,
regardless of which tab is visually active, and a `file_uploader`'s value
persists across reruns within a session. Combined, this means an error
while processing an uploaded photo doesn't just fail once - it re-fires
(and crashes the entire app, not just that tab) on every subsequent
interaction, including simply switching tabs, for as long as that photo
stays uploaded. `app.py` guards against this two ways: recognition results
are cached by `(image bytes, thresholds)` via `st.cache_data` so a rerun
with an unchanged photo doesn't recompute at all, and the verification call
itself is wrapped in a try/except that surfaces a friendly in-app error
(with technical details in an expander) instead of raising - so a
model-loading failure shows up once, in context, rather than taking down
the whole page.

**App startup speed:** `deepface` (which pulls in TensorFlow) and
`mediapipe` are both imported *lazily* - inside `EmbeddingEngine.embed()`
and `LandmarkEngine.__init__()` respectively, not at module level. Both
libraries take real time to import (several seconds for TensorFlow
especially), and previously that cost was paid on every single app boot,
before the UI even rendered, regardless of whether anyone had uploaded a
photo yet. Now the sidebar, tabs, and database stats appear immediately;
the model-loading cost is only paid the first time someone actually runs a
recognition (or clicks the sidebar's **Warm up models** button, which lets
you pay that cost proactively right after a cold start instead of on your
first real photo).

**Theme:** dark, configured in `.streamlit/config.toml`
(`base = "dark"` plus custom accent/background colors) - Streamlit's native
widgets (sliders, buttons, tabs) pick this up automatically, and `app.py`
layers a light custom-CSS pass on top for card depth, status-pill glow, and
tightened typography.

## "Who is this?" — person info lookup

After a confirmed match, the Recognize tab shows a short bio card for the
identified person - name, summary, thumbnail, and a link to read more.
This uses **Wikipedia's public REST API** (`src/person_info.py`):
`action=query&list=search` to find the best-matching page title, then the
`page/summary` endpoint for the actual bio.

**Why Wikipedia and not a general web/Google search:** a real Google
Search API (Custom Search JSON API, SerpAPI, etc.) needs an API key,
billing setup, and per-query cost - not something that should be a
required step just to get this project running. Wikipedia's API needs
none of that and works immediately on a fresh deploy, which matters more
here than search-result breadth. The real limitation: it only returns
something for identities with a Wikipedia page - fine for the celebrities
in this project's dataset, not for a private individual. Swapping in a
paid search API later is a drop-in change: implement the same
`fetch_person_summary(name) -> {title, extract, url, thumbnail} | None`
contract in `src/person_info.py` against whichever API you choose, and
nothing else in `app.py` needs to change.

Lookups are cached per name for 24 hours (`st.cache_data(ttl=...)` in
`app.py`) so switching tabs or re-running recognition on the same person
doesn't re-hit the API every time.

## Image Lab — OpenCV playground

A separate tab (independent of the recognition pipeline - works on any
image, not just faces) for classic image-processing operations, all in
`src/image_lab.py`:

- **Rotation** - arbitrary angle, canvas auto-expanded so corners aren't
  cropped off (unlike a naive same-size `warpAffine`).
- **Morphological transforms** - Erode, Dilate, Opening, Closing,
  Gradient, Top Hat, Black Hat, with adjustable kernel size, shape
  (rectangle/ellipse/cross), and iteration count.
- **Edge detection** - Canny, with both thresholds exposed as sliders.
- **Contour detection** - built on Canny edges + `cv2.findContours`,
  with a minimum-area filter to drop noise; draws kept contours on the
  image and reports the count.
- **Image characteristics** - dimensions, file size, mean brightness,
  and a sharpness/blur estimate (variance of the Laplacian - a standard,
  cheap proxy: a sharp image has a lot of high-frequency edge content, a
  blurry one doesn't), plus mean and sampled-dominant color.

Operations chain in order (rotate → morphology → edges/contours), so you
can e.g. rotate first and then run edge detection on the rotated result.

## Rigorous evaluation: ROC, EER, and a learned Stage 2

`config.py`'s original thresholds (`EMBEDDING_MATCH_THRESHOLD = 0.30`, etc.)
were reasonable starting guesses, not values derived from data.
`train_classifier.py` replaces guessing with the standard face-verification
evaluation methodology - cross-validated ROC curves and Equal Error Rate
(EER) - and, honestly, the result changed the story of this project.

**Methodology.** `src/pairs.py` mines genuine/impostor pairs from
`data/test/` (511 images, never used to build the training database) against
`data/embeddings_db.pkl`. For each test image, the *genuine* sample is its
distance to its own identity; the *impostor* sample is its distance to the
**nearest wrong identity** - the hardest case, i.e. exactly the situation
Stage 2 exists to catch, not a random unrelated person that would be
trivially easy to reject.

**What running it actually found**, on 1,009 pairs from this dataset (see
`data/evaluation_report.md` and `data/roc_curve.png` after running it):

| | AUC | EER |
|---|---|---|
| A) Stage 1 alone (embedding distance) | 0.993 | 3.4% |
| C) Learned classifier (embedding + landmark) | 0.992 | 3.8% |

The two ROC curves sit almost exactly on top of each other. **On this
dataset, MediaPipe's geometry signal adds essentially nothing measurable
beyond the embedding distance alone** - the fitted classifier's own weights
confirm it (`w_embedding_distance` ≈ -13.4 vs. `w_landmark_distance` ≈
-0.9, roughly a 14:1 ratio). This dataset's photos are mostly clean,
frontal, high-quality celebrity images - exactly the case FaceNet512
already handles well on its own, leaving little room for a second signal
to add value. A harder test set (poor lighting, extreme angles,
adversarial look-alikes) would very plausibly show geometry mattering
more; this one doesn't.

**What *did* clearly matter:**

| | FPR | FNR | Accuracy |
|---|---|---|---|
| B) Original hand-tuned rule (`config.py`) | 27.1% | 23.3% | 74.8% |
| C) Learned classifier at target 1% FPR | 1.0% | 7.3% (93% TPR) | ~96% |

The hand-tuned nested-threshold rule sits **far** off the achievable ROC
curve - not a close call. The real improvement this evaluation produced
isn't "two stages beat one," it's "a properly calibrated decision boundary
beats hand-picked thresholds," which is why `USE_LEARNED_STAGE2 = True` is
now the default in `config.py`: `FaceVerifier` uses the trained classifier
(`data/stage2_classifier.json`) instead of the nested-threshold rule.
Inference stays dependency-free - `src/learned_gate.py` is just a sigmoid
over three saved numbers, no sklearn needed in the deployed app (see
`requirements-dev.txt` vs. `requirements.txt`).

**To reproduce or retrain:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
python train_classifier.py                      # default: target 1% FPR
python train_classifier.py --target-fpr 0.005    # stricter operating point
```
Rerun this after changing the training set, the embedding model, or the
landmark feature set - the saved classifier is only valid for the data
distribution it was fit on.

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
