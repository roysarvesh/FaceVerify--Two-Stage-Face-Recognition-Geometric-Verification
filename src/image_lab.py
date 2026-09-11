"""
image_lab.py
-------------
Classic OpenCV image-processing operations exposed as a playground:
rotation, morphological transforms (erode/dilate/open/close/gradient/
top-hat/black-hat), edge detection, contour detection, and basic image
diagnostics (blur estimate, brightness, dominant color, etc.).

Kept separate from the recognition pipeline (embedding_engine.py /
landmark_engine.py) - this module is about *exploring* an image, not about
identifying who's in it.
"""

import cv2
import numpy as np

MORPH_SHAPES = {
    "Rectangle": cv2.MORPH_RECT,
    "Ellipse": cv2.MORPH_ELLIPSE,
    "Cross": cv2.MORPH_CROSS,
}

MORPH_OPERATIONS = {
    "Erode": cv2.MORPH_ERODE,
    "Dilate": cv2.MORPH_DILATE,
    "Opening": cv2.MORPH_OPEN,
    "Closing": cv2.MORPH_CLOSE,
    "Gradient": cv2.MORPH_GRADIENT,
    "Top Hat": cv2.MORPH_TOPHAT,
    "Black Hat": cv2.MORPH_BLACKHAT,
}


def rotate_image(image_bgr: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotates around the image center, expanding the canvas so corners
    aren't cropped off (unlike a naive same-size warpAffine)."""
    h, w = image_bgr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image_bgr, matrix, (new_w, new_h), borderValue=(30, 30, 30))


def apply_morphology(image_bgr: np.ndarray, operation: str, kernel_size: int = 5,
                      iterations: int = 1, kernel_shape: str = "Rectangle") -> np.ndarray:
    """Applies a morphological operation. Erode/Dilate are the primitives;
    Opening/Closing/Gradient/Top Hat/Black Hat are the standard compound
    operations built from them (cv2.morphologyEx)."""
    kernel_size = max(1, kernel_size | 1)  # force odd, >=1
    shape = MORPH_SHAPES.get(kernel_shape, cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (kernel_size, kernel_size))
    op = MORPH_OPERATIONS.get(operation)

    if op in (cv2.MORPH_ERODE, cv2.MORPH_DILATE):
        fn = cv2.erode if op == cv2.MORPH_ERODE else cv2.dilate
        return fn(image_bgr, kernel, iterations=iterations)
    return cv2.morphologyEx(image_bgr, op, kernel, iterations=iterations)


def detect_edges(image_bgr: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Canny edge detection. Returns a 3-channel BGR image (edges in white
    on black) so it displays consistently alongside the other operations."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def detect_contours(image_bgr: np.ndarray, threshold1: int = 100, threshold2: int = 200,
                     min_area: int = 50):
    """Finds contours via Canny edges + cv2.findContours, draws the ones
    above min_area on a copy of the original image, and returns
    (annotated_image, contour_count, areas_sorted_desc)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)  # close small gaps

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]

    annotated = image_bgr.copy()
    cv2.drawContours(annotated, kept, -1, (0, 255, 0), 2)

    areas = sorted((cv2.contourArea(c) for c in kept), reverse=True)
    return annotated, len(kept), areas


def image_characteristics(image_bgr: np.ndarray, file_size_bytes: int | None = None) -> dict:
    """Basic diagnostic stats about an image - the kind of thing you'd
    otherwise have to compute ad hoc: sharpness, brightness, dominant
    color, etc."""
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Variance of the Laplacian is a standard, cheap blur/sharpness proxy:
    # a sharp image has a lot of high-frequency edge content (high
    # variance); a blurry one is smoothed out (low variance).
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    mean_bgr = image_bgr.reshape(-1, 3).mean(axis=0)
    dominant_bgr = image_bgr.reshape(-1, 3)[
        np.random.default_rng(0).choice(image_bgr.reshape(-1, 3).shape[0], size=min(2000, h * w), replace=False)
    ]
    # cheap "dominant color" proxy: mean of a random sample, rounded -
    # a full k-means would be more accurate but is overkill for a quick stat
    dominant_bgr = dominant_bgr.mean(axis=0)

    return {
        "width": w,
        "height": h,
        "channels": image_bgr.shape[2] if image_bgr.ndim == 3 else 1,
        "aspect_ratio": round(w / h, 3) if h else None,
        "file_size_kb": round(file_size_bytes / 1024, 1) if file_size_bytes else None,
        "mean_brightness": round(float(gray.mean()), 1),
        "brightness_std": round(float(gray.std()), 1),
        "sharpness_laplacian_var": round(laplacian_var, 1),
        "likely_blurry": laplacian_var < 100.0,  # empirical rule of thumb
        "mean_color_bgr": tuple(round(float(c), 1) for c in mean_bgr),
        "dominant_color_bgr": tuple(round(float(c), 1) for c in dominant_bgr),
    }
