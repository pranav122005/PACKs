import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# If the image is smaller than this on its longest side, upscale it before
# processing. Small/low-res label photos are a major cause of OCR misses.
MIN_LONG_EDGE_PX = 1600


class ImagePreprocessError(Exception):
    """Raised when an image cannot be loaded or processed."""


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from disk as a BGR numpy array.

    Raises:
        ImagePreprocessError: if the file doesn't exist or can't be decoded.
    """
    path = Path(image_path)
    if not path.exists():
        raise ImagePreprocessError(f"Image file not found: {image_path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImagePreprocessError(
            f"Could not decode image (unsupported or corrupt file): {image_path}"
        )
    return image


def _upscale_if_small(gray: np.ndarray) -> np.ndarray:
    """Upscales the image if its longest edge is below MIN_LONG_EDGE_PX."""
    h, w = gray.shape[:2]
    long_edge = max(h, w)
    if long_edge >= MIN_LONG_EDGE_PX:
        return gray

    scale = MIN_LONG_EDGE_PX / float(long_edge)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Estimates and corrects small rotational skew using the minimum-area
    bounding rectangle of thresholded foreground pixels. Falls back to the
    original image if skew estimation fails or is negligible.
    """
    try:
        inverted = cv2.bitwise_not(gray)
        thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 50:
            return gray  # not enough foreground to estimate skew reliably

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Ignore negligible or wildly implausible skew estimates
        if abs(angle) < 0.5 or abs(angle) > 15:
            return gray

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    except Exception as exc:  # noqa: BLE001 - deskew is best-effort, never fatal
        logger.warning("Deskew step failed, continuing with un-deskewed image: %s", exc)
        return gray


def preprocess_for_ocr(image_path: str, save_debug_path: str | None = None) -> np.ndarray:
    """
    Loads and preprocesses a nutrition label image for OCR.

    Pipeline:
        1. Load as BGR, convert to grayscale
        2. Upscale small images
        3. Denoise (fast non-local means)
        4. CLAHE adaptive contrast enhancement (handles glare / low contrast)
        5. Deskew (corrects minor photo rotation)
        6. Adaptive Gaussian thresholding -> clean black-on-white binary image

    Args:
        image_path: path to the source image file.
        save_debug_path: optional path to write the processed image to disk,
                          useful for debugging OCR quality issues.

    Returns:
        A single-channel (grayscale/binary) numpy array ready for EasyOCR.

    Raises:
        ImagePreprocessError: if the image can't be loaded or processed.
    """
    image = load_image(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _upscale_if_small(gray)

    # Denoise before contrast enhancement so CLAHE doesn't amplify noise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE: local adaptive contrast enhancement, good for glossy/glare-prone
    # packaging and faint printed text
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    deskewed = _deskew(enhanced)

    # Adaptive thresholding handles uneven lighting across the label better
    # than a single global threshold value would.
    binary = cv2.adaptiveThreshold(
        deskewed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )

    # Light morphological close to reconnect thin broken character strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if save_debug_path:
        try:
            cv2.imwrite(save_debug_path, processed)
        except Exception as exc:  # noqa: BLE001 - debug output is non-critical
            logger.warning("Failed to write debug preprocessed image: %s", exc)

    return processed
