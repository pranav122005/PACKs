"""
backend/ocr/image_preprocessing.py

Image Preprocessing — prepares a raw photo of a packaged food label for
OCR using OpenCV:

    1. Resize (upscale small images for better character detail, cap huge ones)
    2. Grayscale conversion
    3. Noise removal (fastNlMeansDenoising + median blur)
    4. Contrast enhancement (CLAHE)
    5. Rotation / skew correction (minAreaRect based deskew, Hough fallback)
    6. Adaptive thresholding / binarization

Each step is exposed individually (so callers can compose a custom
pipeline) as well as via a single `preprocess()` convenience method that
runs the full recommended pipeline and returns intermediate artifacts
for debugging/inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessingResult:
    """Holds the final processed image plus intermediate stages for debugging."""

    original: np.ndarray
    grayscale: np.ndarray
    denoised: np.ndarray
    contrast_enhanced: np.ndarray
    deskewed: np.ndarray
    binarized: np.ndarray
    rotation_angle_degrees: float
    final: np.ndarray


class ImagePreprocessor:
    """
    Encapsulates the full OpenCV preprocessing pipeline applied to label
    images before they are handed to EasyOCR.

    All tunable parameters are constructor arguments with sane production
    defaults, so behaviour can be adjusted per-deployment without editing
    code (e.g. from a config-driven factory upstream).
    """

    def __init__(
        self,
        target_max_dimension: int = 1600,
        min_upscale_dimension: int = 800,
        clahe_clip_limit: float = 3.0,
        clahe_tile_grid_size: int = 8,
        denoise_strength: int = 10,
        adaptive_threshold_block_size: int = 35,
        adaptive_threshold_c: int = 11,
    ) -> None:
        self._target_max_dimension = target_max_dimension
        self._min_upscale_dimension = min_upscale_dimension
        self._clahe_clip_limit = clahe_clip_limit
        self._clahe_tile_grid_size = clahe_tile_grid_size
        self._denoise_strength = denoise_strength
        self._adaptive_threshold_block_size = adaptive_threshold_block_size
        self._adaptive_threshold_c = adaptive_threshold_c

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> PreprocessingResult:
        """
        Run the complete preprocessing pipeline on a BGR image (as read by
        cv2.imread / cv2.imdecode) and return every intermediate stage.
        """
        if image is None or image.size == 0:
            raise ValueError("Received an empty or invalid image array.")

        original = image.copy()
        resized = self._resize(original)
        grayscale = self._to_grayscale(resized)
        denoised = self._remove_noise(grayscale)
        contrast_enhanced = self._enhance_contrast(denoised)
        deskewed, angle = self._correct_rotation(contrast_enhanced)
        binarized = self._binarize(deskewed)

        return PreprocessingResult(
            original=original,
            grayscale=grayscale,
            denoised=denoised,
            contrast_enhanced=contrast_enhanced,
            deskewed=deskewed,
            binarized=binarized,
            rotation_angle_degrees=angle,
            final=binarized,
        )

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Upscale small images and downscale very large ones for consistent OCR input size."""
        height, width = image.shape[:2]
        largest_dim = max(height, width)

        if largest_dim < self._min_upscale_dimension:
            scale = self._min_upscale_dimension / largest_dim
            interpolation = cv2.INTER_CUBIC
        elif largest_dim > self._target_max_dimension:
            scale = self._target_max_dimension / largest_dim
            interpolation = cv2.INTER_AREA
        else:
            return image

        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=interpolation)

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _remove_noise(self, gray_image: np.ndarray) -> np.ndarray:
        """Remove sensor/JPEG noise while preserving text edges."""
        denoised = cv2.fastNlMeansDenoising(
            gray_image, h=self._denoise_strength, templateWindowSize=7, searchWindowSize=21
        )
        # Light median blur mops up remaining salt-and-pepper speckle
        # without softening character edges as much as a gaussian blur would.
        return cv2.medianBlur(denoised, 3)

    def _enhance_contrast(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip_limit,
            tileGridSize=(self._clahe_tile_grid_size, self._clahe_tile_grid_size),
        )
        return clahe.apply(gray_image)

    def _correct_rotation(self, gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct skew using the minimum-area bounding rectangle
        of foreground (text) pixels. Falls back to Hough-line based
        estimation if the bounding-box approach yields no usable angle.
        """
        angle = self._estimate_skew_via_min_area_rect(gray_image)
        if angle is None:
            angle = self._estimate_skew_via_hough_lines(gray_image)
        if angle is None or abs(angle) < 0.1:
            return gray_image, 0.0

        (h, w) = gray_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            gray_image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, float(angle)

    @staticmethod
    def _estimate_skew_via_min_area_rect(gray_image: np.ndarray) -> Optional[float]:
        # Invert so text becomes the foreground (white) for contour detection.
        thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(thresh)
        if coords is None or len(coords) < 20:
            return None

        rect_angle = cv2.minAreaRect(coords)[-1]
        # cv2.minAreaRect returns angles in [-90, 0); normalize to a
        # human-intuitive small rotation correction.
        if rect_angle < -45:
            angle = -(90 + rect_angle)
        else:
            angle = -rect_angle

        # Ignore implausible corrections (likely noise, not real skew).
        if abs(angle) > 20:
            return None
        return angle

    @staticmethod
    def _estimate_skew_via_hough_lines(gray_image: np.ndarray) -> Optional[float]:
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)
        if lines is None:
            return None

        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = (theta * 180 / np.pi) - 90
            if -20 <= angle_deg <= 20:
                angles.append(angle_deg)

        if not angles:
            return None
        return float(np.median(angles))

    def _binarize(self, gray_image: np.ndarray) -> np.ndarray:
        """Adaptive thresholding — robust to uneven lighting across a label/package."""
        block_size = self._adaptive_threshold_block_size
        if block_size % 2 == 0:
            block_size += 1  # cv2 requires an odd block size

        return cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self._adaptive_threshold_c,
        )
