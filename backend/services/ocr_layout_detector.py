"""
backend/services/ocr_layout_detector.py

PACKS - AI Ingredient Intelligence Platform
====================================================
Uses OpenCV to detect coarse regions of interest on packaging images:
ingredient section, nutrition facts section, serving size area,
manufacturer section, and barcode area. Detection is based on
text-density heuristics (contour density per horizontal band) rather
than deep layout models, so it degrades gracefully without GPU/ML
dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("packs.ocr_layout_detector")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

BoundingBox = Tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class LayoutRegion:
    """A detected region of interest on the packaging image."""

    label: str
    bbox: BoundingBox
    confidence: float
    text_density: float

    def to_dict(self) -> Dict[str, object]:
        """Return the region as a plain dictionary."""
        return asdict(self)


@dataclass
class LayoutDetectionResult:
    """Full layout detection result for a single image."""

    image_width: int
    image_height: int
    ingredient_region: Optional[LayoutRegion] = None
    nutrition_region: Optional[LayoutRegion] = None
    serving_size_region: Optional[LayoutRegion] = None
    manufacturer_region: Optional[LayoutRegion] = None
    barcode_region: Optional[LayoutRegion] = None

    def to_dict(self) -> Dict[str, object]:
        """Return the full detection result as a plain dictionary."""
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "ingredient_region": self.ingredient_region.to_dict() if self.ingredient_region else None,
            "nutrition_region": self.nutrition_region.to_dict() if self.nutrition_region else None,
            "serving_size_region": self.serving_size_region.to_dict() if self.serving_size_region else None,
            "manufacturer_region": self.manufacturer_region.to_dict() if self.manufacturer_region else None,
            "barcode_region": self.barcode_region.to_dict() if self.barcode_region else None,
        }


class OCRLayoutDetector:
    """
    Detects coarse layout regions on a packaging image using OpenCV
    text-density and structural heuristics.

    Strategy:
        1. Convert to grayscale and binarize.
        2. Compute a horizontal-band text-density profile via
           morphological dilation + contour counting.
        3. Segment the image into candidate bands.
        4. Classify bands into ingredient / nutrition / manufacturer /
           serving-size candidates using relative density and position
           heuristics (nutrition tables tend to have grid-like line
           structure; barcode areas have high-frequency vertical edges).
    """

    def __init__(
        self,
        num_bands: int = 12,
        min_band_density: float = 0.02,
    ) -> None:
        """
        Args:
            num_bands: Number of horizontal bands to divide the image
                into for density analysis.
            min_band_density: Minimum text density (fraction of
                non-zero pixels) for a band to be considered textual.
        """
        self._num_bands = num_bands
        self._min_band_density = min_band_density

    def detect(self, image_path: str) -> LayoutDetectionResult:
        """
        Detect layout regions on the given image.

        Args:
            image_path: Filesystem path to the packaging image.

        Returns:
            LayoutDetectionResult describing detected regions. Any
            region that could not be confidently located is left as
            None.

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError: If the image could not be read/decoded.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode image at path: {path}")

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = self._binarize(gray)

        bands = self._compute_band_densities(binary, height)
        text_bands = [b for b in bands if b["density"] >= self._min_band_density]

        ingredient_region = self._detect_ingredient_region(text_bands, width, height)
        nutrition_region = self._detect_nutrition_region(binary, text_bands, width, height)
        serving_size_region = self._detect_serving_size_region(
            nutrition_region, width, height
        )
        manufacturer_region = self._detect_manufacturer_region(text_bands, width, height)
        barcode_region = self._detect_barcode_region(gray, width, height)

        result = LayoutDetectionResult(
            image_width=width,
            image_height=height,
            ingredient_region=ingredient_region,
            nutrition_region=nutrition_region,
            serving_size_region=serving_size_region,
            manufacturer_region=manufacturer_region,
            barcode_region=barcode_region,
        )

        logger.info(
            "Layout detection complete for %s: ingredients=%s nutrition=%s barcode=%s",
            path,
            bool(ingredient_region),
            bool(nutrition_region),
            bool(barcode_region),
        )
        return result

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Binarize a grayscale image using adaptive thresholding, tuned
        for printed packaging text under varying lighting.

        Args:
            gray: Grayscale image array.

        Returns:
            Binary image array (0/255) with text as white foreground.
        """
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        return dilated

    def _compute_band_densities(
        self, binary: np.ndarray, height: int
    ) -> List[Dict[str, object]]:
        """
        Divide the image into horizontal bands and compute the
        foreground pixel density of each band.

        Args:
            binary: Binarized image array.
            height: Image height in pixels.

        Returns:
            List of dicts, one per band, each with 'index', 'y_start',
            'y_end', and 'density'.
        """
        band_height = max(1, height // self._num_bands)
        bands: List[Dict[str, object]] = []

        for i in range(self._num_bands):
            y_start = i * band_height
            y_end = height if i == self._num_bands - 1 else min(height, (i + 1) * band_height)
            if y_start >= y_end:
                continue

            band_slice = binary[y_start:y_end, :]
            density = float(np.count_nonzero(band_slice)) / float(band_slice.size)

            bands.append({
                "index": i,
                "y_start": y_start,
                "y_end": y_end,
                "density": density,
            })

        return bands

    def _detect_ingredient_region(
        self, text_bands: List[Dict[str, object]], width: int, height: int
    ) -> Optional[LayoutRegion]:
        """
        Identify the ingredient region as the largest contiguous run of
        text-dense bands in the upper-to-middle portion of the image
        (ingredients typically appear before the nutrition table on
        most packaging layouts).

        Args:
            text_bands: List of text-dense band dicts.
            width: Image width.
            height: Image height.

        Returns:
            LayoutRegion for the ingredient section, or None if no
            sufficiently dense contiguous run was found.
        """
        run = self._largest_contiguous_run(text_bands, max_index_fraction=0.65)
        if not run:
            return None

        y_start = int(run[0]["y_start"])
        y_end = int(run[-1]["y_end"])
        avg_density = float(np.mean([b["density"] for b in run]))

        return LayoutRegion(
            label="ingredient_section",
            bbox=(0, y_start, width, y_end - y_start),
            confidence=round(min(1.0, avg_density * 5.0), 3),
            text_density=round(avg_density, 4),
        )

    def _detect_nutrition_region(
        self,
        binary: np.ndarray,
        text_bands: List[Dict[str, object]],
        width: int,
        height: int,
    ) -> Optional[LayoutRegion]:
        """
        Identify the nutrition table region using grid-line structure
        detection (nutrition tables commonly have horizontal separator
        lines) combined with band density in the lower portion of the
        image.

        Args:
            binary: Binarized image array.
            text_bands: List of text-dense band dicts.
            width: Image width.
            height: Image height.

        Returns:
            LayoutRegion for the nutrition table, or None if not found.
        """
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 10), 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        line_rows = np.where(np.count_nonzero(horizontal_lines, axis=1) > (width * 0.3))[0]
        if line_rows.size < 2:
            run = self._largest_contiguous_run(
                text_bands, max_index_fraction=1.0, min_index_fraction=0.4
            )
            if not run:
                return None
            y_start = int(run[0]["y_start"])
            y_end = int(run[-1]["y_end"])
            avg_density = float(np.mean([b["density"] for b in run]))
            return LayoutRegion(
                label="nutrition_facts",
                bbox=(0, y_start, width, y_end - y_start),
                confidence=round(min(1.0, avg_density * 4.0), 3),
                text_density=round(avg_density, 4),
            )

        y_start = int(line_rows.min())
        y_end = int(line_rows.max())
        region_height = max(1, y_end - y_start)
        density = float(
            np.count_nonzero(binary[y_start:y_end, :]) / float(width * region_height)
        )

        return LayoutRegion(
            label="nutrition_facts",
            bbox=(0, y_start, width, region_height),
            confidence=round(min(1.0, (line_rows.size / 20.0)), 3),
            text_density=round(density, 4),
        )

    def _detect_serving_size_region(
        self,
        nutrition_region: Optional[LayoutRegion],
        width: int,
        height: int,
    ) -> Optional[LayoutRegion]:
        """
        Approximate the serving size region as a narrow band just above
        the detected nutrition table, where serving size is
        conventionally printed.

        Args:
            nutrition_region: Previously detected nutrition table region.
            width: Image width.
            height: Image height.

        Returns:
            LayoutRegion approximating the serving size line, or None
            if the nutrition region was not detected.
        """
        if nutrition_region is None:
            return None

        nx, ny, nw, nh = nutrition_region.bbox
        band_height = max(15, int(height * 0.03))
        y_start = max(0, ny - band_height)

        return LayoutRegion(
            label="serving_size",
            bbox=(0, y_start, width, ny - y_start if ny > y_start else band_height),
            confidence=0.5,
            text_density=0.0,
        )

    def _detect_manufacturer_region(
        self, text_bands: List[Dict[str, object]], width: int, height: int
    ) -> Optional[LayoutRegion]:
        """
        Approximate the manufacturer/legal info region as the
        text-dense band(s) closest to the bottom of the image, which is
        the conventional placement on most packaging.

        Args:
            text_bands: List of text-dense band dicts.
            width: Image width.
            height: Image height.

        Returns:
            LayoutRegion for the manufacturer section, or None if no
            text-dense bands exist near the bottom.
        """
        if not text_bands:
            return None

        bottom_threshold = self._num_bands * 0.75
        bottom_bands = [b for b in text_bands if b["index"] >= bottom_threshold]
        if not bottom_bands:
            return None

        y_start = int(bottom_bands[0]["y_start"])
        y_end = int(bottom_bands[-1]["y_end"])
        avg_density = float(np.mean([b["density"] for b in bottom_bands]))

        return LayoutRegion(
            label="manufacturer_section",
            bbox=(0, y_start, width, y_end - y_start),
            confidence=round(min(1.0, avg_density * 4.0), 3),
            text_density=round(avg_density, 4),
        )

    def _detect_barcode_region(
        self, gray: np.ndarray, width: int, height: int
    ) -> Optional[LayoutRegion]:
        """
        Detect a probable barcode region using gradient-based texture
        analysis: barcodes exhibit strong, densely-packed vertical
        edges concentrated in a small area.

        Args:
            gray: Grayscale image array.
            width: Image width.
            height: Image height.

        Returns:
            LayoutRegion for the probable barcode area, or None if no
            sufficiently strong candidate region was found.
        """
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.subtract(cv2.convertScaleAbs(grad_x), cv2.convertScaleAbs(grad_y))
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = (width * height) * 0.005

        if area < min_area:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        confidence = round(min(1.0, area / (width * height)), 3)

        return LayoutRegion(
            label="barcode_area",
            bbox=(x, y, w, h),
            confidence=confidence,
            text_density=0.0,
        )

    def _largest_contiguous_run(
        self,
        bands: List[Dict[str, object]],
        max_index_fraction: float = 1.0,
        min_index_fraction: float = 0.0,
    ) -> Optional[List[Dict[str, object]]]:
        """
        Find the largest contiguous run of bands (by band index) within
        an allowed index-fraction window of the total band count.

        Args:
            bands: List of text-dense band dicts.
            max_index_fraction: Upper bound (as a fraction of
                num_bands) for band indices to be considered.
            min_index_fraction: Lower bound (as a fraction of
                num_bands) for band indices to be considered.

        Returns:
            The largest contiguous run as a list of band dicts, or None
            if no bands fall within the allowed window.
        """
        max_index = self._num_bands * max_index_fraction
        min_index = self._num_bands * min_index_fraction

        candidates = [b for b in bands if min_index <= b["index"] <= max_index]
        if not candidates:
            return None

        candidates.sort(key=lambda b: b["index"])

        runs: List[List[Dict[str, object]]] = []
        current_run: List[Dict[str, object]] = [candidates[0]]

        for prev, curr in zip(candidates, candidates[1:]):
            if curr["index"] - prev["index"] <= 1:
                current_run.append(curr)
            else:
                runs.append(current_run)
                current_run = [curr]
        runs.append(current_run)

        return max(runs, key=len)