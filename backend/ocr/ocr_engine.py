"""
backend/ocr/ocr_engine.py

OCR Engine — thin, production-hardened wrapper around EasyOCR.

Responsibilities:
    - Lazily initialize and cache a single EasyOCR Reader per process
      (model loading is expensive; we do it once).
    - Run text detection + recognition on a preprocessed image.
    - Normalize EasyOCR's raw output into typed, sorted `OCRTextBlock`s
      (reading-order: top-to-bottom, then left-to-right).
    - Isolate all EasyOCR-specific API details behind this class so the
      rest of PACKS never imports `easyocr` directly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class OCRTextBlock:
    """A single detected text region with its bounding box and confidence."""

    text: str
    confidence: float
    bounding_box: List[Tuple[float, float]]  # 4 (x, y) corner points

    @property
    def top_y(self) -> float:
        return min(point[1] for point in self.bounding_box)

    @property
    def left_x(self) -> float:
        return min(point[0] for point in self.bounding_box)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bounding_box": [[round(x, 1), round(y, 1)] for x, y in self.bounding_box],
        }


@dataclass
class OCRResult:
    """Complete raw OCR output for one image."""

    blocks: List[OCRTextBlock] = field(default_factory=list)
    full_text: str = ""
    average_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "full_text": self.full_text,
            "average_confidence": round(self.average_confidence, 4),
            "block_count": len(self.blocks),
        }


class OCREngineError(RuntimeError):
    """Raised when the EasyOCR reader fails to initialize or run."""


class OCREngine:
    """
    Wraps `easyocr.Reader` with lazy, thread-safe initialization and a
    normalized output format.

    Usage:
        engine = OCREngine(languages=["en"], gpu=False)
        result = engine.extract_text(preprocessed_image)
    """

    _init_lock = threading.Lock()

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
        min_confidence: float = 0.35,
        paragraph: bool = False,
    ) -> None:
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._min_confidence = min_confidence
        self._paragraph = paragraph
        self._reader: Optional[Any] = None

    def _get_reader(self) -> Any:
        """Lazily create the EasyOCR Reader exactly once (thread-safe)."""
        if self._reader is None:
            with self._init_lock:
                if self._reader is None:
                    try:
                        import easyocr  # imported lazily to keep module import light
                    except ImportError as exc:
                        raise OCREngineError(
                            "easyocr is not installed. Install it with `pip install easyocr`."
                        ) from exc
                    try:
                        self._reader = easyocr.Reader(self._languages, gpu=self._gpu)
                    except Exception as exc:  # pragma: no cover - defensive
                        raise OCREngineError(f"Failed to initialize EasyOCR reader: {exc}") from exc
        return self._reader

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Run OCR on a preprocessed image (grayscale or binarized numpy
        array) and return a normalized, reading-order-sorted OCRResult.
        """
        if image is None or image.size == 0:
            raise ValueError("Received an empty or invalid image array for OCR.")

        reader = self._get_reader()
        try:
            raw_results = reader.readtext(image, detail=1, paragraph=self._paragraph)
        except Exception as exc:
            raise OCREngineError(f"EasyOCR failed to process the image: {exc}") from exc

        blocks: List[OCRTextBlock] = []
        for entry in raw_results:
            bounding_box, text, confidence = self._unpack_entry(entry)
            if not text or confidence < self._min_confidence:
                continue
            blocks.append(
                OCRTextBlock(
                    text=text.strip(),
                    confidence=float(confidence),
                    bounding_box=[(float(x), float(y)) for x, y in bounding_box],
                )
            )

        blocks = self._sort_reading_order(blocks)
        full_text = "\n".join(b.text for b in blocks)
        average_confidence = (
            sum(b.confidence for b in blocks) / len(blocks) if blocks else 0.0
        )

        return OCRResult(blocks=blocks, full_text=full_text, average_confidence=average_confidence)

    @staticmethod
    def _unpack_entry(entry: Any) -> Tuple[List[Tuple[float, float]], str, float]:
        """
        EasyOCR's `paragraph=False` mode returns (bbox, text, confidence).
        `paragraph=True` mode returns (bbox, text) with no confidence, so
        we default confidence to 1.0 in that case.
        """
        if len(entry) == 3:
            bounding_box, text, confidence = entry
        else:
            bounding_box, text = entry
            confidence = 1.0
        return bounding_box, text, confidence

    @staticmethod
    def _sort_reading_order(blocks: List[OCRTextBlock], row_tolerance_px: float = 15.0) -> List[OCRTextBlock]:
        """
        Sort detected blocks into natural reading order: group blocks into
        rows based on vertical proximity, then order each row left-to-right.
        """
        if not blocks:
            return blocks

        sorted_by_y = sorted(blocks, key=lambda b: b.top_y)
        rows: List[List[OCRTextBlock]] = []
        current_row: List[OCRTextBlock] = [sorted_by_y[0]]
        current_row_y = sorted_by_y[0].top_y

        for block in sorted_by_y[1:]:
            if abs(block.top_y - current_row_y) <= row_tolerance_px:
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]
                current_row_y = block.top_y
        rows.append(current_row)

        ordered: List[OCRTextBlock] = []
        for row in rows:
            ordered.extend(sorted(row, key=lambda b: b.left_x))
        return ordered
