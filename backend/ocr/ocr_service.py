from __future__ import annotations

import logging
from typing import Optional

import cv2

from backend.ocr.image_preprocessing import ImagePreprocessor
from backend.ocr.ocr_engine import OCREngine as EasyOCREngine
from backend.ocr.text_cleaner import TextCleaner

logger = logging.getLogger("packs.ocr_service")


class OCRService:
    """Extracts text from images using EasyOCR with OpenCV preprocessing."""

    def __init__(self) -> None:
        self._preprocessor = ImagePreprocessor()
        self._ocr_engine: Optional[EasyOCREngine] = None
        self._text_cleaner = TextCleaner()

    def _get_ocr(self) -> EasyOCREngine:
        if self._ocr_engine is None:
            self._ocr_engine = EasyOCREngine(languages=["en"], gpu=False)
        return self._ocr_engine

    def extract_text(self, image_path: str) -> Optional[str]:
        logger.info("OCR requested for: %s", image_path)
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error("Failed to read image: %s", image_path)
                return None

            preprocessed = self._preprocessor.preprocess(image)
            ocr_engine = self._get_ocr()
            ocr_result = ocr_engine.extract_text(preprocessed.final)

            if not ocr_result.full_text.strip():
                logger.warning("OCR produced no text for: %s", image_path)
                return None

            cleaned = self._text_cleaner.process(ocr_result.full_text)
            return cleaned.cleaned_text
        except Exception:
            logger.exception("OCR extraction failed for: %s", image_path)
            return None
