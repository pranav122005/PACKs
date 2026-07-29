"""
PACKS - OCR Scanner & LLM Extraction
Runs preprocessed nutrition label images through EasyOCR to get raw text,
then sends that raw text to Groq (llama-3.1-8b-instant) to extract a
strict, structured JSON payload of macro/ingredient/allergen data.
"""

import json
import logging
import threading

import easyocr
from groq import Groq, APIError, APIConnectionError, APITimeoutError

from backend.config import get_config
from backend.ocr.preprocessor import preprocess_for_ocr, ImagePreprocessError

logger = logging.getLogger(__name__)

_config = get_config()

REQUIRED_KEYS = ("calories", "protein", "carbs", "fats", "ingredients", "allergens")
NUMERIC_KEYS = ("calories", "protein", "carbs", "fats")
LIST_KEYS = ("ingredients", "allergens")

SYSTEM_PROMPT = """You are a nutrition label data extraction engine. You will be given raw, \
possibly noisy OCR text extracted from a photo of a food product's nutrition facts label and \
ingredients list. Your ONLY job is to extract structured nutrition data from it.

Rules you MUST follow:
1. Respond with ONLY a single valid JSON object. No prose, no markdown, no code fences, no explanation.
2. The JSON object must contain EXACTLY these six keys, no more, no fewer:
   - "calories": number (float). Total calories per serving. Use 0 if not present in the text.
   - "protein": number (float), in grams per serving. Use 0 if not present.
   - "carbs": number (float), in grams per serving (total carbohydrates). Use 0 if not present.
   - "fats": number (float), in grams per serving (total fat). Use 0 if not present.
   - "ingredients": array of strings. Each element is a single ingredient name, cleaned up \
(proper capitalization, no OCR artifacts, no percentages or parenthetical sub-ingredients \
unless they are themselves a distinct named ingredient). Use an empty array [] if no \
ingredients list is present in the text.
   - "allergens": array of strings. Common allergens either explicitly declared (e.g. \
"Contains: Milk, Soy") or clearly inferable from the ingredients list (e.g. "whey" implies \
Milk, "wheat flour" implies Wheat/Gluten). Use standard allergen names: Milk, Eggs, Fish, \
Shellfish, Tree Nuts, Peanuts, Wheat, Soy, Sesame. Use an empty array [] if none are found.
3. Never fabricate numbers. If a value is genuinely absent or unreadable from the OCR text, use 0 \
for numeric fields and [] for list fields rather than guessing.
4. OCR text may contain artifacts, misspellings, or broken line breaks — use your best judgment \
to reconstruct the intended values, but do not invent data that isn't reasonably present in the text.
5. Numeric values must be plain numbers with no units, commas, or symbols (e.g. 12.5, not "12.5g").

Return ONLY the JSON object."""


class OCRExtractionError(Exception):
    """Raised when OCR text extraction fails."""


class LLMExtractionError(Exception):
    """Raised when the Groq structuring step fails or returns an invalid payload."""


# --- EasyOCR reader: lazy singleton -----------------------------------------------------------
# Model loading is expensive (~seconds), so we initialize it once per process,
# guarded by a lock since Flask's dev server can be multi-threaded.

_reader_lock = threading.Lock()
_reader_instance = None


def _get_reader() -> easyocr.Reader:
    global _reader_instance
    if _reader_instance is None:
        with _reader_lock:
            if _reader_instance is None:  # double-checked locking
                logger.info("Initializing EasyOCR reader (en, CPU)...")
                _reader_instance = easyocr.Reader(["en"], gpu=False)
    return _reader_instance


# --- Groq client: lazy singleton ---------------------------------------------------------------

_groq_client_lock = threading.Lock()
_groq_client_instance = None


def _get_groq_client() -> Groq:
    global _groq_client_instance
    if _groq_client_instance is None:
        with _groq_client_lock:
            if _groq_client_instance is None:
                if not _config.GROQ_API_KEY:
                    raise LLMExtractionError(
                        "GROQ_API_KEY is not configured. Set it in backend/.env"
                    )
                _groq_client_instance = Groq(api_key=_config.GROQ_API_KEY)
    return _groq_client_instance


def extract_text_from_image(image_path: str) -> str:
    """
    Preprocesses the image with OpenCV, then runs EasyOCR on it to
    extract raw text.

    Args:
        image_path: path to the source nutrition label image.

    Returns:
        The raw OCR text, with detected text lines joined by newlines.

    Raises:
        OCRExtractionError: if preprocessing or OCR fails, or no text is found.
    """
    try:
        processed_image = preprocess_for_ocr(image_path)
    except ImagePreprocessError as exc:
        raise OCRExtractionError(f"Image preprocessing failed: {exc}") from exc

    try:
        reader = _get_reader()
        # detail=0 returns only the recognized strings (no boxes/confidence),
        # paragraph=True merges nearby text into readable blocks/lines.
        results = reader.readtext(processed_image, detail=0, paragraph=True)
    except Exception as exc:  # noqa: BLE001 - surface any EasyOCR runtime failure uniformly
        raise OCRExtractionError(f"EasyOCR failed to process the image: {exc}") from exc

    raw_text = "\n".join(line.strip() for line in results if line and line.strip())

    if not raw_text:
        raise OCRExtractionError(
            "No readable text was detected in the image. Try a clearer, well-lit photo."
        )

    return raw_text


def _validate_and_normalize(payload: dict) -> dict:
    """Validates the LLM's JSON payload against the required schema and coerces types."""
    if not isinstance(payload, dict):
        raise LLMExtractionError("LLM response was not a JSON object.")

    normalized = {}

    for key in NUMERIC_KEYS:
        value = payload.get(key, 0)
        try:
            normalized[key] = round(float(value), 2) if value not in (None, "") else 0.0
        except (TypeError, ValueError):
            logger.warning("Non-numeric value for %s: %r — defaulting to 0.0", key, value)
            normalized[key] = 0.0

    for key in LIST_KEYS:
        value = payload.get(key, [])
        if isinstance(value, list):
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(value, str) and value.strip():
            # Occasionally a model returns a comma-separated string instead of an array
            normalized[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            normalized[key] = []

    return normalized


def extract_nutrition_json(raw_text: str) -> dict:
    """
    Sends raw OCR text to Groq (llama-3.1-8b-instant) and returns a validated,
    strictly-typed nutrition data dict.

    Args:
        raw_text: raw OCR output from extract_text_from_image().

    Returns:
        dict with keys: calories, protein, carbs, fats (floats),
        ingredients, allergens (lists of strings).

    Raises:
        LLMExtractionError: if the Groq call fails or returns invalid JSON/schema.
    """
    client = _get_groq_client()

    try:
        response = client.chat.completions.create(
            model=_config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR TEXT:\n\n{raw_text}"},
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
    except APITimeoutError as exc:
        raise LLMExtractionError("Groq API request timed out.") from exc
    except APIConnectionError as exc:
        raise LLMExtractionError(f"Could not connect to Groq API: {exc}") from exc
    except APIError as exc:
        raise LLMExtractionError(f"Groq API returned an error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - catch-all so callers get a single exception type
        raise LLMExtractionError(f"Unexpected error calling Groq API: {exc}") from exc

    try:
        content = response.choices[0].message.content
    except (IndexError, AttributeError) as exc:
        raise LLMExtractionError("Groq API response was empty or malformed.") from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMExtractionError(f"Groq response was not valid JSON: {exc}") from exc

    missing_keys = [k for k in REQUIRED_KEYS if k not in parsed]
    if missing_keys:
        logger.warning("Groq response missing keys %s — filling with defaults.", missing_keys)

    return _validate_and_normalize(parsed)


def scan_nutrition_label(image_path: str) -> dict:
    """
    Full pipeline: preprocess -> OCR -> Groq structured extraction.

    Args:
        image_path: path to the uploaded nutrition label image.

    Returns:
        Validated nutrition data dict (see extract_nutrition_json), plus the
        raw OCR text under the "raw_ocr_text" key for debugging/traceability.

    Raises:
        OCRExtractionError: if OCR fails.
        LLMExtractionError: if the Groq structuring step fails.
    """
    raw_text = extract_text_from_image(image_path)
    logger.info("OCR extracted %d characters of raw text.", len(raw_text))

    nutrition_data = extract_nutrition_json(raw_text)
    nutrition_data["raw_ocr_text"] = raw_text

    return nutrition_data
