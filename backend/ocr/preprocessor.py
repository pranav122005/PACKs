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

