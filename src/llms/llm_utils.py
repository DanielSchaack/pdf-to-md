import base64
import logging

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image file to a base64 string.

    Args:
        image_path: The path to the image file.

    Returns:
        The base64 encoded string of the image.

    Raises:
        FileNotFoundError: If the image_path does not exist.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}", exc_info=True)
        raise

