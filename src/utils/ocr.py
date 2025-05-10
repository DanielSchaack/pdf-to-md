import cv2
import pytesseract
import logging
import json
from pytesseract import Output
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)


def get_text_column_data(image_path: str, language: str = 'eng', tesseract_config: str = '--psm 3') -> Dict[str, Tuple[int, int, int, int, List[str], str, int]]:
    """
    Analyzes an image to identify potential text columns using OpenCV and Tesseract,
    with added logging print statements.

    Args:
        image_path: Path to the input image file.
        language: Language code for Tesseract OCR (e.g., 'eng', 'deu').
        tesseract_config: Custom Tesseract configuration string (e.g., page segmentation mode).

    Returns:
        A dictionary containing the estimated number of columns and their bounding boxes
        based on Tesseract's block-level output. Bounding boxes are in pixel coordinates
        with a top-left origin.
        {
            columns:
            [
                {
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                    "words",
                    "full_text",
                    "tesseract_block_num"
                }
            ]
        }
        Returns an empty structure if an error occurs or no blocks are found.
    """
    logger.info(f"Starting column analysis for image: '{image_path}', Language: '{language}', Tesseract Config: '{tesseract_config}'")
    empty_result = {"columns": []}

    try:
        logger.debug("Starting image loading...")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image from path: {image_path}", exc_info=True)
            return {"numberOfTextColumns": 0, "columnBoundingBoxes_Pixels": []}
        logger.info(f"Image loaded successfully. Dimensions: {img.shape[1]}x{img.shape[0]} (WxH)")

        logger.debug("Calling Tesseract (pytesseract.image_to_data)...")
        data = pytesseract.image_to_data(
            img,
            lang=language,
            config=tesseract_config,
            output_type=Output.DICT
        )
        logger.info("Tesseract completed")
        logger.debug(f"Raw Tesseract data:\n{data}")

        num_items = len(data['level'])
        if num_items == 0 or 'block_num' not in data:
            logger.info("No text elements or block data found by Tesseract.")
            return empty_result

        # This dictionary will store data for each block_num.
        # Value: {'x_min', 'y_min', 'x_max', 'y_max', 'elements_count', 'words'}
        found_blocks: Dict[int, Dict[str, Any]] = {}

        logger.info(f"Processing {num_items} items from Tesseract output to identify blocks and words...")
        for i in range(num_items):
            block_num = data['block_num'][i]
            # Consider elements that are part of a block (block_num > 0)
            # and have valid geometry (width and height > 0).
            if block_num > 0 and data['width'][i] > 0 and data['height'][i] > 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                if block_num not in found_blocks:
                    found_blocks[block_num] = {
                        'x_min': x,
                        'y_min': y,
                        'x_max': x + w,
                        'y_max': y + h,
                        'elements_count': 1,
                        'words': []  # Initialize list to store words for this block
                    }

                # Add the text of the current element (word) to the block's word list
                word_text = data['text'][i].strip()
                if word_text:
                    found_blocks[block_num]['words'].append(word_text)

        if not found_blocks:
            logger.warning("No valid blocks identified after processing Tesseract data.")
            return empty_result

        columns_data: List[Dict[str, Any]] = []
        logger.info(f"Consolidating {len(found_blocks)} Tesseract blocks into columns with their words")

        sorted_block_nums = sorted(found_blocks.keys())

        for block_num in sorted_block_nums:
            block_info = found_blocks[block_num]

            words_in_block = block_info.get("words", [])
            full_text_of_block = " ".join(words_in_block)
            current_block = {
                "x_min": block_info["x_min"],
                "y_min": block_info["y_min"],
                "x_max": block_info["x_max"],
                "y_max": block_info["y_max"],
                "words": words_in_block,
                "full_text": full_text_of_block,
                "tesseract_block_num": block_num  # Retain original block number for reference
            }
            columns_data.append(current_block)
            logger.debug(f"Block {block_num}: \n{columns_data}")

        logger.info(f"Analysis complete. Detected {len(columns_data)} blocks.")
        return {
            "columns": columns_data
        }

    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract is not installed or not in your PATH.", exc_info=True)
        return empty_result
    except Exception as e:
        logger.error(f"An error occurred during image processing of Tesseract OCR: {e}", e, exc_info=True)
        return empty_result


def get_full_text_of_blocks(column_data) -> str:
    current_blocks = column_data["columns"]
    combined_full_text = "\n".join(block["full_text"] for block in current_blocks)
    return combined_full_text
