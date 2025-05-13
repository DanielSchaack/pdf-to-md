import cv2
import pytesseract
import logging
import os
import base64
import fitz
from pytesseract import Output
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class FileService:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir

    def get_text_column_data(self,
                             image_path: str,
                             language: str = 'eng',
                             tesseract_config: str = '--psm 3') -> List[Tuple[int, int, int, int, List[str], str, int]]:
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
            Returns an empty structure if an error occurs or no blocks are found.
        """
        logger.info(f"Starting column analysis for image: '{image_path}', Language: '{language}', Tesseract Config: '{tesseract_config}'")
        empty_result = []

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
                        found_blocks[block_num] = {'x_min': x,
                                                   'y_min': y,
                                                   'x_max': x + w,
                                                   'y_max': y + h,
                                                   'elements_count': 1,
                                                   'words': []}

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

                if len(words_in_block) == 0:
                    logger.debug(f"Block {block_num} does not contain any words, skipping")
                    continue

                full_text_of_block = " ".join(words_in_block)
                current_block = {"x_min": block_info["x_min"],
                                 "y_min": block_info["y_min"],
                                 "x_max": block_info["x_max"],
                                 "y_max": block_info["y_max"],
                                 "words": words_in_block,
                                 "full_text": full_text_of_block,
                                 "tesseract_block_num": block_num}
                columns_data.append(current_block)
                logger.debug(f"Block {block_num}: \n{current_block}")

            logger.info(f"Analysis complete. Detected {len(columns_data)} blocks.")
            return columns_data

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed or not in your PATH.", exc_info=True)
            return empty_result
        except Exception as e:
            logger.error(f"An error occurred during image processing of Tesseract OCR: {e}", exc_info=True)
            return empty_result

    def get_filename_from_path(self, file_path: str) -> str:
        """
          Extracts the filename from a given file path, without the extension.

          Args:
              file_path: The full path to the file (e.g., "/path/to/your/file.txt").

          Returns:
              The filename part of the path, excluding the extension (e.g., "file").
      """
        base_name = os.path.basename(file_path)
        file_name_without_extension, _ = os.path.splitext(base_name)
        logger.debug(f"Base filename is {file_name_without_extension}")
        return file_name_without_extension

    def convert_pdf_to_images(self,
                              pdf_path: str,
                              filename: str,
                              image_format: str = "png",
                              dpi: int = 300,
                              colorspace: str = "rgb",  # "rgb", "gray", "cmyk"
                              use_alpha: bool = False,
                              page_numbers: Optional[List[int]] = None) -> List[str]:  # None for all, or list of 1-based page numbers [1, 3, 5]
        """
        Converts PDF pages to images using PyMuPDF (Fitz) with configurable quality settings.

        Args:
            pdf_path (str): Path to the PDF file.
            filename_prefix (str): Prefix for the output image filenames.
            image_format (str): Desired image format (e.g., "png", "jpeg", "tiff", "ppm").
            dpi (int): Dots Per Inch for rendering. Higher DPI means better quality.
            colorspace_str (str): Colorspace to use ("rgb", "gray", "cmyk").
            use_alpha (bool): Whether to include an alpha channel (transparency).
            page_numbers (list, optional): List of 1-based page numbers to convert.
                                           If None, all pages are converted.

        Returns:
            list: A list of paths to the saved image files, or an empty list on failure.
        """
        # Input validation
        if colorspace.lower() == "rgb":
            fitz_colorspace = fitz.csRGB
        elif colorspace.lower() == "gray":
            fitz_colorspace = fitz.csGRAY
        elif colorspace.lower() == "cmyk":
            fitz_colorspace = fitz.csCMYK
        else:
            logger.warn(f"Unknown colorspace '{colorspace}'. Defaulting to RGB.")
            fitz_colorspace = fitz.csRGB

        image_format = image_format.lower()
        if image_format == "jpg":
            image_format = "jpeg"

        if image_format not in ["png", "jpeg", "tiff", "ppm", "pnm", "pgm", "pbm"]:
            logger.warn(f"Image format '{image_format}' might not be directly supported by pix.save(). "
                        f"Ensure the filename extension is appropriate. Using '{image_format}' as extension.")

        if not os.path.exists(pdf_path):
            logger.error(f"Error: PDF file not found at {pdf_path}")
            return []

        image_dir_path = os.path.join(self.output_dir,
                                      filename,
                                      image_format)
        logger.debug(f"Image dir path is '{image_dir_path}'")

        if not os.path.exists(image_dir_path):
            logger.info(f"Creating output folder: {image_dir_path}")
            os.makedirs(image_dir_path)

        saved_image_paths = []
        pages_to_process = []
        with fitz.open(pdf_path) as doc:
            total_pages_in_doc = len(doc)

            if page_numbers:
                for p_num_1_based in page_numbers:
                    if 1 <= p_num_1_based <= total_pages_in_doc:
                        pages_to_process.append(p_num_1_based - 1)  # Convert to 0-based index
                    else:
                        logger.warn(f"Page number {p_num_1_based} is out of range (1-{total_pages_in_doc}). Skipping.")
            else:
                pages_to_process = range(total_pages_in_doc)

            if not pages_to_process:
                logger.warn("No valid pages selected for conversion.")
                return []

            logger.debug(f"Converting PDF: {pdf_path}")
            logger.debug(f"Settings: DPI={dpi}, Format={image_format}, Colorspace={colorspace}, Alpha={use_alpha}")
            logger.debug(f"Outputting to: {image_dir_path}")

            # Determine padding for filenames based on the max page number being processed
            max_page_num_for_naming = max(p + 1 for p in pages_to_process) if pages_to_process else 0
            logger.debug(f"Provided maximal page number is {max_page_num_for_naming}")

            page_num_padding = len(str(max_page_num_for_naming))
            logger.debug(f"Based on maximal page number is the padding of zeroes {page_num_padding}")

            for page_index_0_based in pages_to_process:
                page_num_1_based = page_index_0_based + 1
                try:
                    page = doc.load_page(page_index_0_based)

                    # Render page to an image (pixmap)
                    pix = page.get_pixmap(dpi=dpi,
                                          colorspace=fitz_colorspace,
                                          alpha=use_alpha)

                    # Naming: page_001.png, page_002.png etc.
                    image_filename = os.path.join(image_dir_path,
                                                  f"{filename}_{str(page_num_1_based).zfill(page_num_padding)}.{image_format}")

                    # Save the pixmap as an image file
                    pix.save(image_filename)

                    saved_image_paths.append(image_filename)
                    logger.info(f"Saved: {image_filename}")

                except Exception as e:
                    logger.error(f"Error converting page {page_num_1_based}: {e}", exc_info=True)
                    # Continue with other pages if one fails

            if saved_image_paths:
                logger.info(f"Successfully converted {len(saved_image_paths)} pages.")
            else:
                logger.warn("No images were successfully converted.")
        return saved_image_paths

    def convert_chunks_to_files(self,
                                filename: str,
                                filetype: str,
                                chunks: List[List[str]]):
        """
        Convert a list of Markdown chunks into individual files with sequential filenames.

        Args:
            filename_prefix (str): The prefix to use for each chunk file name.
            suffix (str): The file extension to use for the chunk files.
            chunks (List[List[str]]): A list of chunks, where each chunk is a list of strings representing lines in that chunk.

        Raises:
            ValueError: If the output directory does not exist or cannot be created.
            PermissionError: If there are insufficient permissions to write to the output directory.
        """

        amount_chunks: int = len(chunks)
        logger.debug(f"Creating {amount_chunks} chunks")
        page_num_padding = len(str(amount_chunks))
        logger.debug(f"Padding chunk number to {page_num_padding}")

        for index, chunk in enumerate(chunks):
            chunk_path = os.path.join(self.output_dir,
                                      filename,
                                      filetype)
            logger.debug(f"Chunk path is '{chunk_path}'")

            chunk_filepath = os.path.join(chunk_path,
                                          f"{filename}_{str(index).zfill(page_num_padding)}.{filetype}")
            logger.debug(f"Chunk filepath is '{chunk_filepath}'")

            chunk_text = "\n".join(chunk)
            logger.debug(f"Chunk text is '{chunk_text}'")
            save(chunk_path, chunk_filepath, chunk_text)

    def save_to_filesystem(self,
                           filename: str,
                           text: str,
                           filetype: str):
        file_dir = os.path.join(self.output_dir,
                                filename,
                                filetype)
        logger.debug(f"Chunk path is '{file_dir}'")

        filepath = os.path.join(file_dir, f"{filename}.{filetype}")
        logger.debug(f"Chunk filepath is '{filepath}'")

        save(file_dir, filepath, text)


def save(file_dir: str, filepath: str, content: str):
    # Ensure the output directory exists
    if not os.path.exists(file_dir):
        try:
            os.makedirs(file_dir)
            logger.info(f"Created directory '{filepath}'")
        except Exception as e:
            logger.error(f"Failed to create output directory at {filepath}: {e}", exc_info=True)
            raise

    try:
        with open(filepath, "w") as chunk_file:
            chunk_file.write(content)
            logger.info(f"Successfully written to file '{filepath}'")

    except PermissionError as e:
        logger.error(f"Permission denied when trying to write file at {filepath}: {e}", exc_info=True)
        raise

    except Exception as e:
        logger.error(f"An error occurred while writing file at {filepath}: {e}", exc_info=True)
        raise


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
