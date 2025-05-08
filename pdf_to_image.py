import fitz
import cv2
import pytesseract
from pytesseract import Output
from typing import List, Dict, Any, Tuple
import json
import os


def convert_pdf_to_images(
    pdf_path,
    output_folder="output",
    image_format="png",
    dpi=300,
    colorspace_str="rgb",  # "rgb", "gray", "cmyk"
    use_alpha=False,
    page_numbers=None, # None for all, or list of 1-based page numbers [1, 3, 5]
    filename_prefix="page_"
):
    """
    Converts PDF pages to images using PyMuPDF (Fitz) with configurable quality settings.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Directory to save the output images.
        image_format (str): Desired image format (e.g., "png", "jpeg", "tiff", "ppm").
        dpi (int): Dots Per Inch for rendering. Higher DPI means better quality.
        colorspace_str (str): Colorspace to use ("rgb", "gray", "cmyk").
        use_alpha (bool): Whether to include an alpha channel (transparency).
        page_numbers (list, optional): List of 1-based page numbers to convert.
                                       If None, all pages are converted.
        filename_prefix (str): Prefix for the output image filenames.

    Returns:
        list: A list of paths to the saved image files, or an empty list on failure.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return []

    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder)

    # --- Parameter Mapping ---
    if colorspace_str.lower() == "rgb":
        fitz_colorspace = fitz.csRGB
    elif colorspace_str.lower() == "gray":
        fitz_colorspace = fitz.csGRAY
    elif colorspace_str.lower() == "cmyk":
        fitz_colorspace = fitz.csCMYK
    else:
        print(f"Warning: Unknown colorspace '{colorspace_str}'. Defaulting to RGB.")
        fitz_colorspace = fitz.csRGB

    image_format = image_format.lower()
    if image_format == "jpg":
        image_format = "jpeg"

    if image_format not in ["png", "jpeg", "tiff", "ppm", "pnm", "pgm", "pbm"]: # Common raster formats
        print(f"Warning: Image format '{image_format}' might not be directly supported by pix.save(). "
              "Ensure the filename extension is appropriate. Using '{image_format}' as extension.")

    saved_image_paths = []
    total_pages_in_doc = len(doc)
    pages_to_process = []

    if page_numbers:
        for p_num_1_based in page_numbers:
            if 1 <= p_num_1_based <= total_pages_in_doc:
                pages_to_process.append(p_num_1_based - 1) # Convert to 0-based index
            else:
                print(f"Warning: Page number {p_num_1_based} is out of range (1-{total_pages_in_doc}). Skipping.")
    else:
        pages_to_process = range(total_pages_in_doc)

    if not pages_to_process:
        print("No valid pages selected for conversion.")
        doc.close()
        return []

    print(f"Converting PDF: {pdf_path}")
    print(f"Settings: DPI={dpi}, Format={image_format}, Colorspace={colorspace_str}, Alpha={use_alpha}")
    print(f"Outputting to: {output_folder}")

    # Determine padding for filenames based on the max page number being processed
    max_page_num_for_naming = max(p + 1 for p in pages_to_process) if pages_to_process else 0
    page_num_padding = len(str(max_page_num_for_naming))

    for page_index_0_based in pages_to_process:
        page_num_1_based = page_index_0_based + 1
        try:
            page = doc.load_page(page_index_0_based)

            # Render page to an image (pixmap)
            pix = page.get_pixmap(
                dpi=dpi,
                colorspace=fitz_colorspace,
                alpha=use_alpha
            )

            # Naming: page_001.png, page_002.png etc.
            image_filename = os.path.join(
                output_folder,
                f"{filename_prefix}{str(page_num_1_based).zfill(page_num_padding)}.{image_format}"
            )

            # Save the pixmap as an image file
            pix.save(image_filename)

            saved_image_paths.append(image_filename)
            print(f"  Saved: {image_filename}")

        except Exception as e:
            print(f"Error converting page {page_num_1_based}: {e}")
            # Continue with other pages if one fails

    doc.close()
    if saved_image_paths:
        print(f"\nSuccessfully converted {len(saved_image_paths)} pages.")
    else:
        print("\nNo images were successfully converted.")
    return saved_image_paths


def get_text_column_data(image_path: str, language: str = 'eng', tesseract_config: str = '--psm 6') -> Dict[str, Any]:
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
            "numberOfTextColumns": int,
            "columnBoundingBoxes_Pixels": [[x_min, y_min, x_max, y_max], ...]
        }
        Returns an empty structure if an error occurs or no blocks are found.
    """
    print(f"\n[LOG] Starting column analysis for image: '{image_path}'")
    print(f"[LOG] Language: '{language}', Tesseract Config: '{tesseract_config}'")
    empty_result = {"numberOfTextColumns": 0, "columns": []}

    try:
        print("[LOG] Starting image loading...")
        img = cv2.imread(image_path)
        if img is None:
            print(f"[LOG] Error: Could not read image from path: {image_path}")
            return {"numberOfTextColumns": 0, "columnBoundingBoxes_Pixels": []}
        print(f"[LOG] Image loaded successfully. Dimensions: {img.shape[1]}x{img.shape[0]} (WxH)")

        print("[LOG] Calling Tesseract (pytesseract.image_to_data)...")
        data = pytesseract.image_to_data(
            img,
            lang=language,
            config=tesseract_config,
            output_type=Output.DICT
        )
        print("[LOG] Raw Tesseract data")
        for key in data:
            print(f"[LOG]   {key}: {data[key][:5]}...")

        num_items = len(data['level'])
        if num_items == 0 or 'block_num' not in data:
            print("[LOG] No text elements or block data found by Tesseract.")
            return empty_result

        # This dictionary will store data for each block_num.
        # Value: {'x_min', 'y_min', 'x_max', 'y_max', 'elements_count', 'words'}
        found_blocks: Dict[int, Dict[str, Any]] = {}

        print(f"[LOG] Processing {num_items} items from Tesseract output to identify blocks and words...")
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
                else:
                    current_block = found_blocks[block_num]
                    current_block['x_min'] = min(current_block['x_min'], x)
                    current_block['y_min'] = min(current_block['y_min'], y)
                    current_block['x_max'] = max(current_block['x_max'], x + w)
                    current_block['y_max'] = max(current_block['y_max'], y + h)
                    current_block['elements_count'] += 1
                
                # Add the text of the current element (word) to the block's word list
                word_text = data['text'][i].strip()
                if word_text: # Ensure there's actual text
                    found_blocks[block_num]['words'].append(word_text)

        if not found_blocks:
            print("[LOG] No valid blocks identified after processing Tesseract data.")
            return empty_result

        columns_data: List[Dict[str, Any]] = []
        print(f"[LOG] Consolidating {len(found_blocks)} Tesseract blocks into columns with their words:")

        sorted_block_nums = sorted(found_blocks.keys())

        for block_num in sorted_block_nums:
            block_info = found_blocks[block_num]

            # Optional filtering heuristics (as before) can be applied here
            # e.g., if block_info['elements_count'] < min_elements or len(block_info['words']) < min_words:
            #     print(f"[LOG]   Skipping block {block_num} due to filtering criteria.")
            #     continue

            col_bbox = [
                block_info['x_min'],
                block_info['y_min'],
                block_info['x_max'],
                block_info['y_max']
            ]

            words_in_block = block_info.get('words', []) # Should exist due to initialization
            full_text_of_block = " ".join(words_in_block)

            columns_data.append({
                "boundingBox_Pixels": col_bbox,
                "words": words_in_block,
                "full_text": full_text_of_block,
                "tesseract_block_num": block_num # Retain original block number for reference
            })
            print(f"[LOG]   Block {block_num}: BBox={col_bbox}, Elements={block_info['elements_count']}, Words Found={len(words_in_block)}")

        # Sort the identified columns by their horizontal position (left-to-right)
        # columns_data.sort(key=lambda col: col["boundingBox_Pixels"][0])
        columns_data.sort(key=lambda col: col["tesseract_block_num"])

        print(f"[LOG] Analysis complete. Detected {len(columns_data)} blocks.")
        return {
            "numberOfTextColumns": len(columns_data),
            "columns": columns_data
        }

    except pytesseract.TesseractNotFoundError:
        print("[LOG] Error: Tesseract is not installed or not in your PATH.")
        return empty_result
    except Exception as e:
        print(f"[LOG] An error occurred during image processing or Tesseract OCR: {e}")
        return empty_result


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_pdf_file = "examples/Ergänzungstarif für Wahlleistungen im Krankenhaus SW101.pdf"
    if os.path.exists(test_pdf_file):
        output_dir1 = os.path.join(current_dir, "examples/png")
        images1 = convert_pdf_to_images(
            pdf_path=test_pdf_file,
            output_folder=output_dir1,
            image_format="png",
            dpi=300,
            colorspace_str="rgb",
            use_alpha=False
        )
        if images1:
            json_data = json.dumps(get_text_column_data(images1[0], "deu", "--psm 3"), indent=4)
            print(json_data)
    else:
        print(f"Test PDF '{test_pdf_file}' not found. Please place a PDF there or update path.")
        # if images1: print(f"Generated: {images1}")

