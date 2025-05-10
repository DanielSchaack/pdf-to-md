import fitz
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

