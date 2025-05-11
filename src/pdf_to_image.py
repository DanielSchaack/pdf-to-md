

import fitz
import json
import os
mport fitz
import json
import os



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

