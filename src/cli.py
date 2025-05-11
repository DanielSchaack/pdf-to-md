import logging
import logging.config
import argparse
import os
from llms.llm_providers import LLMProviderFactory, LLMProvider
from utils.strings import get_markdown_headers_and_tables, convert_markdown_to_chunks
from utils.files import get_text_column_data, get_full_text_of_blocks, convert_pdf_to_images, convert_chunks_to_files, get_filename_from_path

logging.config.fileConfig("src/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an AI assistant specialized in OCR for German legal and contractual documents (e.g., "Allgemeine Vertragsbedingungen," "Tarife"). Your primary mission is **absolute textual fidelity** and **faithful Markdown structuring.**

**Key Priorities:**

1.  **Verbatim Text Extraction (Correctness):**
    *   Extract ALL text with **100% character-for-character accuracy.** This is paramount for legal German.
    *   **Meticulous Detail:** Transcribe exactly:
        *   German characters: ä, ö, ü, Ä, Ö, Ü, ß.
        *   Punctuation, capitalization, numbers (policy numbers, amounts, dates, §, Abs., Nr.).
        *   Special symbols.
    *   **No Alterations:**
        *   Do NOT omit, add, paraphrase, summarize, or "correct" anything.
        *   Preserve original wording, spelling (even apparent typos), and grammar verbatim.
    *   **Illegibility:** If truly unreadable, use `[unleserlich]` or `[unklar: Grund]`. **Do not guess.**
2.  **Markdown Formatting (Structure & Readability):**
    *   **Goal:** Reflect the document's original structure to aid readability while preserving verbatim text.
    *   **Headings:** Use `# H1`, `## H2`, etc., for visual hierarchy. Text must be verbatim.
    *   **Paragraphs:** Separate with a blank line. Replicate original numbering.
    *   **Lists:**
        *   Use `*` or `-` for unordered lists.
        *   Use original numbering/lettering (e.g., `1.`, `a)`, `(i)`) for ordered lists.
        *   Preserve indentation for nested lists.
    *   **Tables:** Use Markdown table syntax if possible. All cell content must be verbatim.
        ```markdown
        | Header 1 | Header 2 |
        |---|---|
        | Exakter Text 1.1 | €123,45 |
        ```
    *   **Inline Formatting:**
        *   Bold: `**text**`
        *   Italic: `*text*`
        *   Underline: `<u>text</u>`
        *   Strikethrough: `~~text~~`
3.  **Output Requirements:**
    *   **ONLY the verbatim transcribed German text, formatted in Markdown.**
    *   No conversational preamble, summaries, or explanations (unless it's an `[unleserlich]` note).

Proceed with OCR and Markdown formatting, understanding that **precision and exactness of German contractual wording and formatting are non-negotiable.
"""


def test_llm_provider(image_path: str, api_key: str):
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    user_message = "No previous page."
    provider = "openrouter"
    text_model = "mistralai/mistral-small-3.1-24b-instruct:free"

    try:
        logger.info(f"Creating text provider with model: {text_model}")
        text_provider: LLMProvider = LLMProviderFactory.create_provider(
            provider_name=provider,
            api_url=OPENROUTER_API_URL,
            api_key=api_key,
        )

        logger.info(f"User: {user_message}")
        response_message = text_provider.get_completion(model=text_model, user_message=user_message, system_prompt=SYSTEM_PROMPT, image_path=image_path)
        logger.info(f"LLM ({text_model}): {response_message}")

        return response_message

    except Exception as e:
        logger.error(e, exc_info=True)
        raise


def test_logic(pdf_path: str, output_dir: str, api_key: str):
    filename = get_filename_from_path(pdf_path)

    image_paths = convert_pdf_to_images(
        pdf_path=pdf_path,
        output_dir=output_dir,
        filename=filename,
        image_format="png",
        dpi=300,
        colorspace="rgb",
        page_numbers=[1],
        use_alpha=False
    )

    for index, path in enumerate(image_paths):
        blocks = get_text_column_data(path, language="deu")
        block_text = get_full_text_of_blocks(blocks)
        response_message = test_llm_provider(path, api_key)
        headers, tables, is_table = get_markdown_headers_and_tables(response_message)
        chunks = convert_markdown_to_chunks(filename=filename, markdown_text=response_message, header_level_cutoff=3)
        convert_chunks_to_files(filename_prefix=filename, chunk_suffix="md", output_dir=output_dir, chunks=chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PDF processing with an LLM provider."
    )

    # Argument for the PDF file path
    # Use a flag (e.g., --pdf-path) and set a default value
    parser.add_argument(
        "--pdf-path", "-p",
        type=str,
        required=True,
        help="Path to the PDF document to be processed. "
    )

    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key for the Large Language Model (LLM) provider. "
             "This is a required argument. Alternatively, you can set the "
             "'API_KEY' environment variable."
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Directory where output files will be saved. Defaults to './output'."
    )

    args = parser.parse_args()

    path = args.pdf_path

    # Get API key from argument or environment variable
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("API_KEY")
        if not api_key:
            parser.error(
                "Error: LLM API key is required. Please provide it using "
                "--api-key or by setting the API_KEY environment variable."
            )

    # Ensure the output directory exists, create it if necessary
    output_directory = args.output_dir
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
            print(f"Info: Created output directory: {output_directory}")
        except OSError as e:
            parser.error(f"Error creating output directory '{output_directory}': {e}")
    elif not os.path.isdir(output_directory):
        parser.error(f"Error: '{output_directory}' exists but is not a directory.")

    test_logic(pdf_path=path, output_dir=output_directory, api_key=args.api_key)

