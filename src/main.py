import logging
import logging.config
from llms.llm_providers import LLMProviderFactory, LLMProvider
from utils.strings import get_markdown_headers_and_tables
from utils.ocr import get_text_column_data, get_full_text_of_blocks

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
        *   Italic: `_text_`
        *   Underline: `<u>text</u>`
        *   Strikethrough: `~~text~~`
3.  **Output Requirements:**
    *   **ONLY the verbatim transcribed German text, formatted in Markdown.**
    *   No conversational preamble, summaries, or explanations (unless it's an `[unleserlich]` note).

Proceed with OCR and Markdown formatting, understanding that **precision and exactness of German contractual wording and formatting are non-negotiable.
"""


def test_llm_provider():
    API_KEY = "<API_KEY>"
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    user_message = "No previous page."
    provider = "openrouter"
    text_model = "mistralai/mistral-small-3.1-24b-instruct:free"
    image_path = "./examples/png/page_1.png"

    try:
        logger.info(f"Creating text provider with model: {text_model}")
        text_provider: LLMProvider = LLMProviderFactory.create_provider(
            provider_name=provider,
            api_url=OPENROUTER_API_URL,
            api_key=API_KEY,
        )

        logger.info(f"User: {user_message}")
        response_message = text_provider.get_completion(model=text_model, user_message=user_message, system_prompt=SYSTEM_PROMPT, image_path=image_path)
        logger.info(f"LLM ({text_model}): {response_message}")

        return response_message

    except Exception as e:
        logger.error(e, exc_info=True)
        raise


if __name__ == '__main__':
    blocks = get_text_column_data("examples/png/page_1.png", language="deu")
    block_text = get_full_text_of_blocks(blocks)
    logger.info(f"Tesseract provided the following text: \n{block_text}")
    response_message = test_llm_provider()
    headers, tables, is_table = get_markdown_headers_and_tables(response_message)
    logger.info(headers)
    logger.info(tables)

