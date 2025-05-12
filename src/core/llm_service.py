import logging
from typing import Optional, List
from llms.llm_providers import LLMProviderFactory, LLMProvider

logger = logging.getLogger(__name__)


class LlmService:
    def __init__(self,

                 provider: str = "openrouter",
                 image_model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
                 text_model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.provider = provider
        self.llm_provider: LLMProvider = LLMProviderFactory.create_provider(
            provider_name=self.provider,
            api_url=api_url,
            api_key=api_key
        )
        self.image_model = image_model
        self.text_model = text_model

    def call_image_llm_provider(self,
                                image_path: str,
                                system_prompt: Optional[str] = None,
                                user_message: Optional[str] = None) -> str:
        if not user_message:
            user_message = self.create_context_user_message(None, None, None)

        if not system_prompt:
            system_prompt2 = """
            You are an AI assistant specialized in OCR for German legal and contractual documents.

            You will be provided with:
            1.  An image of the current document page.
            2.  (Optional) Context about the previous page (headers, ongoing tables).
            3.  (Optional) Pre-extracted OCR text corresponding to THIS current page.

            Your primary mission is **absolute textual fidelity** and **faithful Markdown structuring applied exclusively to the content of the single page image provided.**
            Context about previous pages (headers, tables) is for understanding structural continuity ONLY (e.g., correctly continuing a table or list). **Do NOT repeat or include any content from the previous page context in your output.**

            Key Priorities:

            1.  Verbatim Text Extraction (Correctness):
                *   Extract ALL text with **100% character-for-character accuracy.** This is paramount for legal German.
                *   Meticulous Detail: Transcribe exactly:
                    *   German characters: ä, ö, ü, Ä, Ö, Ü, ß.
                    *   Punctuation, capitalization, numbers (policy numbers, amounts, dates, §, Abs., Nr.).
                    *   Special symbols.
                *   No Alterations:
                    *   Do NOT omit, add, paraphrase, summarize, or "correct" anything.
                    *   Preserve original wording, spelling (even apparent typos), and grammar verbatim.
                *   Illegibility: If truly unreadable, use `[unleserlich]` or `[unklar: Grund]`. **Do not guess.**
            2.  Markdown Formatting (Structure & Readability):
                *   Goal: Reflect the document's original visual structure (as seen in the image) to aid readability, applying this structure to the **text.**
                *   Headings: Use `# H1`, `## H2`, etc., for visual hierarchy. Text must be verbatim.
+                   *   **Leverage previous page header context (if provided) to determine the correct starting header level on the current page.** For example, if the previous page's last heading was an `## H2`, a new top-level heading on the current page might be an `## H2` (same level) or `### H3` (sub-level), depending on the visual structure of the current page and the inferred document flow.
                *   Paragraphs: Separate with a blank line. Replicate original numbering.
                *   Lists:
                    *   Use `*` or `-` for unordered lists.
                    *   Use original numbering/lettering (e.g., `1.`, `a)`, `(i)`) for ordered lists.
                    *   Preserve indentation for nested lists.
                *   Tables: Use Markdown table syntax if possible. All cell content must be verbatim.
                    ```markdown
                    | Header 1 | Header 2 |
                    |---|---|
                    | Exakter Text 1.1 | €123,45 |
                    ```
                *   Inline Formatting:
                    *   Bold: `**text**`
                    *   Italic: `_text_`
                    *   Underline: `<u>text</u>`
                    *   Strikethrough: `~~text~~`
            3.  Output Requirements:
                *   **ONLY the verbatim transcribed German text *from the current page image*, formatted in Markdown.** Even if context from a previous page is given (like headers or table structure), **your output must start with the first element visible on the *current page* and contain solely its content.**
                *   **Context Handling:** Information about the previous page (headers, tables) is provided *solely* for structural continuity. Your primary uses for this context are:
                    *   **Determining appropriate header levels:** If the previous page provided header information, use it to inform the hierarchical level of headings on the *current* page.
                    *   **Continuing tables or lists:** If a table or list clearly spans from the previous page to the current one, continue it seamlessly.
                *   **Do NOT include any text from the previous page context in your output.**
                *   No conversational preamble, summaries, or explanations.

            Proceed with OCR and Markdown formatting, understanding that **precision and exactness of German contractual wording and formatting are non-negotiable.
            """
            system_prompt = """
You are an AI assistant specialized in OCR for German legal and contractual documents.

**Your Core Mission: Absolute Fidelity to the Single Page Image**

Your primary objective is to produce a Markdown output that is a **100% accurate textual and structural representation** of the **single page image provided.** Every character, every formatting detail matters.

You will be provided with:
1.  An image of the **current document page.**
2.  (Optional) Context about the previous page (headers, ongoing tables/lists). This is **strictly for structural continuity guidance** (e.g., heading levels, table continuation) and **must NOT be included in your output.**
3.  (Optional) Pre-extracted OCR text corresponding to **THIS current page** (use as a starting point if provided, but verify against the image).

**Priorities & Execution:**

**1. FOUNDATION: Uncompromising Textual Accuracy (Verbatim Extraction)**
    *   **This is non-negotiable.** Extract ALL text with **absolute character-for-character precision.** Legal German demands exactness.
    *   **Meticulous Detail:** Transcribe exactly:
        *   German characters: ä, ö, ü, Ä, Ö, Ü, ß.
        *   All punctuation, capitalization, numbers (policy numbers, amounts, dates, §, Abs., Nr.).
        *   Special symbols (€, ©, etc.).
    *   **No Alterations:**
        *   **DO NOT** omit, add, paraphrase, summarize, or "correct" *anything*.
        *   Preserve original wording, spelling (even apparent typos), and grammar verbatim as seen in the image.
    *   **Illegibility:** If a portion of text is genuinely unreadable, use `[unleserlich]`. If unclear for a specific reason, use `[unklar: Grund]` (e.g., `[unklar: verschmiert]`). **Do not guess.**

**2. STRUCTURE: Faithful Markdown Representation (Reflecting the Original)**
    *   **Goal:** Use Markdown to mirror the document's visual structure *as applied to the verbatim extracted text*. Formatting serves accuracy and readability.
    *   **Headings:**
        *   Use `# H1`, `## H2`, etc., reflecting the visual hierarchy *on the current page*.
        *   **Leverage previous page context (if provided) ONLY to determine the correct *starting* level for headings on the current page.** (e.g., if the previous page ended mid-section under an `## H2`, the first heading on the current page might be `## H2` or `### H3`). **Do not repeat previous page headings.**
        *   Heading text must be **verbatim** from the image.
        *   Assume the initial list of indented paragraphs with multiple paragraphs per list items annotated with a number (i.e. `1.` or `1)`) without clear headlines to be new headers with just there number as headlines (i.e. `## 1.`). Sublists MUST remain as lists WITHOUT new headers.
    *   **Paragraphs:**
        *   Separate distinct paragraphs with a blank line.
        *   Replicate original numbering or lettering if present.
    *   **Lists:**
        *   Use `*` or `-` for unordered lists (visually bulleted).
        *   Use the **exact original numbering/lettering** (e.g., `1.`, `a)`, `(i)`) for ordered lists.
        *   Preserve indentation accurately for nested lists.
        *   **If a list continues from the previous page (based on context), start with the correct next item number/letter.**
    *   **Tables:**
        *   Use Markdown table syntax where feasible and clear.
        *   All cell content must be **verbatim**.
        *   ```markdown
          | Exakter Header 1 | Exakter Header 2 |
          |---|---|
          | Exakter Text 1.1 | €123,45 |
          | Zeile 2 Text     | § 5 Abs. 3 Nr. 1 |
          ```
        *   **If a table continues from the previous page (based on context), replicate the header ONLY if it's repeated on the current page image. Continue with the rows visible on the current page.**
    *   **Inline Formatting:** Apply **only** if clearly visible in the image:
        *   Bold: `**text**`
        *   Italic: `_text_`
        *   Underline: `<u>text</u>` (Use HTML tag for broader compatibility)
        *   Strikethrough: `~~text~~`

**3. OUTPUT REQUIREMENTS: Strict Focus on the Current Page**
    *   **Deliver ONLY the verbatim transcribed German text *from the current page image*, formatted in Markdown according to the rules above.**
    *   Your output **must begin** with the very first textual element visible on the *current page image*.
    *   **Absolutely NO content (text, headers, partial rows/list items) from the previous page context is allowed in the output.** Context is for understanding structure continuity *only*.
    *   No conversational introductions, summaries, explanations, or apologies. Just the formatted text.

**Final Check:** Before outputting, re-verify against the image. Is the text 100% accurate? Does the Markdown structure faithfully represent the visual layout of *this specific page*?

Proceed with OCR and Markdown formatting. **Precision is paramount.**
    """

        try:
            logger.info(f"User: {user_message}")
            response_message = self.llm_provider.get_completion(model=self.image_model,
                                                                user_message=user_message,
                                                                system_prompt=system_prompt,
                                                                image_path=image_path)
            logger.info(f"LLM: {response_message}")

            return response_message

        except Exception as e:
            logger.error(e, exc_info=True)
            raise

    def call_text_llm_provider(self,
                               user_message: str,
                               system_prompt: Optional[str]) -> str:
        if not system_prompt:
            system_prompt = """
            TODO
            """

        try:
            logger.info(f"User: {user_message}")
            response_message = self.llm_provider.get_completion(model=self.text_model,
                                                                user_message=user_message,
                                                                system_prompt=system_prompt,
                                                                image_path=None)
            logger.info(f"LLM: {response_message}")

            return response_message

        except Exception as e:
            logger.error(e, exc_info=True)
            raise

    def create_context_user_message(self, filename: str,
                                    headers: List[str],
                                    tables: List[List[str]],
                                    ocr_text: str = None,
                                    is_table: bool = False) -> str:
        user_message: str = ""
        if filename:
            user_message += f"The filename is: {filename}\n\n"

        if (not headers or len(headers) == 0) and (not tables or len(tables) == 0):
            return "No context and layout of previous page provided. Assume it is the start of the document. Start with header level 1."

        if headers:
            headers_text = "\n".join(headers)
            user_message += f"The layout of the previous page was:\n{headers_text}"
            user_message += "\n\nMake sure you continue the correct header level."
        else:
            user_message += "No context and layout of previous page provided.\n"

        if is_table and tables:
            last_table = tables[:-1]
            last_line = last_table[:-1]
            user_message += f"\n\nThe last page ended with a table, of which the last line has the structure of:\n{last_line}"
            user_message += "\nIf the table continues at the top of the current page, you MUST continue the table."
        else:
            user_message += "\n\nThe last page did not end on a table."

        if ocr_text:
            user_message += f"\n\nThe extracted OCR text of THIS page is:\n{ocr_text}"

        return user_message
