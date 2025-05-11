import logging
from typing import Optional
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
            user_message = "No previous page provided. Assume it is at the start of the document."

        if not system_prompt:
            system_prompt = """
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
                               system_prompt: Optional[str],
                               image_path: str) -> str:
        if not system_prompt:
            system_prompt = """
            TODO
            convert Tables to text
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

