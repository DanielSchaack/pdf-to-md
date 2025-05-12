from core.markdown_service import MarkdownService
from core.file_service import FileService
from core.llm_service import LlmService
from core.repository_service import RepositoryService
from typing import List
import io
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, markdown_service: MarkdownService,
                 file_service: FileService,
                 llm_service: LlmService,
                 repository_service: RepositoryService):
        self.markdown_service = markdown_service
        self.file_service = file_service
        self.llm_service = llm_service
        self.repository_service = repository_service

    def process_pdf(self, pdf_path: str):
        filename = self.file_service.get_filename_from_path(pdf_path)

        image_paths = self.file_service.convert_pdf_to_images(pdf_path=pdf_path,
                                                              filename=filename,
                                                              image_format="png",
                                                              dpi=300,
                                                              colorspace="rgb",
                                                              page_numbers=None,
                                                              use_alpha=False)
        # TODO save file paths

        markdowns_responses: List[str] = []
        headers: List[str] = []
        tables: List[List[str]] = []
        is_table: bool = False

        for index, image_path in enumerate(image_paths):

            blocks = self.file_service.get_text_column_data(image_path, language="deu")
            combined_full_text = "\n".join(block["full_text"] for block in blocks)
            logger.debug(f"Tesseract provided the following text: \n{combined_full_text}")
            # TODO save tesseract result

            user_message = self.llm_service.create_context_user_message(filename=filename,
                                                                        headers=headers,
                                                                        tables=tables,
                                                                        ocr_text=combined_full_text,
                                                                        is_table=is_table)
            response_message = self.llm_service.call_image_llm_provider(image_path=image_path,
                                                                        user_message=user_message)
            cleansed_response_message = self.markdown_service.remove_lines_starting_with(response_message, "`")

            new_headers, tables, is_table = self.markdown_service.get_markdown_headers_and_tables(cleansed_response_message)
            if len(headers) > 0:
                headers = self.correct_header_levels(headers, new_headers)
                cleansed_response_message = self.markdown_service.replace_headers(cleansed_response_message, headers)
            else:
                headers = new_headers

            if len(tables) > 0 and not is_table:
                context = "\n".join(headers)
                table_texts = self.convert_tables_to_texts(tables, context)
                cleansed_response_message = self.markdown_service.replace_tables(cleansed_response_message, table_texts)
            # TODO save markdown_text

            markdowns_responses.append(cleansed_response_message)
            # TODO save response

        complete_markdown = "\n".join(markdowns_responses)
        headers, tables, is_table = self.markdown_service.get_markdown_headers_and_tables(complete_markdown)

        if len(tables) > 0:
            table_texts = self.convert_tables_to_texts(tables)
            complete_markdown = self.markdown_service.replace_tables(complete_markdown, table_texts)

        self.file_service.save_to_filesystem(filename_prefix=filename,
                                             file_suffix="md",
                                             text=complete_markdown)
        # TODO save complete markdown

        chunks = self.markdown_service.convert_markdown_to_chunks(filename=filename,
                                                                  markdown_text=complete_markdown,
                                                                  header_level_cutoff=3)
        # TODO save chunks
        self.file_service.convert_chunks_to_files(filename_prefix=filename,
                                                  chunk_suffix="md",
                                                  chunks=chunks)

    def convert_tables_to_texts(self, tables: List[List[str]], context: str = None) -> List[str]:
        system_prompt = """
# Role: Table-to-Text Converter

**Objective:** Convert the provided data table into a series of descriptive sentences in plain text, formatted in Markdown. The goal is to make table information easily accessible and searchable for Large Language Models within a knowledge base.

**Task:**
1.  Identify the data cells within the table (cells containing values, not just row or column headers).
2.  For **each** data cell, generate **one single sentence**.
3.  This sentence must clearly state the value of the cell within the context of its corresponding row header and column header. If an overall context or title for the table is provided, include that as well.
4.  Iterate through all data cells, generating one sentence per cell.
5.  Output **only** the generated sentences in Markdown format. Do not include any preamble, explanation, commentary, or formatting other than the sentences themselves.
6.  **Strict Adherence:** Do **not** interpret, infer, or add any information not explicitly present in the table. Ensure the generated sentence accurately reflects the cell's value and its corresponding headers.

**Example:**

    **Input**
    ```
    (Optional) Context:
    Visual Aids

    Table:
    | Tariff | ET10 | ET15 | ET20 | ET25 | ET30 | ET35 | ET40 | ET45 | ET50 |
    |---|---|---|---|---|---|---|---|---|---|
    | Reimbursement Rate in % | 90 | 85 | 80 | 75 | 70 | 65 | 60 | 55 | 50 |
    ```

    **Expected Output:**
    ```markdown
    - For the Tariff ET10, the Reimbursement Rate in % for Visual Aids is 90%.
    - For the Tariff ET15, the Reimbursement Rate in % for Visual Aids is 85%.
    - For the Tariff ET20, the Reimbursement Rate in % for Visual Aids is 80%.
    - For the Tariff ET25, the Reimbursement Rate in % for Visual Aids is 75%.
    - For the Tariff ET30, the Reimbursement Rate in % for Visual Aids is 70%.
    - For the Tariff ET35, the Reimbursement Rate in % for Visual Aids is 65%.
    - For the Tariff ET40, the Reimbursement Rate in % for Visual Aids is 60%.
    - For the Tariff ET45, the Reimbursement Rate in % for Visual Aids is 55%.
    - For the Tariff ET50, the Reimbursement Rate in % for Visual Aids is 50%.
    ```

**Constraint Checklist:**
*   Output is Markdown.
*   One sentence per data cell.
*   Sentence includes row header, column header, cell value, and table context (if available).
*   The language MUST be the same as the table.
*   Output contains ONLY the generated sentences.
*   Information is strictly from the table, no interpretation.
"""
        texts: str = []
        for table in tables:
            text = "\n".join(table)
            user_message = ""
            if context:
                user_message += f"Context:\n{context}\n\n"
            user_message += f"Table:\n{text}"

            new_table_text = self.llm_service.call_text_llm_provider(user_message=user_message, system_prompt=system_prompt)
            new_table_text = self.markdown_service.remove_lines_starting_with(new_table_text, "`")
            logger.debug(f"Existing table: {text}, new table: {new_table_text}")
            texts.append(new_table_text)
        return texts

    def correct_header_levels(self, headers: List[str], new_headers: List[str]) -> List[str]:
        system_prompt = """
# Task Description
Your task is to analyze a list of new markdown headers and correct their hierarchical level based on their content, numbering structure, and the context provided by a preceding list of headers.

# Input
The input consists of two parts:

1.  **`Existing Headers`**: A list of markdown headers representing the established structure *before* the headers that need correction. This provides the immediate context.
2.  **`New Headers`**: A list of subsequent markdown headers that need to be analyzed and potentially corrected based on the `Existing Headers` and their own internal sequence.

# Task Steps
1.  Examine the `Existing Headers` to understand the current hierarchical level and numbering/theming patterns at the point where the `New Headers` begin.
2.  Analyze the *first* header in the `New Headers` list. Determine its correct level relative to the *last* header(s) in the `Existing Headers`.
3.  Examine the subsequent headers within the `New Headers` list sequentially.
4.  Identify headers within `New Headers` that appear to be sub-points or continuations of a preceding header (either the last `Existing Header` or a header within `New Headers` itself) based on their numbering (e.g., "A.3." following "A.2.") or clear thematic similarity indicated by consistent phrasing or prefixes (e.g., "Section A.*").
5.  Adjust the markdown header level (#, ##, ###, etc.) of the headers within the `New Headers` list to correctly reflect their position within the overall hierarchy. A sub-point should typically be one level deeper than its parent header.

# Output Requirements
*   Output **only** the corrected version of the **`New Headers`** list.
*   Do **not** include the `Existing Headers` in the output.
*   Do **not** include any preamble, introductory text, explanations, comments (like <- Corrected level), or labels (like "Corrected Headers:") in the output.
*   The output should consist *solely* of the corrected markdown header lines originally provided in the `New Headers` input.

# Example

**Input:**

**`Existing Headers`:**
```markdown
# Main Topic
## Section A
### Section A.1. Sub-point One
### Section A.2. Sub-point Two
```

**`New Headers`:**
```markdown
## Section A.3. Sub-point Three <- Incorrect level
## Section A.4. Sub-point Four <- Incorrect level
## Section B
### Section B.1. Another Sub-point
```

**Correct Output:**
```markdown
### Section A.3. Sub-point Three <- Corrected level
### Section A.4. Sub-point Four <- Corrected level
## Section B
### Section B.1. Another Sub-point
```

# Focus
Focus on maintaining the logical structure implied by the content and numbering, determining the correct levels for the `New Headers` based on the `Existing Headers` context and internal relationships within `New Headers`.
        """
        old_headers_listed = "\n".join(headers)
        new_headers_listed = "\n".join(new_headers)
        user_message = f"Existing Headers:\n{old_headers_listed}\n\nNew Headers:\n{new_headers_listed}"

        new_new_headers = self.llm_service.call_text_llm_provider(user_message=user_message, system_prompt=system_prompt)
        new_new_headers = self.markdown_service.remove_lines_starting_with(new_new_headers, "`")
        logger.debug(f"Existing headers: {old_headers_listed}, old new headers: {new_headers_listed}, new new headers: {new_new_headers}")

        n_headers = []
        lines = io.StringIO(new_new_headers)
        for line in lines:
            n_headers.append(line.strip())

        if len(new_headers) == len(n_headers):
            logger.debug(f"Returning new headers corrected: {n_headers}")
            return n_headers

        return new_headers

