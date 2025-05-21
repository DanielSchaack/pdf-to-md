from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from core.markdown_service import MarkdownService
from core.llm_service import LlmService
import io
import logging

logger = logging.getLogger(__name__)


class Processor(ABC):
    def __init__(self,
                 filetype: str,
                 llm_service: LlmService):
        self.filetype = filetype
        self.llm_service = llm_service
        self.markdown_service = MarkdownService()

    @abstractmethod
    def call_ocr(self,
                 filename: str,
                 image_path: str,
                 headers: List[str],
                 tables: List[List[str]],
                 ocr_text: str = None,
                 is_table: bool = False) -> str:
        pass

    @abstractmethod
    def refine_headers(self,
                       input_text: str,
                       headers: List[str]) -> Tuple[str, List[str], List[List[str]], bool]:
        pass

    @abstractmethod
    def convert_tables_to_texts(self, tables: List[List[str]], context: str = None) -> List[str]:
        pass

    @abstractmethod
    def correct_header_levels(self,
                              headers: List[str],
                              new_headers: List[str]) -> List[str]:
        pass

    @abstractmethod
    def convert_text_to_chunks(self,
                               filename: str,
                               aggregated_text: str,
                               header_level_cutoff: int = 3) -> Tuple[List[str], List[str], List[str]]:
        pass

    @abstractmethod
    def integrate_messages(self, extracted_images: List[str]) -> str:
        pass


class ProcessorFactory:
    @staticmethod
    def create_processor(filetype: str, llm_service: LlmService) -> Processor:
        if filetype.lower() == "md":
            logger.debug("Returning md Processor")
            return MarkdownProcessor(filetype=filetype.lower(), llm_service=llm_service)
        # TODO
        # elif filetype.lower() == "tex":
        #     logger.debug("Returning tex Processor")
        #     return TexProcessor()
        else:
            raise ValueError(f"Unsupported filetype Processor: {filetype}")


class MarkdownProcessor(Processor):
    def call_ocr(self,
                 filename: str,
                 image_path: str,
                 headers: List[str],
                 tables: List[List[str]],
                 ocr_text: str = None,
                 is_table: bool = False) -> str:
        system_prompt = """
# Role
You are an AI assistant specialized in OCR for German legal and contractual documents.

# Core Mission: Absolute Fidelity to the Single Page Image

Your primary objective is to produce a Markdown output that is a **100% accurate textual and structural representation** of the **single page image provided.** Every character, every formatting detail matters.

You will be provided with:
1.  An image of the **current document page.**
2.  Pre-extracted OCR text corresponding to **THIS current page** (use as a starting point if provided, but verify against the image).
3.  (Optional) Context about the previous page (headers, ongoing tables/lists). This is **strictly for structural continuity guidance** (e.g., continuing header levels, table continuation) and **must NOT be included in your output.**

# Priorities & Execution

## FOUNDATION: Uncompromising Textual Accuracy (Verbatim Extraction)
This is non-negotiable. Extract ALL text with **absolute character-for-character precision.** Legal German demands exactness.

The transcription process demands meticulous attention to detail, requiring an exact replication of the original text. This includes faithfully transcribing all characters, specifically the umlauts (ä, ö, ü) and the Eszett (ß), as well as their uppercase counterparts. Furthermore, it’s crucial to capture every instance of punctuation, ensuring consistent capitalization, and accurately recording numbers, including policy numbers, monetary amounts, dates, legal sections (denoted by § and Abs.), and identification numbers (Nr.). Finally, the transcription must incorporate all special symbols such as currency signs (€) and copyright symbols (©).

Absolutely no alterations are permitted – You MUST NOT omit, add, paraphrase, summarize, or attempt to “correct” any portion of the text.

## STRUCTURE: Faithful Markdown Representation (Reflecting the Original image)
Use Markdown to mirror the document's visual structure as applied to the verbatim extracted text.

### Headings:
Multiline headers with bold text and the same font size are possible. You MUST treat these as a singular header. Example: `**General Topic**\n**Part of Header**\n**Deeper Description of the header**` MUST be `# General Topic, Part of Header, Deeper Description of the header`

Next to being above paragraphs, headers can also be to the side of the paragraphs. The headers must be above the paragraphs based on the same height.

Headers can also be leadings signs like `I`, `1.`, `A)` without any additional text. This is especially important for pages that numerically lists paragraphs. Treat these as headers WITHOUT additional text and with a sublevelof the current one.
Example:
```
**The Current Main Topic**
1)      Paragraph start right here[...]\n\n and ends way later.
2)      Second paragraph start right here[...]\n\n and ends way later.
```
Should be:
```markdown
# The Current Main Topic
## 1)
Paragraph start right here[...]\n\n and ends way later.
## 2)
Second paragraph start right here[...]\n\n and ends way later.
```

Headers within paragraphs MUST be treated as sublevel of the current header level, for example you're in header `### A3 Third Subtopic` and a bolded line appears of `**f) The Reasons Why**`, it MUST be of header level 4 - `#### f) The Reasons Why`

The guidelines to header-levels are as follows:
1. Bold text in their own paragraph _without_ any leading signs like roman numerals, letters or number MUST ALWAYS be of header level 1, i.e. `**General Topic**\n**Part of Header**` MUST be `# General Topic, Part of Header`
2. Bold text in their own paragraph _with_ exactly one leading sign like `III` or `B.` MUST ALWAYS be of header level 2, i.e. `**II. Second Subtopic**` MUST be `## II. Second Subtopic`
3. Bold text in their own paragraph _with_ exactly two leading signs like `A.3` or `I.2` AND also `C3` MUST ALWAYS be of header level 3, i.e. `**C.1 First Subsubtopic**` MUST be `### C.1 First Subsubtopic`.

You MUST follow the guidelines even if subtopics are the first headlines that appear!

### Lists:
Use `-` for unordered lists (visually bulleted). Use the **exact original numbering/lettering** (e.g., `1.`, `a)`, `(i)`) for ordered lists.

Preserve indentation accurately for nested lists.

### Paragraphs:
Separate distinct paragraphs with a blank line. When paragraphs are given as a list, replicate the original numbering or lettering.

### Tables:
Use Markdown table syntax where feasible and clear. All cell content must be **verbatim**. Cells can also be left empty if no text is provided inside a cell.

If a table continues from the previous page, continue with the same structure rows visible on the current page without a header. 

Example for a table:
```markdown
| Exakter Header 1 | Exakter Header 2 | Exakter Header 3 |
|---|---|---|
| Exakter Text 1.1 | €123,45 | |
| Zeile 2 Text | § 5 Abs. 3 Nr. 1 | The above cell is empty |
```

### Inline Formatting:
Apply **only** if clearly visible in the image:
-   Bold: `**text**`
-   Italic: `_text_`
-   Strikethrough: `~~text~~`

## OUTPUT REQUIREMENTS:
-   Deliver ONLY the verbatim transcribed text _from the current page image_, formatted in Markdown according to the rules above.**
-   Absolutely NO content (text, headers, partial rows/list items) from the previous page context is allowed in the output. Context is for understanding structure continuity *only*.
-   The Text MUST be in the language provided by the text on the image.
-   No conversational introductions, summaries, explanations, or apologies. Just the formatted text.
"""
        user_message = self.create_context_user_message(filename=filename,
                                                        headers=headers,
                                                        tables=tables,
                                                        ocr_text=ocr_text,
                                                        is_table=is_table)
        response_message = self.llm_service.call_image_llm_provider(image_path=image_path,
                                                                    system_prompt=system_prompt,
                                                                    user_message=user_message)
        return self.markdown_service.remove_lines_starting_with(response_message, "`")

    def refine_headers(self,
                       input_text: str,
                       headers: List[str]) -> Tuple[str, List[str], List[List[str]], bool]:
        new_headers, tables, is_table = self.markdown_service.get_markdown_headers_and_tables(input_text)
        if len(headers) > 0:
            headers = new_headers  # self.correct_header_levels(headers, new_headers)
            # input_text = self.markdown_service.replace_headers(input_text, headers)
        else:
            headers = new_headers

        return input_text, headers, tables, is_table

    def convert_tables_to_texts(self,
                                tables: List[List[str]],
                                context: str = None) -> List[str]:
        system_prompt = """
# Role: Table-to-Text Converter

# Objective
Convert the provided data table into a series of descriptive sentences in plain text, formatted in Markdown. The goal is to make table information easily accessible and searchable for Large Language Models within a knowledge base.

# Task
1.  Identify the data cells within the table (cells containing values, not just row or column headers).
2.  For **each** data cell, generate **one single sentence**.
3.  This sentence must clearly state the value of the cell within the context of its corresponding row header and column header. If an overall context or title for the table is provided, include that as well.
4.  Iterate through all data cells, generating one sentence per cell.
5.  Output **only** the generated sentences in Markdown format. Do not include any preamble, explanation, commentary, or formatting other than the sentences themselves.
6.  **Strict Adherence:** Do **not** interpret, infer, or add any information not explicitly present in the table. Ensure the generated sentence accurately reflects the cell's value and its corresponding headers.

# Examples

**Input**
```markdown
| Tariff | ET10 | ET15 | ET20 | ET25 | ET30 | ET35 | ET40 | ET45 | ET50 |
|---|---|---|---|---|---|---|---|---|---|
| Reimbursement Rate in % | 90 | 85 | 80 | 75 | 70 | 65 | 60 | 55 | 50 |
```

**Expected Output:**
```markdown
- For the Tariff ET10, the Reimbursement Rate in % for Visual Aids is 90%.
- For the Tariff ET15, the Reimbursement Rate in % for Visual Aids is 85%.
- For the Tariff ET20, the Reimbursement Rate in % for Visual Aids is 80%.
[...]
```

---

**Input**
```markdown
| Tarif | Erwachsene ab Alter 21 | Kinder/Jugendliche bis Alter 20 |
|-------|------------------------|--------------------------------|
| GUP0  | 0 EUR                  | 0 EUR                          |
| GUP500| 500 EUR                | 250 EUR                        |
| GUP900| 900 EUR                | 450 EUR                        |
| GUP1.8| 1.800 EUR              | 900 EUR                        |
```

**Expected Output:**
```markdown
- Der Tarif GUP0 für Erwachsene ab Alter 21 beträgt 0 Euro.
- Der Tarif GUP0 für Kinder/Jugendliche bis Alter 20 beträgt 0 Euro.
- Der Tarif GUP500 für Erwachsene ab Alter 21 beträgt 500 Euro.
- Der Tarif GUP500 für Kinder/Jugendliche bis Alter 20 beträgt 250 Euro.
- Der Tarif GUP900 für Erwachsene ab Alter 21 beträgt 900 Euro.
- Der Tarif GUP900 für Kinder/Jugendliche bis Alter 20 beträgt 450 Euro.
- Der Tarif GUP1.8 für Erwachsene ab Alter 21 beträgt 1.800 Euro.
- Der Tarif GUP1.8 für Kinder/Jugendliche bis Alter 20 beträgt 900 Euro.
```

---

**Input**
```markdown
| Wann und für wen? | Was? | Wie oft? | Abrechnung nach GOÄ |
|---|---|---|---|
| | Vorsorgeuntersuchungen nach Alter¹ | | |
| 0 – 10 J. | Vorsorgeleistungen für | Jede U-Untersuchung ein Mal | Die Untersuchungen U1, U2, U3, U4, U5, U6, U7, U7a, U8, U9, U10 und U11 werden jeweils nach GOÄ 25 oder 26 abgerechnet. |
| w / m² | Kinder | | |
| | zur Früherkennung von | | |
| | Krankheiten | | |
| | | | GOÄ Leistungsbeschreibung |
| | | | 25 Neugeborenen-Erstuntersuchung |
| | | | 26 Früherkennungsuntersuchung beim Kind |
| | | | 3691.H1 Zu U1 – Screening auf Sichelzellkrankheit |
| | | | 4872 Zu U1 – Screening auf Spinale Muskelatrophie |
| 13 – 14 J. | Vorsorgeleistungen für | ein Mal | GOÄ |
| w / m | Jugendliche | | 32 Untersuchung nach Jugendarbeitsschutzgesetz |
| | Jugendgesundheitsuntersuchung (J1) | | 250 Blutabnahme |
| 16 – 17 J. | Vorsorgeleistungen für | ein Mal | GOÄ Leistungsbeschreibung |
| w / m | Jugendliche | | 32 Untersuchung nach Jugendarbeitsschutzgesetz |
| | Jugendgesundheitsuntersuchung (J2) | | 250 Blutabnahme |
| | | | 3514 Glukose |
```

**Expected Output:**
```markdown
- Für das Alter von 0 bis 10 Jahren wird die Vorsorgeleistung für Kinder zur Früherkennung von Krankheiten jede U-Untersuchung ein Mal nach GOÄ-Ziffer 25 Neugeborenen-Erstuntersuchung abgerechnet.
- Für das Alter von 0 bis 10 Jahren wird die Vorsorgeleistung für Kinder zur Früherkennung von Krankheiten jede U-Untersuchung ein Mal nach GOÄ-Ziffer 26 Früherkennungsuntersuchung beim Kind abgerechnet.
- Für das Alter von 0 bis 10 Jahren wird die Vorsorgeleistung für Kinder zur Früherkennung von Krankheiten jede U-Untersuchung ein Mal nach GOÄ-Ziffer 3691.H1 Screening auf Sichelzellenkrankheit abgerechnet.
- Für das Alter von 0 bis 10 Jahren wird die Vorsorgeleistung für Kinder zur Früherkennung von Krankheiten jede U-Untersuchung ein Mal nach GOÄ-Ziffer 4872 Screening auf Spinale Muskelerkrankung abgerechnet.
- Für das Alter von 13 bis 14 Jahren wird die Vorsorgeleistung für Jugendliche Jugendgesundheitsuntersuchung (J1) ein Mal bei der GOÄ-Ziffer 32 "Untersuchung nach Jugendarbeitsschutzgesetz" abgerechnet
- Für das Alter von 13 bis 14 Jahren wird die Vorsorgeleistung für Jugendliche Jugendgesundheitsuntersuchung (J1) ein Mal bei der GOÄ-Ziffer 250 "Blutabnahme" abgerechnet
- Für das Alter von 16 bis 17 Jahren wird die Vorsorgeleistung für Jugendliche Jugendgesundheitsuntersuchung (J2) ein Mal bei der GOÄ-Ziffer 32 "Untersuchung nach Jugendarbeitsschutzgesetz" abgerechnet
- Für das Alter von 16 bis 17 Jahren wird die Vorsorgeleistung für Jugendliche Jugendgesundheitsuntersuchung (J2) ein Mal bei der GOÄ-Ziffer 250 "Blutabnahme" abgerechnet
- Für das Alter von 16 bis 17 Jahren wird die Vorsorgeleistung für Jugendliche Jugendgesundheitsuntersuchung (J2) ein Mal bei der GOÄ-Ziffer 3514 "Glukose" abgerechnet
```

---

**Constraint Checklist:**
-   Output is Markdown.
-   One sentence per data cell.
-   Sentence includes row header, column header, cell value, and table context (if available).
-   The language MUST be the same as the table.
-   Output contains ONLY the generated sentences.
-   Information is strictly from the table, no interpretation.
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

# Example

Input:

Existing Headers:
```markdown
# Main Topic
## Section A
### Section A.1. Sub-point One
### Section A.2. Sub-point Two
```

New Headers:
```markdown
## Section A.3. Sub-point Three <- Incorrect level, shares same subsection as A.1./A.2.
## Section A.4. Sub-point Four <- Incorrect level, shares same subsection as A.1./A.2.
## Section B <- Correct level, equal to A
### Section B.1. Another Sub-point <- Correct level, equal to A.1./A.2.
```

Correct Output:
```markdown
### Section A.3. Sub-point Three <- Corrected level equal to A.1.
### Section A.4. Sub-point Four <- Corrected level equal to A.1.
## Section B
### Section B.1. Another Sub-point
```

Input:

Existing Headers:
```markdown
## Topic A
### Subtopic A
### Subtopic B
### Subtopic C
#### C.1 Specific title 1
#### C.2 Specific title 2
#### C.3 Specific title 3
```

New Headers::
```markdown
## C.4 Specific title 4 <- Incorrect level, shares same subsection as C.1/C.2/C.3
## C.5 Specific title 5 <- Incorrect level, shares same subsection as C.1/C.2/C.3
## C.6 Specific title 6 <- Incorrect level, shares same subsection as C.1/C.2/C.3
## C.7 Specific title 7 <- Incorrect level, shares same subsection as C.1/C.2/C.3
# Subtopic D <- Incorrect level, shares same subtopic as A/B/C
```

Correct Output:
```markdown
#### C.4 Specific title 4 <- Corrected level equal to C.1/C.2/C.3
#### C.5 Specific title 5 <- Corrected level equal to C.1/C.2/C.3
#### C.6 Specific title 6 <- Corrected level equal to C.1/C.2/C.3
#### C.7 Specific title 7 <- Corrected level equal to C.1/C.2/C.3
### Subtopic D <- Corrected level equal to A/B/C
```

# Output Requirements
-   Output **only** the corrected version of the **`New Headers`** list.
-   Do **not** include the `Existing Headers` in the output.
-   Do **not** include any preamble, introductory text, explanations, comments (like <- Corrected level), or labels (like "Corrected Headers:") in the output.
-   The output should consist *solely* of the corrected markdown header lines originally provided in the `New Headers` input.

# Focus
Focus on maintaining the logical structure implied by the content and numbering, determining the correct levels for the `New Headers` based on the `Existing Headers` levels.
        """
        old_headers_listed = "\n".join(headers)
        new_headers_listed = "\n".join(new_headers)
        user_message = f"Existing Headers:\n```markdown\n{old_headers_listed}```\n\nNew Headers:\n```markdown\n{new_headers_listed}```"

        new_new_headers = self.llm_service.call_text_llm_provider(user_message=user_message, system_prompt=system_prompt)
        new_new_headers = self.markdown_service.remove_lines_starting_with(new_new_headers, "`")
        logger.debug(f"Existing headers:\n{old_headers_listed}, old new headers:\n{new_headers_listed}, new new headers:\n{new_new_headers}")

        n_headers = []
        lines = io.StringIO(new_new_headers)
        for line in lines:
            n_headers.append(line.strip())

        return n_headers

    def convert_text_to_chunks(self,
                               filename: str,
                               aggregated_text: str,
                               header_level_cutoff: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:

        if not header_level_cutoff:
            header_level_cutoff = self.markdown_service.get_header_level_cutoff(aggregated_text=aggregated_text)
            pass
        return self.markdown_service.convert_markdown_to_chunks(filename=filename,
                                                                markdown_text=aggregated_text,
                                                                header_level_cutoff=header_level_cutoff)

    def integrate_messages(self, extracted_images: List[str]) -> str:
        complete_markdown = "\n".join(extracted_images)
        headers, tables, is_table = self.markdown_service.get_markdown_headers_and_tables(complete_markdown)

        if len(tables) > 0:
            table_texts = self.convert_tables_to_texts(tables)
            complete_markdown = self.markdown_service.replace_tables(complete_markdown, table_texts)
        return complete_markdown

    def create_context_user_message(self,
                                    filename: str,
                                    headers: List[str],
                                    tables: List[List[str]],
                                    ocr_text: str = None,
                                    is_table: bool = False) -> str:
        user_message: str = ""
        if filename:
            user_message += f"The filename is: {filename}\n\n"
            logger.debug(f"The user_message is updated with a filename to:\n{user_message}")

        if ocr_text:
            user_message += f"\n\nThe extracted OCR text of THIS page is:\n{ocr_text}"
            logger.debug(f"The user_message is updated with provided ocr text to:\n{user_message}")

        if (not headers or len(headers) == 0) and (not tables or len(tables) == 0):
            user_message += "No context and layout of previous page provided. Assume it is the start of the document. Start with header level 1."
            logger.debug(f"The user_message is updated without headers or tables to:\n{user_message}")

        if (headers or len(headers) > 0):
            headers_text = "\n".join(headers)
            user_message += f"The layout of the previous page was:\n{headers_text}"
            user_message += "\n\nMake sure you continue the correct header level."
            logger.debug(f"The user_message is updated with headers to:\n{user_message}")

        if (tables or len(tables) > 0):
            last_table = tables[:-1]
            last_line = last_table[:-1]
            user_message += f"\n\nThe last page ended with a table, of which the last line has the structure of:\n{last_line}"
            user_message += "\nIf the table continues at the top of the current page, you MUST continue the table."
            logger.debug(f"The user_message is updated with tables to:\n{user_message}")

        return user_message

