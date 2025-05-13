import io
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MarkdownService:
    def __init__(self):
        pass

    def count_consecutive_chars(self, s: str, start_pos: int, char: str) -> tuple[int, int]:
        """
        Counts consecutive occurrences of a character starting from a specified position in a string.

        Args:
            s (str): The input string to search within.
            start_pos (int): The starting index from which to begin counting consecutive characters.
            char (str): The character whose consecutive occurrences need to be counted.

        Returns:
            tuple: A tuple containing two elements:
                - count (int): The number of consecutive occurrences of the specified character.
                - end_position (int): The index at which the last occurrence in the sequence ends.

        Raises:
            ValueError: If the start position is out of bounds of the string.

        Example:
            >>> count_consecutive_chars_and_position("aaabbbaaac", 2, 'b')
            (3, 4)
        """
        if start_pos < 0 or start_pos >= len(s):
            raise ValueError("Start position out of bounds")

        count = 0
        for i in range(start_pos, len(s)):
            if s[i] == char:
                count += 1
            else:
                break

        logger.debug(f"Character {char}, starting at {start_pos}, appears {count} times, ending at {i-1}")
        return count, i - 1

    def get_markdown_headers_and_tables(self, s: str) -> Tuple[List[str], List[List[str]], bool]:
        """
        Extracts markdown headers and tables from a given string.

        Args:
            s (str): The input string containing markdown content.

        Returns:
            tuple: A tuple containing three elements:
                - headers (List[str]): A list of markdown headers found in the string.
                - tables (List[List[str]]): A list of lists representing tables found in the string, where each sublist contains rows of a table.
                - is_currently_table (bool): A boolean indicating whether the function is currently parsing a table.

        Raises:
            ValueError: If the input string `s` is empty.

        Example:
            >>> get_markdown_headers_and_tables("# Header 1\n| Row 1 |\n| Row 2 |\n## Header 2\n| Row 3 |\n| Row 4 |")
            (['# Header 1', '## Header 2'], [['| Row 1 |', '| Row 2 |'], ['| Row 3 |', '| Row 4 |']], False)
        """
        if s and not s.strip():
            raise ValueError("Input string cannot be empty")

        headers: List[str] = []
        tables: List[List[str]] = []
        lines_with_pipe: List[str] = []
        is_currently_table: bool = False

        lines = io.StringIO(s)
        for line in lines:
            # Headers
            if line.startswith("#"):
                headers.append(line.strip())
                logger.debug(f"Adding new header: {line.strip()}")

            # Currently a table
            if line.startswith("|"):
                is_currently_table = True
                lines_with_pipe.append(line.strip())
                logger.debug(f"Adding new line to table: {line.strip()}")

            # End of tables
            if len(lines_with_pipe) > 0 and not line.startswith("|"):
                is_currently_table = False
                tables.append(lines_with_pipe)
                logger.debug(f"Adding new table: {lines_with_pipe}")
                lines_with_pipe = []

        # If last line is still a table, it wasn't added in the loop
        if len(lines_with_pipe) > 0:
            tables.append(lines_with_pipe)

        logger.debug(f"Returning headers: {headers}")
        logger.debug(f"Returning tables: {tables}")
        logger.debug(f"Returning is_currently_table: {is_currently_table}")
        return headers, tables, is_currently_table

    def convert_markdown_to_chunks(self,
                                   filename: str,
                                   markdown_text: str,
                                   header_level_cutoff: int = 3) -> List[List[str]]:
        """
        Convert Markdown text into chunks based on headers and a specified cutoff level.

        Args:
            file_name (str): The name of the file containing the Markdown content.
            markdown_text (str): The Markdown text to be converted.
            header_level_cutoff (int, optional): The maximum header level to consider for chunking.
                                                 Cuts chunk with each header level equal or less than to the cutoff. Defaults to 3.

        Returns:
            List[List[str]]: A list of chunks, where each chunk is a list of strings representing lines in that chunk.

        Raises:
            ValueError: If the header_level_cutoff is less than or equal to 0.
        """
        header_map: Dict[int, Any] = {}
        current_chunk: List[str] = []
        current_chunk.append(filename)
        chunks: List[List[str]] = []

        lines = io.StringIO(markdown_text)
        for line in lines:

            if line.startswith("#"):
                header_level, end_position = self.count_consecutive_chars(line, 0, "#")
                header_map[header_level] = line.strip()

                if header_level > header_level_cutoff:
                    current_chunk.append(line.strip())
                    logger.debug(f"Added header line '{line.strip()}' to current chunk")
                    continue  # continue with current chunk

                # Append header for first chunk, otherwise missing
                if len(chunks) == 0:
                    current_chunk.append(line.strip())
                    logger.debug(f"Added header line '{line.strip()}' to first chunk")

                # Start of next chunk
                if len(current_chunk) > len(header_map) + 1:    # More lines than just the context

                    chunks.append(current_chunk)
                    logger.debug(f"Added current chunk '{current_chunk}' to chunks")

                    # Reset current chunk - add filename and headers as context
                    current_chunk = []
                    current_chunk.append(filename)
                    for header in header_map.values():
                        current_chunk.append(header)
                    logger.debug(f"Cleared and added context to current chunk, now: '{current_chunk}'")

                continue  # Don't add regular line

            # Regular text, add to chunk
            current_chunk.append(line.strip())
            logger.debug(f"Added regular line '{line.strip()}' to current_chunk")

        if len(current_chunk) > len(header_map) + 1:    # More lines than just the context
            chunks.append(current_chunk)
            logger.debug(f"Added leftover chunk '{current_chunk}' to all chunks")

        logger.debug(f"Returning chunks '{chunks}'")
        return chunks

    def remove_lines_starting_with(self, text: str, char: str = "`"):
        filtered_lines = []
        lines = io.StringIO(text)
        for line in lines:
            if not line.lstrip().startswith(char):
                filtered_lines.append(line.strip())

        filtered_text = "\n".join(filtered_lines)
        logger.debug(f"Returning filtered text {filtered_text}")
        return filtered_text

    def replace_headers(self, markdown_text: str, headers: List[str]) -> str:
        filtered_lines = []
        i: int = 0
        lines = io.StringIO(markdown_text)
        for line in lines:
            if line.startswith("#") and i < len(headers):
                filtered_lines.append(headers[i])
                i += 1
            else:
                filtered_lines.append(line.strip())

        replaced_headers_text = "\n".join(filtered_lines)
        logger.debug(f"Returning text with updated headers: {replaced_headers_text}")
        return replaced_headers_text

    def replace_tables(self, markdown_text: str, table_texts: List[str]) -> str:
        filtered_lines = []
        i: int = 0
        is_table = False
        lines = io.StringIO(markdown_text)
        for line in lines:
            if line.startswith("|") and not is_table and i < len(table_texts):
                filtered_lines.append(table_texts[i])
                i += 1
                is_table = True
            elif line.startswith("|") and is_table:  # Skip table lines
                continue
            elif is_table:  # Reset is_table
                is_table = False
                filtered_lines.append(line.strip())
            else:
                filtered_lines.append(line.strip())

        replaced_tables_text = "\n".join(filtered_lines)
        logger.debug(f"Returning text with updated headers: {replaced_tables_text}")
        return replaced_tables_text
