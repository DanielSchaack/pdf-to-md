import io
from typing import List, Tuple


def count_consecutive_chars(s: str, start_pos: int, char: str) -> tuple[int, int]:
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

    return count, i - 1


def get_markdown_headers_and_tables(s: str) -> Tuple[List[str], List[List[str]], bool]:
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
    if not s.strip():
        raise ValueError("Input string cannot be empty")
    tables = []
    lines_with_pipe = []
    lines_with_pound = []
    is_currently_table = False
    lines = io.StringIO(s)
    for line in lines:
        if line.startswith("#"):
            lines_with_pound.append(line.strip())
        if line.startswith("|"):
            is_currently_table = True
            lines_with_pipe.append(line.strip())
        if len(lines_with_pipe) > 0 and not line.startswith("|"):
            is_currently_table = False
            tables.append(lines_with_pipe)

    return lines_with_pound, lines_with_pipe, is_currently_table
