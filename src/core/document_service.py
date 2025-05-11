from core.markdown_service import MarkdownService
from core.file_service import FileService
from core.llm_service import LlmService
from core.repository_service import RepositoryService
from typing import List
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
                                                              page_numbers=[1],
                                                              use_alpha=False)
        # save file paths

        headers: List[str] = []
        for index, image_path in enumerate(image_paths):

            blocks = self.file_service.get_text_column_data(image_path, language="deu")
            combined_full_text = "\n".join(block["full_text"] for block in blocks)
            logger.debug(f"Tesseract provided the following text: \n{combined_full_text}")
            # save Tesseract result

            response_message = self.llm_service.call_image_llm_provider(image_path)
            cleansed_response_message = self.markdown_service.remove_lines_starting_with(response_message, "`")
            # save response

            headers, tables, is_table = self.markdown_service.get_markdown_headers_and_tables(cleansed_response_message)
            if len(tables) > 0:
                table_texts = self.convert_tables_to_texts(tables)
                markdown_text = self.markdown_service.replace_tables_with_texts(cleansed_response_message, table_texts)
            # save markdown_text

            chunks = self.markdown_service.convert_markdown_to_chunks(filename=filename,
                                                                      markdown_text=markdown_text,
                                                                      header_level_cutoff=3)
            # save chunks
            self.file_service.convert_chunks_to_files(filename_prefix=filename,
                                                      chunk_suffix="md",
                                                      chunks=chunks)

    # TODO
    def convert_tables_to_texts(self, tables: List[List[str]]) -> List[str]:
        texts: str = []
        for table in tables:
            text = "\n".join(table)
            texts.append(text)
        return texts
