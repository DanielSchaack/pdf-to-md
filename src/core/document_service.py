from core.file_service import FileService
from core.repository_service import RepositoryService
from core.document_processors import Processor
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self,
                 file_service: FileService):
        self.file_service = file_service
        self.repository_service = RepositoryService()

    def process_pdf(self, pdf_path: str, processor: Processor):
        filename = self.file_service.get_filename_from_path(pdf_path)

        image_paths = self.file_service.convert_pdf_to_images(pdf_path=pdf_path,
                                                              filename=filename,
                                                              image_format="png",
                                                              dpi=300,
                                                              colorspace="rgb",
                                                              page_numbers=None,
                                                              use_alpha=False)
        # TODO save file paths

        ocr_responses: List[str] = []
        headers: List[str] = []
        tables: List[List[str]] = []
        is_table: bool = False

        for index, image_path in enumerate(image_paths):

            blocks = self.file_service.get_text_column_data(image_path, language="deu")
            combined_tesseract_text = "\n".join(block["full_text"] for block in blocks)
            logger.debug(f"Tesseract provided the following text: \n{combined_tesseract_text}")
            # TODO save tesseract result

            ocr_response = processor.call_ocr(filename=filename,
                                              image_path=image_path,
                                              headers=headers,
                                              tables=tables,
                                              ocr_text=combined_tesseract_text,
                                              is_table=is_table)

            ocr_response, headers, tables, is_table = self.process_image(filename=filename,
                                                                         image_path=image_path,
                                                                         processor=processor,
                                                                         headers=headers,
                                                                         tables=tables,
                                                                         is_table=is_table)

            ocr_responses.append(ocr_response)

        complete_text = self.aggregate_texts(processor=processor,
                                             ocr_responses=ocr_responses,
                                             filename=filename)

        self.convert_to_chunks(processor=processor,
                               filename=filename,
                               complete_text=complete_text)

    def process_image(self,
                      filename: str,
                      image_path: str,
                      processor: Processor,
                      headers: List[str],
                      tables: List[List[str]],
                      is_table: bool = False) -> Tuple[str, List[str], List[List[str]], bool]:
        blocks = self.file_service.get_text_column_data(image_path, language="deu")
        combined_tesseract_text = "\n".join(block["full_text"] for block in blocks)
        logger.debug(f"Tesseract provided the following text: \n{combined_tesseract_text}")
        # TODO save tesseract result

        ocr_response = processor.call_ocr(filename=filename,
                                          image_path=image_path,
                                          headers=headers,
                                          tables=tables,
                                          ocr_text=combined_tesseract_text,
                                          is_table=is_table)

        ocr_response, headers, tables, is_table = processor.refine_headers_tables(ocr_response, headers)
        # TODO save response
        return ocr_response, headers, tables, is_table

    def aggregate_texts(self, processor: Processor, ocr_responses: List[str], filename: str):
        complete_text = processor.integrate_messages(ocr_responses=ocr_responses)
        # TODO save complete text

        self.file_service.save_to_filesystem(filename=filename,
                                             filetype=processor.filetype,
                                             text=complete_text)
        return complete_text

    def convert_to_chunks(self,
                          processor: Processor,
                          filename: str,
                          complete_text: str,
                          header_level_cutoff: Optional[int] = None):
        chunks = processor.convert_text_to_chunks(filename=filename,
                                                  input_text=complete_text,
                                                  header_level_cutoff=3)
        # TODO save chunks

        self.file_service.convert_chunks_to_files(filename=filename,
                                                  filetype=processor.filetype,
                                                  chunks=chunks)
