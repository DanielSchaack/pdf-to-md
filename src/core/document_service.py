from core.file_service import FileService
from core.document_processors import Processor
from core.repository_models import ProcessedFile, ExtractedImage, TextChunk
from core import document_schemas
from core.repository_service import RepositoryService
from typing import List, Tuple, Optional
import requests
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self,
                 file_service: FileService,
                 repository_service: RepositoryService):
        self.file_service = file_service
        self.repository_service = repository_service

    def process_pdf(self, pdf_path: str, processor: Processor):
        file_to_process, images_to_process = self.convert_to_images(pdf_path=pdf_path,
                                                                    processor=processor)

        images_to_process: List[str] = self.get_ocr_of_images(processor=processor,
                                                              file_to_process=file_to_process,
                                                              images_to_process=images_to_process)

        file_to_process = self.aggregate_texts(processor=processor,
                                               extracted_images=images_to_process,
                                               file_to_process=file_to_process)

        self.convert_to_chunks(processor=processor,
                               file_to_process=file_to_process)

    def convert_to_images(self, pdf_path: str, processor: Processor) -> Tuple[ProcessedFile,
                                                                              List[ExtractedImage]]:
        filename = self.file_service.get_filename_from_path(pdf_path)
        processed_file_create = document_schemas.ProcessedFileCreate(original_pdf_path=pdf_path,
                                                                     filename=filename,
                                                                     status=document_schemas.ProcessingStatus.PENDING)

        db_file = self.repository_service.create_processed_file(processed_file_create)
        logger.info(f"Created ProcessedFile record ID: {db_file.id} for {filename}")

        image_paths = self.file_service.convert_pdf_to_images(pdf_path=pdf_path,
                                                              filename=filename,
                                                              file_id=db_file.id,
                                                              image_format="png",
                                                              dpi=300,
                                                              colorspace="rgb",
                                                              page_numbers=None,
                                                              use_alpha=False)
        images_to_create: List[document_schemas.ExtractedImageCreate] = []
        images_to_process: List[ExtractedImage] = []

        for i, img_path in enumerate(image_paths):
            images_to_create.append(
                document_schemas.ExtractedImageCreate(
                    image_path=img_path,
                    page_number=i + 1
                )
            )

        if images_to_create:
            images_to_process = self.repository_service.bulk_create_extracted_images(db_file.id, images_to_create)
            logger.info(f"[{db_file.id}] Saved {len(images_to_create)} image paths to DB.")

        self.repository_service.update_processed_file(
            db_file,
            document_schemas.ProcessedFileUpdate(status=document_schemas.ProcessingStatus.IMAGES_EXTRACTED)
        )
        return db_file, images_to_process

    def get_ocr_of_images(self, processor: Processor,
                          file_to_process: ProcessedFile,
                          images_to_process: List[ExtractedImage]) -> List[ExtractedImage]:
        current_headers: List[str] = []
        current_tables: List[List[str]] = []
        current_is_table: bool = False

        for index, image in enumerate(images_to_process):
            image = self.process_image(file_to_process=file_to_process,
                                       image_to_process=image,
                                       processor=processor,
                                       headers=current_headers,
                                       tables=current_tables,
                                       is_table=current_is_table)
            current_headers = image.headers
            current_tables = image.tables
            current_is_table = image.is_table

        self.repository_service.update_processed_file(
            file_to_process,
            document_schemas.ProcessedFileUpdate(status=document_schemas.ProcessingStatus.OCR_COMPLETED)
        )
        logger.info(f"[{file_to_process.id}] OCR completed for images.")
        return images_to_process

    def process_image(self,
                      file_to_process: ProcessedFile,
                      image_to_process: ExtractedImage,
                      processor: Processor,
                      headers: List[str],
                      tables: List[List[str]],
                      is_table: bool = False) -> ExtractedImage:
        blocks = self.file_service.get_text_column_data(image_to_process.image_path, language="deu")
        combined_tesseract_text = "\n".join(block["full_text"] for block in blocks)
        logger.debug(f"Tesseract provided the following text: \n{combined_tesseract_text}")

        ocr_response = processor.call_ocr(filename=file_to_process.filename,
                                          image_path=image_to_process.image_path,
                                          headers=headers,
                                          tables=tables,
                                          ocr_text=image_to_process.tesseract_raw_text,
                                          is_table=is_table)

        ocr_response, headers, tables, is_table = processor.refine_headers(ocr_response, headers)
        image_to_process = self.repository_service.update_extracted_image(image_to_process,
                                                                          document_schemas.ExtractedImageUpdate(
                                                                              image_path=image_to_process.image_path,
                                                                              tesseract_raw_text=combined_tesseract_text,
                                                                              refined_ocr_text=ocr_response,
                                                                              headers=headers,
                                                                              tables=tables,
                                                                              is_table=is_table))
        return image_to_process

    def aggregate_texts(self, processor: Processor, extracted_images: List[ExtractedImage], file_to_process: ProcessedFile) -> ProcessedFile:
        complete_text = processor.integrate_messages(extracted_images=[image.refined_ocr_text for image in extracted_images])

        self.repository_service.update_processed_file(file_to_process,
                                                      document_schemas.ProcessedFileUpdate(
                                                          status=document_schemas.ProcessingStatus.TEXT_AGGREGATED,
                                                          aggregated_text=complete_text))

        self.file_service.save_to_filesystem(filename=file_to_process.filename,
                                             file_id=file_to_process.id,
                                             filetype=processor.filetype,
                                             text=complete_text)
        return file_to_process

    def convert_to_chunks(self,
                          processor: Processor,
                          file_to_process: ProcessedFile,
                          header_level_cutoff: Optional[int] = None) -> List[TextChunk]:
        chunks, chunk_titles = processor.convert_text_to_chunks(filename=file_to_process.filename,
                                                                aggregated_text=file_to_process.aggregated_text,
                                                                header_level_cutoff=3)
        assert len(chunks) == len(chunk_titles), "Each chunk needs its own title"

        self.repository_service.update_processed_file(
            file_to_process,
            document_schemas.ProcessedFileUpdate(status=document_schemas.ProcessingStatus.CHUNKING_COMPLETED)
        )

        self.repository_service.bulk_create_text_chunks(processed_file_id=file_to_process.id,
                                                        chunks_data=[document_schemas.TextChunkCreate(chunk_text=chunks[i], chunk_title=chunk_titles[i]) for i in range(len(chunks))])

        processed_chunks = self.file_service.convert_chunks_to_files(filename=file_to_process.filename,
                                                                     file_id=file_to_process.id,
                                                                     filetype=processor.filetype,
                                                                     chunks=chunks,
                                                                     titles=chunk_titles)

        self.repository_service.update_processed_file(
            file_to_process,
            document_schemas.ProcessedFileUpdate(status=document_schemas.ProcessingStatus.COMPLETED)
        )
        return processed_chunks


def is_ollama_available(ollama_url: str) -> bool:
    """
    Checks if the Ollama service is available at the given URL.
    """
    try:
        # Ollama's root endpoint usually returns "Ollama is running" with a 200 OK.
        response = requests.get(ollama_url, timeout=5)
        if response.status_code == 200:
            print(f"Ollama is available at {ollama_url}. Response: {response.text[:100]}")
            return True
        else:
            print(f"Ollama returned status {response.status_code} at {ollama_url}.")
            return False
    except Exception as e:
        print(f"An unexpected error occurred while checking Ollama availability: {e}")
        return False
