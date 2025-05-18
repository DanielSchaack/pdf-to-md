from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum
import enum


class ProcessingStatus(str, enum.Enum):
    QUEUED_FOR_PROCESSING = "queued_for_processing"
    PENDING = "pending"
    IMAGE_EXTRACTION_IN_PROGRESS = "image_extraction_in_progress "
    IMAGES_EXTRACTED = "images_extracted"
    OCR_IN_PROGRESS = "ocr_in_progress"
    OCR_COMPLETED = "ocr_completed"
    TEXT_AGGREGATION_IN_PROGRESS = "text_aggregation_in_progress "
    TEXT_AGGREGATED = "text_aggregated"
    CHUNKING_IN_PROGRESS = "chunking_in_progress"
    CHUNKING_COMPLETED = "chunking_completed"
    COMPLETED = "completed"
    FAILED = "failed"


# --- ProcessedFile Schemas ---
class ProcessedFileBase(BaseModel):
    original_pdf_path: str
    filename: str
    status: Optional[ProcessingStatus] = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    aggregated_text: Optional[str] = None


class ProcessedFileCreate(ProcessedFileBase):
    pass


class ProcessedFileUpdate(BaseModel):
    filename: Optional[str] = None
    status: Optional[ProcessingStatus] = None
    error_message: Optional[str] = None
    aggregated_text: Optional[str] = None


class ProcessedFileInDB(ProcessedFileBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- ExtractedImage Schemas ---
class ExtractedImageBase(BaseModel):
    image_path: str
    page_number: Optional[int] = None
    tesseract_raw_text: Optional[str] = None
    refined_ocr_text: Optional[str] = None
    headers: Optional[List[str]] = None
    tables: Optional[List[str]] = None
    is_table: Optional[bool] = None


class ExtractedImageCreate(ExtractedImageBase):
    pass


class ExtractedImageUpdate(ExtractedImageBase):
    image_path: Optional[str] = None
    page_number: Optional[int] = None
    tesseract_raw_text: Optional[str] = None
    refined_ocr_text: Optional[str] = None
    headers: Optional[List[str]] = None
    tables: Optional[List[str]] = None
    is_table: Optional[bool] = None


class ExtractedImageInDB(ExtractedImageBase):
    id: int
    processed_file_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- TextChunk Schemas ---
class TextChunkBase(BaseModel):
    chunk_text: str
    chunk_title: Optional[str] = None
    chunk_number: Optional[int] = None


class TextChunkCreate(TextChunkBase):
    pass


class TextChunkUpdate(BaseModel):
    chunk_text: Optional[str] = None
    chunk_title: Optional[str] = None
    chunk_number: Optional[int] = None


class TextChunkInDB(TextChunkBase):
    id: int
    processed_file_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ProcessedFileWithDetails(ProcessedFileInDB):
    images: List[ExtractedImageInDB] = []
    chunks: List[TextChunkInDB] = []

    class Config:
        from_attributes = True


class ProcessingStep(str, Enum):
    SCAN = "scan"
    OCR = "ocr"
    AGGREGATE = "aggregate"
    CHUNK = "chunk"
    COMPLETE = "complete"


class PdfProcessingRequestParams(BaseModel):
    # model_config = {"extra": "forbid"}

    output_type: str = Field(default="md", description="Desired output type (e.g., 'md', 'txt').")
    provider: Optional[str] = Field(default=None, description="LLM provider (e.g., 'Ollama', 'Openrouter'). Overrides environment settings.")
    image_model: Optional[str] = Field(default=None, description="Multimodal LLM model name. Overrides environment settings.")
    text_model: Optional[str] = Field(default=None, description="Text LLM model name. Overrides environment settings.")
    api_url: Optional[str] = Field(default=None, description="LLM API URL. Overrides environment settings.")
    api_key: Optional[str] = Field(default=None, description="LLM API Key. Overrides environment settings. (Use with caution)")
    processing_steps: List[ProcessingStep] = Field(
        default=[ProcessingStep.COMPLETE],
        description="List of processing steps to perform (scan, ocr, aggregate, chunk, complete)."
    )


class OllamaStatusResponse(BaseModel):
    ollama_url_checked: str
    is_available: bool
    message: str


class BatchPdfProcessingRequest(BaseModel):
    common_params: PdfProcessingRequestParams = Field(description="Common processing parameters for all files in the batch.")
    file_ids: Optional[List[int]] = Field(default=None, description="List of specific file IDs to process.")
    target_statuses: Optional[List[ProcessingStatus]] = Field(default=None, description="Process files currently in one of these statuses.")

    # Validator to ensure at least one selection method is provided
    @model_validator(mode='after')
    def check_selection_method(cls, values):
        if not values.file_ids and not values.target_statuses:
            raise ValueError("Either 'file_ids' or 'target_statuses' must be provided.")
        if values.file_ids and values.target_statuses:
            # Decide on precedence or raise error. Let's prioritize file_ids.
            print("Warning: Both 'file_ids' and 'target_statuses' provided. 'file_ids' will be used.")
            values.target_statuses = None
        return values


class FilterParams(BaseModel):
    filename: str = Field(None)
    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["id", "filename", "created_at", "updated_at"] = "id"
    order_direction: Literal["asc", "desc"] = "asc"
