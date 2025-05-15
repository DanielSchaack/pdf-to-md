from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import enum


class ProcessingStatus(enum.Enum):
    PENDING = "pending"
    IMAGES_EXTRACTED = "images_extracted"
    OCR_IN_PROGRESS = "ocr_in_progress"
    OCR_COMPLETED = "ocr_completed"
    TEXT_AGGREGATED = "text_aggregated"
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
