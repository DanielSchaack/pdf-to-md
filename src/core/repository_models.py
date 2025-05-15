from sqlalchemy import Column, Integer, String, Text, JSON, ForeignKey, DateTime, Boolean, Enum as SQLAlchemyEnum
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from core.document_schemas import ProcessingStatus


Base = declarative_base()


class ProcessedFile(Base):
    __tablename__ = "processed_files"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    original_pdf_path = Column(String, nullable=False, index=True)
    filename = Column(String, index=True)
    status = Column(SQLAlchemyEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    error_message = Column(Text, nullable=True)
    aggregated_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    images = relationship("ExtractedImage", back_populates="processed_file", cascade="all, delete-orphan")
    chunks = relationship("TextChunk", back_populates="processed_file", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ProcessedFile(id={self.id}, filename='{self.filename}', status='{self.status.value}')>"


class ExtractedImage(Base):
    __tablename__ = "extracted_images"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    processed_file_id = Column(Integer, ForeignKey("processed_files.id"), nullable=False)
    image_path = Column(String, nullable=False, unique=True)
    page_number = Column(Integer, nullable=True)
    tesseract_raw_text = Column(Text, nullable=True)
    refined_ocr_text = Column(Text, nullable=True)
    headers = Column(JSON, nullable=True)
    tables = Column(JSON, nullable=True)
    is_table = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    processed_file = relationship("ProcessedFile", back_populates="images")

    def __repr__(self):
        return f"<ExtractedImage(id={self.id}, path='{self.image_path}', file_id={self.processed_file_id})>"


class TextChunk(Base):
    __tablename__ = "text_chunks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    processed_file_id = Column(Integer, ForeignKey("processed_files.id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_title = Column(Text, nullable=True)
    chunk_number = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_file = relationship("ProcessedFile", back_populates="chunks")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<TextChunk(id={self.id}, file_id={self.processed_file_id}, order={self.chunk_number})>"

