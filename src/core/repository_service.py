from sqlalchemy import create_engine, asc, desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, selectinload, Session
from typing import List, Optional, TypeVar
from core import repository_models, document_schemas
import logging

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=repository_models.Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=document_schemas.BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=document_schemas.BaseModel)
DATABASE_URL = "sqlite:///./data/document_processing.db"


class RepositoryService:
    def __init__(self, db_session: Session):
        self.db: Session = db_session

    def _commit_and_refresh(self, db_obj: ModelType) -> ModelType:
        try:
            self.db.add(db_obj)
            self.db.commit()
            self.db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database commit/refresh error: {e}")
            raise

    def _bulk_commit_and_refresh(self, db_objs: List[ModelType]) -> List[ModelType]:
        try:
            self.db.add_all(db_objs)
            self.db.commit()
            for db_obj in db_objs:
                self.db.refresh(db_obj)
            return db_objs
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database bulk commit/refresh error: {e}")
            raise

    def _delete_and_commit(self, db_obj: ModelType) -> bool:
        try:
            self.db.delete(db_obj)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database delete error for object {db_obj}: {e}")
            return False

    def _bulk_delete_and_commit(self, db_objs: List[ModelType]) -> bool:
        try:
            for db_obj in db_objs:
                self.db.delete(db_obj)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database bulk delete error for objects: {e}")
            return False

# --- ProcessedFile Operations ---
    def create_processed_file(self, file_data: document_schemas.ProcessedFileCreate) -> repository_models.ProcessedFile:
        db_file = repository_models.ProcessedFile(
            original_pdf_path=file_data.original_pdf_path,
            filename=file_data.filename,
            status=file_data.status if file_data.status else repository_models.ProcessingStatus.PENDING
        )
        return self._commit_and_refresh(db_file)

    def get_processed_file(self, file_id: int) -> Optional[repository_models.ProcessedFile]:
        return self.db.query(repository_models.ProcessedFile).filter(repository_models.ProcessedFile.id == file_id).first()

    def get_files_by_status(self, status: List[document_schemas.ProcessingStatus]) -> List[repository_models.ProcessedFile]:
        return self.db.query(repository_models.ProcessedFile).filter(repository_models.ProcessedFile.status.in_(status)).all()

    def get_processed_file_by_path(self, original_pdf_path: str) -> Optional[repository_models.ProcessedFile]:
        return self.db.query(repository_models.ProcessedFile).filter(repository_models.ProcessedFile.original_pdf_path == original_pdf_path).first()

    def get_processed_file_by_id_with_details(self, file_id: int) -> Optional[repository_models.ProcessedFile]:
        stmt = (
            select(repository_models.ProcessedFile)
            .where(repository_models.ProcessedFile.id == file_id)
            .options(
                selectinload(repository_models.ProcessedFile.images),
                selectinload(repository_models.ProcessedFile.chunks)
            )
        )
        result = self.db.execute(stmt)
        return result.scalar_one_or_none()

    def get_processed_files(self,
                            filename: str,
                            offset: int = 0,
                            limit: int = 100,
                            order_by: str = "id",
                            order_direction: str = "asc"
                            ) -> List[repository_models.ProcessedFile]:
        query = self.db.query(repository_models.ProcessedFile)

        if filename:
            query = query.filter(repository_models.ProcessedFile.filename.like('%' + filename + '%'))

        # Apply ordering based on parameters
        if order_by == "id":
            if order_direction == "desc":
                query = query.order_by(desc(repository_models.ProcessedFile.id))
            else:
                query = query.order_by(asc(repository_models.ProcessedFile.id))
        elif order_by == "filename":
            if order_direction == "desc":
                query = query.order_by(desc(repository_models.ProcessedFile.filename))
            else:
                query = query.order_by(asc(repository_models.ProcessedFile.filename))
        else:
            query = query.order_by(asc(repository_models.ProcessedFile.id))  # Default to sorting by ID ascending

        query = query.offset(offset).limit(limit)
        return query.all()

    def delete_processed_file_by_id(self, file_id: int) -> bool:
        db_file = self.get_processed_file(file_id=file_id)
        if db_file:
            self._bulk_delete_and_commit(self.get_extracted_images_for_file(file_id))
            self._bulk_delete_and_commit(self.get_text_chunks_for_file(file_id))
            self._delete_and_commit(db_file)
            return True
        else:
            return False

    def update_processed_file(self,
                              db_file: repository_models.ProcessedFile,
                              update_data: document_schemas.ProcessedFileUpdate) -> Optional[repository_models.ProcessedFile]:
        if db_file:
            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(db_file, key, value)
            return self._commit_and_refresh(db_file)
        return None

    def update_processed_file_by_id(self,
                                    file_id: int,
                                    update_data: document_schemas.ProcessedFileUpdate) -> Optional[repository_models.ProcessedFile]:
        db_file = self.get_processed_file(file_id)
        if db_file:
            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(db_file, key, value)
            return self._commit_and_refresh(db_file)
        return None

    # --- ExtractedImage Operations ---
    def create_extracted_image(self,
                               processed_file_id: int,
                               image_data: document_schemas.ExtractedImageCreate) -> repository_models.ExtractedImage:
        db_image = repository_models.ExtractedImage(
            processed_file_id=processed_file_id,
            **image_data.model_dump()
        )
        return self._commit_and_refresh(db_image)

    def bulk_create_extracted_images(self,
                                     processed_file_id: int,
                                     images_data: List[document_schemas.ExtractedImageCreate]) -> List[repository_models.ExtractedImage]:
        db_images = [
            repository_models.ExtractedImage(
                processed_file_id=processed_file_id,
                **image_data.model_dump()
            ) for image_data in images_data
        ]
        return self._bulk_commit_and_refresh(db_images)

    def get_extracted_image(self, image_id: int) -> Optional[repository_models.ExtractedImage]:
        return self.db.query(repository_models.ExtractedImage).filter(repository_models.ExtractedImage.id == image_id).first()

    def get_extracted_image_by_path(self, image_path: str) -> Optional[repository_models.ExtractedImage]:
        return self.db.query(repository_models.ExtractedImage).filter(repository_models.ExtractedImage.image_path == image_path).first()

    def update_extracted_image(self,
                               db_image: repository_models.ExtractedImage,
                               update_data: document_schemas.ExtractedImageUpdate) -> Optional[repository_models.ExtractedImage]:
        if db_image:
            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(db_image, key, value)
            return self._commit_and_refresh(db_image)
        return None

    def update_extracted_image_by_id(self,
                                     image_id: int,
                                     update_data: document_schemas.ExtractedImageUpdate) -> Optional[repository_models.ExtractedImage]:
        db_image = self.get_extracted_image(image_id)
        if db_image:
            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(db_image, key, value)
            return self._commit_and_refresh(db_image)
        return None

    def get_extracted_images_for_file(self, processed_file_id: int) -> List[repository_models.ExtractedImage]:
        return self.db.query(repository_models.ExtractedImage).filter(repository_models.ExtractedImage.processed_file_id == processed_file_id).order_by(repository_models.ExtractedImage.page_number).all()

    def delete_extracted_images_for_processed_file(self, file_id: int) -> bool:
        """
        Deletes all ExtractedImage objects associated with a given ProcessedFile.
        """
        try:
            extracted_images = self.db.query(repository_models.ExtractedImage).filter(
                repository_models.ExtractedImage.processed_file_id == file_id
            ).all()
            if extracted_images:
                logger.info(f"Deleting {len(extracted_images)} extracted images for ProcessedFile ID: {file_id}")
                return self._bulk_delete_and_commit(extracted_images)
            else:
                logger.info(f"No extracted images found for ProcessedFile ID: {file_id} to delete.")
                return True
        except Exception as e:
            logger.error(f"Error deleting extracted images for ProcessedFile ID {file_id}: {e}")
            self.db.rollback()
            raise

    # --- TextChunk Operations ---
    def create_text_chunk(self, processed_file_id: int,
                          chunk_data: document_schemas.TextChunkCreate) -> repository_models.TextChunk:
        db_chunk = repository_models.TextChunk(
            processed_file_id=processed_file_id,
            **chunk_data.model_dump()
        )
        return self._commit_and_refresh(db_chunk)

    def bulk_create_text_chunks(self,
                                processed_file_id: int,
                                chunks_data: List[document_schemas.TextChunkCreate]) -> List[repository_models.TextChunk]:
        db_chunks = [
            repository_models.TextChunk(
                processed_file_id=processed_file_id,
                **chunk_data.model_dump()
            ) for chunk_data in chunks_data
        ]
        return self._bulk_commit_and_refresh(db_chunks)

    def get_text_chunks_for_file(self, processed_file_id: int) -> List[repository_models.TextChunk]:
        return self.db.query(repository_models.TextChunk).filter(repository_models.TextChunk.processed_file_id == processed_file_id).order_by(repository_models.TextChunk.chunk_number).all()

    def delete_text_chunks_for_processed_file_id(self, file_id: int) -> bool:
        """
        Deletes all TextChunk objects associated with a given ProcessedFile.
        """
        try:
            text_chunks = self.db.query(repository_models.TextChunk).filter(
                repository_models.TextChunk.processed_file_id == file_id
            ).all()
            if text_chunks:
                logger.info(f"Deleting {len(text_chunks)} text chunks for ProcessedFile ID: {file_id}")
                return self._bulk_delete_and_commit(text_chunks)
            else:
                logger.info(f"No text chunks found for ProcessedFile ID: {file_id} to delete.")
                return True
        except Exception as e:
            logger.error(f"Error deleting text chunks for ProcessedFile ID {file_id}: {e}")
            self.db.rollback()
            raise

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_db_and_tables():
    repository_models.Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
