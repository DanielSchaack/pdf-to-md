import logging
import logging.config
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import Annotated, Optional

from core.file_service import FileService
from core.document_service import DocumentService, is_ollama_available
from core.document_processors import Processor, ProcessorFactory
from core.document_schemas import BatchPdfProcessingRequest, PdfProcessingRequestParams, ProcessedFileWithDetails, ProcessedFileCreate, ProcessedFileUpdate, ProcessedFileInDB, FilterParams, ProcessingStatus, OllamaStatusResponse
from core.llm_service import LlmService
from core.repository_service import RepositoryService, get_db, create_db_and_tables
from core.repository_models import ProcessedFile


logging.config.fileConfig("src/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def background_pdf_processing_worker(
    file_id: int,
    pdf_path: str,
    params: PdfProcessingRequestParams,
    file_service: FileService,
    repository_service: RepositoryService
):
    logger.debug(f"Endpoint called for file_id: {file_id} with params: {params.model_dump()}")
    db_file: ProcessedFile = repository_service.get_processed_file(file_id)

    pdf_path_to_process = db_file.original_pdf_path

    llm_service = LlmService(
        provider=params.provider,
        image_model=params.image_model,
        text_model=params.text_model,
        api_url=params.api_url,
        api_key=params.api_key
    )

    processor: Processor = ProcessorFactory.create_processor(
        filetype=params.output_type,
        llm_service=llm_service
    )

    document_service = DocumentService(
        file_service=file_service,
        repository_service=repository_service
    )

    try:
        logger.debug(f"Calling document_service.process_pdf for path: {pdf_path_to_process}")
        document_service.process_pdf(
            pdf_path=None,
            file_id=file_id,
            processor=processor,
            steps=params.processing_steps
        )

    except Exception as e:
        logger.error(f"Error during PDF processing: {e}", exc_info=True)
        if repository_service:
            repository_service.update_processed_file(db_file,
                                                     ProcessedFileUpdate(status=ProcessingStatus.FAILED,
                                                                         error_message=str(e)))


def get_api_repository_service(db: Session = Depends(get_db)) -> RepositoryService:
    return RepositoryService(db_session=db)


def get_api_file_service():
    return FileService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI application startup...")
    create_db_and_tables()
    logger.info("Database tables checked/created.")
    yield
    # Shutdown
    logger.info("FastAPI application shutdown...")

app = FastAPI(lifespan=lifespan)


@app.get("/v1/documents/pdfs/",
         response_model=List[ProcessedFileInDB],
         summary="List all documents")
async def list_documents(
    filter_params: Annotated[FilterParams, Query()],
    repository_service: RepositoryService = Depends(get_api_repository_service)
):
    db_files = repository_service.get_processed_files(filename=filter_params.filename,
                                                      offset=filter_params.offset,
                                                      limit=filter_params.limit,
                                                      order_by=filter_params.order_by,
                                                      order_direction=filter_params.order_direction)
    return [ProcessedFileInDB.model_validate(f) for f in db_files]


@app.get("/v1/documents/pdfs/{file_id}",
         response_model=ProcessedFileWithDetails,
         summary="Get the details of a document, including extracted images and text chunks")
async def get_document_details(
    file_id: int,
    repository_service: RepositoryService = Depends(get_api_repository_service)
):
    db_file_with_details = repository_service.get_processed_file_by_id_with_details(file_id)

    if not db_file_with_details:
        raise HTTPException(status_code=404, detail=f"ProcessedFile with ID {file_id} not found.")
    return db_file_with_details


@app.post("/v1/documents/pdfs/",
          response_model=List[ProcessedFileInDB],
          status_code=202,
          summary="Upload PDF documents")
async def api_process_pdf(
    pdf_files: List[UploadFile] = File(..., description="PDF file to process."),
    repository_service: RepositoryService = Depends(get_api_repository_service),
    file_service: FileService = Depends(get_api_file_service)
):
    files: List[ProcessedFileInDB] = []
    for pdf_file in pdf_files:
        if not pdf_file.filename:
            raise HTTPException(status_code=400, detail="Filename not provided with upload.")
        if not pdf_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Invalid file type for {pdf_file.filename}. Only PDF files are accepted.")

        filename = file_service.get_filename_from_path(pdf_file.filename)
        filepath = ""
        try:
            content = pdf_file.file.read()
            filepath = file_service.save_to_filesystem(filename=filename, file_id=None, content=content, filetype="pdf")
        except Exception:
            raise HTTPException(status_code=500, detail=f'Something went wrong during writing of {pdf_file.filename}')
        finally:
            pdf_file.file.close()

        db_file = repository_service.create_processed_file(ProcessedFileCreate(original_pdf_path=filepath,
                                                                               filename=filename))

        logger.info(f"File and entry for '{pdf_file.filename}' (ID: {db_file.id}) added to files.")
        files.append(ProcessedFileInDB.model_validate(db_file))
    return files


@app.put("/v1/documents/pdfs/", status_code=204)
async def batch_process_pdf_documents_endpoint(
    batch_request: BatchPdfProcessingRequest,
    background_tasks: BackgroundTasks,
    file_service: FileService = Depends(get_api_file_service),
    repository_service: RepositoryService = Depends(get_api_repository_service)
):
    logger.debug(f"ENDPOINT (BATCH): Called with batch_request: {batch_request.model_dump_json(indent=2)}")
    files_to_process: List[ProcessedFile] = []

    if batch_request.file_ids:
        files_to_process = repository_service.get_files_by_ids(batch_request.file_ids)
        if len(files_to_process) != len(batch_request.file_ids):
            print("Warning: Some requested file_ids were not found.")
    elif batch_request.target_statuses:
        files_to_process = repository_service.get_files_by_status(batch_request.target_statuses)

    if not files_to_process or len(files_to_process) == 0:
        raise HTTPException(status_code=404, detail="No files found matching the criteria for batch processing.")

    queued_tasks = []
    for db_file in files_to_process:
        if not db_file.original_pdf_path:
            print(f"Skipping file_id {db_file.id} due to missing path.")
            continue

        background_tasks.add_task(
            background_pdf_processing_worker,
            file_id=db_file.id,
            pdf_path=db_file.original_pdf_path,
            params=batch_request.common_params,
            file_service=file_service,
            repository_service=repository_service
        )
        queued_tasks.append({"file_id": db_file.id})

        repository_service.update_processed_file(db_file,
                                                 ProcessedFileUpdate(status=ProcessingStatus.QUEUED_FOR_PROCESSING))

    return {
        "message": f"{len(queued_tasks)} PDF processing tasks have been queued.",
        "queued_tasks_count": len(queued_tasks),
        "tasks_details": queued_tasks
    }


@app.put("/v1/documents/pdfs/{file_id}", status_code=204)
async def process_pdf_document(
    file_id: int,
    params: PdfProcessingRequestParams = Query(),
    repository_service: RepositoryService = Depends(get_api_repository_service),
    file_service: FileService = Depends(get_api_file_service),
    background_tasks: BackgroundTasks = BackgroundTasks()
):

    logger.debug(f"Endpoint called for file_id: {file_id} with params: {params.model_dump()}")
    db_file: ProcessedFile = repository_service.get_processed_file(file_id)
    if not db_file:
        raise HTTPException(status_code=404, detail=f"ProcessedFile with ID {file_id} not found.")

    if not hasattr(db_file, 'original_pdf_path') or not db_file.original_pdf_path:
        raise HTTPException(status_code=500, detail=f"File path not found for ProcessedFile ID {file_id}.")

    background_tasks.add_task(
        background_pdf_processing_worker,
        file_id=db_file.id,
        pdf_path=db_file.original_pdf_path,
        params=params,
        file_service=file_service,
        repository_service=repository_service
    )

    repository_service.update_processed_file(db_file,
                                             ProcessedFileUpdate(status=ProcessingStatus.QUEUED_FOR_PROCESSING))
    return {"message": f"PDF document with ID {file_id} is being processed.",
            "file_id": file_id,
            "output_type": params.output_type,
            "provider_used": params.provider}


@app.delete("/v1/documents/pdfs/{file_id}", status_code=204)
async def delete_pdf_document(
    file_id: int,
    repository_service: RepositoryService = Depends(get_api_repository_service)
):
    is_deleted = repository_service.delete_processed_file_by_id(file_id)
    if not is_deleted:
        raise HTTPException(status_code=500, detail=f"ProcessedFile with ID {file_id} not deleted.")


@app.get("/v1/health/ollama",
         response_model=OllamaStatusResponse,
         summary="Check Ollama Service Availability")
async def check_ollama_status(
    ollama_url: Optional[str] = Query(None, description="Ollama URL to check (e.g., http://localhost:11434). If not provided, uses configured default.")
):
    """
    Verifies if the Ollama service is reachable and responding correctly.
    You can provide a specific `ollama_url` to check, or it will use the
    system's default configured Ollama URL.
    """
    target_url_to_check = ollama_url if ollama_url else "http://localhost:11434"

    if not target_url_to_check:
        raise HTTPException(
            status_code=400, 
            detail="Ollama URL not provided and no default is configured in the service."
        )

    available = is_ollama_available(ollama_url=target_url_to_check)

    if available:
        return OllamaStatusResponse(
            ollama_url_checked=target_url_to_check,
            is_available=True,
            message="Ollama service is available and responding."
        )
    else:
        return OllamaStatusResponse(
            ollama_url_checked=target_url_to_check,
            is_available=False,
            message="Ollama service is not available or not responding as expected. Check server logs for details."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=42069, reload=True)
