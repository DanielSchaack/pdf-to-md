from core.document_processors import Processor, ProcessorFactory
from core.document_service import DocumentService
from core.document_schemas import ProcessingStep
from core.file_service import FileService
from core.repository_service import RepositoryService, get_db, create_db_and_tables
from core.llm_service import LlmService
import logging
import logging.config
import argparse

logging.config.fileConfig("src/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PDF processing with an LLM provider."
    )

    parser.add_argument(
        "--pdf-path",
        type=str,
        required=True,
        help="Path to the PDF document to be processed. "
    )

    parser.add_argument(
        "--provider",
        type=str,
        help="The llm provider to use. Currently available: Openrouter, Ollama. Alternatively, you can set the 'API_PROVIDER' environment variable."
    )

    parser.add_argument(
        "--image-model",
        type=str,
        help="The multimodal model from the provider to use. Tested with Mistral 3.1 Small. Alternatively, you can set the 'API_IMAGE_MODEL' environment variable."
    )

    parser.add_argument(
        "--text-model",
        type=str,
        help="The text model from the provider to use. Tested with Mistral 3.1 Small. Alternatively, you can set the 'API_TEXT_MODEL' environment variable."
    )

    parser.add_argument(
        "--api-url",
        type=str,
        help="Filetype to convert to. Defaults to 'md'. Alternatively, you can set the 'API_URL' environment variable."
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for the Large Language Model (LLM) provider. Alternatively, you can set the 'API_KEY' environment variable."
    )

    parser.add_argument(
        "--output-type",
        type=str,
        default="md",
        help="Filetype to convert to. Defaults to 'md'."
    )

    args = parser.parse_args()
    create_db_and_tables()
    db_generator = get_db()
    db_session_for_cli = next(db_generator)
    repository_service = RepositoryService(db_session=db_session_for_cli)
    file_service = FileService()
    llm_service = LlmService(provider=args.provider,
                             image_model=args.image_model,
                             text_model=args.text_model,
                             api_url=args.api_url,
                             api_key=args.api_key)
    processor: Processor = ProcessorFactory.create_processor(filetype=args.output_type,
                                                             llm_service=llm_service)

    document_service = DocumentService(file_service=file_service, repository_service=repository_service)
    document_service.process_pdf(steps=[ProcessingStep.COMPLETE],
                                 processor=processor,
                                 pdf_path=args.pdf_path)

