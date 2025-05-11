import logging
import logging.config
import argparse
from core.document_service import DocumentService
from core.markdown_service import MarkdownService
from core.file_service import FileService
from core.llm_service import LlmService
from core.repository_service import RepositoryService
import os

logging.config.fileConfig("src/logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PDF processing with an LLM provider."
    )

    # Argument for the PDF file path
    parser.add_argument(
        "--pdf-path", "-p",
        type=str,
        required=True,
        help="Path to the PDF document to be processed. "
    )

    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="API key for the Large Language Model (LLM) provider. "
             "This is a required argument. Alternatively, you can set the "
             "'API_KEY' environment variable."
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Directory where output files will be saved. Defaults to './output'."
    )

    args = parser.parse_args()

    path = args.pdf_path

    # Get API key from argument or environment variable
    api_key = args.api_key
    if not api_key:
        api_key = os.getenv("API_KEY")
        if not api_key:
            parser.error(
                "Error: LLM API key is required. Please provide it using "
                "--api-key or by setting the API_KEY environment variable."
            )

    # Ensure the output directory exists, create it if necessary
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Info: Created output directory: {output_dir}")
        except OSError as e:
            parser.error(f"Error creating output directory '{output_dir}': {e}")
    elif not os.path.isdir(output_dir):
        parser.error(f"Error: '{output_dir}' exists but is not a directory.")

    markdown_service = MarkdownService()
    file_service = FileService(output_dir=output_dir)
    llm_service = LlmService(api_key=api_key)
    repository_service = RepositoryService()
    document_service = DocumentService(markdown_service=markdown_service,
                                       file_service=file_service,
                                       llm_service=llm_service,
                                       repository_service=repository_service)
    document_service.process_pdf(pdf_path=path)

