# pdf-to-md
**From PDF chaos to RAG clarity. Built to handle PDFs that clearly skipped formatting school**

Using Tesseract and multimodal LLMs to transform pdfs into markdown
- Required packages:
    - [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
    - [pytesseract](https://github.com/h/pytesseract)
    - [opencv-python](https://github.com/opencv/opencv-python)
    - [uvicorn](https://github.com/encode/uvicorn)
    - [Pydantic](https://github.com/pydantic/pydantic)
    - [sqlalchemy](https://github.com/sqlalchemy/sqlalchemy)
    - [python-multipart](https://github.com/Kludex/python-multipart)
    - [requests](https://pypi.org/project/requests/)
- Dependencies:
    - [Tesseract](https://github.com/tesseract-ocr/tesseract)
    - [OpenCV](https://github.com/opencv/opencv)

# Example - How to run
```bash
git clone https://github.com/DanielSchaack/pdf-to-md.git
cd pdf-to-md
podman image build -t pipeline-service .
podman run -d -p 127.0.0.1:42069:42069 -e API_PROVIDER="openrouter" -e API_KEY=key -v pipeline:/app/data --name pipeline --restart always localhost/pipeline-service:latest
```

Personal use:
```bash
podman run -d -p 127.0.0.1:42069:42069 --network=llm -e API_PROVIDER=openrouter -e API_KEY=key -v pipeline:/app/data --name pipeline --restart always localhost/pipeline-service:latest
# Or with more VRAM
podman run -d -p 127.0.0.1:42069:42069 --network=llm -e API_PROVIDER=ollama -e API_URL="http://ollama:11434" -e API_IMAGE_MODEL="gemma3:4b-it-qat" -e API_TEXT_MODEL="gemma3:4b-it-qat" -v pipeline:/app/data --name pipeline --restart always localhost/pipeline-service:latest
chromium http://127.0.0.1:42069/docs

# Redo image
podman stop pipeline && podman rm pipeline && podman image build -t pipeline-service .

```



# TODOs
## MUST DO
- API response cleanup - use Pydantic for all
- Add default cutoff header level based on maximum level in MD
- Options for processing inside of ProcessingFile
- Add a splitter to images with multiple columns, like scientific literature

## NICE TO HAVEs
- Frontend
    - Configurable options on UI
- File cleanup when deleting
- Options for model calls
    - temperature
    - top k
    - top p
    - num_predict
    - num_gpu (ollama)
- Support for reasoning?

