# pdf-to-md
**From PDF chaos to RAG clarity. Built to handle PDFs that clearly skipped formatting school**

Using Tesseract and multimodal LLMs to transform pdfs into markdown
- Required packages:
    - [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
    - [pytesseract](https://github.com/h/pytesseract)
    - [opencv-python](https://github.com/opencv/opencv-python)
    - [requests](https://pypi.org/project/requests/)
- Dependencies:
    - [Tesseract](https://github.com/tesseract-ocr/tesseract)

# TODOs
## MUST DO
- Add a splitter to images with multiple columns, like scientific literature
- Add support for Ollama
- Make use of env variables

## NICE TO HAVEs
- FastAPI + Frontend
    - Docker image
- File cleanup
- Database integration
