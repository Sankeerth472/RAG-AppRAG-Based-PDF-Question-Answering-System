from fastapi import UploadFile
from pypdf import PdfReader
import io


def extract_text_from_pdf(file: UploadFile) -> str:
    pdf_bytes = file.file.read()

    reader = PdfReader(io.BytesIO(pdf_bytes))

    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text_parts.append(f"\n--- Page {page_number} ---\n{page_text}")

    return "\n".join(text_parts)