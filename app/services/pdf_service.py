import io
import logging

from fastapi import UploadFile
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file: UploadFile) -> str:

    if not file:
        logger.warning("No file provided for PDF extraction")
        return ""

    try:
        logger.info(
            "Starting PDF text extraction for file: %s",
            file.filename
        )

        pdf_bytes = file.file.read()

        if not pdf_bytes:
            logger.warning(
                "Uploaded PDF file is empty: %s",
                file.filename
            )
            return ""

        reader = PdfReader(io.BytesIO(pdf_bytes))

        text_parts = []

        logger.info(
            "PDF loaded successfully with %s pages",
            len(reader.pages)
        )

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                page_text = page.extract_text()

                if page_text and page_text.strip():

                    text_parts.append(
                        f"\n--- Page {page_number} ---\n{page_text}"
                    )

                    logger.info(
                        "Extracted text from page %s",
                        page_number
                    )

                else:
                    logger.warning(
                        "No readable text found on page %s",
                        page_number
                    )

            except Exception as e:
                logger.exception(
                    "Failed to extract text from page %s",
                    page_number
                )
                continue

        final_text = "\n".join(text_parts)

        logger.info(
            "PDF extraction completed successfully. Total extracted characters: %s",
            len(final_text)
        )

        return final_text

    except Exception as e:
        logger.exception(
            "Unexpected error during PDF extraction for file: %s",
            file.filename
        )

        raise RuntimeError(
            "PDF text extraction failed"
        ) from e