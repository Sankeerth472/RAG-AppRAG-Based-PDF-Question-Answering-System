import logging

logger = logging.getLogger(__name__)


def chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200
) -> list[str]:

    if not text or not text.strip():
        logger.warning("No valid text provided for chunking")
        return []

    if chunk_size <= 0:
        logger.error("Invalid chunk_size: %s", chunk_size)
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        logger.error("Invalid overlap: %s", overlap)
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        logger.error(
            "Overlap (%s) cannot be greater than or equal to chunk_size (%s)",
            overlap,
            chunk_size
        )

        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    try:
        logger.info(
            "Starting text chunking with chunk_size=%s and overlap=%s",
            chunk_size,
            overlap
        )

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end - overlap

        logger.info(
            "Chunking completed successfully. Total chunks created: %s",
            len(chunks)
        )

        return chunks

    except Exception as e:
        logger.exception("Unexpected error during text chunking")

        raise RuntimeError(
            "Text chunking failed"
        ) from e