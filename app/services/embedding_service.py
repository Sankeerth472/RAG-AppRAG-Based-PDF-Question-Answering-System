import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

try:
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model loaded successfully: %s", MODEL_NAME)
except Exception as e:
    logger.exception("Failed to load embedding model: %s", MODEL_NAME)
    raise RuntimeError("Embedding model initialization failed") from e


def generate_embeddings(chunks: list[str]) -> list[dict]:
    if not chunks:
        logger.warning("No chunks provided for embedding generation")
        return []

    embeddings = []

    for index, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            logger.warning("Skipping empty chunk at index %s", index)
            continue

        try:
            vector = model.encode(chunk).tolist()

            embeddings.append({
                "text": chunk,
                "embedding": vector
            })

            logger.info(
                "Generated embedding for chunk %s with dimension %s",
                index,
                len(vector)
            )

        except Exception as e:
            logger.exception(
                "Failed to generate embedding for chunk index %s",
                index
            )
            continue

    logger.info(
        "Embedding generation completed. Input chunks: %s, successful embeddings: %s",
        len(chunks),
        len(embeddings)
    )

    return embeddings