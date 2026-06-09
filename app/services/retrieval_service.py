import logging
import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_client() -> QdrantClient:
    try:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )

        logger.info("Qdrant client initialized successfully")

        return client

    except Exception as e:
        logger.exception("Failed to initialize Qdrant client")

        raise RuntimeError(
            "Vector database connection failed"
        ) from e


try:
    model = SentenceTransformer(MODEL_NAME)

    logger.info(
        "Retrieval embedding model loaded successfully: %s",
        MODEL_NAME
    )

except Exception as e:
    logger.exception(
        "Failed to load retrieval embedding model"
    )

    raise RuntimeError(
        "Retrieval model initialization failed"
    ) from e


def retrieve_relevant_chunks(
        query: str,
        limit: int = 3
) -> list[dict]:

    if not query.strip():
        logger.warning("Empty retrieval query received")

        return []

    try:
        client = get_client()

        logger.info(
            "Generating embedding for retrieval query"
        )

        query_vector = model.encode(query).tolist()

        logger.info(
            "Searching Qdrant collection '%s' with limit=%s",
            COLLECTION_NAME,
            limit
        )

        search_results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points

        results = []

        for index, result in enumerate(search_results):

            payload = result.payload or {}

            text = payload.get("text")

            if not text:
                logger.warning(
                    "Missing text payload in search result index %s",
                    index
                )
                continue

            results.append({
                "score": result.score,
                "text": text
            })

        logger.info(
            "Retrieved %s relevant chunks for query",
            len(results)
        )

        return results

    except Exception as e:
        logger.exception(
            "Failed during semantic retrieval"
        )

        raise RuntimeError(
            "Chunk retrieval failed"
        ) from e
