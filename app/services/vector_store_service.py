import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
QDRANT_PATH = "qdrant_storage_v2"


def get_client() -> QdrantClient:
    try:
        client = QdrantClient(path=QDRANT_PATH)
        logger.info("Qdrant client initialized successfully")
        return client

    except Exception as e:
        logger.exception("Failed to initialize Qdrant client")
        raise RuntimeError("Vector database connection failed") from e


def create_collection_if_not_exists(client: QdrantClient, vector_size: int) -> None:
    try:
        collections = client.get_collections().collections

        collection_names = [
            collection.name for collection in collections
        ]

        if COLLECTION_NAME not in collection_names:
            logger.info(
                "Creating Qdrant collection '%s' with vector size %s",
                COLLECTION_NAME,
                vector_size
            )

            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

            logger.info("Qdrant collection created successfully")

        else:
            logger.info("Qdrant collection '%s' already exists", COLLECTION_NAME)

    except Exception as e:
        logger.exception("Failed to create/check Qdrant collection")
        raise RuntimeError("Vector collection setup failed") from e


def store_embeddings(embedded_chunks: list[dict]) -> int:
    if not embedded_chunks:
        logger.warning("No embedded chunks provided for storage")
        return 0

    try:
        client = get_client()

        first_embedding = embedded_chunks[0].get("embedding")

        if not first_embedding:
            logger.error("First embedded chunk is missing embedding vector")
            raise ValueError("Embedding vector missing")

        vector_size = len(first_embedding)

        create_collection_if_not_exists(client, vector_size)

        points = []

        for index, item in enumerate(embedded_chunks):
            text = item.get("text")
            embedding = item.get("embedding")

            if not text or not embedding:
                logger.warning(
                    "Skipping invalid embedded chunk at index %s",
                    index
                )
                continue

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": text
                    }
                )
            )

        if not points:
            logger.warning("No valid points created for Qdrant upsert")
            return 0

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        logger.info(
            "Stored %s embeddings in Qdrant collection '%s'",
            len(points),
            COLLECTION_NAME
        )

        return len(points)

    except Exception as e:
        logger.exception("Failed to store embeddings in Qdrant")
        raise RuntimeError("Embedding storage failed") from e