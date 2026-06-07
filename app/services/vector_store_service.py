from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

COLLECTION_NAME = "documents"


def get_client():
    return QdrantClient(path="qdrant_storage_v2")


def create_collection_if_not_exists(client, vector_size: int):

    collections = client.get_collections().collections

    collection_names = [
        collection.name for collection in collections
    ]

    if COLLECTION_NAME not in collection_names:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


def store_embeddings(embedded_chunks: list[dict]):

    if not embedded_chunks:
        return 0

    client = get_client()

    vector_size = len(embedded_chunks[0]["embedding"])

    create_collection_if_not_exists(client, vector_size)

    points = []

    for item in embedded_chunks:

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=item["embedding"],
                payload={
                    "text": item["text"]
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(points)