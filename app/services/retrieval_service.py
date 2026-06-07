from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

def get_client():
    return QdrantClient(path="qdrant_storage_v2")

model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "documents"


def retrieve_relevant_chunks(query: str, limit: int = 3):
    client = get_client()
    query_vector = model.encode(query).tolist()

    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    ).points

    results = []

    for result in search_results:
        results.append({
            "score": result.score,
            "text": result.payload["text"]
        })

    return results