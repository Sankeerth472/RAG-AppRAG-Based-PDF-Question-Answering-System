# from openai import OpenAI
# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
# def generate_embeddings(chunks:list[str]):
#     embeddings = []
#     for chunk in chunks :
#         response = client.embeddings.create(
#             model="text-embedding-3-small",
#             input=chunk
#         )
#         vector = response.data[0].embedding
#         embeddings.append({
#             "text": chunk,
#             "embedding": vector
#         })
#
#
#     return embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks: list[str]):

    embeddings = []

    for chunk in chunks:

        vector = model.encode(chunk).tolist()

        embeddings.append({
            "text": chunk,
            "embedding": vector
        })

    return embeddings
