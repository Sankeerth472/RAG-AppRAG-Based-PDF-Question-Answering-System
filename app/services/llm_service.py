import requests


def generate_answer(question: str, context_chunks: list):

    context_text = "\n\n".join(
        [chunk["text"] for chunk in context_chunks]
    )

    prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the provided context.

Context:
{context_text}

Question:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return result["response"]