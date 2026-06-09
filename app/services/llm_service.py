import requests
import logging
import os

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")


def generate_answer(question: str, context_chunks: list) -> str:

    if not question.strip():
        logger.warning("Empty question received")
        return "Question cannot be empty."

    if not context_chunks:
        logger.warning("No context chunks retrieved for question: %s", question)
        return "No relevant context found."

    try:

        context_text = "\n\n".join(
            [chunk["text"] for chunk in context_chunks]
        )

        logger.info(
            "Generating answer using %s context chunks",
            len(context_chunks)
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
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get("response")

        if not answer:
            logger.error("LLM response missing 'response' field")
            return "Failed to generate answer."

        logger.info("Answer generated successfully")

        return answer

    except requests.exceptions.Timeout:
        logger.exception("LLM request timed out")
        return "LLM request timed out."

    except requests.exceptions.ConnectionError:
        logger.exception("Could not connect to Ollama server")
        return "Could not connect to LLM service."

    except requests.exceptions.HTTPError as e:
        logger.exception("HTTP error from Ollama: %s", str(e))
        return "LLM service returned an HTTP error."

    except Exception as e:
        logger.exception("Unexpected error during answer generation")
        return "Unexpected error occurred while generating answer."
