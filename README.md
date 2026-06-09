# RAG-AppRAG-Based-PDF-Question-Answering-System

## Docker

1. Create an env file:

```bash
cp .env.example .env
```

2. Make sure Ollama is running on the host and the selected model is available:

```bash
ollama serve
ollama pull llama3
```

3. Start the API and Qdrant:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000` and will connect to:

- Qdrant at `qdrant:6333` inside Docker
- Ollama on the host through `host.docker.internal:11434`
