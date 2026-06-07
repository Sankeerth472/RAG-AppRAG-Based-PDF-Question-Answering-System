from fastapi import FastAPI
from app.routes.document import router as document_router

app = FastAPI(title="RAG App")
app.include_router(document_router, prefix="/documents", tags=["documents"])

@app.get("/")
def health_check():
    return {"status": "ok"}