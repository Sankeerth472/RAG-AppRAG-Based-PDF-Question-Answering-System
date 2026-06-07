from fastapi import UploadFile,File,HTTPException,APIRouter,Body
from app.services.llm_service import generate_answer
from app.services.pdf_service import extract_text_from_pdf
from app.services.chunk_service import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.vector_store_service import store_embeddings

router = APIRouter()

@router.post("/ask")
def ask_document(question: str = Body(...)):
    relevant_chunks=retrieve_relevant_chunks(question)
    answer=generate_answer(question,relevant_chunks)
    return {
        "question": question,
        "answer": answer,
        "matches": relevant_chunks
    }

# UploadFile : object file wrapper
@router.post("/upload")
def upload_file(file : UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Unsupported file type")
    extracted_text = extract_text_from_pdf(file)

    if not extracted_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in PDF"
        )
    chunks = chunk_text(extracted_text)
    embeddings = generate_embeddings(chunks)
    stored_count = store_embeddings(embeddings)
    return {
        "filename": file.filename,
        "chunk_count": len(chunks),
        "embedding_count": len(embeddings),
        "stored_count": stored_count,
        "sample_embedding_dimension": len(embeddings[0]["embedding"])
    }
