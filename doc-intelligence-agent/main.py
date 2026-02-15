"""
FastAPI Backend — Autonomous Document Intelligence Agent
"""

import config  # noqa: F401 — load .env

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from graph import graph
from rag import load_documents, get_vectorstore

app = FastAPI(
    title="Autonomous Document Intelligence Agent",
    description="Multi-agent RAG system using LangGraph — Planner, Retriever, Analyst, Writer, Critic",
    version="1.0.0",
)


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""

    question: str


class AskResponse(BaseModel):
    """Response from /ask endpoint."""

    answer: str
    plan: str | None = None
    iterations: int = 0


@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "status": "ok",
        "message": "Autonomous Document Intelligence Agent",
        "endpoints": {
            "POST /upload": "Upload a PDF document",
            "POST /ask": "Ask a question about uploaded documents",
        },
    }


@app.post("/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and build the RAG index."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        num_chunks = load_documents(content=content)
        return {
            "message": "Document processed successfully",
            "chunks_indexed": num_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Ask a question. Uses RAG + multi-agent pipeline with self-reflection."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    vs = get_vectorstore()
    if vs is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload a PDF first via POST /upload",
        )

    try:
        result = graph.invoke({"question": question})
        answer = result.get("draft_answer", result.get("final_answer", "No answer generated."))
        plan = result.get("plan")
        iterations = result.get("iteration", 0)

        return AskResponse(
            answer=answer,
            plan=plan,
            iterations=iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
