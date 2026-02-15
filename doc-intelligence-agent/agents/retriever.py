"""Retriever Agent — Tool usage: fetches relevant context from vector store."""

from rag import similarity_search

from state import AgentState


def retriever(state: AgentState) -> dict:
    """Retrieve relevant document chunks for the question."""
    question = state["question"]
    chunks = similarity_search(question, k=4)
    context = "\n\n---\n\n".join(chunks) if chunks else "(No documents loaded. Please upload a PDF first.)"
    return {"context": context}
