"""
RAG Pipeline — Retrieval-Augmented Generation Core
Loads documents, chunks text, creates embeddings, stores in FAISS.
"""

import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Global vectorstore (in-memory, replaced when new docs are loaded)
_vectorstore: FAISS | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings model."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create text splitter for chunking documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def load_and_chunk_pdf(file_path: str) -> list:
    """Load PDF and split into chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    return chunks


def load_and_chunk_from_bytes(content: bytes, filename: str = "document.pdf") -> list:
    """Load PDF from bytes (e.g., uploaded file) and split into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        return load_and_chunk_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def build_vectorstore(chunks: list) -> FAISS:
    """Build FAISS vectorstore from document chunks."""
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def load_documents(file_path: str | None = None, content: bytes | None = None) -> int:
    """
    Load documents and build/update vectorstore.
    Returns number of chunks indexed.
    """
    global _vectorstore

    if file_path:
        chunks = load_and_chunk_pdf(file_path)
    elif content:
        chunks = load_and_chunk_from_bytes(content)
    else:
        raise ValueError("Either file_path or content must be provided")

    _vectorstore = build_vectorstore(chunks)
    return len(chunks)


def get_vectorstore() -> FAISS | None:
    """Get the current vectorstore. Returns None if no documents loaded."""
    return _vectorstore


def similarity_search(question: str, k: int = 4) -> list[str]:
    """
    Retrieve relevant document chunks for a question.
    Returns list of page content strings.
    """
    vs = get_vectorstore()
    if vs is None:
        return []
    docs = vs.similarity_search(question, k=k)
    return [doc.page_content for doc in docs]
