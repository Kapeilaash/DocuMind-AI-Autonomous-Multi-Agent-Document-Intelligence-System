# Autonomous Document Intelligence Agent

Multi-agent RAG system using LangGraph — from zero knowledge to production-ready.

## Architecture

```
Frontend (React) → FastAPI → LangGraph Engine → Multi-Agent System → RAG (FAISS) → LLM
```

**Agent Flow:** Planner → Retriever → Analyst → Writer → Critic → (Writer loop | End)

## Tech Stack

- **Backend:** FastAPI, LangGraph, LangChain, FAISS, OpenAI
- **Deployment:** Docker, AWS EC2 / Render / Railway

## Quick Start

### 1. Setup

```bash
cd doc-intelligence-agent
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run

Set `OPENAI_API_KEY` in `.env`, then:

```bash
# Option A: Direct
uvicorn main:app --reload

# Option B: With env check
python run.py
```

### 4. Use

**Upload a PDF:**
```bash
curl -X POST http://localhost:8000/upload -F "file=@your-document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is the main finding of this document?\"}"
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Upload PDF (builds RAG index) |
| POST | `/ask` | Ask a question (runs full pipeline) |

## Docker

```bash
docker build -t doc-intelligence-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key doc-intelligence-agent
```

## Project Structure

```
doc-intelligence-agent/
├── main.py           # FastAPI app
├── graph.py          # LangGraph orchestration
├── state.py          # Agent state definition
├── rag.py            # RAG pipeline (load, chunk, embed, FAISS)
├── config.py         # Env loading
├── agents/
│   ├── planner.py    # Step-by-step reasoning plan
│   ├── retriever.py  # Vector search (tool usage)
│   ├── analyst.py    # Deep analysis
│   ├── writer.py     # Structured answer
│   ├── critic.py     # Reflection / quality check
│   └── routing.py    # Conditional: loop or end
├── requirements.txt
├── Dockerfile
└── README.md
```

## Interview Prep

- **What is RAG?** Retrieval-Augmented Generation — retrieve relevant chunks, then generate with LLM to reduce hallucination.
- **Why LangGraph?** Graph-based flows with conditional routing, loops, and state — enables self-correcting agents.
- **How does routing work?** Critic returns IMPROVE or FINAL; router sends to Writer or END.
- **Chunk size impact?** Larger = more context but less granular retrieval; smaller = more precise but can miss coherence.

## CV Line

> **Autonomous Document Intelligence Agent** — Built multi-agent LLM system using LangGraph. Implemented planner, retriever, analyst, writer, critic agents with conditional routing and self-reflection. Integrated RAG pipeline with FAISS. Deployed with FastAPI and Docker.
