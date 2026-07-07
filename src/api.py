import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from loader import load_text
from preprocessor import clean_text
from chunker import chunk_text
from embedder import create_embeddings
from metadata_store import build_metadata_chunks, filter_chunks
from hybrid_search import hybrid_search
from reranker import rerank
from prompt_builder import build_prompt
from llm import ask_llm

# --- Load and index documents at startup ---
DATA_DIR = Path(__file__).parent.parent / "data"

DOCS = [
    {"file": str(DATA_DIR / "college_notes.txt"), "source": "college_notes.txt", "topic": "databases"},
    {"file": str(DATA_DIR / "python_notes.txt"),  "source": "python_notes.txt",  "topic": "python"},
]

all_metadata_chunks = []
all_embeddings = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global all_metadata_chunks, all_embeddings
    print("Loading and indexing documents...")
    for doc in DOCS:
        raw     = load_text(doc["file"])
        chunks  = chunk_text(clean_text(raw), chunk_size=200)
        meta    = {"source": doc["source"], "topic": doc["topic"]}
        mc      = build_metadata_chunks(chunks, meta)
        embs    = create_embeddings([c["text"] for c in mc])
        all_metadata_chunks.extend(mc)
        all_embeddings.extend(embs)
    print(f"Ready. {len(all_metadata_chunks)} chunks indexed.")
    yield

app = FastAPI(title="RAG API", lifespan=lifespan)


# --- Request / Response models ---
class AskRequest(BaseModel):
    question: str
    topic: str = None       # optional filter
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(all_metadata_chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Filter by topic if provided
    if request.topic:
        filtered = filter_chunks(all_metadata_chunks, topic=request.topic)
    else:
        filtered = all_metadata_chunks

    if not filtered:
        raise HTTPException(status_code=404, detail=f"No chunks found for topic '{request.topic}'.")

    texts      = [c["text"] for c in filtered]
    embeddings = create_embeddings(texts)

    # Hybrid search
    query_embedding = create_embeddings([request.question])[0]
    candidates = hybrid_search(
        query=request.question,
        query_embedding=query_embedding,
        chunks=texts,
        embeddings=embeddings,
        top_k=min(5, len(texts)),
        alpha=0.5
    )

    # Re-rank
    reranked = rerank(request.question, [r["text"] for r in candidates], top_k=request.top_k)

    retrieved_chunks = [r["text"] for r in reranked]

    # Build prompt and call LLM
    prompt = build_prompt(request.question, retrieved_chunks)
    try:
        answer = ask_llm(prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {str(e)}")

    return AskResponse(answer=answer, sources=retrieved_chunks)
