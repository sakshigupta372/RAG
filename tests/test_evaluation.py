import sys
sys.path.insert(0, "src")

from loader import load_text
from preprocessor import clean_text
from chunker import chunk_text
from embedder import create_embeddings
from hybrid_search import hybrid_search
from reranker import rerank
from evaluator import evaluate

# --- Setup ---
raw = load_text("data/college_notes.txt")
chunks = chunk_text(clean_text(raw), chunk_size=200)
embeddings = create_embeddings(chunks)

def retrieval_fn(query):
    qe = create_embeddings([query])[0]
    candidates = hybrid_search(query, qe, chunks, embeddings, top_k=5, alpha=0.5)
    reranked = rerank(query, [r["text"] for r in candidates], top_k=3)
    return [r["text"] for r in reranked]

# --- Test cases ---
test_cases = [
    {"query": "What is SQL?",          "expected_keywords": ["SQL", "language"]},
    {"query": "What is DBMS?",         "expected_keywords": ["DBMS", "software"]},
    {"query": "What are ACID properties?", "expected_keywords": ["ACID", "Atomicity"]},
    {"query": "How are indexes used?", "expected_keywords": ["Indexes", "query"]},
]

print("=" * 40)
print("RAG Retrieval Evaluation")
print("=" * 40 + "\n")

evaluate(test_cases, chunks, retrieval_fn)
