from loader import load_text
from preprocessor import clean_text
from chunker import chunk_text
from embedder import create_embeddings
from metadata_store import build_metadata_chunks, filter_chunks
from hybrid_search import hybrid_search
from reranker import rerank
from prompt_builder import build_prompt
from llm import ask_llm

# --- Step 1: Load two documents with metadata ---
docs = [
    {"file": "data/college_notes.txt", "source": "college_notes.txt", "topic": "databases"},
    {"file": "data/python_notes.txt",  "source": "python_notes.txt",  "topic": "python"},
]

all_metadata_chunks = []

for doc in docs:
    raw_text = load_text(doc["file"])
    cleaned  = clean_text(raw_text)
    chunks   = chunk_text(cleaned, chunk_size=200)
    metadata = {"source": doc["source"], "topic": doc["topic"]}
    all_metadata_chunks.extend(build_metadata_chunks(chunks, metadata))

print(f"Total chunks: {len(all_metadata_chunks)}\n")

# --- Step 2: Query loop ---
queries = [
    {"question": "What is SQL used for?",     "topic": "databases"},
    {"question": "How are functions defined?", "topic": "python"},
]

for q in queries:
    # Filter by topic
    filtered = filter_chunks(all_metadata_chunks, topic=q["topic"])
    texts     = [c["text"] for c in filtered]
    embeddings = create_embeddings(texts)

    # Hybrid search → top-5 candidates
    query_embedding = create_embeddings([q["question"]])[0]
    candidates = hybrid_search(
        query=q["question"],
        query_embedding=query_embedding,
        chunks=texts,
        embeddings=embeddings,
        top_k=5,
        alpha=0.5
    )

    candidate_texts = [r["text"] for r in candidates]

    # Re-rank → pick best 2
    reranked = rerank(q["question"], candidate_texts, top_k=2)

    print(f"Q: {q['question']}")
    print("Re-ranked results:")
    for r in reranked:
        print(f"  [{r['score']}] {r['text'][:70]}")

    # Send to LLM
    prompt = build_prompt(q["question"], [r["text"] for r in reranked])
    answer = ask_llm(prompt)

    print(f"A: {answer}")
    print("-" * 50)
