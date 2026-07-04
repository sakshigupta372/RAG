from embedder import create_embeddings
from vector_store import build_vector_store
from retriever import retrieve

chunks = [
    "Python is a programming language.",
    "Cats are animals.",
    "SQL stores data."
]

embeddings = create_embeddings(chunks)

store = build_vector_store(chunks, embeddings)

query = "What language is used for programming?"
query_embedding = create_embeddings([query])[0]

results = retrieve(query_embedding, store, top_k=3)

for score, doc in results:
    print(f"{score:.4f} -> {doc['text']}")
