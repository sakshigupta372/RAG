from embedder import create_embeddings
from vector_store import build_vector_store
from retriever import retrieve
from prompt_builder import build_prompt
from llm import ask_llm

chunks = [
    "Python is a programming language.",
    "Cats are animals.",
    "SQL stores data."
]

embeddings = create_embeddings(chunks)
store = build_vector_store(chunks, embeddings)

query = "What language is used for programming?"
query_embedding = create_embeddings([query])[0]

results = retrieve(query_embedding, store, top_k=2)
retrieved_chunks = [doc["text"] for score, doc in results]

prompt = build_prompt(query, retrieved_chunks)

answer = ask_llm(prompt)

print("Question:", query)
print("Answer:", answer)
