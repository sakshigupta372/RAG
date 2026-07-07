import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def rerank(query, chunks, top_k=3):
    """
    Re-ranks chunks by scoring each (query, chunk) pair using
    cosine similarity between their concatenated embeddings.
    Returns top_k most relevant chunks.
    """
    query_embedding = model.encode(query)
    chunk_embeddings = model.encode(chunks)

    scores = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        dot = np.dot(query_embedding, chunk_emb)
        norm = np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
        score = dot / norm if norm != 0 else 0.0
        scores.append((score, chunks[i]))

    scores.sort(reverse=True, key=lambda x: x[0])

    return [{"text": text, "score": round(score, 4)} for score, text in scores[:top_k]]
