import numpy as np
from rank_bm25 import BM25Okapi


def bm25_search(query, chunks, top_k=3):
    """
    Keyword-based search using BM25.
    Returns list of (score, chunk_index) sorted by score descending.
    """
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def vector_search(query_embedding, embeddings, top_k=3):
    """
    Semantic search using cosine similarity.
    Returns list of (score, chunk_index) sorted by score descending.
    """
    query = np.array(query_embedding)
    scores = []

    for i, emb in enumerate(embeddings):
        dot = np.dot(query, emb)
        norm = np.linalg.norm(query) * np.linalg.norm(emb)
        similarity = dot / norm if norm != 0 else 0.0
        scores.append((i, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in scores[:top_k]]


def hybrid_search(query, query_embedding, chunks, embeddings, top_k=3, alpha=0.5):
    """
    Combines BM25 and vector search scores.
    alpha=1.0 -> pure BM25, alpha=0.0 -> pure vector, alpha=0.5 -> equal mix.
    Returns top-k chunks sorted by combined score.
    """
    bm25_results = bm25_search(query, chunks, top_k=len(chunks))
    vector_results = vector_search(query_embedding, embeddings, top_k=len(chunks))

    bm25_scores = {idx: score for idx, score in bm25_results}
    vector_scores = {idx: score for idx, score in vector_results}

    bm25_max = max(bm25_scores.values()) or 1
    vector_max = max(vector_scores.values()) or 1

    combined = {}
    for idx in range(len(chunks)):
        bm25_norm = bm25_scores.get(idx, 0) / bm25_max
        vector_norm = vector_scores.get(idx, 0) / vector_max
        combined[idx] = alpha * bm25_norm + (1 - alpha) * vector_norm

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return [{"text": chunks[idx], "score": score} for idx, score in ranked[:top_k]]
