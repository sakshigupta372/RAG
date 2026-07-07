import numpy as np
import faiss


def build_faiss_index(embeddings):
    """
    Builds a FAISS flat L2 index from a numpy array of embeddings.
    Returns the index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index


def search_faiss(index, query_embedding, chunks, top_k=3):
    """
    Searches the FAISS index and returns the top-K matching chunks.
    """
    query = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx],
            "distance": float(dist)
        })

    return results
