import numpy as np


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


def retrieve(query_embedding, store, top_k=3):
    scores = []

    for document in store:
        similarity = cosine_similarity(query_embedding, document["embedding"])
        scores.append((similarity, document))

    scores.sort(reverse=True, key=lambda x: x[0])

    return scores[:top_k]
