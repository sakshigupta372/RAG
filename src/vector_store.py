def build_vector_store(chunks, embeddings):
    store = []

    for i in range(len(chunks)):
        store.append({
            "id": i,
            "text": chunks[i],
            "embedding": embeddings[i]
        })

    return store