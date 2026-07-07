def is_relevant(chunk_text, expected_keywords):
    """
    A chunk is relevant if it contains ALL expected keywords (case-insensitive).
    """
    text_lower = chunk_text.lower()
    return all(kw.lower() in text_lower for kw in expected_keywords)


def hit_rate(retrieved_chunks, expected_keywords):
    """
    1 if at least one retrieved chunk is relevant, else 0.
    """
    for chunk in retrieved_chunks:
        if is_relevant(chunk, expected_keywords):
            return 1
    return 0


def precision_at_k(retrieved_chunks, expected_keywords):
    """
    Fraction of retrieved chunks that are relevant.
    """
    if not retrieved_chunks:
        return 0.0
    relevant = sum(1 for c in retrieved_chunks if is_relevant(c, expected_keywords))
    return relevant / len(retrieved_chunks)


def recall_at_k(retrieved_chunks, all_chunks, expected_keywords):
    """
    Fraction of all relevant chunks that were retrieved.
    """
    total_relevant = sum(1 for c in all_chunks if is_relevant(c, expected_keywords))
    if total_relevant == 0:
        return 0.0
    found_relevant = sum(1 for c in retrieved_chunks if is_relevant(c, expected_keywords))
    return found_relevant / total_relevant


def evaluate(test_cases, all_chunks, retrieval_fn):
    """
    Runs all test cases and prints Precision, Recall, Hit Rate for each.

    retrieval_fn: function(query) -> list of chunk texts
    """
    results = []

    for case in test_cases:
        query    = case["query"]
        keywords = case["expected_keywords"]

        retrieved = retrieval_fn(query)

        hr  = hit_rate(retrieved, keywords)
        p   = precision_at_k(retrieved, keywords)
        r   = recall_at_k(retrieved, all_chunks, keywords)

        results.append({"query": query, "hit_rate": hr, "precision": p, "recall": r})

        print(f"Query    : {query}")
        print(f"Hit Rate : {hr}")
        print(f"Precision: {p:.2f}")
        print(f"Recall   : {r:.2f}")
        print("-" * 40)

    avg_hr  = sum(r["hit_rate"]  for r in results) / len(results)
    avg_p   = sum(r["precision"] for r in results) / len(results)
    avg_r   = sum(r["recall"]    for r in results) / len(results)

    print(f"\nAverage Hit Rate : {avg_hr:.2f}")
    print(f"Average Precision: {avg_p:.2f}")
    print(f"Average Recall   : {avg_r:.2f}")

    return results
