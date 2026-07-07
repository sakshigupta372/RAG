def build_metadata_chunks(chunks, metadata):
    """
    Attaches metadata to each chunk.

    chunks   : list of text strings
    metadata : dict of extra fields (e.g. source, topic)

    Returns list of dicts: {"text": ..., "source": ..., ...}
    """
    result = []
    for i, chunk in enumerate(chunks):
        entry = {"chunk_id": i, "text": chunk}
        entry.update(metadata)
        result.append(entry)
    return result


def filter_chunks(metadata_chunks, **filters):
    """
    Filters chunks by any metadata field.

    Example:
        filter_chunks(chunks, source="college_notes.txt")
        filter_chunks(chunks, topic="databases")
    """
    filtered = metadata_chunks
    for key, value in filters.items():
        filtered = [c for c in filtered if c.get(key) == value]
    return filtered
