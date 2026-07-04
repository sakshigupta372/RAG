def build_prompt(question, retrieved_chunks):
    """
    Builds a RAG prompt by injecting retrieved context and the user question
    into a structured template.
    """

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a helpful assistant.
Answer the question using only the context provided.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

    return prompt
