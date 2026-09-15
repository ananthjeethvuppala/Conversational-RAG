def create_prompt(context, query):
    prompt = f"""
You are a helpful AI assistant.

Answer the question using only the information provided in the context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt