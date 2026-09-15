def create_prompt(context, query, conversation_history):

    history = ""

    for conversation in conversation_history:

        history += f"""
User: {conversation["user"]}
Assistant: {conversation["assistant"]}
"""

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the information provided
in the retrieved document context and the conversation history.

Use the conversation history to understand references and
follow-up questions.

Do not use information that is not supported by the
retrieved document context.

If the answer cannot be found in the retrieved context, say:
"I could not find the answer in the provided documents."

Conversation History:
{history}

Retrieved Context:
{context}

Current Question:
{query}

Answer:
"""
    return prompt