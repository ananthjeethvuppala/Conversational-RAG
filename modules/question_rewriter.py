from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_question(query, conversation_history):

    if not conversation_history:
        return query

    history = ""

    for converstion in conversation_history:
        history += f"""
User: {converstion["user"]}
Assistant: {converstion["assistant"]}
"""

    prompt = f"""
You are a question rewriting assistant.

Your task is to rewrite the user's latest question into a
standalone question that can be understood without the
previous conversation.

Use the conversation history to resolve references such as:
"it", "this", "that", "they", "he", "she", etc.

Do not answer the question.

Return only the rewritten question.

Conversation history:
{history}

Latest user question:
{query}

Standalone question:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content.strip()