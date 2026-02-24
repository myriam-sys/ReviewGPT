"""
LLM service — wraps the Groq API for chat completion.

Responsibilities:
- Build a RAG prompt by combining a system instruction, the retrieved review
  context (passages), and the user's question.
- Call the Groq API using the configured model (e.g. llama3-8b-8192).
- Return the model's text response.
- Optionally support streaming responses for a better UX.
"""

# TODO: Initialize Groq client with GROQ_API_KEY from config
# TODO: Implement build_prompt(context_chunks, user_query) -> list[dict] (messages)
# TODO: Implement complete(messages) -> str
# TODO: Implement stream_complete(messages) -> AsyncGenerator[str, None]
