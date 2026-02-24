"""
Chat router — handles conversational queries against embedded review data.

Responsibilities:
- Accept a user message and a business_id from the client.
- Embed the user query via embedding_service.
- Retrieve the most relevant review chunks from Supabase via retrieval_service.
- Build a prompt from the retrieved context and send it to the LLM via llm_service.
- Stream or return the LLM response to the client.
"""

# TODO: Implement POST /chat endpoint
# TODO: Embed the incoming query
# TODO: Perform similarity search scoped to business_id
# TODO: Construct RAG prompt with retrieved review context
# TODO: Call llm_service.complete() and return the response
