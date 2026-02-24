"""
Embedding service — generates dense vector embeddings for review text.

Responsibilities:
- Load the multilingual-e5-large model from HuggingFace via sentence-transformers.
- Expose an embed() function that accepts a string or list of strings and returns
  numpy arrays of shape (n, EMBEDDING_DIMENSION).
- Handle the instruction prefix required by E5 models:
    query:   "query: <text>"
    passage: "passage: <text>"
- Cache the model in memory across requests to avoid repeated loading.
"""

# TODO: Initialize SentenceTransformer model from config.EMBEDDING_MODEL
# TODO: Implement embed_query(text: str) -> list[float]
# TODO: Implement embed_passages(texts: list[str]) -> list[list[float]]
