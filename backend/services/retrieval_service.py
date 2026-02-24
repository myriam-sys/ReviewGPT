"""
Retrieval service — manages vector storage and similarity search in Supabase pgvector.

Responsibilities:
- Upsert review embeddings alongside metadata (business_id, date, author, rating,
  language, original text) into the Supabase vector table.
- Perform nearest-neighbor search filtered by business_id to enforce tenant isolation.
- Return the top-k most similar review chunks for a given query embedding.
"""

# TODO: Initialize Supabase vecs client from supabase_client.py
# TODO: Implement upsert(vectors, metadata, business_id)
# TODO: Implement search(query_vector, business_id, top_k) -> list[dict]
# TODO: Implement delete_by_business(business_id) for data cleanup
