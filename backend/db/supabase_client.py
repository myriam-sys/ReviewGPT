"""
Supabase client — initializes and exposes singleton clients for Supabase services.

Responsibilities:
- Create a Supabase REST client (for auth and table operations) using the
  project URL and service_role key from config.
- Create a vecs client pointed at the direct PostgreSQL connection string
  for pgvector operations.
- Export get_supabase_client() and get_vecs_client() for use across services.
"""

# TODO: Initialize supabase.create_client() with SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY
# TODO: Initialize vecs.create_client() with SUPABASE_DB_URL
# TODO: Wrap in lazy singletons to avoid re-connecting on every request
