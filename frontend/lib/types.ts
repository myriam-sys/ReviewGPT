export interface RowError {
  row: number;
  reason: string;
}

export interface UploadResponse {
  session_id: string;
  total_rows: number;
  valid_rows: number;
  invalid_rows: number;
  reviews_with_text: number;
  inserted_rows: number;
  skipped_rows: number;
  embedding_status: string; // "embedding_queued" | "nothing_to_embed" | "embedding_skipped"
  errors: RowError[];
}

export interface EmbeddingStatusResponse {
  session_id: string;
  total_with_text: number;
  embedded: number;
  pending: number;
  status: string; // "complete" | "processing" | "empty"
}

export interface ReviewSource {
  review_id: string;
  author: string | null;
  rating: number;
  date: string | null;
  text: string;
  language: string | null;
  similarity: number | null;
}

export interface ChatResponse {
  answer: string;
  model: string;
  tokens_used: number;
  sources_count: number;
  retrieved_reviews: ReviewSource[];
  session_id: string;
  question: string;
  question_type: string; // "analytical" | "temporal" | "comparative" | "author" | "search"
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  chatResponse?: ChatResponse;
  isLoading?: boolean;
  error?: string;
}
