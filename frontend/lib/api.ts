import type {
  ChatResponse,
  EmbeddingStatusResponse,
  UploadResponse,
} from "./types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail ?? "Upload failed");
  }

  return res.json() as Promise<UploadResponse>;
}

export async function getEmbeddingStatus(
  sessionId: string,
): Promise<EmbeddingStatusResponse> {
  const res = await fetch(`${API_URL}/upload/${sessionId}/embedding-status`);

  if (!res.ok) {
    const err = await res
      .json()
      .catch(() => ({ detail: "Failed to fetch status" }));
    throw new Error(err.detail ?? "Failed to fetch embedding status");
  }

  return res.json() as Promise<EmbeddingStatusResponse>;
}

export async function sendChat(
  sessionId: string,
  question: string,
  topK = 10,
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question, top_k: topK }),
  });

  if (!res.ok) {
    const err = await res
      .json()
      .catch(() => ({ detail: "Chat request failed" }));
    throw new Error(err.detail ?? "Chat request failed");
  }

  return res.json() as Promise<ChatResponse>;
}
