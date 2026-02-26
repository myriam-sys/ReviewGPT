"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { getEmbeddingStatus, uploadFile } from "@/lib/api";
import type { EmbeddingStatusResponse, UploadResponse } from "@/lib/types";

// ── Shared primitives ─────────────────────────────────────────────────────────

function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg
      className={`animate-spin ${className}`}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

function StatRow({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-zinc-400">{label}</span>
      <span className="text-sm font-medium text-zinc-100">{value}</span>
    </div>
  );
}

// ── Embedding status badge ────────────────────────────────────────────────────

function EmbeddingBadge({
  uploadStatus,
  embStatus,
}: {
  uploadStatus: string;
  embStatus: EmbeddingStatusResponse | null;
}) {
  if (uploadStatus === "nothing_to_embed") {
    return (
      <span className="text-xs text-zinc-400 bg-zinc-800 px-2.5 py-0.5 rounded-full">
        no text to embed
      </span>
    );
  }
  if (uploadStatus === "embedding_skipped") {
    return (
      <span className="text-xs text-amber-400 bg-amber-950/40 border border-amber-800/40 px-2.5 py-0.5 rounded-full">
        skipped (DB write failed)
      </span>
    );
  }
  if (!embStatus) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-indigo-300 bg-indigo-950/40 border border-indigo-800/40 px-2.5 py-0.5 rounded-full">
        <Spinner className="h-3 w-3" />
        queued
      </span>
    );
  }
  if (embStatus.status === "complete") {
    return (
      <span className="text-xs text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2.5 py-0.5 rounded-full">
        ✓ complete
      </span>
    );
  }
  if (embStatus.status === "empty") {
    return (
      <span className="text-xs text-zinc-400 bg-zinc-800 px-2.5 py-0.5 rounded-full">
        empty
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-indigo-300 bg-indigo-950/40 border border-indigo-800/40 px-2.5 py-0.5 rounded-full">
      <Spinner className="h-3 w-3" />
      processing
    </span>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  const router = useRouter();

  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [embStatus, setEmbStatus] = useState<EmbeddingStatusResponse | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const [sessionInput, setSessionInput] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Cleanup polling on unmount
  useEffect(
    () => () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    },
    [],
  );

  const selectFile = useCallback((f: File) => {
    setFile(f);
    setError(null);
    setUploadResult(null);
    setEmbStatus(null);
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };
  const handleDragLeave = () => setDragging(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) selectFile(f);
  };

  const startPolling = (sessionId: string) => {
    pollingRef.current = setInterval(async () => {
      try {
        const status = await getEmbeddingStatus(sessionId);
        setEmbStatus(status);
        if (status.status === "complete" || status.status === "empty") {
          if (pollingRef.current) clearInterval(pollingRef.current);
        }
      } catch {
        // transient errors during polling are non-fatal
      }
    }, 5000);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const result = await uploadFile(file);
      setUploadResult(result);
      if (result.embedding_status === "embedding_queued") {
        startPolling(result.session_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleResume = () => {
    const id = sessionInput.trim();
    if (id) router.push(`/chat/${id}`);
  };

  const copySessionId = () => {
    if (!uploadResult) return;
    navigator.clipboard.writeText(uploadResult.session_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const resetUpload = () => {
    setUploadResult(null);
    setEmbStatus(null);
    setFile(null);
    setError(null);
    if (pollingRef.current) clearInterval(pollingRef.current);
  };

  const isEmbDone =
    embStatus?.status === "complete" ||
    embStatus?.status === "empty" ||
    uploadResult?.embedding_status === "nothing_to_embed" ||
    uploadResult?.embedding_status === "embedding_skipped";

  return (
    <main className="min-h-screen bg-zinc-950 flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-zinc-800/60 px-6 py-4">
        <div className="max-w-2xl mx-auto">
          <h1 className="text-base font-semibold tracking-tight text-white">
            ReviewGPT
          </h1>
          <p className="text-xs text-zinc-500 mt-0.5">
            Conversational analytics for your Google Maps reviews
          </p>
        </div>
      </header>

      <div className="flex-1 max-w-2xl mx-auto w-full px-6 py-10 space-y-10">
        {!uploadResult ? (
          <>
            {/* ── Upload zone ── */}
            <section className="space-y-3">
              <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-widest">
                Upload Reviews
              </h2>

              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !file && fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-10 text-center transition-all ${
                  dragging
                    ? "border-indigo-500 bg-indigo-950/20"
                    : file
                      ? "border-zinc-600 bg-zinc-900/40"
                      : "border-zinc-700 hover:border-zinc-500 hover:bg-zinc-900/30 cursor-pointer"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) selectFile(f);
                  }}
                />
                {file ? (
                  <div>
                    <p className="text-zinc-200 font-medium">{file.name}</p>
                    <p className="text-zinc-500 text-sm mt-1">
                      {(file.size / 1024).toFixed(0)} KB
                    </p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setFile(null);
                      }}
                      className="mt-3 text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
                    >
                      × Remove
                    </button>
                  </div>
                ) : (
                  <div>
                    <p className="text-zinc-400 text-sm">
                      Drop a{" "}
                      <span className="text-zinc-200 font-medium">.csv</span> or{" "}
                      <span className="text-zinc-200 font-medium">.xlsx</span>{" "}
                      file here, or click to browse
                    </p>
                    <p className="text-zinc-600 text-xs mt-1.5">Max 10 MB</p>
                  </div>
                )}
              </div>

              {file && (
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium text-sm transition-colors flex items-center justify-center gap-2"
                >
                  {uploading ? (
                    <>
                      <Spinner /> Uploading…
                    </>
                  ) : (
                    "Upload"
                  )}
                </button>
              )}

              {error && (
                <p className="text-sm text-red-400 bg-red-950/30 border border-red-800/40 rounded-lg px-4 py-2.5">
                  {error}
                </p>
              )}
            </section>

            {/* ── Resume session ── */}
            <section className="space-y-3">
              <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-widest">
                Resume Session
              </h2>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={sessionInput}
                  onChange={(e) => setSessionInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleResume()}
                  placeholder="Paste an existing session ID…"
                  className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500 font-mono transition-colors"
                />
                <button
                  onClick={handleResume}
                  disabled={!sessionInput.trim()}
                  className="px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg text-sm transition-colors"
                >
                  Go →
                </button>
              </div>
            </section>
          </>
        ) : (
          /* ── Upload result panel ── */
          <section className="space-y-5">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold text-zinc-500 uppercase tracking-widest">
                Upload Complete
              </h2>
              <button
                onClick={resetUpload}
                className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                ← Upload another
              </button>
            </div>

            {/* Session ID card */}
            <div className="bg-zinc-900 border border-zinc-700/50 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500 uppercase tracking-wider font-medium">
                  Session ID
                </span>
                <button
                  onClick={copySessionId}
                  className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  {copied ? "✓ Copied" : "Copy"}
                </button>
              </div>
              <code className="block text-sm text-zinc-300 font-mono break-all">
                {uploadResult.session_id}
              </code>
            </div>

            {/* Stats card */}
            <div className="bg-zinc-900 border border-zinc-700/50 rounded-xl px-4 divide-y divide-zinc-800">
              <StatRow label="Total rows" value={uploadResult.total_rows} />
              <StatRow label="Valid rows" value={uploadResult.valid_rows} />
              {uploadResult.invalid_rows > 0 && (
                <StatRow
                  label="Invalid rows"
                  value={
                    <span className="text-amber-400">
                      {uploadResult.invalid_rows}
                    </span>
                  }
                />
              )}
              <StatRow
                label="Reviews with text"
                value={uploadResult.reviews_with_text}
              />
              <StatRow
                label="Inserted to DB"
                value={uploadResult.inserted_rows}
              />
              {uploadResult.skipped_rows > 0 && (
                <StatRow
                  label="Skipped (duplicate)"
                  value={
                    <span className="text-zinc-500">
                      {uploadResult.skipped_rows}
                    </span>
                  }
                />
              )}
            </div>

            {/* Embedding status card */}
            <div className="bg-zinc-900 border border-zinc-700/50 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-zinc-300">Embeddings</span>
                <EmbeddingBadge
                  uploadStatus={uploadResult.embedding_status}
                  embStatus={embStatus}
                />
              </div>

              {/* Progress bar (only while processing) */}
              {embStatus &&
                embStatus.status === "processing" &&
                embStatus.total_with_text > 0 && (
                  <div>
                    <div className="flex justify-between text-xs text-zinc-600 mb-1.5">
                      <span>
                        {embStatus.embedded} / {embStatus.total_with_text}{" "}
                        embedded
                      </span>
                      <span>{embStatus.pending} pending</span>
                    </div>
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full transition-all duration-700"
                        style={{
                          width: `${(embStatus.embedded / embStatus.total_with_text) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                )}
            </div>

            {/* CTA */}
            {isEmbDone ? (
              <button
                onClick={() =>
                  router.push(`/chat/${uploadResult.session_id}`)
                }
                className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-semibold text-sm transition-colors"
              >
                Start chatting →
              </button>
            ) : (
              <p className="text-center text-sm text-zinc-500 flex items-center justify-center gap-2">
                <Spinner />
                Generating embeddings — this may take a minute
              </p>
            )}
          </section>
        )}
      </div>
    </main>
  );
}
