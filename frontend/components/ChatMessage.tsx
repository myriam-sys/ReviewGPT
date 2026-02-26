"use client";

import { useState } from "react";
import type { Message, ReviewSource } from "@/lib/types";

// ── Question-type badge styles ────────────────────────────────────────────────

const TYPE_STYLES: Record<string, string> = {
  analytical:
    "bg-blue-950/60 text-blue-300 border-blue-700/40",
  temporal:
    "bg-purple-950/60 text-purple-300 border-purple-700/40",
  comparative:
    "bg-orange-950/60 text-orange-300 border-orange-700/40",
  author:
    "bg-cyan-950/60 text-cyan-300 border-cyan-700/40",
  search:
    "bg-zinc-800/80 text-zinc-400 border-zinc-700/40",
};

// ── Sub-components ────────────────────────────────────────────────────────────

function LoadingSpinner() {
  return (
    <svg
      className="animate-spin h-4 w-4 text-zinc-400"
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

function Stars({ rating }: { rating: number }) {
  const filled = Math.round(Math.min(5, Math.max(0, rating)));
  return (
    <span className="text-amber-400 text-xs tracking-tight">
      {"★".repeat(filled)}
      {"☆".repeat(5 - filled)}
    </span>
  );
}

function SourceCard({ review }: { review: ReviewSource }) {
  return (
    <div className="bg-zinc-900 border border-zinc-700/50 rounded-lg p-3 text-xs space-y-1.5">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-zinc-300 font-medium truncate">
          {review.author || "Anonymous"}
        </span>
        <Stars rating={review.rating} />
        {review.date && (
          <span className="text-zinc-500 ml-auto flex-none">
            {review.date.split("T")[0]}
          </span>
        )}
      </div>
      <p className="text-zinc-400 leading-relaxed line-clamp-3">{review.text}</p>
      {review.similarity !== null && (
        <p className="text-zinc-600">
          {(review.similarity * 100).toFixed(0)}% match
        </p>
      )}
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────

export function ChatMessage({ message }: { message: Message }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);

  // User message — right-aligned
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[70%] bg-indigo-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant message — left-aligned
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] space-y-2">
        {/* Bubble */}
        <div className="bg-zinc-800/80 text-zinc-100 rounded-2xl rounded-tl-sm px-4 py-3 text-sm leading-relaxed">
          {message.isLoading ? (
            <span className="flex items-center gap-2 text-zinc-400">
              <LoadingSpinner />
              Thinking…
            </span>
          ) : message.error ? (
            <span className="text-red-400">⚠ {message.error}</span>
          ) : (
            <div className="whitespace-pre-wrap">{message.content}</div>
          )}
        </div>

        {/* Footer: type badge + sources toggle + token count */}
        {!message.isLoading && !message.error && message.chatResponse && (
          <div className="flex flex-wrap items-center gap-2 px-1">
            {/* Question type */}
            <span
              className={`text-xs px-2 py-0.5 rounded-full border ${
                TYPE_STYLES[message.chatResponse.question_type] ??
                TYPE_STYLES.search
              }`}
            >
              {message.chatResponse.question_type}
            </span>

            {/* Sources toggle */}
            {message.chatResponse.sources_count > 0 && (
              <button
                onClick={() => setSourcesOpen((o) => !o)}
                className="text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
              >
                {sourcesOpen ? "▾" : "▸"} Sources (
                {message.chatResponse.sources_count})
              </button>
            )}

            {/* Token count */}
            <span className="text-xs text-zinc-600 ml-auto">
              {message.chatResponse.tokens_used} tokens
            </span>
          </div>
        )}

        {/* Sources list */}
        {sourcesOpen && message.chatResponse?.retrieved_reviews.length ? (
          <div className="space-y-2 pl-1">
            {message.chatResponse.retrieved_reviews.map((review, i) => (
              <SourceCard key={review.review_id || String(i)} review={review} />
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
