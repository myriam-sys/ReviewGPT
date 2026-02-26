"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { sendChat } from "@/lib/api";
import { ChatMessage } from "@/components/ChatMessage";
import type { Message } from "@/lib/types";

export default function ChatPage({
  params,
}: {
  params: { session_id: string };
}) {
  const { session_id } = params;

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [copied, setCopied] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom whenever messages update
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || sending) return;

    const loadingId = `loading-${Date.now()}`;

    setMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: "user", content: question },
      { id: loadingId, role: "assistant", content: "", isLoading: true },
    ]);
    setInput("");
    setSending(true);

    try {
      const response = await sendChat(session_id, question, 10);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? {
                id: loadingId,
                role: "assistant",
                content: response.answer,
                chatResponse: response,
              }
            : m,
        ),
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Request failed";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? { id: loadingId, role: "assistant", content: "", error: msg }
            : m,
        ),
      );
    } finally {
      setSending(false);
      textareaRef.current?.focus();
    }
  };

  const copySession = () => {
    navigator.clipboard.writeText(session_id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="h-screen flex flex-col bg-zinc-950">
      {/* ── Header ── */}
      <header className="flex-none border-b border-zinc-800/60 px-4 py-3 flex items-center gap-3">
        <Link
          href="/"
          className="text-zinc-500 hover:text-zinc-200 text-sm transition-colors flex-none"
        >
          ←
        </Link>

        <div className="flex-1 flex items-center gap-2 min-w-0">
          <span className="text-xs text-zinc-600 flex-none">Session</span>
          <code className="text-xs text-zinc-400 font-mono truncate">
            {session_id}
          </code>
          <button
            onClick={copySession}
            className="flex-none text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            {copied ? "✓" : "copy"}
          </button>
        </div>

        <span className="text-sm font-semibold text-white flex-none">
          ReviewGPT
        </span>
      </header>

      {/* ── Messages ── */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-20 space-y-2">
              <p className="text-zinc-400">Ask anything about your reviews</p>
              <p className="text-zinc-600 text-sm">
                e.g. &ldquo;What do customers complain about most?&rdquo;
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Input ── */}
      <div className="flex-none border-t border-zinc-800/60 px-4 py-4">
        <div className="max-w-3xl mx-auto flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="Ask a question about your reviews…"
            rows={1}
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500 resize-none transition-colors"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || sending}
            className="flex-none px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl text-sm font-medium transition-colors"
          >
            Send
          </button>
        </div>
        <p className="text-center text-xs text-zinc-700 mt-2">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
