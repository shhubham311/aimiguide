"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { marked } from "marked";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import bash from "highlight.js/lib/languages/bash";
import sql from "highlight.js/lib/languages/sql";
import json from "highlight.js/lib/languages/json";
import { X, BookOpen, ChevronRight, Loader2, AlertCircle } from "lucide-react";

hljs.registerLanguage("python", python);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("sql", sql);
hljs.registerLanguage("json", json);

marked.setOptions({
  highlight(code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
  gfm: true,
});

interface LessonViewerProps {
  topicId: string;
  topicTitle: string;
  isOpen: boolean;
  onClose: () => void;
}

export default function LessonViewer({
  topicId,
  topicTitle,
  isOpen,
  onClose,
}: LessonViewerProps) {
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  const fetchContent = useCallback(async () => {
    if (!topicId || !isOpen) return;
    setLoading(true);
    setError("");
    setContent("");
    try {
      const res = await fetch(`/api/lesson?topic=${encodeURIComponent(topicId)}`);
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setContent(data.content);
      }
    } catch {
      setError("Failed to load lesson");
    } finally {
      setLoading(false);
    }
  }, [topicId, isOpen]);

  useEffect(() => {
    fetchContent();
  }, [fetchContent]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [topicId]);

  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  if (!isOpen) return null;

  const htmlContent = content ? marked.parse(content) as string : "";

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="relative ml-0 w-full sm:ml-8 lg:ml-16 bg-[#0d0d14] border-l border-white/10 flex flex-col h-full max-w-5xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/10 bg-[#0d0d14]/90 backdrop-blur-sm shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center shrink-0">
              <BookOpen className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="min-w-0">
              <h2 className="text-sm font-bold text-white truncate">
                {topicTitle}
              </h2>
              <p className="text-[10px] text-gray-500 font-mono">
                /learn {topicId}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/10 transition-all shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {loading && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-500">
              <Loader2 className="w-6 h-6 animate-spin" />
              <span className="text-sm">Loading lesson...</span>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center h-full gap-3 text-gray-500">
              <AlertCircle className="w-8 h-8 text-rose-400" />
              <span className="text-sm text-rose-300">{error}</span>
            </div>
          )}

          {!loading && !error && htmlContent && (
            <article
              className="prose-custom px-6 py-6 sm:px-8 sm:py-8"
              dangerouslySetInnerHTML={{ __html: htmlContent }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
