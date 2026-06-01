"use client";

import { ExternalLink, Maximize2 } from "lucide-react";

export default function StudyTrackApp() {
  return (
    <div className="relative h-[calc(100dvh-42px)] overflow-hidden bg-[#0a0a0f] text-gray-100">
      <div className="absolute right-3 top-3 z-10 flex gap-1 rounded-full border border-white/10 bg-black/60 p-1 shadow-2xl backdrop-blur-md">
        <a
          href="/studytrack/index.html"
          target="_blank"
          rel="noopener noreferrer"
          title="Open StudyTrack in a new page"
          aria-label="Open StudyTrack in a new page"
          className="inline-flex h-9 w-9 items-center justify-center rounded-full text-gray-300 transition hover:bg-white/10 hover:text-white"
        >
          <ExternalLink className="h-4 w-4" />
        </a>
        <button
          onClick={() => document.getElementById("studytrack-frame")?.requestFullscreen?.()}
          title="Fullscreen StudyTrack"
          aria-label="Fullscreen StudyTrack"
          className="inline-flex h-9 w-9 items-center justify-center rounded-full text-indigo-100 transition hover:bg-indigo-500/25"
        >
          <Maximize2 className="h-4 w-4" />
        </button>
      </div>
      <iframe
        id="studytrack-frame"
        src="/studytrack/index.html"
        title="StudyTrack full app"
        className="h-full w-full border-0 bg-[#0a0a1a]"
      />
    </div>
  );
}
