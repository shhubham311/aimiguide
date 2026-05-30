"use client";

import { useState } from "react";
import { Terminal, Rocket } from "lucide-react";
import RoadmapDashboard from "@/components/roadmap-dashboard";
import ProjectTracker from "@/components/project-tracker";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"roadmap" | "projects">("roadmap");

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      {/* Tab Switcher - fixed top bar */}
      <div className="sticky top-0 z-40 bg-[#0a0a0f]/90 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-1 py-2">
            <button
              onClick={() => setActiveTab("roadmap")}
              className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "roadmap"
                  ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                  : "text-gray-500 hover:text-gray-300 border border-transparent hover:bg-white/5"
              }`}
            >
              <Terminal className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span>Learning Roadmap</span>
            </button>
            <button
              onClick={() => setActiveTab("projects")}
              className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === "projects"
                  ? "bg-violet-500/15 text-violet-300 border border-violet-500/30"
                  : "text-gray-500 hover:text-gray-300 border border-transparent hover:bg-white/5"
              }`}
            >
              <Rocket className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span>Project Tracker</span>
              <span className="hidden sm:inline-flex items-center justify-center w-5 h-5 rounded-full bg-violet-500/20 text-violet-300 text-[10px] font-bold border border-violet-500/30">
                93
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === "roadmap" ? (
        <RoadmapDashboard />
      ) : (
        <ProjectTracker />
      )}
    </div>
  );
}
