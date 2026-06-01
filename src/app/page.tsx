"use client";

import { useState, useEffect } from "react";
import { BookOpenCheck, Flame, Rocket, GraduationCap, Target, Terminal } from "lucide-react";
import RoadmapDashboard from "@/components/roadmap-dashboard";
import ProjectTracker from "@/components/project-tracker";
import LearningTracker from "@/components/learning-tracker";
import StudyTrackApp from "@/components/studytrack-app";
import ResetTrackerApp from "@/components/reset-tracker-app";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"roadmap" | "projects" | "learning" | "studytrack" | "reset">("roadmap");
  const [projectCount, setProjectCount] = useState(101);
  const [subjectCount, setSubjectCount] = useState(0);

  // Load counts dynamically
  useEffect(() => {
    import("@/lib/projects-data").then((mod) => {
      setProjectCount(mod.mlProjects.length);
    });
    import("@/lib/subjects-data").then((mod) => {
      setSubjectCount(mod.learningSubjects.length);
    });
  }, []);

  const tabs = [
    { id: "roadmap" as const, label: "Learning Roadmap", icon: Terminal, colorClass: "emerald", badge: null },
    { id: "projects" as const, label: "Project Tracker", icon: Rocket, colorClass: "violet", badge: projectCount },
    { id: "learning" as const, label: "Subject Tracker", icon: GraduationCap, colorClass: "teal", badge: subjectCount },
    { id: "studytrack" as const, label: "StudyTrack", icon: Target, colorClass: "indigo", badge: null },
    { id: "reset" as const, label: "Reset Tracker", icon: Flame, colorClass: "amber", badge: null },
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      {/* Tab Switcher - fixed top bar */}
      <div className="sticky top-0 z-40 bg-[#0a0a0f]/90 backdrop-blur-md border-b border-white/10">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
          <div className="flex items-center gap-3 py-2">
            <div className="hidden shrink-0 items-center gap-2 pr-3 text-sm font-black text-white sm:flex">
              <BookOpenCheck className="h-4 w-4 text-indigo-300" />
              Personal Recordbook
            </div>
            <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto scrollbar-none">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              const activeColorMap: Record<string, string> = {
                emerald: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
                violet: "bg-violet-500/15 text-violet-300 border-violet-500/30",
                teal: "bg-teal-500/15 text-teal-300 border-teal-500/30",
                indigo: "bg-indigo-500/15 text-indigo-300 border-indigo-500/30",
                amber: "bg-amber-500/15 text-amber-300 border-amber-500/30",
              };
              const badgeColorMap: Record<string, string> = {
                emerald: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
                violet: "bg-violet-500/20 text-violet-300 border-violet-500/30",
                teal: "bg-teal-500/20 text-teal-300 border-teal-500/30",
                indigo: "bg-indigo-500/20 text-indigo-300 border-indigo-500/30",
                amber: "bg-amber-500/20 text-amber-300 border-amber-500/30",
              };
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all shrink-0 ${
                    isActive
                      ? activeColorMap[tab.colorClass]
                      : "text-gray-500 hover:text-gray-300 border border-transparent hover:bg-white/5"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                  <span className="sm:hidden">{tab.label.split(" ")[0]}</span>
                  {tab.badge !== null && tab.badge > 0 && (
                    <span className={`hidden sm:inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full text-[10px] font-bold border ${badgeColorMap[tab.colorClass]}`}>
                      {tab.badge}
                    </span>
                  )}
                </button>
              );
            })}
            </div>
          </div>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === "roadmap" && <RoadmapDashboard />}
      {activeTab === "projects" && <ProjectTracker />}
      {activeTab === "learning" && <LearningTracker />}
      <div className={activeTab === "studytrack" ? "block" : "hidden"} aria-hidden={activeTab !== "studytrack"}>
        <StudyTrackApp />
      </div>
      {activeTab === "reset" && <ResetTrackerApp />}
    </div>
  );
}
