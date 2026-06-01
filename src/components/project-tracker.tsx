"use client";

import { useState, useMemo, useCallback, useEffect, useSyncExternalStore } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ExternalLink, Check, Search, X, Trophy, Rocket,
  TrendingUp, MessageSquare, Eye, Heart, ThumbsUp,
  BrainCircuit, Cpu, Shield, LayoutGrid, List, Filter,
  ChevronDown, ChevronRight, Zap, Target, Star,
  Flame, BookOpen, Send, ArrowUpRight
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  mlProjects,
  projectCategories,
  searchProjects,
  getProjectsByCategory,
  type MLProject,
  type ProjectCategory,
} from "@/lib/projects-data";
import { loadRemoteState, saveRemoteState } from "@/lib/supabase-state";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  TrendingUp, MessageSquare, Eye, Heart, ThumbsUp, BrainCircuit, Cpu, Shield,
};

const depthColors = {
  beginner: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
  intermediate: "bg-amber-500/20 text-amber-300 border-amber-500/30",
  advanced: "bg-rose-500/20 text-rose-300 border-rose-500/30",
};

const depthLabels = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  advanced: "Advanced",
};

interface ProjectProgress {
  completedProjects: string[];
  expandedCategories: string[];
}

const emptyProjectProgress: ProjectProgress = { completedProjects: [], expandedCategories: [] };

let pCachedRaw: string | null = null;
let pCachedProgress: ProjectProgress = emptyProjectProgress;

function subscribeProjectStorage(callback: () => void) {
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

function getStoredProjectProgress(): ProjectProgress {
  try {
    const raw = localStorage.getItem("ml-projects-progress");
    if (raw === pCachedRaw) return pCachedProgress;
    pCachedRaw = raw;
    pCachedProgress = raw ? JSON.parse(raw) : emptyProjectProgress;
    return pCachedProgress;
  } catch {
    pCachedRaw = null;
    pCachedProgress = emptyProjectProgress;
    return emptyProjectProgress;
  }
}

function getServerProjectProgress(): ProjectProgress {
  return emptyProjectProgress;
}

function getStreakCount(completed: Set<string>): number {
  // Simple streak: count total completed
  return completed.size;
}

function getLevel(completedCount: number): { level: number; title: string; nextAt: number; color: string } {
  if (completedCount >= 80) return { level: 5, title: "ML Master", nextAt: 93, color: "text-purple-400" };
  if (completedCount >= 60) return { level: 4, title: "ML Expert", nextAt: 80, color: "text-amber-400" };
  if (completedCount >= 40) return { level: 3, title: "ML Practitioner", nextAt: 60, color: "text-cyan-400" };
  if (completedCount >= 20) return { level: 2, title: "ML Builder", nextAt: 40, color: "text-emerald-400" };
  if (completedCount >= 5) return { level: 1, title: "ML Explorer", nextAt: 20, color: "text-blue-400" };
  return { level: 0, title: "ML Beginner", nextAt: 5, color: "text-gray-400" };
}

export default function ProjectTracker() {
  const [searchQuery, setSearchQuery] = useState("");
  const [depthFilter, setDepthFilter] = useState<"all" | "beginner" | "intermediate" | "advanced">("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("list");

  useEffect(() => {
    let cancelled = false;
    loadRemoteState("ml_projects_progress", emptyProjectProgress).then((data) => {
      if (cancelled) return;
      try {
        const raw = JSON.stringify(data);
        localStorage.setItem("ml-projects-progress", raw);
        pCachedRaw = raw;
        pCachedProgress = data;
        window.dispatchEvent(new StorageEvent("storage"));
      } catch {}
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const storedProgress = useSyncExternalStore(
    subscribeProjectStorage,
    getStoredProjectProgress,
    getServerProjectProgress
  );

  const completedProjects = useMemo(
    () => new Set(storedProgress.completedProjects),
    [storedProgress.completedProjects]
  );
  const expandedCategories = useMemo(
    () => new Set(storedProgress.expandedCategories),
    [storedProgress.expandedCategories]
  );

  const writeProgress = useCallback((data: ProjectProgress) => {
    try {
      localStorage.setItem("ml-projects-progress", JSON.stringify(data));
      window.dispatchEvent(new StorageEvent("storage"));
    } catch {}
    saveRemoteState("ml_projects_progress", data);
  }, []);

  const toggleProjectComplete = useCallback((projectId: string) => {
    const current = storedProgress.completedProjects;
    const next = current.includes(projectId)
      ? current.filter((id) => id !== projectId)
      : [...current, projectId];
    writeProgress({ ...storedProgress, completedProjects: next });
  }, [storedProgress, writeProgress]);

  const toggleCategory = useCallback((categoryId: string) => {
    const current = storedProgress.expandedCategories;
    const next = current.includes(categoryId)
      ? current.filter((id) => id !== categoryId)
      : [...current, categoryId];
    writeProgress({ ...storedProgress, expandedCategories: next });
  }, [storedProgress, writeProgress]);

  const totalProjects = mlProjects.length;
  const completedCount = completedProjects.size;
  const totalProgress = totalProjects > 0 ? Math.round((completedCount / totalProjects) * 100) : 0;
  const levelInfo = getLevel(completedCount);
  const levelProgress = Math.min(
    100,
    Math.round(((completedCount % (levelInfo.nextAt > 0 ? 20 : 93)) / (levelInfo.level === 0 ? 5 : 20)) * 100)
  );

  // Filter projects
  const filteredCategories = useMemo(() => {
    return projectCategories
      .map((cat) => {
        const catProjects = getProjectsByCategory(cat.id);
        const filtered = catProjects.filter((p) => {
          const matchesSearch =
            cat.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            p.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            p.tags.some((t) => t.includes(searchQuery.toLowerCase()));
          const matchesDepth = depthFilter === "all" || p.difficulty === depthFilter;
          const matchesCategory = categoryFilter === "all" || cat.id === categoryFilter;
          return matchesSearch && matchesDepth && matchesCategory;
        });
        const catCompleted = filtered.filter((p) => completedProjects.has(p.id)).length;
        return { ...cat, filteredProjects: filtered, catCompleted };
      })
      .filter((c) => c.filteredProjects.length > 0);
  }, [searchQuery, depthFilter, categoryFilter, completedProjects]);

  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-screen bg-[#0a0a0f] text-gray-100 font-mono relative overflow-hidden">
        {/* Background effects */}
        <div className="fixed inset-0 pointer-events-none">
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
              backgroundSize: "60px 60px",
            }}
          />
          <div className="absolute top-0 right-1/4 w-96 h-96 bg-violet-500/5 rounded-full blur-[120px]" />
          <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-[120px]" />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 py-3 sm:py-6">
          {/* Header */}
          <motion.header
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 sm:mb-8"
          >
            <div className="flex items-center justify-between gap-3 mb-3 sm:mb-5">
              <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
                <div className="relative shrink-0">
                  <div className="w-9 h-9 sm:w-11 sm:h-11 rounded-xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 flex items-center justify-center border border-violet-500/30">
                    <Rocket className="w-4 h-4 sm:w-5 sm:h-5 text-violet-400" />
                  </div>
                </div>
                <div className="min-w-0">
                  <h1 className="text-base sm:text-xl lg:text-2xl font-bold tracking-tight text-white truncate">
                    ML <span className="text-violet-400">Projects</span> Tracker
                  </h1>
                  <p className="text-[10px] sm:text-xs text-gray-500 truncate">{mlProjects.length} real-world projects from GeeksforGeeks — track your progress</p>
                </div>
              </div>
            </div>

            {/* Stats Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3 mb-4">
              {/* Total Progress */}
              <div className="bg-white/[0.04] rounded-xl p-3 sm:p-4 border border-white/10">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-2">
                  <Trophy className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-amber-400" />
                  <span className="text-[10px] sm:text-xs text-gray-500">Completed</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-lg sm:text-2xl font-bold text-white">{completedCount}</span>
                  <span className="text-xs sm:text-sm text-gray-500">/ {totalProjects}</span>
                </div>
                <Progress value={totalProgress} className="h-1 bg-white/10 mt-2" />
                <span className="text-[10px] sm:text-xs text-emerald-400 font-mono">{totalProgress}%</span>
              </div>

              {/* Level */}
              <div className="bg-white/[0.04] rounded-xl p-3 sm:p-4 border border-white/10">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-2">
                  <Zap className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-violet-400" />
                  <span className="text-[10px] sm:text-xs text-gray-500">Level</span>
                </div>
                <div className={`text-lg sm:text-2xl font-bold ${levelInfo.color}`}>
                  Lv.{levelInfo.level}
                </div>
                <span className="text-[10px] sm:text-xs text-gray-400">{levelInfo.title}</span>
                <div className="flex items-center gap-1 mt-1">
                  <Star className="w-2.5 h-2.5 text-amber-400 fill-amber-400" />
                  <span className="text-[9px] sm:text-[10px] text-gray-500">Next: {levelInfo.nextAt} projects</span>
                </div>
              </div>

              {/* Category Progress */}
              <div className="bg-white/[0.04] rounded-xl p-3 sm:p-4 border border-white/10">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-2">
                  <Target className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-cyan-400" />
                  <span className="text-[10px] sm:text-xs text-gray-500">Categories</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-lg sm:text-2xl font-bold text-white">
                    {projectCategories.filter((c) => {
                      const catProjects = getProjectsByCategory(c.id);
                      return catProjects.every((p) => completedProjects.has(p.id));
                    }).length}
                  </span>
                  <span className="text-xs sm:text-sm text-gray-500">/ {projectCategories.length}</span>
                </div>
                <span className="text-[10px] sm:text-xs text-gray-400">100% done</span>
              </div>

              {/* Streak / Fire */}
              <div className="bg-white/[0.04] rounded-xl p-3 sm:p-4 border border-white/10">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-2">
                  <Flame className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-orange-400" />
                  <span className="text-[10px] sm:text-xs text-gray-500">Streak</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-lg sm:text-2xl font-bold text-white">{completedCount}</span>
                </div>
                <span className="text-[10px] sm:text-xs text-gray-400">projects built</span>
              </div>
            </div>

            {/* Search & Filters */}
            <div className="flex gap-2 sm:gap-3">
              <div className="relative flex-1 min-w-0">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <Input
                  placeholder="Search projects, tags..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 sm:pl-10 bg-white/5 border-white/10 text-gray-200 placeholder:text-gray-600 focus:border-violet-500/50 focus:ring-violet-500/20 h-9 sm:h-10 rounded-lg font-mono text-xs sm:text-sm"
                />
                {searchQuery && (
                  <button onClick={() => setSearchQuery("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {/* Category filter dropdown - desktop */}
                <div className="hidden md:flex bg-white/5 rounded-lg p-0.5 border border-white/10 overflow-x-auto scrollbar-none max-w-[200px]">
                  <button
                    onClick={() => setCategoryFilter("all")}
                    className={`px-2.5 py-1.5 rounded-md text-[10px] font-medium transition-all whitespace-nowrap ${
                      categoryFilter === "all" ? "bg-violet-500/20 text-violet-300 border border-violet-500/30" : "text-gray-500 hover:text-gray-300"
                    }`}
                  >
                    All
                  </button>
                  {projectCategories.map((cat) => (
                    <button
                      key={cat.id}
                      onClick={() => setCategoryFilter(cat.id)}
                      className={`px-2 py-1.5 rounded-md text-[10px] font-medium transition-all whitespace-nowrap ${
                        categoryFilter === cat.id ? "bg-violet-500/20 text-violet-300 border border-violet-500/30" : "text-gray-500 hover:text-gray-300"
                      }`}
                    >
                      {cat.name.split(" ")[0]}
                    </button>
                  ))}
                </div>

                {/* Depth filter */}
                <div className="hidden sm:flex bg-white/5 rounded-lg p-0.5 border border-white/10">
                  {(["all", "beginner", "intermediate", "advanced"] as const).map((d) => (
                    <button
                      key={d}
                      onClick={() => setDepthFilter(d)}
                      className={`px-2.5 py-1.5 rounded-md text-[10px] sm:text-xs font-medium transition-all whitespace-nowrap ${
                        depthFilter === d ? "bg-violet-500/20 text-violet-300 border border-violet-500/30" : "text-gray-500 hover:text-gray-300"
                      }`}
                    >
                      {d === "all" ? "All" : depthLabels[d]}
                    </button>
                  ))}
                </div>

                {/* View toggle */}
                <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10">
                  <button onClick={() => setViewMode("list")} className={`p-1.5 rounded-md transition-all ${viewMode === "list" ? "bg-white/10 text-white" : "text-gray-500"}`}>
                    <List className="w-4 h-4" />
                  </button>
                  <button onClick={() => setViewMode("grid")} className={`p-1.5 rounded-md transition-all ${viewMode === "grid" ? "bg-white/10 text-white" : "text-gray-500"}`}>
                    <LayoutGrid className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Mobile filters */}
            <div className="sm:hidden mt-2">
              <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10 overflow-x-auto scrollbar-none gap-1">
                {(["all", ...projectCategories.map((c) => c.id)] as string[]).map((cId) => (
                  <button
                    key={cId}
                    onClick={() => setCategoryFilter(cId)}
                    className={`px-2 py-1.5 rounded-md text-[10px] font-medium transition-all whitespace-nowrap shrink-0 ${
                      categoryFilter === cId ? "bg-violet-500/20 text-violet-300 border border-violet-500/30" : "text-gray-500 hover:text-gray-300 border border-transparent"
                    }`}
                  >
                    {cId === "all" ? "All" : projectCategories.find((c) => c.id === cId)?.name.split(" ")[0] || cId}
                  </button>
                ))}
              </div>
              <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10 overflow-x-auto scrollbar-none mt-2 gap-1">
                {(["all", "beginner", "intermediate", "advanced"] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => setDepthFilter(d)}
                    className={`px-2.5 py-1.5 rounded-md text-[10px] font-medium transition-all whitespace-nowrap shrink-0 ${
                      depthFilter === d ? "bg-violet-500/20 text-violet-300 border border-violet-500/30" : "text-gray-500 hover:text-gray-300 border border-transparent"
                    }`}
                  >
                    {d === "all" ? "All Levels" : depthLabels[d]}
                  </button>
                ))}
              </div>
            </div>
          </motion.header>

          {/* Category + Projects */}
          <div className="space-y-3 sm:space-y-4">
            {filteredCategories.map((cat, idx) => (
              <CategorySection
                key={cat.id}
                category={cat}
                expandedCategories={expandedCategories}
                completedProjects={completedProjects}
                index={idx}
                viewMode={viewMode}
                onToggle={() => toggleCategory(cat.id)}
                onToggleProject={toggleProjectComplete}
              />
            ))}
          </div>

          {filteredCategories.length === 0 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center py-16 sm:py-20 text-gray-600">
              <Search className="w-10 h-10 sm:w-12 sm:h-12 mb-3 opacity-30" />
              <p className="text-base sm:text-lg">No projects found</p>
              <p className="text-xs sm:text-sm mt-1">Try a different search or filter</p>
            </motion.div>
          )}

          {/* Footer info */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="mt-8 mb-4 text-center"
          >
            <p className="text-[10px] sm:text-xs text-gray-600">
              Projects sourced from{" "}
              <a
                href="https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-violet-400 hover:text-violet-300 underline underline-offset-2"
              >
                GeeksforGeeks ML Projects
              </a>{" "}
              — Click any project to open the tutorial
            </p>
          </motion.div>
        </div>
      </div>
    </TooltipProvider>
  );
}

// ── Category Section ──

interface CategorySectionProps {
  category: ProjectCategory & { filteredProjects: MLProject[]; catCompleted: number };
  expandedCategories: Set<string>;
  completedProjects: Set<string>;
  index: number;
  viewMode: "grid" | "list";
  onToggle: () => void;
  onToggleProject: (id: string) => void;
}

function CategorySection({
  category: cat,
  expandedCategories,
  completedProjects,
  index,
  viewMode,
  onToggle,
  onToggleProject,
}: CategorySectionProps) {
  const isExpanded = expandedCategories.has(cat.id);
  const Icon = iconMap[cat.icon] || BrainCircuit;
  const catProgress = cat.filteredProjects.length > 0
    ? Math.round((cat.catCompleted / cat.filteredProjects.length) * 100)
    : 0;
  const isDone = cat.catCompleted === cat.filteredProjects.length && cat.filteredProjects.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.05 }}
      className={`rounded-xl border transition-all duration-300 ${
        isExpanded
          ? "bg-white/[0.06] border-white/15 shadow-lg " + cat.glowColor
          : "bg-white/[0.03] border-white/8 hover:bg-white/[0.05] hover:border-white/12"
      }`}
    >
      <button onClick={onToggle} className="w-full p-3 sm:p-4 lg:p-5 text-left flex items-center gap-2.5 sm:gap-3 lg:gap-4">
        <div className={`relative shrink-0 w-9 h-9 sm:w-10 sm:h-10 lg:w-11 lg:h-11 rounded-lg flex items-center justify-center border transition-all ${cat.bgAccent} ${isExpanded ? "border-white/20" : "border-white/10"}`}>
          <Icon className={`w-4 h-4 sm:w-5 sm:h-5 ${cat.color}`} />
          {isDone && (
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-500 rounded-full flex items-center justify-center">
              <Check className="w-2.5 h-2.5 text-white" strokeWidth={3} />
            </div>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className={`font-bold text-xs sm:text-sm lg:text-base ${isExpanded ? "text-white" : "text-gray-200"}`}>
              {cat.name}
            </h2>
            <span className="text-[9px] sm:text-[10px] px-1.5 py-0.5 rounded-full border bg-white/5 border-white/10 text-gray-400">
              {cat.filteredProjects.length} projects
            </span>
          </div>
          <p className={`text-[10px] sm:text-xs mt-0.5 truncate ${isExpanded ? "text-gray-400" : "text-gray-600"}`}>
            {cat.description}
          </p>
          {cat.catCompleted > 0 && (
            <div className="flex items-center gap-2 mt-1.5 sm:mt-2">
              <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full rounded-full ${isDone ? "bg-emerald-500" : cat.color.replace("text-", "bg-")}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${catProgress}%` }}
                  transition={{ duration: 0.8 }}
                />
              </div>
              <span className="text-[9px] sm:text-[10px] text-emerald-400 font-mono tabular-nums">{catProgress}%</span>
            </div>
          )}
        </div>
        <motion.div animate={{ rotate: isExpanded ? 180 : 0 }} transition={{ duration: 0.3 }} className="shrink-0 text-gray-500">
          <ChevronDown className="w-4 h-4 sm:w-5 sm:h-5" />
        </motion.div>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-3 sm:px-4 lg:px-5 pb-3 sm:pb-4 lg:pb-5">
              <Separator className="bg-white/10 mb-3 sm:mb-4" />
              <div className={viewMode === "grid" ? "grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3" : "space-y-1.5 sm:space-y-2"}>
                {cat.filteredProjects.map((project, pIdx) => (
                  <ProjectCard
                    key={project.id}
                    project={project}
                    isCompleted={completedProjects.has(project.id)}
                    index={pIdx}
                    viewMode={viewMode}
                    onToggle={() => onToggleProject(project.id)}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ── Project Card ──

interface ProjectCardProps {
  project: MLProject;
  isCompleted: boolean;
  index: number;
  viewMode: "grid" | "list";
  onToggle: () => void;
}

function ProjectCard({ project, isCompleted, index, viewMode, onToggle }: ProjectCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.15, delay: index * 0.02 }}
      className={`group rounded-lg border transition-all duration-200 ${
        isCompleted
          ? "bg-emerald-500/[0.06] border-emerald-500/20"
          : "bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.04] hover:border-white/10"
      }`}
    >
      <div className="flex items-center gap-2 sm:gap-3 p-2.5 sm:p-3">
        {/* Checkbox */}
        <button
          onClick={onToggle}
          className={`shrink-0 w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all ${
            isCompleted
              ? "bg-emerald-500 border-emerald-500"
              : "border-gray-600 hover:border-gray-400 group-hover:border-gray-500"
          }`}
        >
          {isCompleted && <Check className="w-3 h-3 text-white" strokeWidth={3} />}
        </button>

        {/* Project Info */}
        <a
          href={project.url}
          target="_blank"
          rel="noopener noreferrer"
          className={`flex-1 text-left min-w-0 transition-colors ${
            isCompleted ? "opacity-70" : ""
          }`}
        >
          <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
            <span className={`text-xs sm:text-sm font-medium leading-snug ${isCompleted ? "text-emerald-300 line-through decoration-emerald-500/40" : "text-gray-200"}`}>
              {project.title}
            </span>
            <ArrowUpRight className="w-3 h-3 text-gray-600 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <div className="flex items-center gap-1 mt-0.5 flex-wrap">
            <span className={`text-[9px] sm:text-[10px] px-1.5 py-0.5 rounded-full border shrink-0 ${depthColors[project.difficulty]}`}>
              {depthLabels[project.difficulty]}
            </span>
            {project.tags.slice(0, viewMode === "grid" ? 2 : 3).map((tag) => (
              <span key={tag} className="text-[9px] sm:text-[10px] px-1.5 py-0.5 rounded-full bg-white/5 border border-white/10 text-gray-500 shrink-0">
                {tag}
              </span>
            ))}
            {project.tags.length > 3 && viewMode === "list" && (
              <span className="text-[9px] text-gray-600">+{project.tags.length - 3}</span>
            )}
          </div>
        </a>
      </div>
    </motion.div>
  );
}
