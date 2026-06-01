"use client";

import { useState, useMemo, useCallback, useEffect, useSyncExternalStore } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ExternalLink, Check, Search, X, Trophy, GraduationCap,
  Code2, Sigma, Database, BarChart3, BrainCircuit, Settings2,
  Cpu, MessageSquare, Eye, Sparkles, Wand2, Container,
  Server, Binary, ChevronDown, Zap, Target, Star,
  Flame, LayoutGrid, List, ArrowUpRight, BookOpen, ShieldCheck
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  subjectCategories,
  learningSubjects,
  getSubjectsByCategorySimple,
  searchSubjects,
  type LearningSubject,
  type SubjectCategory,
} from "@/lib/subjects-data";
import { loadRemoteState, saveRemoteState } from "@/lib/supabase-state";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  Code2, Sigma, Database, BarChart3, BrainCircuit, Settings2,
  Cpu, MessageSquare, Eye, Sparkles, Wand2, Container,
  Server, Binary,
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

const roleColors: Record<string, string> = {
  "Data Scientist": "bg-blue-500/15 text-blue-300 border-blue-500/25",
  "ML Engineer": "bg-emerald-500/15 text-emerald-300 border-emerald-500/25",
  "AI Engineer": "bg-violet-500/15 text-violet-300 border-violet-500/25",
  "GenAI Engineer": "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/25",
};

interface SubjectProgress {
  completedSubjects: string[];
  expandedCategories: string[];
  roleFilter: string;
}

const emptySubjectProgress: SubjectProgress = {
  completedSubjects: [],
  expandedCategories: [],
  roleFilter: "all",
};

let sCachedRaw: string | null = null;
let sCachedProgress: SubjectProgress = emptySubjectProgress;

function subscribeSubjectStorage(callback: () => void) {
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

function getStoredSubjectProgress(): SubjectProgress {
  try {
    const raw = localStorage.getItem("ml-subjects-progress");
    if (raw === sCachedRaw) return sCachedProgress;
    sCachedRaw = raw;
    sCachedProgress = raw ? JSON.parse(raw) : emptySubjectProgress;
    return sCachedProgress;
  } catch {
    sCachedRaw = null;
    sCachedProgress = emptySubjectProgress;
    return emptySubjectProgress;
  }
}

function getServerSubjectProgress(): SubjectProgress {
  return emptySubjectProgress;
}

function getLevel(completedCount: number, total: number): { level: number; title: string; nextAt: number; color: string } {
  const pct = total > 0 ? (completedCount / total) * 100 : 0;
  if (pct >= 90) return { level: 5, title: "AI Master", nextAt: total, color: "text-purple-400" };
  if (pct >= 70) return { level: 4, title: "AI Expert", nextAt: total, color: "text-amber-400" };
  if (pct >= 50) return { level: 3, title: "AI Practitioner", nextAt: Math.round(total * 0.7), color: "text-cyan-400" };
  if (pct >= 25) return { level: 2, title: "AI Builder", nextAt: Math.round(total * 0.5), color: "text-emerald-400" };
  if (pct >= 5) return { level: 1, title: "AI Explorer", nextAt: Math.round(total * 0.25), color: "text-blue-400" };
  return { level: 0, title: "AI Beginner", nextAt: Math.round(total * 0.05) || 5, color: "text-gray-400" };
}

const allRoles = ["all", "Data Scientist", "ML Engineer", "AI Engineer", "GenAI Engineer"];

export default function LearningTracker() {
  const [searchQuery, setSearchQuery] = useState("");
  const [depthFilter, setDepthFilter] = useState<"all" | "beginner" | "intermediate" | "advanced">("all");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("list");
  const [roleFilter, setRoleFilter] = useState<string>("all");

  useEffect(() => {
    let cancelled = false;
    loadRemoteState("ml_subjects_progress", emptySubjectProgress).then((data) => {
      if (cancelled) return;
      try {
        const raw = JSON.stringify(data);
        localStorage.setItem("ml-subjects-progress", raw);
        sCachedRaw = raw;
        sCachedProgress = data;
        window.dispatchEvent(new StorageEvent("storage"));
      } catch {}
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const storedProgress = useSyncExternalStore(
    subscribeSubjectStorage,
    getStoredSubjectProgress,
    getServerSubjectProgress
  );

  const completedSubjects = useMemo(
    () => new Set(storedProgress.completedSubjects),
    [storedProgress.completedSubjects]
  );
  const expandedCategories = useMemo(
    () => new Set(storedProgress.expandedCategories),
    [storedProgress.expandedCategories]
  );

  const writeProgress = useCallback((data: SubjectProgress) => {
    try {
      localStorage.setItem("ml-subjects-progress", JSON.stringify(data));
      window.dispatchEvent(new StorageEvent("storage"));
    } catch {}
    saveRemoteState("ml_subjects_progress", data);
  }, []);

  const toggleSubjectComplete = useCallback((subjectId: string) => {
    const current = storedProgress.completedSubjects;
    const next = current.includes(subjectId)
      ? current.filter((id) => id !== subjectId)
      : [...current, subjectId];
    writeProgress({ ...storedProgress, completedSubjects: next });
  }, [storedProgress, writeProgress]);

  const toggleCategory = useCallback((categoryId: string) => {
    const current = storedProgress.expandedCategories;
    const next = current.includes(categoryId)
      ? current.filter((id) => id !== categoryId)
      : [...current, categoryId];
    writeProgress({ ...storedProgress, expandedCategories: next });
  }, [storedProgress, writeProgress]);

  const toggleRoleFilter = useCallback((role: string) => {
    setRoleFilter(role);
    writeProgress({ ...storedProgress, roleFilter: role });
  }, [storedProgress, writeProgress]);

  const totalSubjects = learningSubjects.length;
  const completedCount = completedSubjects.size;
  const totalProgress = totalSubjects > 0 ? Math.round((completedCount / totalSubjects) * 100) : 0;
  const levelInfo = getLevel(completedCount, totalSubjects);

  // Filter categories and subjects
  const filteredCategories = useMemo(() => {
    return subjectCategories
      .map((cat) => {
        const catSubjects = getSubjectsByCategorySimple(cat.id);
        const filtered = catSubjects.filter((s) => {
          const matchesSearch =
            cat.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            s.tags.some((t) => t.includes(searchQuery.toLowerCase()));
          const matchesDepth = depthFilter === "all" || s.difficulty === depthFilter;
          const matchesCategory = categoryFilter === "all" || cat.id === categoryFilter;
          const matchesRole = roleFilter === "all" || cat.roleRelevance.includes(roleFilter);
          return matchesSearch && matchesDepth && matchesCategory && matchesRole;
        });
        const catCompleted = filtered.filter((s) => completedSubjects.has(s.id)).length;
        return { ...cat, filteredSubjects: filtered, catCompleted };
      })
      .filter((c) => c.filteredSubjects.length > 0);
  }, [searchQuery, depthFilter, categoryFilter, roleFilter, completedSubjects]);

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
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-[120px]" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-fuchsia-500/5 rounded-full blur-[120px]" />
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
                  <div className="w-9 h-9 sm:w-11 sm:h-11 rounded-xl bg-gradient-to-br from-emerald-500/20 to-fuchsia-500/20 flex items-center justify-center border border-emerald-500/30">
                    <GraduationCap className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-400" />
                  </div>
                </div>
                <div className="min-w-0">
                  <h1 className="text-base sm:text-xl lg:text-2xl font-bold tracking-tight text-white truncate">
                    Learning <span className="text-emerald-400">Tracker</span>
                  </h1>
                  <p className="text-[10px] sm:text-xs text-gray-500 truncate">
                    {totalSubjects} subjects for Data Scientist / ML / AI / GenAI Engineer roles
                  </p>
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
                  <span className="text-xs sm:text-sm text-gray-500">/ {totalSubjects}</span>
                </div>
                <Progress value={totalProgress} className="h-1 bg-white/10 mt-2" />
                <span className="text-[10px] sm:text-xs text-emerald-400 font-mono">{totalProgress}%</span>
              </div>

              {/* Level */}
              <div className="bg-white/[0.04] rounded-xl p-3 sm:p-4 border border-white/10">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-2">
                  <Zap className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-400" />
                  <span className="text-[10px] sm:text-xs text-gray-500">Level</span>
                </div>
                <div className={`text-lg sm:text-2xl font-bold ${levelInfo.color}`}>
                  Lv.{levelInfo.level}
                </div>
                <span className="text-[10px] sm:text-xs text-gray-400">{levelInfo.title}</span>
                <div className="flex items-center gap-1 mt-1">
                  <Star className="w-2.5 h-2.5 text-amber-400 fill-amber-400" />
                  <span className="text-[9px] sm:text-[10px] text-gray-500">Next: {levelInfo.nextAt} subjects</span>
                </div>
              </div>

              {/* Category Progress */}
              <div className="bg-white/[0.04] rounded-xl p-3 sm:p-4 border border-white/10">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-2">
                  <ShieldCheck className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-cyan-400" />
                  <span className="text-[10px] sm:text-xs text-gray-500">Domains</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-lg sm:text-2xl font-bold text-white">
                    {subjectCategories.filter((c) => {
                      const catSubjects = getSubjectsByCategorySimple(c.id);
                      return catSubjects.length > 0 && catSubjects.every((s) => completedSubjects.has(s.id));
                    }).length}
                  </span>
                  <span className="text-xs sm:text-sm text-gray-500">/ {subjectCategories.length}</span>
                </div>
                <span className="text-[10px] sm:text-xs text-gray-400">mastered</span>
              </div>

              {/* Streak */}
              <div className="bg-white/[0.04] rounded-xl p-3 sm:p-4 border border-white/10">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-2">
                  <Flame className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-orange-400" />
                  <span className="text-[10px] sm:text-xs text-gray-500">Topics</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-lg sm:text-2xl font-bold text-white">{completedCount}</span>
                </div>
                <span className="text-[10px] sm:text-xs text-gray-400">topics completed</span>
              </div>
            </div>

            {/* Role Filter */}
            <div className="mb-3">
              <div className="flex items-center gap-1.5 mb-1.5">
                <Target className="w-3 h-3 text-gray-500" />
                <span className="text-[10px] text-gray-500">Filter by Role:</span>
              </div>
              <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10 overflow-x-auto scrollbar-none gap-1">
                {allRoles.map((role) => (
                  <button
                    key={role}
                    onClick={() => toggleRoleFilter(role)}
                    className={`px-2.5 py-1.5 rounded-md text-[10px] sm:text-xs font-medium transition-all whitespace-nowrap shrink-0 ${
                      roleFilter === role
                        ? role === "all"
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                          : roleColors[role] || "bg-white/10 text-gray-300 border border-white/20"
                        : "text-gray-500 hover:text-gray-300 border border-transparent"
                    }`}
                  >
                    {role === "all" ? "All Roles" : role}
                  </button>
                ))}
              </div>
            </div>

            {/* Search & Filters */}
            <div className="flex gap-2 sm:gap-3">
              <div className="relative flex-1 min-w-0">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <Input
                  placeholder="Search subjects, tags..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 sm:pl-10 bg-white/5 border-white/10 text-gray-200 placeholder:text-gray-600 focus:border-emerald-500/50 focus:ring-emerald-500/20 h-9 sm:h-10 rounded-lg font-mono text-xs sm:text-sm"
                />
                {searchQuery && (
                  <button onClick={() => setSearchQuery("")} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {/* Depth filter */}
                <div className="hidden sm:flex bg-white/5 rounded-lg p-0.5 border border-white/10">
                  {(["all", "beginner", "intermediate", "advanced"] as const).map((d) => (
                    <button
                      key={d}
                      onClick={() => setDepthFilter(d)}
                      className={`px-2.5 py-1.5 rounded-md text-[10px] sm:text-xs font-medium transition-all whitespace-nowrap ${
                        depthFilter === d ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" : "text-gray-500 hover:text-gray-300"
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
                {(["all", ...subjectCategories.map((c) => c.id)] as string[]).slice(0, 8).map((cId) => (
                  <button
                    key={cId}
                    onClick={() => setCategoryFilter(cId)}
                    className={`px-2 py-1.5 rounded-md text-[10px] font-medium transition-all whitespace-nowrap shrink-0 ${
                      categoryFilter === cId ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" : "text-gray-500 hover:text-gray-300 border border-transparent"
                    }`}
                  >
                    {cId === "all" ? "All" : subjectCategories.find((c) => c.id === cId)?.name.split(" ")[0] || cId}
                  </button>
                ))}
              </div>
              <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10 overflow-x-auto scrollbar-none mt-2 gap-1">
                {(["all", "beginner", "intermediate", "advanced"] as const).map((d) => (
                  <button
                    key={d}
                    onClick={() => setDepthFilter(d)}
                    className={`px-2.5 py-1.5 rounded-md text-[10px] font-medium transition-all whitespace-nowrap shrink-0 ${
                      depthFilter === d ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" : "text-gray-500 hover:text-gray-300 border border-transparent"
                    }`}
                  >
                    {d === "all" ? "All Levels" : depthLabels[d]}
                  </button>
                ))}
              </div>
            </div>
          </motion.header>

          {/* Category + Subjects */}
          <div className="space-y-3 sm:space-y-4">
            {filteredCategories.map((cat, idx) => (
              <SubjectCategorySection
                key={cat.id}
                category={cat}
                expandedCategories={expandedCategories}
                completedSubjects={completedSubjects}
                index={idx}
                viewMode={viewMode}
                onToggle={() => toggleCategory(cat.id)}
                onToggleSubject={toggleSubjectComplete}
              />
            ))}
          </div>

          {filteredCategories.length === 0 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center py-16 sm:py-20 text-gray-600">
              <Search className="w-10 h-10 sm:w-12 sm:h-12 mb-3 opacity-30" />
              <p className="text-base sm:text-lg">No subjects found</p>
              <p className="text-xs sm:text-sm mt-1">Try a different search, filter, or role</p>
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
              Subjects sourced from{" "}
              <a
                href="https://www.geeksforgeeks.org/machine-learning/machine-learning-tutorial/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-emerald-400 hover:text-emerald-300 underline underline-offset-2"
              >
                GeeksforGeeks
              </a>{" "}
              — Click any subject to open the tutorial
            </p>
          </motion.div>
        </div>
      </div>
    </TooltipProvider>
  );
}

// ── Subject Category Section ──

interface SubjectCategorySectionProps {
  category: SubjectCategory & { filteredSubjects: LearningSubject[]; catCompleted: number };
  expandedCategories: Set<string>;
  completedSubjects: Set<string>;
  index: number;
  viewMode: "grid" | "list";
  onToggle: () => void;
  onToggleSubject: (id: string) => void;
}

function SubjectCategorySection({
  category: cat,
  expandedCategories,
  completedSubjects,
  index,
  viewMode,
  onToggle,
  onToggleSubject,
}: SubjectCategorySectionProps) {
  const isExpanded = expandedCategories.has(cat.id);
  const Icon = iconMap[cat.icon] || BrainCircuit;
  const catProgress = cat.filteredSubjects.length > 0
    ? Math.round((cat.catCompleted / cat.filteredSubjects.length) * 100)
    : 0;
  const isDone = cat.catCompleted === cat.filteredSubjects.length && cat.filteredSubjects.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.04 }}
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
              {cat.filteredSubjects.length} topics
            </span>
            {/* Role badges (shown when expanded) */}
            {isExpanded && (
              <div className="flex gap-1">
                {cat.roleRelevance.map((role) => (
                  <span key={role} className={`text-[8px] sm:text-[9px] px-1.5 py-0.5 rounded-full border ${roleColors[role] || "bg-white/5 border-white/10 text-gray-400"}`}>
                    {role}
                  </span>
                ))}
              </div>
            )}
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
                {cat.filteredSubjects.map((subject, sIdx) => (
                  <SubjectCard
                    key={subject.id}
                    subject={subject}
                    isCompleted={completedSubjects.has(subject.id)}
                    index={sIdx}
                    viewMode={viewMode}
                    onToggle={() => onToggleSubject(subject.id)}
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

// ── Subject Card ──

interface SubjectCardProps {
  subject: LearningSubject;
  isCompleted: boolean;
  index: number;
  viewMode: "grid" | "list";
  onToggle: () => void;
}

function SubjectCard({ subject, isCompleted, index, viewMode, onToggle }: SubjectCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.15, delay: index * 0.015 }}
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

        {/* Subject Info */}
        <a
          href={subject.url}
          target="_blank"
          rel="noopener noreferrer"
          className={`flex-1 text-left min-w-0 transition-colors ${
            isCompleted ? "opacity-70" : ""
          }`}
        >
          <div className="flex items-center gap-1.5 sm:gap-2 flex-wrap">
            <span className={`text-xs sm:text-sm font-medium leading-snug ${isCompleted ? "text-emerald-300 line-through decoration-emerald-500/40" : "text-gray-200"}`}>
              {subject.title}
            </span>
            <ArrowUpRight className="w-3 h-3 text-gray-600 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <div className="flex items-center gap-1 mt-0.5 flex-wrap">
            <span className={`text-[9px] sm:text-[10px] px-1.5 py-0.5 rounded-full border shrink-0 ${depthColors[subject.difficulty]}`}>
              {depthLabels[subject.difficulty]}
            </span>
            {subject.tags.slice(0, viewMode === "grid" ? 2 : 3).map((tag) => (
              <span key={tag} className="text-[9px] sm:text-[10px] px-1.5 py-0.5 rounded-full bg-white/5 border border-white/10 text-gray-500 shrink-0">
                {tag}
              </span>
            ))}
            {subject.tags.length > 3 && viewMode === "list" && (
              <span className="text-[9px] text-gray-600">+{subject.tags.length - 3}</span>
            )}
          </div>
        </a>
      </div>
    </motion.div>
  );
}
