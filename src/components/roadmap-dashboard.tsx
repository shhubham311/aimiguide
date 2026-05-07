"use client";

import { useState, useMemo, useCallback, useSyncExternalStore } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Code2, Sigma, Grid3x3, BarChart3, BrainCircuit, Cpu, Sparkles,
  Wrench, Cloud, ThumbsUp, MessageSquareCode, GitBranch,
  Search, ChevronDown, ChevronRight, Copy, Check, Terminal,
  BookOpen, Zap, Trophy, Filter, X, GraduationCap, ArrowRight, Send,
  LayoutGrid, List, Command as CmdIcon
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { courseModules, generateCommand, generateTeachingPrompt, type Module, type Topic } from "@/lib/course-data";

const iconMap: Record<string, React.ComponentType<{ className?: string; size?: number }>> = {
  Code2, Sigma, Grid3x3, BarChart3, BrainCircuit, Cpu, Sparkles,
  Wrench, Cloud, ThumbsUp, MessageSquareCode, GitBranch,
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

interface ProgressData {
  completedTopics: string[];
  expandedModules: string[];
}

function subscribeToStorage(callback: () => void) {
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

function getStoredProgress(): ProgressData {
  try {
    const stored = localStorage.getItem("roadmap-progress");
    return stored ? JSON.parse(stored) : { completedTopics: [], expandedModules: [] };
  } catch {
    return { completedTopics: [], expandedModules: [] };
  }
}

function getServerProgress(): ProgressData {
  return { completedTopics: [], expandedModules: [] };
}

export default function RoadmapDashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [depthFilter, setDepthFilter] = useState<"all" | "beginner" | "intermediate" | "advanced">("all");

  // Read from localStorage using useSyncExternalStore (no hydration mismatch)
  const storedProgress = useSyncExternalStore(
    subscribeToStorage,
    getStoredProgress,
    getServerProgress
  );

  const completedTopics = useMemo(
    () => new Set(storedProgress.completedTopics),
    [storedProgress.completedTopics]
  );
  const expandedModules = useMemo(
    () => new Set(storedProgress.expandedModules),
    [storedProgress.expandedModules]
  );

  // Trigger a storage event to re-read from localStorage after any write
  const writeProgress = useCallback((data: ProgressData) => {
    try {
      localStorage.setItem("roadmap-progress", JSON.stringify(data));
      // Dispatch a storage event so useSyncExternalStore re-reads
      window.dispatchEvent(new StorageEvent("storage"));
    } catch {
      // localStorage not available
    }
  }, []);

  const totalTopics = courseModules.reduce((sum, m) => sum + m.topics.length, 0);
  const totalProgress = totalTopics > 0 ? Math.round((completedTopics.size / totalTopics) * 100) : 0;

  const toggleModule = useCallback((moduleId: string) => {
    const current = storedProgress.expandedModules;
    const next = current.includes(moduleId)
      ? current.filter((id) => id !== moduleId)
      : [...current, moduleId];
    writeProgress({
      ...storedProgress,
      expandedModules: next,
    });
    if (!current.includes(moduleId)) {
      setSelectedModule(moduleId);
    } else {
      setSelectedModule(null);
    }
  }, [storedProgress, writeProgress]);

  const toggleComplete = useCallback((topicKey: string) => {
    const current = storedProgress.completedTopics;
    const next = current.includes(topicKey)
      ? current.filter((id) => id !== topicKey)
      : [...current, topicKey];
    writeProgress({
      ...storedProgress,
      completedTopics: next,
    });
  }, [storedProgress, writeProgress]);

  const sendCommand = useCallback((moduleId: string, topicId: string, topicTitle: string) => {
    // Find the module and topic to get full context
    const mod = courseModules.find((m) => m.id === moduleId);
    const topic = mod?.topics.find((t) => t.id === topicId);
    const cmd = generateCommand(moduleId, topicId);
    const moduleTitle = mod?.title || "Unknown";
    const topicDescription = topic?.description || "";

    const teachingPrompt = generateTeachingPrompt(
      moduleId, topicId, topicTitle, moduleTitle, topicDescription
    );
    const fullMessage = `${cmd}\n\n${teachingPrompt}`;

    // Send to parent chat window
    try {
      window.parent.postMessage(
        { type: "chat-send", text: fullMessage },
        "*"
      );
    } catch {
      // Silently fail if blocked
    }

    // Also copy to clipboard as fallback
    try {
      navigator.clipboard.writeText(fullMessage).catch(() => {
        const textArea = document.createElement("textarea");
        textArea.value = fullMessage;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand("copy");
        document.body.removeChild(textArea);
      });
    } catch {
      // clipboard not available
    }

    setCopiedId(`${moduleId}.${topicId}`);
    setTimeout(() => setCopiedId(null), 2000);
  }, []);

  // Filter modules based on search and depth filter
  const filteredModules = useMemo(() => {
    return courseModules
      .map((mod) => {
        const moduleMatches =
          mod.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
          mod.description.toLowerCase().includes(searchQuery.toLowerCase());

        const filteredTopics = mod.topics.filter((topic) => {
          const matchesSearch =
            moduleMatches ||
            topic.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            topic.description.toLowerCase().includes(searchQuery.toLowerCase());
          const matchesDepth = depthFilter === "all" || topic.depth === depthFilter;
          return matchesSearch && matchesDepth;
        });

        return { ...mod, topics: filteredTopics, moduleMatches };
      })
      .filter((mod) => mod.topics.length > 0);
  }, [searchQuery, depthFilter]);

  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-screen bg-[#0a0a0f] text-gray-100 font-mono relative overflow-hidden">
        {/* Animated background grid */}
        <div className="fixed inset-0 pointer-events-none">
          <div
            className="absolute inset-0 opacity-[0.03]"
            style={{
              backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
              backgroundSize: "60px 60px",
            }}
          />
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-[120px]" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-violet-500/5 rounded-full blur-[120px]" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/3 rounded-full blur-[150px]" />
        </div>

        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Header */}
          <motion.header
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-8"
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center border border-emerald-500/30">
                    <Terminal className="w-5 h-5 text-emerald-400" />
                  </div>
                  <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-400 rounded-full animate-pulse" />
                </div>
                <div>
                  <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
                    AI/ML <span className="text-emerald-400">Learning</span> Terminal
                  </h1>
                  <p className="text-xs text-gray-500">Select a topic to deep dive — teaching happens in this chat</p>
                </div>
              </div>

              {/* Global Progress */}
              <div className="flex items-center gap-3 bg-white/5 rounded-lg px-4 py-2.5 border border-white/10 w-full sm:w-auto">
                <Trophy className="w-4 h-4 text-amber-400 shrink-0" />
                <div className="flex-1 sm:w-40">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-400">Progress</span>
                    <span className="text-emerald-400 font-bold">{completedTopics.size}/{totalTopics}</span>
                  </div>
                  <Progress value={totalProgress} className="h-1.5 bg-white/10" />
                </div>
                <span className="text-sm font-bold text-emerald-400 tabular-nums">{totalProgress}%</span>
              </div>
            </div>

            {/* Search & Filters Bar */}
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                <Input
                  placeholder="Search topics, modules, concepts..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-white/5 border-white/10 text-gray-200 placeholder:text-gray-600 focus:border-emerald-500/50 focus:ring-emerald-500/20 h-10 rounded-lg font-mono text-sm"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery("")}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              <div className="flex items-center gap-2">
                {/* Depth Filter */}
                <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10">
                  {(["all", "beginner", "intermediate", "advanced"] as const).map((d) => (
                    <button
                      key={d}
                      onClick={() => setDepthFilter(d)}
                      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                        depthFilter === d
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                          : "text-gray-500 hover:text-gray-300"
                      }`}
                    >
                      {d === "all" ? "All" : depthLabels[d]}
                    </button>
                  ))}
                </div>

                {/* View Toggle */}
                <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10">
                  <button
                    onClick={() => setViewMode("grid")}
                    className={`p-1.5 rounded-md transition-all ${viewMode === "grid" ? "bg-white/10 text-white" : "text-gray-500"}`}
                  >
                    <LayoutGrid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode("list")}
                    className={`p-1.5 rounded-md transition-all ${viewMode === "list" ? "bg-white/10 text-white" : "text-gray-500"}`}
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </motion.header>

          {/* Module Stats Bar */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="flex items-center gap-2 mb-6 overflow-x-auto pb-2 scrollbar-none"
          >
            <span className="text-xs text-gray-600 shrink-0">Quick Nav:</span>
            {courseModules.map((mod) => {
              const Icon = iconMap[mod.icon] || Code2;
              const modCompleted = mod.topics.filter((t) => completedTopics.has(`${mod.id}.${t.id}`)).length;
              const isDone = modCompleted === mod.topics.length;
              return (
                <Tooltip key={mod.id}>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => {
                        if (!expandedModules.has(mod.id)) {
                          toggleModule(mod.id);
                        }
                        document.getElementById(`module-${mod.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
                      }}
                      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs whitespace-nowrap transition-all shrink-0 border ${
                        selectedModule === mod.id
                          ? `bg-white/10 border-white/20 text-white`
                          : isDone
                          ? `bg-emerald-500/10 border-emerald-500/20 text-emerald-400`
                          : `bg-white/5 border-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200`
                      }`}
                    >
                      <Icon className="w-3 h-3" />
                      <span className="hidden lg:inline">{mod.number}.</span>
                      <span className="truncate max-w-[100px]">{mod.title}</span>
                      {isDone && <Check className="w-3 h-3 text-emerald-400" />}
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="bg-gray-900 border-gray-700 text-xs">
                    {mod.title} — {modCompleted}/{mod.topics.length} topics
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </motion.div>

          {/* Modules Grid/List */}
          <div className={viewMode === "grid" ? "grid grid-cols-1 lg:grid-cols-2 gap-4" : "flex flex-col gap-3"}>
            {filteredModules.map((mod, idx) => (
              <ModuleCard
                key={mod.id}
                module={mod}
                isExpanded={expandedModules.has(mod.id)}
                completedTopics={completedTopics}
                copiedId={copiedId}
                index={idx}
                viewMode={viewMode}
                onToggle={() => toggleModule(mod.id)}
                onComplete={(topicKey) => toggleComplete(topicKey)}
                onCopy={(modId, topicId, title) => sendCommand(modId, topicId, title)}
              />
            ))}
          </div>

          {filteredModules.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center py-20 text-gray-600"
            >
              <Search className="w-12 h-12 mb-4 opacity-30" />
              <p className="text-lg">No topics found</p>
              <p className="text-sm mt-1">Try a different search or filter</p>
            </motion.div>
          )}

          {/* Terminal Instruction */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="mt-8 mb-4"
          >
            <div className="bg-white/[0.03] rounded-xl border border-white/10 p-5">
              <div className="flex items-center gap-2 mb-3">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500/60" />
                  <div className="w-3 h-3 rounded-full bg-amber-500/60" />
                  <div className="w-3 h-3 rounded-full bg-emerald-500/60" />
                </div>
                <span className="text-xs text-gray-500 font-mono">terminal — how it works</span>
              </div>
              <div className="space-y-2 text-sm font-mono">
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0">$</span>
                  <span className="text-gray-300">1. Browse the roadmap and find a topic you want to learn</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0">$</span>
                  <span className="text-gray-300">2. Click the <span className="text-amber-400">&quot;Send Command&quot;</span> button on any topic</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0">$</span>
                  <span className="text-gray-300">3. Paste the command in this chat window</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-emerald-400 shrink-0">$</span>
                  <span className="text-gray-300">4. Get a full deep-dive: theory, math, code, real datasets, exercises</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="text-amber-400 shrink-0">→</span>
                  <span className="text-gray-400">Example command: <code className="bg-white/10 px-1.5 py-0.5 rounded text-emerald-300">/learn python-programming.numpy</code></span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </TooltipProvider>
  );
}

// ======================== Module Card Component ========================

interface ModuleCardProps {
  module: Module;
  isExpanded: boolean;
  completedTopics: Set<string>;
  copiedId: string | null;
  index: number;
  viewMode: "grid" | "list";
  onToggle: () => void;
  onComplete: (topicKey: string) => void;
  onCopy: (moduleId: string, topicId: string, title: string) => void;
}

function ModuleCard({
  module: mod,
  isExpanded,
  completedTopics,
  copiedId,
  index,
  viewMode,
  onToggle,
  onComplete,
  onCopy,
}: ModuleCardProps) {
  const Icon = iconMap[mod.icon] || Code2;
  const modCompleted = mod.topics.filter((t) => completedTopics.has(`${mod.id}.${t.id}`)).length;
  const modProgress = mod.topics.length > 0 ? Math.round((modCompleted / mod.topics.length) * 100) : 0;

  return (
    <motion.div
      id={`module-${mod.id}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.05 }}
      className={`rounded-xl border transition-all duration-300 ${
        isExpanded
          ? "bg-white/[0.06] border-white/15 shadow-lg " + mod.glowColor
          : "bg-white/[0.03] border-white/8 hover:bg-white/[0.05] hover:border-white/12"
      }`}
    >
      {/* Module Header */}
      <button
        onClick={onToggle}
        className={`w-full p-4 sm:p-5 text-left flex items-center gap-3 sm:gap-4 ${viewMode === "list" ? "flex-row" : ""}`}
      >
        {/* Module Number & Icon */}
        <div className={`relative shrink-0 w-11 h-11 rounded-lg flex items-center justify-center border transition-all ${mod.bgAccent} ${isExpanded ? "border-white/20" : "border-white/10"}`}>
          <Icon className={`w-5 h-5 ${mod.color}`} />
          <span className={`absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full text-[10px] font-bold flex items-center justify-center ${
            modProgress === 100
              ? "bg-emerald-500 text-white"
              : "bg-white/10 text-gray-400 border border-white/20"
          }`}>
            {mod.number}
          </span>
        </div>

        {/* Title & Description */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className={`font-bold text-sm sm:text-base transition-colors ${isExpanded ? "text-white" : "text-gray-200"}`}>
              {mod.title}
            </h2>
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${depthColors[mod.topics[0]?.depth || "beginner"]} opacity-60`}>
              {mod.topics.length} topics
            </span>
          </div>
          <p className={`text-xs mt-0.5 truncate transition-colors ${isExpanded ? "text-gray-400" : "text-gray-600"}`}>
            {mod.description}
          </p>
          {modProgress > 0 && (
            <div className="flex items-center gap-2 mt-2">
              <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-emerald-500 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${modProgress}%` }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                />
              </div>
              <span className="text-[10px] text-emerald-400 font-mono tabular-nums">{modProgress}%</span>
            </div>
          )}
        </div>

        {/* Expand/Collapse Icon */}
        <motion.div
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.3 }}
          className="shrink-0 text-gray-500"
        >
          <ChevronDown className="w-5 h-5" />
        </motion.div>
      </button>

      {/* Topics List */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-4 sm:px-5 pb-4 sm:pb-5">
              <Separator className="bg-white/10 mb-4" />
              <div className="space-y-2">
                {mod.topics.map((topic, topicIdx) => (
                  <TopicItem
                    key={topic.id}
                    module={mod}
                    topic={topic}
                    index={topicIdx}
                    isCompleted={completedTopics.has(`${mod.id}.${topic.id}`)}
                    isCopied={copiedId === `${mod.id}.${topic.id}`}
                    onToggleComplete={() => onComplete(`${mod.id}.${topic.id}`)}
                    onCopy={() => onCopy(mod.id, topic.id, topic.title)}
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

// ======================== Topic Item Component ========================

interface TopicItemProps {
  module: Module;
  topic: Topic;
  index: number;
  isCompleted: boolean;
  isCopied: boolean;
  onToggleComplete: () => void;
  onCopy: () => void;
}

function TopicItem({
  module: mod,
  topic,
  index,
  isCompleted,
  isCopied,
  onToggleComplete,
  onCopy,
}: TopicItemProps) {
  const [showDesc, setShowDesc] = useState(false);
  const cmd = generateCommand(mod.id, topic.id);

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.2, delay: index * 0.03 }}
      className={`group rounded-lg border transition-all duration-200 ${
        isCompleted
          ? "bg-emerald-500/[0.06] border-emerald-500/20"
          : "bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.04] hover:border-white/10"
      }`}
    >
      <div className="flex items-center gap-3 p-3">
        {/* Checkbox */}
        <button
          onClick={onToggleComplete}
          className={`shrink-0 w-5 h-5 rounded-md border-2 flex items-center justify-center transition-all ${
            isCompleted
              ? "bg-emerald-500 border-emerald-500"
              : "border-gray-600 hover:border-gray-400 group-hover:border-gray-500"
          }`}
        >
          {isCompleted && <Check className="w-3 h-3 text-white" strokeWidth={3} />}
        </button>

        {/* Topic Info */}
        <button
          onClick={() => setShowDesc(!showDesc)}
          className="flex-1 text-left min-w-0"
        >
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-sm font-medium transition-colors ${isCompleted ? "text-emerald-300" : "text-gray-200"}`}>
              {topic.title}
            </span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${depthColors[topic.depth]}`}>
              {depthLabels[topic.depth]}
            </span>
          </div>
          <AnimatePresence>
            {showDesc && (
              <motion.p
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="text-xs text-gray-500 mt-1 leading-relaxed overflow-hidden"
              >
                {topic.description}
              </motion.p>
            )}
          </AnimatePresence>
        </button>

        {/* Send Command Button */}
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={onCopy}
              className={`shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-mono transition-all border ${
                isCopied
                  ? "bg-emerald-500/20 border-emerald-500/30 text-emerald-300"
                  : "bg-emerald-500/10 border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 hover:text-emerald-300 hover:border-emerald-500/30"
              }`}
            >
              {isCopied ? (
                <>
                  <Check className="w-3 h-3" />
                  <span className="hidden sm:inline">Sent!</span>
                </>
              ) : (
                <>
                  <Send className="w-3 h-3" />
                  <span className="hidden sm:inline">Send to Chat</span>
                </>
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="top" className="bg-gray-900 border-gray-700 text-xs font-mono max-w-xs">
            Sends &quot;{cmd}&quot; to chat
          </TooltipContent>
        </Tooltip>
      </div>

      {/* Command Preview Bar */}
      {showDesc && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="px-3 pb-3"
        >
          <div className="bg-black/30 rounded-md px-3 py-2 flex items-center gap-2 border border-white/5">
            <span className="text-emerald-500 text-xs">$</span>
            <code className="text-xs text-gray-300 font-mono flex-1 truncate">{cmd}</code>
            <span className="text-[10px] text-gray-600">← paste in chat</span>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
