"use client";

import { useEffect, useState } from "react";
import { BarChart3, Bell, CalendarDays, Check, Copy, Download, Flame, History, Pause, Play, Printer, RotateCcw, Star, Timer, Upload } from "lucide-react";
import { loadRemoteState, saveRemoteState } from "@/lib/supabase-state";

type Status = "" | "done" | "partial" | "missed";
type Block = { status: Status; quality: number; note: string };
type Day = {
  wake: string;
  sleep: string;
  gym: string;
  mood: string;
  focus: string;
  followed: string;
  win: string;
  mistake: string;
  tomorrow: string;
  floor: Record<"quant" | "coding" | "gk" | "jobs" | "plan", boolean>;
  blocks: Record<string, Block>;
  ssc: Record<string, string>;
  job: { apps: Array<{ id: string; company: string; role: string; status: string; link: string; gap: string }> };
  code: Record<string, string>;
  gateT: Record<string, string>;
  dop: Record<string, string>;
};
type ResetState = { days: Record<string, Day>; weekly: Record<string, Record<string, string>>; settings: { sscDate: string; gateDate: string } };

const blocks = [
  { id: "quant", name: "SSC Quant", time: "11:00-12:15", mins: 75 },
  { id: "reasoning", name: "SSC Reasoning", time: "12:30-13:15", mins: 45 },
  { id: "mistake", name: "SSC Mistake Review", time: "13:15-14:00", mins: 45 },
  { id: "coding", name: "Coding / DSA", time: "15:30-16:45", mins: 75 },
  { id: "aiml", name: "AI/ML Project", time: "17:00-18:15", mins: 75 },
  { id: "gk", name: "SSC GK", time: "18:30-19:15", mins: 45 },
  { id: "english", name: "SSC English", time: "19:15-20:00", mins: 45 },
  { id: "job", name: "Job Applications", time: "20:00-20:30", mins: 30 },
  { id: "gate", name: "GATE DA/CSE", time: "22:15-23:15", mins: 60 },
  { id: "review", name: "Daily Review & Plan", time: "23:15-23:40", mins: 25 },
];

const empty: ResetState = { days: {}, weekly: {}, settings: { sscDate: "2026-08-15", gateDate: "2027-02-07" } };
const weights: Record<Status, number> = { done: 1, partial: 0.5, missed: 0, "": 0 };

function today() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
function shift(date: string, n: number) {
  const d = new Date(`${date}T12:00:00`);
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}
function blankDay(): Day {
  const blockState: Record<string, Block> = {};
  blocks.forEach((b) => (blockState[b.id] = { status: "", quality: 0, note: "" }));
  return {
    wake: "", sleep: "", gym: "", mood: "", focus: "", followed: "", win: "", mistake: "", tomorrow: "",
    floor: { quant: false, coding: false, gk: false, jobs: false, plan: false },
    blocks: blockState,
    ssc: { topic: "", attempted: "", correct: "", wrong: "", reasoningAcc: "", gk: "", english: "", mock: "", revise: "" },
    job: { apps: [] },
    code: { topic: "", solved: "", difficulty: "", link: "", concept: "", project: "", github: "", learning: "" },
    gateT: { subject: "", topic: "", concept: "", questions: "", accuracy: "", pyq: "", weak: "", revision: "" },
    dop: { manga: "", gaming: "", social: "", relapse: "", preWork: "", withinLimit: "", trigger: "", nextAction: "" },
  };
}
function score(day: Day) {
  const ssc = ["quant", "reasoning", "mistake", "gk", "english"].reduce((n, id) => n + weights[day.blocks[id].status] * 5, 0);
  const job = Math.min(day.job.apps.length / 5, 1) * 15 + weights[day.blocks.job.status] * 10;
  const code = weights[day.blocks.coding.status] * 10 + weights[day.blocks.aiml.status] * 10;
  const gate = weights[day.blocks.gate.status] * 10;
  const health = (day.gym === "yes" ? 7 : 0) + (day.wake && day.wake <= "08:30" ? 3 : 0);
  const dop = (day.dop.withinLimit === "yes" ? 5 : 0) + (day.dop.preWork === "no" ? 3 : 0) + (day.dop.relapse === "no" ? 2 : 0);
  return Math.round(ssc + job + code + gate + health + dop);
}
function tier(s: number) {
  if (s >= 85) return ["Elite Day", "text-emerald-300"];
  if (s >= 70) return ["Strong Day", "text-emerald-300"];
  if (s >= 50) return ["Survival Day", "text-amber-300"];
  return ["Reset Day", "text-red-300"];
}
function monday(date: string) {
  const d = new Date(`${date}T12:00:00`);
  const day = (d.getDay() + 6) % 7;
  d.setDate(d.getDate() - day);
  return d.toISOString().slice(0, 10);
}
function productiveHours(day: Day) {
  const mins = blocks.reduce((total, block) => total + block.mins * weights[day.blocks[block.id].status], 0);
  return Math.round((mins / 60) * 10) / 10;
}
function floorMet(day: Day) {
  return Object.values(day.floor).every(Boolean);
}
function daysLeft(date: string) {
  return Math.max(0, Math.ceil((new Date(`${date}T12:00:00`).getTime() - new Date(`${today()}T12:00:00`).getTime()) / 86400000));
}
function quantAccuracy(day: Day) {
  const attempted = Number(day.ssc.attempted);
  const correct = Number(day.ssc.correct);
  return attempted ? `${Math.round((correct / attempted) * 100)}%` : "-";
}

export default function ResetTrackerApp() {
  const [state, setState] = useState<ResetState>(empty);
  const [loaded, setLoaded] = useState(false);
  const [date, setDate] = useState(today());
  const [tab, setTab] = useState<"today" | "dash" | "week" | "history">("today");
  const [timerId, setTimerId] = useState<string | null>(null);
  const [remaining, setRemaining] = useState(0);
  const [timerPaused, setTimerPaused] = useState(false);
  const [remindersOn, setRemindersOn] = useState(false);
  const [notice, setNotice] = useState("");
  const day = state.days[date] ?? blankDay();
  const sc = score(day);
  const [tierLabel, tierColor] = tier(sc);

  useEffect(() => {
    loadRemoteState("reset_tracker_next", empty).then((value) => {
      setState({ ...empty, ...value });
      setLoaded(true);
    });
  }, []);
  useEffect(() => {
    if (loaded) saveRemoteState("reset_tracker_next", state);
  }, [loaded, state]);
  useEffect(() => {
    if (!timerId || timerPaused) return;
    const t = window.setInterval(() => {
      setRemaining((value) => {
        if (value <= 1) {
          window.clearInterval(t);
          setTimerId(null);
          setBlock(timerId, "status", "done");
          return 0;
        }
        return value - 1;
      });
    }, 1000);
    return () => window.clearInterval(t);
  }, [timerId, timerPaused]);
  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => setNotice(""), 2600);
    return () => window.clearTimeout(t);
  }, [notice]);
  useEffect(() => {
    if (!remindersOn) return;
    if ("Notification" in window && Notification.permission === "default") Notification.requestPermission().catch(() => {});
    const tick = () => {
      const now = new Date();
      const hm = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
      blocks.forEach((block) => {
        const start = block.time.slice(0, 5);
        const key = `reset-reminder:${today()}:${block.id}`;
        if (hm === start && !localStorage.getItem(key)) {
          localStorage.setItem(key, "1");
          showNotice(`Time for ${block.name}`);
          if ("Notification" in window && Notification.permission === "granted") {
            new Notification(`Time for ${block.name}`, { body: `${block.time} - start the first rep now.` });
          }
        }
      });
    };
    tick();
    const t = window.setInterval(tick, 15000);
    return () => window.clearInterval(t);
  }, [remindersOn]);

  function showNotice(message: string) {
    setNotice(message);
  }

  function patchDay(next: Day) {
    setState((prev) => ({ ...prev, days: { ...prev.days, [date]: next } }));
  }
  function field<K extends keyof Day>(key: K, value: Day[K]) {
    patchDay({ ...day, [key]: value });
  }
  function nested(section: keyof Pick<Day, "ssc" | "code" | "gateT" | "dop">, key: string, value: string) {
    patchDay({ ...day, [section]: { ...day[section], [key]: value } });
  }
  function setBlock(id: string, key: keyof Block, value: string | number) {
    patchDay({ ...day, blocks: { ...day.blocks, [id]: { ...day.blocks[id], [key]: value } } });
  }
  function startTimer(id: string) {
    const block = blocks.find((b) => b.id === id);
    if (!block) return;
    setTimerId(id);
    setRemaining(block.mins * 60);
    setTimerPaused(false);
    showNotice(`Timer started: ${block.name}`);
  }
  function addTimerMinutes(minutes: number) {
    setRemaining((value) => Math.max(30, value + minutes * 60));
  }
  function stopTimer() {
    setTimerId(null);
    setTimerPaused(false);
    setRemaining(0);
  }
  function markTimerDone() {
    if (timerId) setBlock(timerId, "status", "done");
    stopTimer();
  }
  function markAll(status: Status) {
    const nextBlocks = Object.fromEntries(Object.entries(day.blocks).map(([id, block]) => [id, { ...block, status }])) as Record<string, Block>;
    patchDay({ ...day, blocks: nextBlocks });
  }
  function copyYesterday() {
    const yesterday = state.days[shift(date, -1)];
    if (!yesterday) {
      showNotice("No data for yesterday");
      return;
    }
    patchDay(JSON.parse(JSON.stringify(yesterday)) as Day);
    showNotice("Copied yesterday into this day");
  }
  function setWeek(key: string, value: string) {
    const weekKey = monday(date);
    setState((prev) => ({ ...prev, weekly: { ...prev.weekly, [weekKey]: { ...(prev.weekly[weekKey] ?? {}), [key]: value } } }));
  }
  function setExamDate(key: "sscDate" | "gateDate", value: string) {
    setState((prev) => ({ ...prev, settings: { ...prev.settings, [key]: value } }));
  }
  function addApp() {
    patchDay({ ...day, job: { apps: [{ id: crypto.randomUUID(), company: "", role: "", status: "Applied", link: "", gap: "" }, ...day.job.apps] } });
  }
  function updateApp(id: string, key: string, value: string) {
    patchDay({ ...day, job: { apps: day.job.apps.map((app) => (app.id === id ? { ...app, [key]: value } : app)) } });
  }
  function removeApp(id: string) {
    patchDay({ ...day, job: { apps: day.job.apps.filter((app) => app.id !== id) } });
  }
  function resetDay() {
    setState((prev) => {
      const days = { ...prev.days };
      delete days[date];
      return { ...prev, days };
    });
  }
  function exportData() {
    const blob = new Blob([JSON.stringify(state, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `reset-tracker-${today()}.json`;
    a.click();
  }
  function importData(file: File | null) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setState(JSON.parse(String(reader.result)));
    reader.readAsText(file);
  }

  const days = Object.keys(state.days).sort();
  const scores = days.slice(-14).map((d) => score(state.days[d]));
  const weekDays = Array.from({ length: 7 }, (_, i) => shift(monday(date), i));
  const weekEntries = weekDays.map((d) => state.days[d]).filter(Boolean);
  const weekNote = state.weekly[monday(date)] ?? {};
  const allDays = Object.values(state.days);
  const totalApps = allDays.reduce((n, d) => n + d.job.apps.length, 0);
  const totalDsa = allDays.reduce((n, d) => n + Number(d.code.solved || 0), 0);
  const totalHours = allDays.reduce((n, d) => n + productiveHours(d), 0);
  const missedBlocks = allDays.reduce((n, d) => n + blocks.filter((block) => d.blocks[block.id].status === "missed").length, 0);

  return (
    <div className="min-h-screen bg-[#14120e] text-[#f0e9dc]">
      <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        <header className="mb-5 flex flex-col gap-4 border-b border-[#3a342a] pb-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-black uppercase tracking-[0.2em] text-orange-300">
              <Flame className="h-4 w-4" />
              Reset
            </div>
            <h1 className="text-3xl font-black tracking-tight text-white sm:text-4xl">Daily Discipline Tracker</h1>
            <p className="mt-2 text-sm text-[#b3a896]">Consistency over motivation. A reset day is not a failure.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button className={`reset-btn ${remindersOn ? "border-orange-400/60 text-orange-200" : ""}`} onClick={() => setRemindersOn((value) => !value)}>
              <Bell className="h-4 w-4" /> Reminders {remindersOn ? "On" : "Off"}
            </button>
            <button className="reset-btn" onClick={exportData}><Download className="h-4 w-4" /> Backup</button>
            <label className="reset-btn cursor-pointer"><Upload className="h-4 w-4" /> Restore<input className="hidden" type="file" accept="application/json" onChange={(e) => importData(e.target.files?.[0] ?? null)} /></label>
          </div>
        </header>
        {notice && <div className="fixed bottom-5 left-1/2 z-50 -translate-x-1/2 rounded-xl border border-orange-400/50 bg-[#2b261e] px-4 py-3 text-sm font-bold text-orange-100 shadow-2xl">{notice}</div>}

        <nav className="mb-5 flex gap-2 overflow-x-auto">
          {([
            ["today", "Today", CalendarDays],
            ["dash", "Dashboard", BarChart3],
            ["week", "Weekly Review", Check],
            ["history", "History", History],
          ] as const).map(([key, label, Icon]) => (
            <button key={key} onClick={() => setTab(key)} className={`reset-tab ${tab === key ? "active" : ""}`}>
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </nav>

        {tab === "today" && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <button className="reset-btn" onClick={() => setDate(shift(date, -1))}>Prev</button>
              <input className="reset-input" type="date" value={date} onChange={(e) => setDate(e.target.value)} />
              <button className="reset-btn" onClick={() => setDate(shift(date, 1))}>Next</button>
              <button className="reset-btn" onClick={() => setDate(today())}>Today</button>
              <button className="reset-btn" onClick={resetDay}><RotateCcw className="h-4 w-4" /> Reset Day</button>
            </div>
            <Panel>
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-xl border border-orange-400/30 bg-orange-500/10 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.14em] text-orange-200">SSC CGL 2026</div>
                  <div className="mt-1 text-3xl font-black text-white">{daysLeft(state.settings.sscDate)} <span className="text-sm text-[#b3a896]">days left</span></div>
                  <input className="reset-input mt-3" type="date" value={state.settings.sscDate} onChange={(e) => setExamDate("sscDate", e.target.value)} />
                </div>
                <div className="rounded-xl border border-orange-400/30 bg-orange-500/10 p-4">
                  <div className="text-xs font-black uppercase tracking-[0.14em] text-orange-200">GATE 2027 DA/CSE</div>
                  <div className="mt-1 text-3xl font-black text-white">{daysLeft(state.settings.gateDate)} <span className="text-sm text-[#b3a896]">days left</span></div>
                  <input className="reset-input mt-3" type="date" value={state.settings.gateDate} onChange={(e) => setExamDate("gateDate", e.target.value)} />
                </div>
              </div>
            </Panel>
            <Panel>
              <div className="grid gap-5 sm:grid-cols-[150px_1fr] sm:items-center">
                <div className="relative grid h-36 w-36 place-items-center rounded-full" style={{ background: `conic-gradient(#f59e3c ${sc}%, #3a342a 0)` }}>
                  <div className="absolute inset-3 rounded-full bg-[#221e18]" />
                  <div className="relative text-center"><div className="text-4xl font-black">{sc}</div><div className="text-xs text-[#7d745f]">/100</div></div>
                </div>
                <div>
                  <h2 className={`text-2xl font-black ${tierColor}`}>{tierLabel}</h2>
                  <p className="mt-1 text-sm text-[#b3a896]">Productive blocks, SSC, jobs, coding, GATE, health, and dopamine control.</p>
                  <div className="mt-4 grid gap-2 sm:grid-cols-3">
                    <Mini label="Jobs" value={String(day.job.apps.length)} />
                    <Mini label="DSA solved" value={day.code.solved || "0"} />
                    <Mini label="Productive hours" value={`${productiveHours(day)}h`} />
                    <Mini label="Quant accuracy" value={quantAccuracy(day)} />
                    <Mini label="Floor" value={floorMet(day) ? "Cleared" : "Open"} />
                  </div>
                </div>
              </div>
            </Panel>
            <Panel title="Bad-day minimum">
              <div className="grid gap-2 sm:grid-cols-5">
                {(["quant", "coding", "gk", "jobs", "plan"] as const).map((key) => (
                  <label key={key} className={`reset-check ${day.floor[key] ? "on" : ""}`}>
                    <input type="checkbox" checked={day.floor[key]} onChange={(e) => field("floor", { ...day.floor, [key]: e.target.checked })} />
                    {key}
                  </label>
                ))}
              </div>
            </Panel>
            <Panel title="Daily log">
              <div className="grid gap-3 sm:grid-cols-3">
                <Field label="Wake" type="time" value={day.wake} onChange={(v) => field("wake", v)} />
                <Field label="Sleep" type="time" value={day.sleep} onChange={(v) => field("sleep", v)} />
                <Select label="Gym" value={day.gym} options={["", "yes", "no"]} onChange={(v) => field("gym", v)} />
                <Field label="Mood 1-10" type="number" value={day.mood} onChange={(v) => field("mood", v)} />
                <Field label="Focus 1-10" type="number" value={day.focus} onChange={(v) => field("focus", v)} />
                <Select label="Followed timetable" value={day.followed} options={["", "Yes", "Partially", "No"]} onChange={(v) => field("followed", v)} />
                <Area label="Biggest win" value={day.win} onChange={(v) => field("win", v)} />
                <Area label="Biggest mistake" value={day.mistake} onChange={(v) => field("mistake", v)} />
                <Area label="Tomorrow first task" value={day.tomorrow} onChange={(v) => field("tomorrow", v)} />
              </div>
            </Panel>
            <Panel title="Study blocks">
              <div className="mb-3 flex flex-wrap gap-2">
                <button className="reset-btn" onClick={copyYesterday}><Copy className="h-4 w-4" /> Copy yesterday</button>
                <button className="reset-btn" onClick={() => markAll("done")}><Check className="h-4 w-4" /> Mark all done</button>
                <button className="reset-btn" onClick={() => markAll("missed")}><RotateCcw className="h-4 w-4" /> Mark all missed</button>
              </div>
              <div className="space-y-2">
                {blocks.map((block) => (
                  <div key={block.id} className={`reset-block ${day.blocks[block.id].status}`}>
                    <div><b>{block.name}</b><div className="text-xs text-[#7d745f]">{block.time} · {block.mins}m</div></div>
                    <button className="reset-btn" onClick={() => timerId === block.id ? stopTimer() : startTimer(block.id)}>
                      <Timer className="h-4 w-4" /> {timerId === block.id ? "Stop" : "Start"}
                    </button>
                    <select className="reset-input" value={day.blocks[block.id].status} onChange={(e) => setBlock(block.id, "status", e.target.value)}>
                      <option value="">status</option><option value="done">Done</option><option value="partial">Partial</option><option value="missed">Missed</option>
                    </select>
                    <div className="flex gap-1">
                      {[1, 2, 3, 4, 5].map((n) => (
                        <button
                          key={n}
                          className={`rounded-lg p-1 ${day.blocks[block.id].quality >= n ? "text-orange-300" : "text-[#3a342a] hover:text-orange-200"}`}
                          onClick={() => setBlock(block.id, "quality", day.blocks[block.id].quality === n ? 0 : n)}
                          title={`Quality ${n}`}
                        >
                          <Star className="h-4 w-4 fill-current" />
                        </button>
                      ))}
                    </div>
                    <input className="reset-input" placeholder="notes / mistakes" value={day.blocks[block.id].note} onChange={(e) => setBlock(block.id, "note", e.target.value)} />
                  </div>
                ))}
              </div>
              {timerId && (
                <div className="mt-3 flex flex-wrap items-center gap-3 rounded-xl border border-orange-400/40 bg-orange-500/10 p-3">
                  <Bell className="h-4 w-4 text-orange-300" />
                  <b>{blocks.find((b) => b.id === timerId)?.name}</b>
                  <span className="font-mono text-xl text-orange-200">{String(Math.floor(remaining / 60)).padStart(2, "0")}:{String(remaining % 60).padStart(2, "0")}</span>
                  <button className="reset-btn" onClick={() => setTimerPaused((value) => !value)}>{timerPaused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}{timerPaused ? "Resume" : "Pause"}</button>
                  <button className="reset-btn" onClick={() => addTimerMinutes(-5)}>-5</button>
                  <button className="reset-btn" onClick={() => addTimerMinutes(5)}>+5</button>
                  <button className="reset-btn" onClick={() => addTimerMinutes(15)}>+15</button>
                  <button className="reset-btn" onClick={markTimerDone}><Check className="h-4 w-4" /> Mark done</button>
                  <button className="reset-btn" onClick={stopTimer}>Stop</button>
                </div>
              )}
            </Panel>
              <Panel title="SSC / Jobs / Coding / GATE / Dopamine">
              <Section title="SSC CGL" fields={[
                ["Quant topic", "ssc", "topic"], ["Attempted", "ssc", "attempted"], ["Correct", "ssc", "correct"], ["Wrong", "ssc", "wrong"], ["Reasoning accuracy %", "ssc", "reasoningAcc"], ["GK topics", "ssc", "gk"], ["English", "ssc", "english"], ["Mock score", "ssc", "mock"], ["Revise", "ssc", "revise"],
              ]} day={day} nested={nested} />
              <p className="mt-2 text-xs font-bold text-orange-200">Quant accuracy: {quantAccuracy(day)}</p>
              <h3 className="reset-section">Job applications</h3>
              <button className="reset-btn mb-2" onClick={addApp}><Check className="h-4 w-4" /> Add Application</button>
              <div className="space-y-2">
                {day.job.apps.map((app) => (
                  <div key={app.id} className="grid gap-2 rounded-xl border border-[#3a342a] bg-[#14120e] p-3 sm:grid-cols-5">
                    <input className="reset-input" placeholder="Company" value={app.company} onChange={(e) => updateApp(app.id, "company", e.target.value)} />
                    <input className="reset-input" placeholder="Role" value={app.role} onChange={(e) => updateApp(app.id, "role", e.target.value)} />
                    <select className="reset-input" value={app.status} onChange={(e) => updateApp(app.id, "status", e.target.value)}>{["Applied", "Screening", "Interview", "Offer", "Rejected"].map((s) => <option key={s}>{s}</option>)}</select>
                    <input className="reset-input" placeholder="Link" value={app.link} onChange={(e) => updateApp(app.id, "link", e.target.value)} />
                    <button className="reset-btn" onClick={() => removeApp(app.id)}>Remove</button>
                    <input className="reset-input sm:col-span-5" placeholder="Resume version / recruiter / gap noticed" value={app.gap} onChange={(e) => updateApp(app.id, "gap", e.target.value)} />
                  </div>
                ))}
              </div>
              <Section title="Coding / DSA / AI-ML" fields={[["DSA topic", "code", "topic"], ["Problems solved", "code", "solved"], ["Difficulty", "code", "difficulty"], ["Problem link", "code", "link"], ["Concept learned", "code", "concept"], ["AI/ML project / dataset", "code", "project"], ["GitHub commit link", "code", "github"], ["Practical learning", "code", "learning"]]} day={day} nested={nested} />
              <Section title="GATE 2027 DA / CSE" fields={[["Subject", "gateT", "subject"], ["Topic", "gateT", "topic"], ["Questions", "gateT", "questions"], ["Accuracy %", "gateT", "accuracy"], ["Concept", "gateT", "concept"], ["PYQ attempted?", "gateT", "pyq"], ["Weak area", "gateT", "weak"], ["Revision date", "gateT", "revision"]]} day={day} nested={nested} />
              <Section title="Dopamine / Distraction" fields={[["Manga min", "dop", "manga"], ["Gaming min", "dop", "gaming"], ["Social min", "dop", "social"], ["Within healthy limit", "dop", "withinLimit"], ["Dopamine before work", "dop", "preWork"], ["Relapse", "dop", "relapse"], ["Trigger", "dop", "trigger"], ["Next action", "dop", "nextAction"]]} day={day} nested={nested} />
            </Panel>
          </div>
        )}

        {tab === "dash" && (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <Panel title="Countdown">
                <div className="grid gap-3 sm:grid-cols-2">
                  <Mini label="SSC CGL 2026" value={`${daysLeft(state.settings.sscDate)}d`} />
                  <Mini label="GATE 2027" value={`${daysLeft(state.settings.gateDate)}d`} />
                </div>
              </Panel>
              <Panel title="Consistency">
                <div className="grid gap-3 sm:grid-cols-2">
                  <Mini label="Floor cleared" value={String(allDays.filter(floorMet).length)} />
                  <Mini label="Missed blocks" value={String(missedBlocks)} />
                </div>
              </Panel>
            </div>
            <div className="grid gap-3 sm:grid-cols-4">
              <Mini label="Days tracked" value={String(days.length)} />
              <Mini label="Average score" value={String(scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 0)} />
              <Mini label="Best score" value={String(scores.length ? Math.max(...scores) : 0)} />
              <Mini label="Applications" value={String(totalApps)} />
              <Mini label="Productive hours" value={`${totalHours.toFixed(1)}h`} />
              <Mini label="DSA solved" value={String(totalDsa)} />
              <Mini label="GitHub commits" value={String(allDays.filter((d) => d.code.github).length)} />
              <Mini label="Avg distraction" value={`${allDays.length ? Math.round(allDays.reduce((n, d) => n + Number(d.dop.manga || 0) + Number(d.dop.gaming || 0) + Number(d.dop.social || 0), 0) / allDays.length) : 0}m`} />
            </div>
            <Panel title="Last 14 scores">
              <div className="flex h-32 items-end gap-2">
                {scores.map((s, i) => <div key={i} className="w-6 rounded-t bg-orange-400" style={{ height: `${Math.max(4, s)}%` }} title={String(s)} />)}
              </div>
            </Panel>
          </div>
        )}

        {tab === "week" && (
          <Panel title={`Weekly Review · ${monday(date)} to ${shift(monday(date), 6)}`}>
            <div className="mb-3 flex justify-end">
              <button className="reset-btn" onClick={() => window.print()}><Printer className="h-4 w-4" /> Print / Save PDF</button>
            </div>
            <div className="grid gap-3 sm:grid-cols-4">
              <Mini label="Days logged" value={`${weekEntries.length}/7`} />
              <Mini label="Productive blocks" value={String(weekEntries.reduce((n, d) => n + blocks.filter((b) => d.blocks[b.id].status).length, 0))} />
              <Mini label="Jobs" value={String(weekEntries.reduce((n, d) => n + d.job.apps.length, 0))} />
              <Mini label="Best score" value={String(weekEntries.length ? Math.max(...weekEntries.map(score)) : 0)} />
              <Mini label="Productive hours" value={`${weekEntries.reduce((n, d) => n + productiveHours(d), 0).toFixed(1)}h`} />
              <Mini label="SSC questions" value={String(weekEntries.reduce((n, d) => n + Number(d.ssc.attempted || 0), 0))} />
              <Mini label="DSA problems" value={String(weekEntries.reduce((n, d) => n + Number(d.code.solved || 0), 0))} />
              <Mini label="GATE blocks" value={String(weekEntries.filter((d) => d.blocks.gate.status).length)} />
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <Field label="Repeated distraction" value={weekNote.distraction ?? ""} onChange={(v) => setWeek("distraction", v)} />
              <Field label="Weak topic" value={weekNote.weak ?? ""} onChange={(v) => setWeek("weak", v)} />
              <Field label="One improvement next week" value={weekNote.improve ?? ""} onChange={(v) => setWeek("improve", v)} />
            </div>
          </Panel>
        )}

        {tab === "history" && (
          <Panel title="History">
            <div className="space-y-2">
              {[...days].reverse().map((d) => (
                <button key={d} onClick={() => { setDate(d); setTab("today"); }} className="flex w-full items-center justify-between rounded-xl border border-[#3a342a] bg-[#14120e] p-3 text-left">
                  <span>{d}</span><b>{score(state.days[d])}</b>
                </button>
              ))}
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}

function Panel({ title, children }: { title?: string; children: React.ReactNode }) {
  return <section className="rounded-2xl border border-[#3a342a] bg-[#221e18] p-4">{title && <h2 className="mb-3 text-lg font-black text-white">{title}</h2>}{children}</section>;
}
function Mini({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-[#3a342a] bg-[#221e18] p-4"><div className="text-xs font-bold uppercase tracking-[0.14em] text-[#7d745f]">{label}</div><div className="mt-1 text-2xl font-black text-white">{value}</div></div>;
}
function Field({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (v: string) => void; type?: string }) {
  return <label className="reset-label">{label}<input className="reset-input" type={type} value={value} onChange={(e) => onChange(e.target.value)} /></label>;
}
function Area({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return <label className="reset-label">{label}<textarea className="reset-input min-h-20" value={value} onChange={(e) => onChange(e.target.value)} /></label>;
}
function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  return <label className="reset-label">{label}<select className="reset-input" value={value} onChange={(e) => onChange(e.target.value)}>{options.map((o) => <option key={o} value={o}>{o || "-"}</option>)}</select></label>;
}
function Section({ title, fields, day, nested }: { title: string; fields: string[][]; day: Day; nested: (section: keyof Pick<Day, "ssc" | "code" | "gateT" | "dop">, key: string, value: string) => void }) {
  return <><h3 className="reset-section">{title}</h3><div className="grid gap-3 sm:grid-cols-3">{fields.map(([label, section, key]) => <Field key={`${section}.${key}`} label={label} value={day[section as keyof Pick<Day, "ssc" | "code" | "gateT" | "dop">][key] ?? ""} onChange={(v) => nested(section as keyof Pick<Day, "ssc" | "code" | "gateT" | "dop">, key, v)} />)}</div></>;
}
