// ═══════════════════════════════════════════════════════
//  StudyTrack — Complete JavaScript v2
//  Supabase Auth + DB · Pomodoro · Ambient Sounds
//  Command Palette · Focus Mode · Keyboard Shortcuts
//  XP / Levels · Confetti · Sparklines · Heatmap
//  + Offline / localStorage fallback
// ═══════════════════════════════════════════════════════

// ── Supabase Config ──────────────────────────────
const SUPABASE_URL  = 'https://gimzumqhongsksvvjqnr.supabase.co';
const SUPABASE_ANON = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdpbXp1bXFob25nc2tzdnZqcW5yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgyMzM4NDgsImV4cCI6MjA5MzgwOTg0OH0.I1MMkElg2ahLi8NQffBf0Du7qjIzzhDAEiilXc2mfOA';

let sb;
try { sb = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON); }
catch(e) { console.error('Supabase init failed:', e); }

// ── App State ────────────────────────────────────
let CU = null;          // Current user (auth)
let UP = null;          // User profile (DB row)
let isOffline = false;
const SINGLE_USER_ID = "studytrack_single_user";
const SINGLE_USER_NAME = "StudyTrack";
let appStateCache = {};

// Clock-in state
let clockInTime  = null;
let clockInLabel = '';
let clockTimerInt = null;

// Date cursors
let goalDate = todayStr();
let sessDate = todayStr();
let noteDate = todayStr();

// Filter state
let goalsFilter = 'all';  // 'all' or subject_id
let sessionsFilter = 'all';  // 'all' or subject_id

// Calendar
let calYear, calMonth, calSelDate;

// Stats
let statPeriod   = 'weekly';
let chartTime    = null;
let chartGoals   = null;

// Note mood
let selectedMood = 'Good';
let notesSearchQuery = '';
let notesMoodFilter = 'all';

// DB ready flag (for setup banner)
let dbReady = false;

// ── Pomodoro State ───────────────────────────────
let pomoMode        = 'focus';
let pomoRunning     = false;
let pomoSecondsLeft = 25 * 60;
let pomoTotalSeconds= 25 * 60;
let pomoInterval    = null;
let pomoCount       = 0;
let pomoFocusStart  = null;

// ── Settings ─────────────────────────────────────
let settings = { focusMin: 25, shortMin: 5, longMin: 15 };
let weeklyGoal = 600; // minutes (10 hours default)
let reminderSettings = { enabled: false, time: '20:00' };

// ── Subjects System ──────────────────────────────
let subjects = []; // Array of { id, name, color, emoji }
const DEFAULT_SUBJECTS = [
  { id: 'math', name: 'Mathematics', color: '--accent', emoji: '📐' },
  { id: 'physics', name: 'Physics', color: '--violet', emoji: '⚛️' },
  { id: 'chemistry', name: 'Chemistry', color: '--blue', emoji: '🧪' },
  { id: 'english', name: 'English', color: '--green', emoji: '📖' },
  { id: 'history', name: 'History', color: '--amber', emoji: '📜' },
  { id: 'cs', name: 'Computer Science', color: '--cyan', emoji: '💻' },
  { id: 'biology', name: 'Biology', color: '--red', emoji: '🧬' },
  { id: 'other', name: 'Other', color: '--purple', emoji: '📚' }
];
let selectedGoalSubject = 'other';     // Track selected subject for goal modal
let selectedNoteSubject = 'other';     // Track selected subject for note modal
let selectedSessionSubject = 'other';  // Track selected subject for session

function loadSubjects() {
  let loaded = lsGet('subjects');
  if (!loaded || !Array.isArray(loaded) || loaded.length === 0) {
    subjects = [...DEFAULT_SUBJECTS];
    lsSet('subjects', subjects);
  } else {
    subjects = loaded;
  }
  return subjects;
}

async function saveSubjects() {
  lsSet('subjects', subjects);
  if (!isOffline && sb && uid()) {
    try {
      const { error } = await sb.from('subjects').upsert(
        subjects.map(s => ({ id: s.id, user_id: uid(), name: s.name, color: s.color, emoji: s.emoji }))
      );
      if (error) { console.error('saveSubjects:', error); toast('Subject sync failed', 'err'); }
    } catch (e) { console.error('saveSubjects:', e); }
  }
}

// Re-point local goals/sessions/notes off a deleted subject
function reassignSubjectRefs(fromId, toId) {
  ['goals', 'sessions', 'notes'].forEach(key => {
    const arr = lsGetArr(key);
    let changed = false;
    arr.forEach(item => { if (item.subject_id === fromId) { item.subject_id = toId; changed = true; } });
    if (changed) lsSet(key, arr);
  });
}

function getSubjectById(id) {
  return subjects.find(s => s.id === id);
}

function getSubjectName(id) {
  const subj = getSubjectById(id);
  return subj ? subj.name : 'Other';
}

function getSubjectColor(id) {
  const subj = getSubjectById(id);
  return subj ? subj.color : '--text3';
}

function getSubjectEmoji(id) {
  const subj = getSubjectById(id);
  return subj ? subj.emoji : '📚';
}

// ── Subject Picker Component ──────────────────────
function renderSubjectPicker(containerId, selectedSubjectId, onChangeCallback) {
  const container = $(containerId);
  if (!container) return;
  if (typeof onChangeCallback === 'function') window[`onSubjectChange_${containerId}`] = onChangeCallback;
  
  const selected = getSubjectById(selectedSubjectId) || subjects[0];
  const pickerId = `picker_${containerId}`;
  const dropdownId = `dropdown_${containerId}`;
  
  const cssColor = getComputedStyle(document.documentElement).getPropertyValue(selected.color).trim() || '#6366f1';
  
  const html = `
    <div class="subject-picker-wrap">
      <button class="subject-picker-btn" id="${pickerId}">
        <span class="subject-emoji">${selected.emoji}</span>
        <span class="subject-name">${selected.name}</span>
        <svg class="subject-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="subject-picker-dropdown" id="${dropdownId}" style="display:none">
        <div class="subject-picker-list">
          ${subjects.map(s => {
            const sCssColor = getComputedStyle(document.documentElement).getPropertyValue(s.color).trim() || '#6366f1';
            return `<button class="subject-picker-item ${s.id === selectedSubjectId ? 'active' : ''}" onclick="selectSubjectFromPicker('${containerId}', '${s.id}')">
              <span class="subject-dot" style="background-color: ${sCssColor}"></span>
              <span class="subject-emoji">${s.emoji}</span>
              <span>${s.name}</span>
            </button>`;
          }).join('')}
        </div>
      </div>
    </div>
  `;
  
  container.innerHTML = html;
  
  const btn = $(pickerId);
  const dropdown = $(dropdownId);
  
  if (btn) {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const isOpen = dropdown.style.display !== 'none';
      document.querySelectorAll('.subject-picker-dropdown').forEach(d => d.style.display = 'none');
      if (!isOpen) dropdown.style.display = 'block';
    });
  }
  
  document.addEventListener('click', () => {
    if (dropdown) dropdown.style.display = 'none';
  });
}

function selectSubjectFromPicker(containerId, subjectId) {
  const container = $(containerId);
  if (!container) return;
  
  const selected = getSubjectById(subjectId) || subjects[0];
  const btn = container.querySelector('.subject-picker-btn');
  if (btn) {
    btn.innerHTML = `
      <span class="subject-emoji">${selected.emoji}</span>
      <span class="subject-name">${selected.name}</span>
      <svg class="subject-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
    `;
  }
  
  const dropdown = container.querySelector('.subject-picker-dropdown');
  if (dropdown) dropdown.style.display = 'none';
  
  // Trigger callback if exists
  const callback = window[`onSubjectChange_${containerId}`];
  if (callback) callback(subjectId);
}

// ── Ambient Audio ────────────────────────────────
let audioCtx      = null;
let ambientNodes  = {};
let ambientVolume = 0.4;

// ── Command Palette ──────────────────────────────
let cmdItems       = [];
let cmdSelectedIdx = 0;

// ── Heatmap ──────────────────────────────────────
let heatmapYearOffset = 0;
let heatmapTooltipEl  = null;

// ── XP / Level ───────────────────────────────────
function calcXP(profile) {
  if (!profile) return 0;
  return (profile.total_study_minutes || 0) + (profile.total_goals_completed || 0) * 10;
}
function calcLevel(xp) {
  // Level n requires n*500 XP — generous enough to feel progress
  return Math.max(1, Math.floor(xp / 500) + 1);
}
function xpForLevel(lvl) { return (lvl - 1) * 500; }
function xpForNextLevel(lvl) { return lvl * 500; }
function updateXPBar() {
  if (!UP) return;
  const xp    = calcXP(UP);
  const lvl   = calcLevel(xp);
  const curFloor  = xpForLevel(lvl);
  const nextFloor = xpForNextLevel(lvl);
  const pct   = Math.min(100, Math.round((xp - curFloor) / (nextFloor - curFloor) * 100));
  $('sb-level-badge').textContent = `Lv.${lvl}`;
  $('sb-xp-label').textContent    = `${xp - curFloor} / ${nextFloor - curFloor} XP`;
  $('sb-xp-fill').style.width     = pct + '%';
}

// ── localStorage Helpers ─────────────────────────
function lsGet(key)    { try { return JSON.parse(localStorage.getItem('st-' + key)) || null; } catch(e) { return null; } }
function lsSet(key, v) { localStorage.setItem('st-' + key, JSON.stringify(v)); }
function lsGetArr(key) { return lsGet(key) || []; }

// ── Pure Helpers ─────────────────────────────────
function todayStr() {
  const d = new Date();
  return `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}`;
}
function p2(n) { return String(n).padStart(2, '0'); }
function dateShift(str, n) {
  const d = new Date(str + 'T12:00:00');
  d.setDate(d.getDate() + n);
  return `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}`;
}
function fmtDur(m) {
  if (!m || m < 1) return '0m';
  const h = Math.floor(m / 60), mn = Math.floor(m % 60);
  return h ? (mn ? `${h}h ${mn}m` : `${h}h`) : `${mn}m`;
}
function fmtTime(ts) {
  if (!ts) return '';
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function fmtRelative(ts) {
  if (!ts) return '';
  const diff = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (diff <  60)    return 'just now';
  if (diff < 3600)   return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400)  return `${Math.floor(diff / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}
function fmtDayLabel(str) {
  const today = todayStr(), yest = dateShift(today, -1), tmrw = dateShift(today, 1);
  if (str === today) return 'Today';
  if (str === yest)  return 'Yesterday';
  if (str === tmrw)  return 'Tomorrow';
  const [y, m, d] = str.split('-').map(Number);
  return new Date(y, m-1, d).toLocaleDateString([], { weekday: 'long' });
}
function fmtDateSub(str) {
  const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const [y, m, d] = str.split('-').map(Number);
  return `${MONTHS[m-1]} ${d}, ${y}`;
}
function avatarBg(uid) {
  const cols = ['#6366f1','#8b5cf6','#3b82f6','#22c55e','#f59e0b','#ef4444','#06b6d4','#ec4899'];
  return cols[(uid || '').split('').reduce((a, c) => a + c.charCodeAt(0), 0) % cols.length];
}
function initials(name) { return (name || '?').charAt(0).toUpperCase(); }
function uid()  { return CU ? CU.id : SINGLE_USER_ID; }
function $(id)  { return document.getElementById(id); }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function genId(prefix) { return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2,7)}`; }

// ── Toast ─────────────────────────────────────────
function toast(msg, type = 'ok') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icon = type === 'ok'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>';
  el.innerHTML = `${icon}<span>${escHtml(msg)}</span>`;
  $('toast-wrap').appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ── Page / View Navigation ────────────────────────
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  $(id).classList.add('active');
}
function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  $('view-' + id).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === id));
  document.querySelectorAll('.mob-nav-item').forEach(n => n.classList.toggle('active', n.dataset.view === id));
  closeNavDrawer();
  if (id === 'dashboard')   renderDashboardCharts();
  if (id === 'calendar')    renderCalendar();
  if (id === 'subjects')    renderSubjectsPage();
  if (id === 'stats')       renderStats();
  if (id === 'airoadmap')   renderAiRoadmap();
  if (id === 'aiprojects')  renderAiProjects();
  if (id === 'ailearning')  renderAiLearning();
  if (id === 'resettracker') renderResetTracker();
}

function toggleNavDrawer() {
  document.body.classList.toggle('drawer-open');
  const btn = document.querySelector('.mobile-menu-btn');
  if (btn) btn.setAttribute('aria-expanded', document.body.classList.contains('drawer-open') ? 'true' : 'false');
}

function closeNavDrawer() {
  document.body.classList.remove('drawer-open');
  const btn = document.querySelector('.mobile-menu-btn');
  if (btn) btn.setAttribute('aria-expanded', 'false');
}

document.addEventListener('keydown', function(event) {
  if (event.key === 'Escape') closeNavDrawer();
});

function openModal(id) {
  $(id).classList.add('open');
  // Render subject pickers when modals open
  if (id === 'modal-goal') {
    selectedGoalSubject = subjects[0]?.id || 'other';
    renderSubjectPicker('goal-subject-picker', selectedGoalSubject, function(subjectId) { selectedGoalSubject = subjectId; });
  } else if (id === 'modal-note') {
    selectedNoteSubject = subjects[0]?.id || 'other';
    renderSubjectPicker('note-subject-picker', selectedNoteSubject, function(subjectId) { selectedNoteSubject = subjectId; });
  }
}
function closeModal(id) { $(id).classList.remove('open'); }

// ── Theme ─────────────────────────────────────────
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  localStorage.setItem('st-theme', isDark ? 'light' : 'dark');
  // Re-render all charts after theme change
  if (typeof renderDashboardCharts === 'function') renderDashboardCharts();
  if (typeof renderStatsCharts === 'function') renderStatsCharts();
  if (typeof renderSessionsChart === 'function') renderSessionsChart();
}
function initTheme() {
  const saved = localStorage.getItem('st-theme');
  document.documentElement.setAttribute('data-theme', saved || 'dark');
}

// ── Database Setup / SQL Download ────────────────
async function checkDatabase() {
  if (isOffline) { dbReady = true; $('setup-banner').style.display = 'none'; return; }
  try {
    const profileCheck = await sb.from('profiles').select('id').limit(1);
    const stateCheck = profileCheck.error ? { error: null } : await sb.from('app_state').select('id').limit(1);
    if ((profileCheck.error && profileCheck.error.code === 'PGRST205') || (stateCheck.error && stateCheck.error.code === 'PGRST205')) {
      dbReady = false; $('setup-banner').style.display = 'block';
    } else {
      dbReady = true; $('setup-banner').style.display = 'none';
    }
  } catch(e) { dbReady = false; $('setup-banner').style.display = 'block'; }
}

async function setupDatabase() {
  const SQL = `
-- StudyTrack Database Setup
CREATE TABLE IF NOT EXISTS profiles (
  id TEXT PRIMARY KEY,
  display_name TEXT,
  username TEXT UNIQUE,
  email TEXT,
  current_streak INT DEFAULT 0,
  longest_streak INT DEFAULT 0,
  last_streak_date TEXT DEFAULT '',
  total_study_minutes INT DEFAULT 0,
  today_study_minutes INT DEFAULT 0,
  week_study_minutes INT DEFAULT 0,
  month_study_minutes INT DEFAULT 0,
  total_goals_completed INT DEFAULT 0,
  total_goals_added INT DEFAULT 0,
  last_active_date TEXT DEFAULT '',
  active_session JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_username ON profiles(username) WHERE username IS NOT NULL;
CREATE TABLE IF NOT EXISTS goals (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  date TEXT NOT NULL,
  text TEXT NOT NULL,
  completed BOOLEAN DEFAULT false,
  subject_id TEXT DEFAULT 'other',
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  date TEXT NOT NULL,
  clock_in TIMESTAMPTZ,
  clock_out TIMESTAMPTZ,
  duration_minutes INT DEFAULT 0,
  label TEXT DEFAULT 'Study session',
  subject_id TEXT DEFAULT 'other',
  mood TEXT DEFAULT 'Good',
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  date TEXT NOT NULL,
  content TEXT NOT NULL,
  mood TEXT DEFAULT 'Good',
  subject_id TEXT DEFAULT 'other',
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  date TEXT NOT NULL,
  time TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS activity (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  display_name TEXT,
  username TEXT,
  type TEXT,
  detail TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS reminders (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  time TEXT NOT NULL,
  subject_id TEXT DEFAULT 'other',
  message TEXT,
  enabled BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS subjects (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  emoji TEXT DEFAULT '📚',
  color TEXT DEFAULT '--blue',
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS app_state (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  value JSONB DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, key)
);
CREATE TABLE IF NOT EXISTS community_messages (
  id TEXT PRIMARY KEY,
  user_id TEXT REFERENCES profiles(id) ON DELETE CASCADE,
  display_name TEXT,
  username TEXT,
  body TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS username TEXT UNIQUE;
ALTER TABLE activity ADD COLUMN IF NOT EXISTS username TEXT;
ALTER TABLE goals ADD COLUMN IF NOT EXISTS subject_id TEXT DEFAULT 'other';
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS subject_id TEXT DEFAULT 'other';
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS mood TEXT DEFAULT 'Good';
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS notes TEXT;
ALTER TABLE notes ADD COLUMN IF NOT EXISTS subject_id TEXT DEFAULT 'other';
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE goals    ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE notes    ENABLE ROW LEVEL SECURITY;
ALTER TABLE events   ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity ENABLE ROW LEVEL SECURITY;
ALTER TABLE reminders ENABLE ROW LEVEL SECURITY;
ALTER TABLE subjects ENABLE ROW LEVEL SECURITY;
ALTER TABLE community_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_state ENABLE ROW LEVEL SECURITY;
DO $$ BEGIN
  CREATE POLICY "profiles_all" ON profiles FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "goals_all"    ON goals    FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "sessions_all" ON sessions FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "notes_all"    ON notes    FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "events_all"   ON events   FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "activity_all" ON activity FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "reminders_all" ON reminders FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "subjects_all" ON subjects FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "community_messages_all" ON community_messages FOR ALL USING (true) WITH CHECK (true);
  CREATE POLICY "app_state_all" ON app_state FOR ALL USING (true) WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
CREATE INDEX IF NOT EXISTS idx_goals_user_date    ON goals(user_id, date);
CREATE INDEX IF NOT EXISTS idx_sessions_user_date ON sessions(user_id, date);
CREATE INDEX IF NOT EXISTS idx_notes_user_date    ON notes(user_id, date);
CREATE INDEX IF NOT EXISTS idx_events_user_date   ON events(user_id, date);
CREATE INDEX IF NOT EXISTS idx_reminders_user_time ON reminders(user_id, time);
CREATE INDEX IF NOT EXISTS idx_subjects_user ON subjects(user_id);
CREATE INDEX IF NOT EXISTS idx_community_messages_created ON community_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_app_state_user_key ON app_state(user_id, key);
CREATE INDEX IF NOT EXISTS idx_activity_created   ON activity(created_at DESC);
`.trim();
  toast('Downloading SQL setup file...');
  const blob = new Blob([SQL], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'supabase_setup.sql'; a.click();
  URL.revokeObjectURL(url);
  window.open('https://supabase.com/dashboard/project/gimzumqhongsksvvjqnr/sql/new', '_blank');
}

// ── Auth ──────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.auth-tab').forEach((t, i) =>
    t.classList.toggle('active', (i === 0 && tab === 'login') || (i === 1 && tab === 'register')));
  $('form-login').style.display    = tab === 'login'    ? 'block' : 'none';
  $('form-register').style.display = tab === 'register' ? 'block' : 'none';
  $('auth-err').classList.remove('show');
}
function showAuthErr(m) { const e = $('auth-err'); e.textContent = m; e.classList.add('show'); }

async function loginEmail() {
  const em = $('l-email').value.trim(), ps = $('l-pass').value;
  if (!em || !ps) { showAuthErr('Fill in all fields'); return; }
  $('l-btn').disabled = true;
  const { data, error } = await sb.auth.signInWithPassword({ email: em, password: ps });
  $('l-btn').disabled = false;
  if (error) { showAuthErr(error.message); return; }
  CU = { id: data.user.id, displayName: data.user.user_metadata?.display_name || data.user.email, email: data.user.email };
  showPage('page-app'); loadApp();
}

async function registerEmail() {
  const nm = $('r-name').value.trim();
  const un = $('r-username').value.trim().replace(/[^a-zA-Z0-9_]/g, '');
  const em = $('r-email').value.trim();
  const ps = $('r-pass').value;
  if (!nm || !em || !ps) { showAuthErr('Fill in all fields'); return; }
  if (ps.length < 6)     { showAuthErr('Password must be at least 6 characters'); return; }
  if (un && un.length < 3) { showAuthErr('Username must be at least 3 characters'); return; }
  if (un) {
    const { data: existing } = await sb.from('profiles').select('id').eq('username', un).limit(1);
    if (existing && existing.length > 0) { showAuthErr('Username is already taken'); return; }
  }
  $('r-btn').disabled = true;
  const { data, error } = await sb.auth.signUp({ email: em, password: ps, options: { data: { display_name: nm } } });
  $('r-btn').disabled = false;
  if (error) { showAuthErr(error.message); return; }
  const userId = data.user?.id || data.session?.user?.id;
  if (userId) {
    const profileData = { id: userId, display_name: nm, email: em };
    if (un) profileData.username = un;
    await sb.from('profiles').insert(profileData);
    CU = { id: userId, displayName: nm, email: em };
    showPage('page-app'); loadApp(); toast('Account created! Welcome ' + nm);
  } else {
    showAuthErr('Check your email to confirm your account, then sign in.');
  }
}

function loginOffline() {
  isOffline = true;
  CU = { id: 'local_user', displayName: 'Local User', email: '' };
  localStorage.setItem('st-offline', 'true');
  let profile = lsGet('profile');
  if (!profile) {
    profile = {
      id: 'local_user', display_name: 'Local User', email: '', username: null,
      current_streak: 0, longest_streak: 0, last_streak_date: '',
      total_study_minutes: 0, today_study_minutes: 0, week_study_minutes: 0,
      month_study_minutes: 0, total_goals_completed: 0, total_goals_added: 0,
      last_active_date: '', active_session: null
    };
    lsSet('profile', profile);
  }
  showPage('page-app'); loadApp(); toast('Running in Local Mode — data stays on this device');
}

async function logout() {
  toast('This merged build is single-user. No sign out needed.');
  return;
  if (clockInTime) { if (!confirm('You are clocked in. Clock out and sign out?')) return; await clockOut(); }
  stopPomo(); stopAllAmbient();
  if (communityChatChannel && sb) { try { await sb.removeChannel(communityChatChannel); } catch(e) {} communityChatChannel = null; }
  if (midnightCheckInterval) clearInterval(midnightCheckInterval);
  if (isOffline) {
    isOffline = false; localStorage.removeItem('st-offline');
    CU = null; UP = null; showPage('page-auth'); return;
  }
  await sb.auth.signOut(); CU = null; UP = null; showPage('page-auth');
}

// ── Forgot Password ──────────────────────────────
function openForgotPasswordModal(e) {
  e.preventDefault();
  $('forgot-email').value = '';
  $('forgot-err').style.display = 'none';
  $('forgot-success').style.display = 'none';
  openModal('modal-forgot-password');
}

async function sendPasswordReset() {
  const email = $('forgot-email').value.trim();
  const errEl = $('forgot-email').previousElementSibling ? $('forgot-err') : $('forgot-err');
  errEl.style.display = 'none';
  
  if (!email) {
    $('forgot-err').textContent = 'Please enter your email address';
    $('forgot-err').style.display = 'block';
    return;
  }
  
  if (!sb) {
    $('forgot-err').textContent = 'Password reset is only available with Supabase authentication';
    $('forgot-err').style.display = 'block';
    return;
  }
  
  try {
    const { error } = await sb.auth.resetPasswordForEmail(email, {
      redirectTo: window.location.origin
    });
    if (error) {
      $('forgot-err').textContent = error.message || 'Failed to send reset email';
      $('forgot-err').style.display = 'block';
    } else {
      $('forgot-success').textContent = 'Check your email for a password reset link!';
      $('forgot-success').style.display = 'block';
      setTimeout(() => closeModal('modal-forgot-password'), 3000);
    }
  } catch(e) {
    $('forgot-err').textContent = 'An error occurred. Please try again.';
    $('forgot-err').style.display = 'block';
  }
}

if (sb) sb.auth.onAuthStateChange(async (event, session) => {
  if (isOffline) return;
  if (session?.user) {
    CU = { id: session.user.id, displayName: session.user.user_metadata?.display_name || session.user.email, email: session.user.email };
    showPage('page-app'); loadApp();
  } else if (!CU) { showPage('page-auth'); }
});

// ── Load App ──────────────────────────────────────
let lastResetDate = todayStr();
let midnightCheckInterval = null;

async function checkAndResetDaily() {
  const today = todayStr();
  if (today !== lastResetDate) {
    lastResetDate = today;
    if (UP) {
      UP.today_study_minutes = 0;
      if (isOffline) { lsSet('profile', UP); }
      else { await sb.from('profiles').update({ today_study_minutes: 0 }).eq('id', uid()); }
    }
    updateDashStats();
    renderDashGoals();
    setGreeting();
  }
}

function startMidnightCheck() {
  // Clear any existing interval
  if (midnightCheckInterval) clearInterval(midnightCheckInterval);
  
  // Check every minute for date change (lightweight check)
  midnightCheckInterval = setInterval(checkAndResetDaily, 60000);
  
  // Also run once immediately to set baseline
  checkAndResetDaily();
}

async function loadApp() {
  if (isOffline) {
    $('setup-banner').style.display = 'none';
    UP = lsGet('profile') || {
      id: uid(), display_name: CU.displayName, email: CU.email, username: null,
      current_streak: 0, longest_streak: 0, last_streak_date: '',
      total_study_minutes: 0, today_study_minutes: 0, week_study_minutes: 0,
      month_study_minutes: 0, total_goals_completed: 0, total_goals_added: 0,
      last_active_date: '', active_session: null
    };
    lsSet('profile', UP);
  } else {
    await checkDatabase();
    const { data } = await sb.from('profiles').select('*').eq('id', uid()).single();
    UP = data || { id: uid(), display_name: CU.displayName, email: CU.email, username: null };
    if (!data) await sb.from('profiles').insert({ id: uid(), display_name: CU.displayName, email: CU.email, username: null });
  }

  // Restore active session
  if (UP.active_session?.clockIn) {
    clockInTime  = new Date(UP.active_session.clockIn).getTime();
    clockInLabel = UP.active_session.label || '';
  }

  // Reset daily minutes if it's a new day
  const today = todayStr();
  if (UP.last_active_date && UP.last_active_date !== today) {
    UP.today_study_minutes = 0;
    if (isOffline) { lsSet('profile', UP); }
    else { await sb.from('profiles').update({ today_study_minutes: 0 }).eq('id', uid()); }
  }

  loadSubjects();
  if (!isOffline && dbReady) {
    try {
      const { data: remoteSubjects } = await sb.from('subjects').select('*').eq('user_id', uid());
      if (remoteSubjects && remoteSubjects.length) {
        const merged = [...subjects];
        remoteSubjects.forEach(rs => {
          if (!merged.some(s => s.id === rs.id)) merged.push({ id: rs.id, name: rs.name, color: rs.color || '--accent', emoji: rs.emoji || '📚' });
        });
        subjects = merged;
        saveSubjects();
      } else {
        await sb.from('subjects').upsert(subjects.map(s => ({ ...s, user_id: uid() })));
      }
    } catch(e) {}
  }
  loadSettings();
  selectedSessionSubject = UP.active_session?.subject_id || selectedSessionSubject || subjects[0]?.id || 'other';
  selectedGoalSubject = subjects[0]?.id || 'other';
  selectedNoteSubject = subjects[0]?.id || 'other';
  renderSubjectPicker('sess-subject-picker', selectedSessionSubject, subjectId => { selectedSessionSubject = subjectId; });
  renderSubjectPicker('dash-clock-subject-picker', selectedSessionSubject, subjectId => { selectedSessionSubject = subjectId; });
  renderNotesMoodFilter();
  loadReminderSettings();
  initSubjectColorPickers();
  setGreeting();
  updateClockUI();
  updateDashStats();
  updateGoalsDateUI();
  updateSessDateUI();
  updateNotesDateUI();

  const n = new Date();
  calYear = n.getFullYear(); calMonth = n.getMonth(); calSelDate = today;
  $('event-date').value = today;

  renderDashGoals();
  renderDashboardCharts();
  renderActivity();
  renderSessList();
  renderNotesList();
  renderPomoTally();
  buildCommandItems();
  updateXPBar();

  // Start midnight check to automatically reset daily stats
  startMidnightCheck();

  startReminderCheck();

  // Hide loading screen
  setTimeout(() => {
    const loading = $('page-loading');
    if (loading) { loading.classList.add('hidden'); setTimeout(() => loading.remove(), 400); }
  }, 400);
}

// ── Greeting / Profile ────────────────────────────
function setGreeting() {
  const h = new Date().getHours();
  const g = h < 5 ? 'Good night' : h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : h < 21 ? 'Good evening' : 'Good night';
  const firstName = (CU.displayName || 'there').split(' ')[0];
  $('d-greeting').textContent = `${g}, ${firstName}`;
  $('d-date').textContent = new Date().toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });

  // Contextual sub-greeting
  const hour = new Date().getHours();
  const subs = hour < 9
    ? "Early bird gets the worm. Let's go!"
    : hour < 12 ? "Make the most of your morning!"
    : hour < 14 ? "Afternoon focus session incoming?"
    : hour < 17 ? "Keep the momentum going!"
    : hour < 20 ? "Evening study time. You got this!"
    : "Night owl mode activated.";
  $('d-greeting-sub').textContent = subs;

  $('sb-name').textContent     = CU.displayName || 'User';
  const un = UP?.username;
  $('sb-username').textContent = un ? '@' + un : 'Set a username';
  const av = $('sb-avatar');
  av.style.background = avatarBg(CU.id);
  av.textContent = initials(CU.displayName);
  const mobAv = $('mob-profile-btn');
  if (mobAv) { mobAv.style.background = avatarBg(CU.id); mobAv.textContent = initials(CU.displayName); }
  $('sb-offline-badge').style.display = isOffline ? 'inline-flex' : 'none';
}

function openProfileModal() {
  $('profile-name').value         = CU.displayName || '';
  $('profile-username').value     = UP?.username || '';
  $('profile-email').value        = CU.email || '';
  $('profile-current-pass').value = '';
  $('profile-new-pass').value     = '';
  $('profile-confirm-pass').value = '';
  $('profile-err').style.display  = 'none';
  openModal('modal-profile');
}

async function saveProfile() {
  const errEl   = $('profile-err'); errEl.style.display = 'none';
  const newName = $('profile-name').value.trim();
  if (!newName) { errEl.textContent = 'Display name is required.'; errEl.style.display = 'block'; return; }

  if (isOffline) {
    CU.displayName = newName;
    if (UP) { UP.display_name = newName; lsSet('profile', UP); }
    setGreeting(); closeModal('modal-profile'); toast('Profile updated');
    return;
  }

  if (uid() === SINGLE_USER_ID) {
    CU.displayName = newName;
    if (UP) { UP.display_name = newName; UP.username = $('profile-username').value.trim() || null; }
    if (!isOffline && sb && dbReady) await sb.from('profiles').upsert({ id: uid(), display_name: newName, email: '', username: UP?.username || null });
    else lsSet('profile', UP);
    setGreeting(); closeModal('modal-profile'); toast('Profile updated');
    return;
  }
  const newUsername = $('profile-username').value.trim().replace(/[^a-zA-Z0-9_]/g, '');
  const currentPass = $('profile-current-pass').value;
  const newPass     = $('profile-new-pass').value;
  const confirmPass = $('profile-confirm-pass').value;

  if (newUsername && newUsername.length < 3) {
    errEl.textContent = 'Username must be at least 3 characters.'; errEl.style.display = 'block'; return;
  }
  if (newUsername && newUsername !== (UP?.username || '')) {
    const { data: existing } = await sb.from('profiles').select('id').eq('username', newUsername).limit(1);
    if (existing && existing.length > 0) {
      errEl.textContent = 'Username is already taken.'; errEl.style.display = 'block'; return;
    }
  }
  if (currentPass || newPass || confirmPass) {
    if (!currentPass) { errEl.textContent = 'Enter your current password to change it.'; errEl.style.display = 'block'; return; }
    if (newPass.length < 6) { errEl.textContent = 'New password must be at least 6 characters.'; errEl.style.display = 'block'; return; }
    if (newPass !== confirmPass) { errEl.textContent = 'New passwords do not match.'; errEl.style.display = 'block'; return; }
    const { error: reAuthErr } = await sb.auth.signInWithPassword({ email: CU.email, password: currentPass });
    if (reAuthErr) { errEl.textContent = 'Current password is incorrect.'; errEl.style.display = 'block'; return; }
    const { error: passErr } = await sb.auth.updateUser({ password: newPass });
    if (passErr) { errEl.textContent = 'Password update failed: ' + passErr.message; errEl.style.display = 'block'; return; }
  }
  const updates = { display_name: newName, username: newUsername || null };
  await sb.from('profiles').update(updates).eq('id', uid());
  await sb.auth.updateUser({ data: { display_name: newName } });
  CU.displayName = newName;
  if (UP) { UP.display_name = newName; UP.username = newUsername || null; }
  setGreeting(); closeModal('modal-profile'); toast('Profile updated');
}

// ── Dashboard Stats ───────────────────────────────
function updateDashStats() {
  if (!UP) return;
  const streak = UP.current_streak || 0;
  $('d-streak-badge').textContent = `${streak} day streak`;
  const badge = $('d-streak-badge-wrap');
  badge.classList.toggle('active-streak', streak > 0);

  let todayMin = UP.today_study_minutes || 0;
  if (clockInTime) todayMin += Math.floor((Date.now() - clockInTime) / 60000);
  $('d-today-val').textContent = fmtDur(todayMin);
  $('d-week-val').textContent  = fmtDur(UP.week_study_minutes || 0);

  const rate = UP.total_goals_added > 0
    ? Math.round((UP.total_goals_completed / UP.total_goals_added) * 100) : 0;
  $('d-rate-val').textContent = `${rate}%`;

  // Sparklines
  drawSparkline('spark-goals', [0,0,0,0,0,0,0], 'accent');
  drawSparkline('spark-time',  [0,0,0,0,0,0,todayMin], 'green');
  drawSparkline('spark-week',  [0,0,0,0,0,0, UP.week_study_minutes||0], 'violet');
  drawSparkline('spark-rate',  [0,0,0,0,0,0,rate], 'amber');
  updateXPBar();
  updateWeeklyGoalBanner();
}

function drawSparkline(id, data, colorName) {
  const el = $(id); if (!el) return;
  const w = 80, h = 24, max = Math.max(...data, 1);
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - (v / max) * (h - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  const colorMap = { accent:'var(--accent)', green:'var(--green)', violet:'var(--violet)', amber:'var(--amber)' };
  el.innerHTML = `<polyline points="${pts}" fill="none" stroke="${colorMap[colorName]||'var(--accent)'}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>`;
}

// ── Clock In / Out ────────────────────────────────
function updateClockUI() {
  const label  = $('d-clock-label');
  const title  = $('d-clock-title');
  const timer  = $('d-timer');
  const btn    = $('d-clock-btn');
  const banner = $('d-clock-banner');
  if (clockInTime) {
    label.textContent = 'CURRENTLY STUDYING';
    title.textContent = clockInLabel || 'Study session';
    timer.style.display = 'block';
    clearInterval(clockTimerInt);
    clockTimerInt = setInterval(() => {
      const el = Math.max(0, Date.now() - clockInTime);
      const h = Math.floor(el / 3600000), m = Math.floor((el % 3600000) / 60000), s = Math.floor((el % 60000) / 1000);
      timer.textContent = `${p2(h)}:${p2(m)}:${p2(s)}`;
    }, 1000);
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Clock Out';
    btn.classList.add('clocked');
    banner.classList.add('active');
  } else {
    label.textContent = 'READY TO FOCUS?';
    title.textContent = 'Start your session';
    timer.style.display = 'none';
    clearInterval(clockTimerInt);
    timer.textContent = '00:00:00';
    btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg> Clock In';
    btn.classList.remove('clocked');
    banner.classList.remove('active');
  }
}

async function toggleClockDash() { if (clockInTime) await clockOut(); else await clockIn(); }

async function clockIn() {
  clockInLabel = $('sess-label').value.trim() || 'Study session';
  selectedSessionSubject = selectedSessionSubject || subjects[0]?.id || 'other';
  clockInTime  = Date.now();
  const today  = todayStr();
  const updates = { active_session: { clockIn: new Date(clockInTime).toISOString(), label: clockInLabel, subject_id: selectedSessionSubject }, last_active_date: today };
  if (UP.last_active_date !== today) updates.today_study_minutes = 0;

  if (isOffline) {
    Object.assign(UP, updates); lsSet('profile', UP);
    pushActivity('clockin', 'Clocked in');
  } else {
    await sb.from('profiles').update(updates).eq('id', uid());
    await sb.from('activity').insert({ id: genId('act'), user_id: uid(), display_name: CU.displayName, username: UP?.username || null, type: 'clockin', detail: 'Clocked in' });
  }
  updateClockUI(); renderActivity(); toast('Clocked in! Stay focused 🎯');
}

async function clockOut() {
  if (!clockInTime) return;
  await openSessionFeedback(clockInTime, Date.now(), clockInLabel, selectedSessionSubject);
}

// ── Activity Feed ─────────────────────────────────
function pushActivity(type, detail) {
  const activityArr = lsGetArr('activity');
  activityArr.unshift({ id: genId('act'), user_id: uid(), display_name: CU.displayName, username: UP?.username || null, type, detail, created_at: new Date().toISOString() });
  lsSet('activity', activityArr.slice(0, 100));
}

async function renderActivity() {
  let data;
  if (isOffline) {
    data = lsGetArr('activity').sort((a, b) => new Date(b.created_at) - new Date(a.created_at)).slice(0, 12);
  } else {
    const result = await sb.from('activity').select('*').order('created_at', { ascending: false }).limit(12);
    data = result.data || [];
  }
  const el = $('d-activity-list');
  if (!data || !data.length) { el.innerHTML = '<div class="goal-empty">No activity yet</div>'; return; }
  el.innerHTML = data.map(a => {
    const isMe    = a.user_id === uid();
    const showName = isMe ? (a.display_name || 'You') : (a.username ? '@' + a.username : (a.display_name || 'Someone'));
    const dotClass = a.type || 'note';
    return `<div class="activity-item">
      <div class="activity-dot ${dotClass}"></div>
      <div class="activity-text"><strong>${escHtml(showName)}</strong> ${escHtml(a.detail || '')}</div>
      <div class="activity-time">${fmtRelative(a.created_at)}</div>
    </div>`;
  }).join('');
}

// ── Goals ─────────────────────────────────────────
function updateGoalsDateUI() {
  const d = goalDate, today = todayStr();
  $('goals-date-label').textContent = fmtDayLabel(d);
  $('goals-date-sub').textContent   = fmtDateSub(d);
  $('goals-today-btn').style.display = d === today ? 'none' : 'block';
  renderGoalsFilterPills();
  renderGoalsPage();
}
function goalDateNav(n) { goalDate = dateShift(goalDate, n); updateGoalsDateUI(); }
function goalGoToday() { goalDate = todayStr(); updateGoalsDateUI(); }

async function renderDashGoals() {
  let items;
  if (isOffline) { items = lsGetArr('goals').filter(g => g.user_id === uid() && g.date === todayStr()); }
  else { const { data } = await sb.from('goals').select('*').eq('user_id', uid()).eq('date', todayStr()); items = data || []; }

  const done = items.filter(g => g.completed).length;
  $('d-goals-val').textContent = `${done}/${items.length}`;
  $('d-goals-sub').textContent = items.length ? `${done} of ${items.length} completed` : 'No goals yet';

  const el = $('d-goal-list');
  if (!items.length) {
    el.innerHTML = '<div class="goal-empty">No goals for today. <a href="#" onclick="showView(\'goals\');return false">Add one</a></div>';
    return;
  }
  el.innerHTML = items.slice(0, 6).map(g => `
    <div class="goal-item">
      <button class="goal-check ${g.completed ? 'done' : ''}" onclick="toggleGoal('${g.id}')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
      </button>
      <span style="font-size:1.2em;margin:0 6px">${getSubjectEmoji(g.subject_id)}</span>
      <div class="goal-text ${g.completed ? 'done' : ''}">${escHtml(g.text)}</div>
    </div>`).join('');
}

async function renderGoalsPage() {
  let items;
  if (isOffline) { items = lsGetArr('goals').filter(g => g.user_id === uid() && g.date === goalDate); }
  else { const { data } = await sb.from('goals').select('*').eq('user_id', uid()).eq('date', goalDate); items = data || []; }

  // Apply subject filter
  if (goalsFilter !== 'all') {
    items = items.filter(g => g.subject_id === goalsFilter);
  }

  const el = $('goals-list-wrap');
  if (!items.length) {
    el.innerHTML = `<div style="text-align:center;padding:48px 0;color:var(--text3)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="40" height="40" style="display:block;margin:0 auto 12px;opacity:.4"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
      No goals for this day — add one to get started
    </div>`;
    return;
  }
  el.innerHTML = items.map(g => `
    <div class="goal-card ${g.completed ? '' : ''}">
      <button class="goal-check ${g.completed ? 'done' : ''}" onclick="toggleGoal('${g.id}')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
      </button>
      <span style="font-size:1.3em;margin-right:8px">${getSubjectEmoji(g.subject_id)}</span>
      <div class="goal-text ${g.completed ? 'done' : ''}" style="flex:1">${escHtml(g.text)}</div>
      <div class="goal-card-actions">
        <button class="goal-action-btn" onclick="deleteGoal('${g.id}')" title="Delete">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
      </div>
    </div>`).join('');
}

async function addGoal() {
  const text = $('goal-text').value.trim();
  if (!text) { toast('Enter a goal', 'err'); return; }
  const id = genId('g');

  if (isOffline) {
    const arr = lsGetArr('goals');
    arr.push({ id, user_id: uid(), date: goalDate, text, completed: false, subject_id: selectedGoalSubject, created_at: new Date().toISOString() });
    lsSet('goals', arr);
    UP.total_goals_added = (UP.total_goals_added || 0) + 1;
    lsSet('profile', UP);
    pushActivity('goal', `Added: ${text.slice(0, 40)}`);
  } else {
    await sb.from('goals').insert({ id, user_id: uid(), date: goalDate, text, completed: false, subject_id: selectedGoalSubject });
    await sb.from('profiles').update({ total_goals_added: (UP.total_goals_added || 0) + 1 }).eq('id', uid());
    UP.total_goals_added = (UP.total_goals_added || 0) + 1;
    await sb.from('activity').insert({ id: genId('act'), user_id: uid(), display_name: CU.displayName, username: UP?.username || null, type: 'goal', detail: `Added: ${text.slice(0, 40)}` });
  }
  $('goal-text').value = ''; closeModal('modal-goal');
  selectedGoalSubject = subjects[0]?.id || 'other';
  renderGoalsPage(); renderDashGoals(); updateDashStats(); renderActivity();
  toast('Goal added');
}

async function toggleGoal(id) {
  if (isOffline) {
    const arr = lsGetArr('goals');
    const goal = arr.find(g => g.id === id); if (!goal) return;
    const wasCompleted = goal.completed;
    goal.completed = !wasCompleted;
    lsSet('goals', arr);
    if (!wasCompleted) {
      UP.total_goals_completed = (UP.total_goals_completed || 0) + 1;
      pushActivity('goal', `Completed: ${goal.text.slice(0, 40)}`);
      triggerConfetti();
    } else {
      UP.total_goals_completed = Math.max(0, (UP.total_goals_completed || 0) - 1);
    }
    lsSet('profile', UP);
    renderGoalsPage(); renderDashGoals(); updateDashStats(); renderActivity(); updateXPBar();
    return;
  }

  const { data } = await sb.from('goals').select('*').eq('id', id).single();
  if (!data) return;
  const wasCompleted = data.completed;
  await sb.from('goals').update({ completed: !wasCompleted }).eq('id', id);
  if (!wasCompleted) {
    UP.total_goals_completed = (UP.total_goals_completed || 0) + 1;
    await sb.from('profiles').update({ total_goals_completed: UP.total_goals_completed }).eq('id', uid());
    await sb.from('activity').insert({ id: genId('act'), user_id: uid(), display_name: CU.displayName, username: UP?.username || null, type: 'goal', detail: `Completed: ${data.text.slice(0, 40)}` });
    triggerConfetti();
  } else {
    UP.total_goals_completed = Math.max(0, (UP.total_goals_completed || 0) - 1);
    await sb.from('profiles').update({ total_goals_completed: UP.total_goals_completed }).eq('id', uid());
  }
  renderGoalsPage(); renderDashGoals(); updateDashStats(); renderActivity(); updateXPBar();
}

async function deleteGoal(id) {
  if (isOffline) {
    lsSet('goals', lsGetArr('goals').filter(g => g.id !== id));
  } else {
    await sb.from('goals').delete().eq('id', id);
  }
  renderGoalsPage(); renderDashGoals(); updateDashStats();
}

// ── Sessions ──────────────────────────────────────
async function updateSessDateUI() {
  const d = sessDate, today = todayStr();
  $('sess-date-label').textContent  = fmtDayLabel(d);
  $('sess-date-sub').textContent    = fmtDateSub(d);
  $('sess-today-btn').style.display = d === today ? 'none' : 'block';
  renderSessionsFilterPills();
  await renderSessList();
  await renderSessionsChart();
}
function sessDateNav(n) { sessDate = dateShift(sessDate, n); updateSessDateUI(); }
function sessGoToday()  { sessDate = todayStr(); updateSessDateUI(); }

async function renderSessList() {
  let sessions;
  if (isOffline) {
    sessions = lsGetArr('sessions').filter(s => s.user_id === uid() && s.date === sessDate).sort((a, b) => new Date(b.clock_in) - new Date(a.clock_in));
  } else {
    const { data } = await sb.from('sessions').select('*').eq('user_id', uid()).eq('date', sessDate).order('clock_in', { ascending: false });
    sessions = data || [];
  }

  // Apply subject filter
  if (sessionsFilter !== 'all') {
    sessions = sessions.filter(s => s.subject_id === sessionsFilter);
  }

  const total = sessions.reduce((a, s) => a + (s.duration_minutes || 0), 0);
  $('sess-total').textContent = fmtDur(total);
  const el = $('sess-list');
  if (!sessions.length) { el.innerHTML = '<div class="session-empty">No sessions for this day</div>'; renderSessionsChart(); return; }
  el.innerHTML = sessions.map(s => `
    <div class="session-item">
      <div class="session-dot"></div>
      <div style="flex:1;min-width:0">
        <div class="session-dur">${fmtDur(s.duration_minutes)}</div>
        <div class="session-label"><span style="font-size:1.1em;margin-right:6px">${getSubjectEmoji(s.subject_id)}</span>${escHtml(s.label || 'Study session')}</div>
        <div class="session-time">${fmtTime(s.clock_in)} — ${fmtTime(s.clock_out)}</div>
      </div>
      <button class="session-del" onclick="deleteSession('${s.id}')" title="Delete">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
      </button>
    </div>`).join('');
  renderSessionsChart();
}

async function deleteSession(id) {
  if (isOffline) {
    const arr = lsGetArr('sessions');
    const sess = arr.find(s => s.id === id);
    if (sess) {
      const dur = sess.duration_minutes || 0;
      UP.total_study_minutes = Math.max(0, (UP.total_study_minutes || 0) - dur);
      lsSet('profile', UP);
    }
    lsSet('sessions', arr.filter(s => s.id !== id));
  } else {
    await sb.from('sessions').delete().eq('id', id);
  }
  renderSessList(); updateDashStats(); updateXPBar();
}

// ── Notes ─────────────────────────────────────────
function updateNotesDateUI() {
  const d = noteDate, today = todayStr();
  $('notes-date-label').textContent  = fmtDayLabel(d);
  $('notes-date-sub').textContent    = fmtDateSub(d);
  $('notes-today-btn').style.display = d === today ? 'none' : 'block';
  // Clear search when navigating dates
  const searchInput = document.getElementById('notes-search-input');
  if (searchInput) { searchInput.value = ''; notesSearchQuery = ''; }
  renderNotesList();
}
function noteDateNav(n) { noteDate = dateShift(noteDate, n); updateNotesDateUI(); }
function noteGoToday()  { noteDate = todayStr(); updateNotesDateUI(); }

function selectMood(btn, mood) {
  document.querySelectorAll('.mood-opt').forEach(b => b.classList.remove('active'));
  btn.classList.add('active'); selectedMood = mood;
}

async function renderNotesList() {
  let notes;
  if (isOffline) {
    notes = lsGetArr('notes').filter(n => n.user_id === uid() && n.date === noteDate).sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  } else {
    const { data } = await sb.from('notes').select('*').eq('user_id', uid()).eq('date', noteDate).order('created_at', { ascending: false });
    notes = data || [];
  }
  // Search filter for notes
  if (notesSearchQuery && notesSearchQuery.trim()) {
    const query = notesSearchQuery.toLowerCase();
    notes = notes.filter(n => {
      const contentMatch = (n.content && n.content.toLowerCase().includes(query));
      const moodMatch = (n.mood && n.mood.toLowerCase().includes(query));
      const subjName = subjects.find(s => s.id === n.subject_id)?.name?.toLowerCase() || '';
      const subjMatch = subjName.includes(query);
      return contentMatch || moodMatch || subjMatch;
    });
  }
  if (notesMoodFilter !== 'all') notes = notes.filter(n => (n.mood || 'Good') === notesMoodFilter);
  const el = $('notes-list');
  if (!notes.length) {
    el.innerHTML = '<div class="goal-empty" style="padding:40px 0;text-align:center">No notes for this day</div>';
    return;
  }
  el.innerHTML = notes.map(n => `
    <div class="note-card">
      <div class="note-card-header">
        <div style="display:flex;align-items:center;gap:8px">
          <div class="note-mood ${escHtml(n.mood || 'Good')}">
            <div class="note-mood-dot"></div>
            ${escHtml(n.mood || 'Good')}
          </div>
          <span style="font-size:1.1em">${getSubjectEmoji(n.subject_id)}</span>
        </div>
        <div class="note-time">${fmtTime(n.created_at)}</div>
      </div>
      <div class="note-body">${escHtml(n.content)}</div>
      <div class="note-actions">
        <button class="btn-del-note" onclick="deleteNote('${n.id}')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
          Delete
        </button>
      </div>
    </div>`).join('');
}

async function addNote() {
  const text = $('note-text').value.trim();
  if (!text) { toast('Write something', 'err'); return; }
  const id = genId('n');
  const entry = { id, user_id: uid(), date: noteDate, content: text, mood: selectedMood, subject_id: selectedNoteSubject, created_at: new Date().toISOString() };

  if (isOffline) {
    const arr = lsGetArr('notes'); arr.push(entry); lsSet('notes', arr);
    pushActivity('note', 'Added a note');
  } else {
    await sb.from('notes').insert({ id, user_id: uid(), date: noteDate, content: text, mood: selectedMood, subject_id: selectedNoteSubject });
    await sb.from('activity').insert({ id: genId('act'), user_id: uid(), display_name: CU.displayName, username: UP?.username || null, type: 'note', detail: 'Added a note' });
  }
  $('note-text').value = ''; closeModal('modal-note');
  selectedNoteSubject = subjects[0]?.id || 'other';
  renderNotesList(); renderActivity(); toast('Note saved');
}

function filterNotesBySearch() {
  notesSearchQuery = (document.getElementById('notes-search-input')?.value || '').toLowerCase().trim();
  renderNotesList();
}

function renderNotesMoodFilter() {
  const el = $('notes-mood-filter');
  if (!el) return;
  const moods = ['Good','Great','Okay','Tired','Stressed'];
  el.innerHTML = `<button class="filter-pill ${notesMoodFilter === 'all' ? 'active' : ''}" onclick="setNotesMoodFilter('all')">All Moods</button>` +
    moods.map(m => `<button class="filter-pill ${notesMoodFilter === m ? 'active' : ''}" onclick="setNotesMoodFilter('${m}')">${m}</button>`).join('');
}

function setNotesMoodFilter(mood) {
  notesMoodFilter = mood;
  renderNotesMoodFilter();
  renderNotesList();
}

async function deleteNote(id) {
  if (isOffline) { lsSet('notes', lsGetArr('notes').filter(n => n.id !== id)); }
  else { await sb.from('notes').delete().eq('id', id); }
  renderNotesList();
}

// ── Calendar ──────────────────────────────────────
async function renderCalendar() {
  const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  $('cal-month-label').textContent = `${MONTHS[calMonth]} ${calYear}`;
  const mStr = `${calYear}-${p2(calMonth+1)}`;

  let events;
  if (isOffline) {
    events = lsGetArr('events').filter(e => e.user_id === uid() && e.date.startsWith(mStr));
  } else {
    const { data } = await sb.from('events').select('*').eq('user_id', uid()).like('date', mStr + '%');
    events = data || [];
  }

  const evMap = {};
  events.forEach(e => { (evMap[e.date] = evMap[e.date] || []).push(e); });

  const firstDay   = new Date(calYear, calMonth, 1).getDay();
  const offset     = (firstDay + 6) % 7;
  const daysInMonth = new Date(calYear, calMonth + 1, 0).getDate();
  const prevDays   = new Date(calYear, calMonth, 0).getDate();
  const today      = todayStr();
  let html = '';

  for (let i = 0; i < offset; i++) html += `<div class="cal-cell other-month"><span>${prevDays - offset + i + 1}</span></div>`;
  for (let d = 1; d <= daysInMonth; d++) {
    const ds     = `${calYear}-${p2(calMonth+1)}-${p2(d)}`;
    const isTd   = ds === today;
    const isSel  = ds === calSelDate;
    const hasEvt = !!(evMap[ds]?.length);
    html += `<div class="cal-cell ${isTd ? 'today' : ''} ${isSel && !isTd ? 'selected' : ''} ${hasEvt ? 'has-events' : ''}" onclick="selectCalDay('${ds}')"><span>${d}</span></div>`;
  }
  const totalCells = offset + daysInMonth;
  const rem        = (Math.ceil(totalCells / 7) * 7) - totalCells;
  for (let d = 1; d <= rem; d++) html += `<div class="cal-cell other-month"><span>${d}</span></div>`;
  $('cal-grid').innerHTML = html;
  selectCalDay(calSelDate, true);
}

function calNav(n) {
  calMonth += n;
  if (calMonth > 11) { calMonth = 0; calYear++; }
  if (calMonth < 0)  { calMonth = 11; calYear--; }
  renderCalendar();
}

async function selectCalDay(ds, silent = false) {
  calSelDate = ds;
  if (!silent) renderCalendar();
  const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const [y, m, d] = ds.split('-').map(Number);
  $('cal-sel-label').textContent = `${MONTHS[m-1]} ${d}, ${y}`;
  $('event-date').value = ds;

  let evts;
  if (isOffline) { evts = lsGetArr('events').filter(e => e.user_id === uid() && e.date === ds); }
  else { const { data } = await sb.from('events').select('*').eq('user_id', uid()).eq('date', ds); evts = data || []; }

  const el = $('cal-events-list');
  if (!evts.length) { el.innerHTML = '<div class="cal-no-events">No events</div>'; return; }
  el.innerHTML = evts.map(e => `
    <div class="event-item">
      <div class="event-dot"></div>
      <div style="flex:1">
        <div class="event-title">${escHtml(e.title)}</div>
        ${e.time ? `<div class="event-time">${e.time}</div>` : ''}
      </div>
      <button class="goal-action-btn" style="opacity:1;margin-left:auto" onclick="deleteEvent('${e.id}')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
      </button>
    </div>`).join('');
}

async function addEvent() {
  const title = $('event-title').value.trim();
  const date  = $('event-date').value;
  const time  = $('event-time').value;
  if (!title || !date) { toast('Fill in title and date', 'err'); return; }
  const id = genId('ev');

  if (isOffline) {
    const arr = lsGetArr('events'); arr.push({ id, user_id: uid(), title, date, time: time || null }); lsSet('events', arr);
  } else {
    await sb.from('events').insert({ id, user_id: uid(), title, date, time: time || null });
  }
  $('event-title').value = ''; $('event-time').value = '';
  closeModal('modal-event'); renderCalendar(); toast('Event added');
}

async function deleteEvent(id) {
  if (isOffline) { lsSet('events', lsGetArr('events').filter(e => e.id !== id)); }
  else { await sb.from('events').delete().eq('id', id); }
  renderCalendar();
}

// ── Statistics ────────────────────────────────────
function updateStatsUI() {
  if (!UP) return;
  $('st-streak').textContent   = `${UP.current_streak || 0} days`;
  $('st-total').textContent    = fmtDur(UP.total_study_minutes || 0);
  $('st-goals').textContent    = UP.total_goals_completed || 0;
  const rate = UP.total_goals_added > 0 ? Math.round((UP.total_goals_completed / UP.total_goals_added) * 100) : 0;
  $('st-rate').textContent     = `${rate}%`;
  $('st2-streak').textContent  = `${UP.current_streak || 0} days`;
  $('st2-longest').textContent = `${UP.longest_streak || 0} days`;
  $('tb-today').textContent    = fmtDur(UP.today_study_minutes || 0);
  $('tb-week').textContent     = fmtDur(UP.week_study_minutes  || 0);
  $('tb-month').textContent    = fmtDur(UP.month_study_minutes || 0);
  $('tb-all').textContent      = fmtDur(UP.total_study_minutes || 0);
}

function setPeriod(p, btn) {
  statPeriod = p;
  document.querySelectorAll('.period-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  renderStats();
}

// ── Subjects Page ─────────────────────────────
async function renderSubjectsPage() {
  let allGoals, allSessions;
  if (isOffline) {
    allGoals = lsGetArr('goals').filter(g => g.user_id === uid());
    allSessions = lsGetArr('sessions').filter(s => s.user_id === uid());
  } else {
    const [{ data: g }, { data: s }] = await Promise.all([
      sb.from('goals').select('*').eq('user_id', uid()),
      sb.from('sessions').select('*').eq('user_id', uid())
    ]);
    allGoals = g || [];
    allSessions = s || [];
  }

  const el = $('subjects-grid');
  if (!el) return;

  const html = subjects.map(subj => {
    const subjGoals = allGoals.filter(g => g.subject_id === subj.id).length;
    const subjSessions = allSessions.filter(s => s.subject_id === subj.id);
    const totalMinutes = subjSessions.reduce((a, s) => a + (s.duration_minutes || 0), 0);
    const sessCount = subjSessions.length;

    const last7 = [];
    for (let i = 6; i >= 0; i--) {
      const ds = dateShift(todayStr(), -i);
      last7.push(subjSessions.filter(s => s.date === ds).reduce((a, s) => a + (s.duration_minutes || 0), 0));
    }
    const max = Math.max(...last7, 1);
    const pts = last7.map((v, i) => `${(i / 6) * 160},${32 - (v / max) * 28}`).join(' ');
    return `
      <div class="subject-card">
        <div class="subject-card-head" style="background:${subjectColorValue(subj)}"><span>${subj.emoji}</span><span>${escHtml(subj.name)}</span></div>
        <div class="subject-card-stats">
          <div class="subject-stat">
            <span class="subject-stat-label">Study Time:</span>
            <span class="subject-stat-value">${fmtDur(totalMinutes)}</span>
          </div>
          <div class="subject-stat">
            <span class="subject-stat-label">Goals:</span>
            <span class="subject-stat-value">${subjGoals}</span>
          </div>
          <div class="subject-stat">
            <span class="subject-stat-label">Sessions:</span>
            <span class="subject-stat-value">${sessCount}</span>
          </div>
        </div>
        <svg class="subject-card-sparkline" viewBox="0 0 160 34" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="${subjectColorValue(subj)}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <div class="subject-card-actions">
          <button onclick="editSubjectDetails('${subj.id}')">Edit</button>
          <button class="danger" onclick="deleteSubject('${subj.id}')">Delete</button>
        </div>
      </div>
    `;
  }).join('');

  el.innerHTML = html;
}

async function renderStats() {
  updateStatsUI(); renderHeatmap(); await renderStatsCharts();
  const today = new Date();
  let sessions, allGoals = [];

  if (isOffline) {
    sessions = lsGetArr('sessions').filter(s => s.user_id === uid());
    allGoals = lsGetArr('goals').filter(g => g.user_id === uid());
  } else {
    const { data } = await sb.from('sessions').select('*').eq('user_id', uid());
    sessions = data || [];
    const { data: gData } = await sb.from('goals').select('*').eq('user_id', uid());
    allGoals = gData || [];
  }

  let labels = [], sessData = [], goalsData = [];
  if (statPeriod === 'daily') {
    for (let i = 6; i >= 0; i--) {
      const d  = new Date(today); d.setDate(today.getDate() - i);
      const ds = `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}`;
      labels.push(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][d.getDay()]);
      sessData.push(sessions.filter(s => s.date === ds).reduce((a, s) => a + (s.duration_minutes || 0), 0));
      goalsData.push(allGoals.filter(g => g.date === ds && g.completed).length);
    }
  } else if (statPeriod === 'weekly') {
    for (let i = 6; i >= 0; i--) {
      const wEnd   = new Date(today); wEnd.setDate(today.getDate() - i * 7);
      const wStart = new Date(wEnd);  wStart.setDate(wEnd.getDate() - 6);
      const s = `${wStart.getFullYear()}-${p2(wStart.getMonth()+1)}-${p2(wStart.getDate())}`;
      const e = `${wEnd.getFullYear()}-${p2(wEnd.getMonth()+1)}-${p2(wEnd.getDate())}`;
      labels.push(`${p2(wStart.getMonth()+1)}/${p2(wStart.getDate())}`);
      sessData.push(sessions.filter(s2 => s2.date >= s && s2.date <= e).reduce((a, s2) => a + (s2.duration_minutes || 0), 0));
      goalsData.push(allGoals.filter(g => g.date >= s && g.date <= e && g.completed).length);
    }
  } else {
    const MN = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    for (let i = 5; i >= 0; i--) {
      const d    = new Date(today.getFullYear(), today.getMonth() - i, 1);
      const mStr = `${d.getFullYear()}-${p2(d.getMonth()+1)}`;
      labels.push(MN[d.getMonth()]);
      sessData.push(sessions.filter(s2 => (s2.date || '').startsWith(mStr)).reduce((a, s2) => a + (s2.duration_minutes || 0), 0));
      goalsData.push(allGoals.filter(g => (g.date || '').startsWith(mStr) && g.completed).length);
    }
  }

  const isDark      = document.documentElement.getAttribute('data-theme') === 'dark';
  const gridColor   = chartGridColor();
  const tickColor   = isDark ? 'rgba(228,228,240,.48)' : 'rgba(90,90,122,.58)';
  const tooltipBg   = isDark ? '#14142c' : '#fff';
  const tooltipTitle = isDark ? '#e4e4f0' : '#111';
  const tooltipBody  = isDark ? '#9494b4' : '#666';
  const barColor    = isDark ? '#818cf8' : '#6366f1';
  const barColor2   = isDark ? '#8b5cf6' : '#7c3aed';

  const chartCfg = (data, lbl, color) => ({
    type: 'bar',
    data: { labels, datasets: [{ data, backgroundColor: color, borderRadius: 7, borderSkipped: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: c => ` ${c.parsed.y} ${lbl}` },
          backgroundColor: tooltipBg, titleColor: tooltipTitle, bodyColor: tooltipBody,
          borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)', borderWidth: 1,
          cornerRadius: 8, padding: 10
        }
      },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 11 } } },
        y: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 11 } }, beginAtZero: true }
      }
    }
  });

  if (chartTime)  chartTime.destroy();
  if (chartGoals) chartGoals.destroy();
  chartTime  = new Chart($('chart-time').getContext('2d'),  chartCfg(sessData,  'minutes', barColor));
  chartGoals = new Chart($('chart-goals').getContext('2d'), chartCfg(goalsData, 'goals',   barColor2));
}

// ── Activity Heatmap (GitHub-style) ──────────────
function heatmapYearNav(dir) {
  heatmapYearOffset = Math.max(-5, Math.min(0, heatmapYearOffset + dir));
  renderHeatmap();
}

async function renderHeatmap() {
  let sessions;
  if (isOffline) {
    sessions = lsGetArr('sessions').filter(s => s.user_id === uid()).map(s => ({ date: s.date, duration_minutes: s.duration_minutes }));
  } else {
    const { data } = await sb.from('sessions').select('date,duration_minutes').eq('user_id', uid());
    sessions = data || [];
  }
  const map = {};
  sessions.forEach(s => { if (s.date && s.duration_minutes > 0) map[s.date] = (map[s.date] || 0) + s.duration_minutes; });

  const grid = $('heatmap-grid'); if (!grid) return;
  const monthsEl = $('heatmap-months'); if (!monthsEl) return;
  const yearLabelEl = $('heatmap-year-label'); if (!yearLabelEl) return;

  const today      = new Date(); today.setHours(0,0,0,0);
  const targetYear = today.getFullYear() + heatmapYearOffset;
  let startDate, endDate;

  startDate = new Date(targetYear, 0, 1);
  endDate   = new Date(targetYear, 11, 31);
  startDate.setDate(startDate.getDate() - startDate.getDay());
  endDate.setDate(endDate.getDate() + (6 - endDate.getDay()));

  yearLabelEl.textContent = String(targetYear);
  $('heatmap-year-prev').disabled = heatmapYearOffset <= -5;
  $('heatmap-year-next').disabled = heatmapYearOffset >= 0;

  const DAY_NAMES   = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  const cells = []; const monthLabels = []; let currentMonth = -1; let colIdx = 0;
  const d = new Date(startDate);

  while (d <= endDate) {
    if (d.getDay() === 0) {
      if (d.getMonth() !== currentMonth) {
        monthLabels.push({ label: MONTH_NAMES[d.getMonth()], col: colIdx });
        currentMonth = d.getMonth();
      }
      colIdx++;
    }
    const ds   = `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}`;
    const mins = map[ds] || 0;
    const lvl  = mins >= 120 ? 'l4' : mins >= 60 ? 'l3' : mins >= 30 ? 'l2' : mins > 0 ? 'l1' : 'l0';
    const isFuture = d > today;
    const tooltipDate = `${DAY_NAMES[d.getDay()]}, ${MONTH_NAMES[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
    const tooltip     = mins > 0 ? `${mins} min${mins!==1?'s':''} on ${tooltipDate}` : `No activity on ${tooltipDate}`;
    cells.push(`<div class="heatmap-cell ${isFuture ? '' : lvl}" ${!isFuture ? `data-tooltip="${tooltip}"` : ''}></div>`);
    d.setDate(d.getDate() + 1);
  }
  grid.innerHTML = cells.join('');

  const stride = 14; // 12px cell + 2px gap
  let mHtml = '';
  monthLabels.forEach((ml, i) => {
    const left  = ml.col * stride;
    const width = (i < monthLabels.length - 1 ? (monthLabels[i+1].col - ml.col) * stride : (colIdx - ml.col) * stride) - 4;
    mHtml += `<span class="heatmap-month-label" style="left:${left}px;width:${Math.max(20, width)}px">${ml.label}</span>`;
  });
  monthsEl.style.position = 'relative'; monthsEl.style.height = '16px';
  monthsEl.style.width    = (colIdx * stride) + 'px';
  monthsEl.innerHTML      = mHtml;
  grid.style.width        = (colIdx * stride) + 'px';

  setupHeatmapTooltip();
}

function setupHeatmapTooltip() {
  if (!heatmapTooltipEl) {
    heatmapTooltipEl = document.createElement('div');
    heatmapTooltipEl.className = 'heatmap-tooltip';
    document.body.appendChild(heatmapTooltipEl);
  }
  const grid = $('heatmap-grid'); if (!grid) return;
  const newGrid = grid.cloneNode(true);
  grid.parentNode.replaceChild(newGrid, grid);
  newGrid.addEventListener('mouseover', e => {
    const cell = e.target.closest('.heatmap-cell[data-tooltip]');
    if (cell) { heatmapTooltipEl.textContent = cell.dataset.tooltip; heatmapTooltipEl.classList.add('visible'); positionHeatmapTip(cell); }
  });
  newGrid.addEventListener('mouseout', e => {
    if (e.target.closest('.heatmap-cell[data-tooltip]')) heatmapTooltipEl.classList.remove('visible');
  });
}

function positionHeatmapTip(cell) {
  const rect = cell.getBoundingClientRect(), tip = heatmapTooltipEl.getBoundingClientRect();
  let left = rect.left + rect.width / 2 - tip.width / 2;
  let top  = rect.top - tip.height - 8;
  if (left < 4) left = 4;
  if (left + tip.width > window.innerWidth - 4) left = window.innerWidth - tip.width - 4;
  if (top < 4) top = rect.bottom + 8;
  heatmapTooltipEl.style.left = left + 'px';
  heatmapTooltipEl.style.top  = top + 'px';
}

// ── Leaderboard ───────────────────────────────────
async function renderLeaderboard() {
  const MEDALS = ['🥇', '🥈', '🥉'];

  if (isOffline) {
    const name = UP.display_name || 'Local User';
    $('lb-podium').innerHTML = `
      <div class="lb-top-card rank-1">
        <div class="lb-medal">${MEDALS[0]}</div>
        <div class="lb-top-avatar" style="background:${avatarBg(uid())}">${initials(name)}</div>
        <div class="lb-top-name">${escHtml(name)}</div>
        <div class="lb-top-streak">${UP.current_streak || 0}d streak</div>
      </div>`;
    $('lb-rows').innerHTML = `
      <div class="lb-row me top3">
        <div class="lb-rank">1</div>
        <div class="lb-user-wrap">
          <div class="lb-av" style="background:${avatarBg(uid())}">${initials(name)}</div>
          <div><div class="lb-name">${escHtml(name)}</div><div class="lb-username">You</div></div>
        </div>
        <div class="lb-streak-val">${UP.current_streak || 0}d</div>
        <div class="lb-time-val">${fmtDur(UP.total_study_minutes || 0)}</div>
        <div class="lb-goals-val">${UP.total_goals_completed || 0}</div>
      </div>`;
    return;
  }

  const { data } = await sb.from('profiles').select('*').order('total_study_minutes', { ascending: false }).limit(50);
  const users = (data || []).sort((a, b) => {
    const sdiff = (b.current_streak || 0) - (a.current_streak || 0);
    return sdiff !== 0 ? sdiff : (b.total_study_minutes || 0) - (a.total_study_minutes || 0);
  });

  // Build podium (top 3) — rank 1 in center, 2 on left, 3 on right
  const podiumSlots = [users[1], users[0], users[2]].map((u, slot) => {
    if (!u) return '';
    const rank   = slot === 0 ? 2 : slot === 1 ? 1 : 3;
    const rankClass = `rank-${rank}`;
    const isMe   = u.id === uid();
    const name   = isMe ? (u.display_name || 'User') : (u.username ? '@' + u.username : (u.display_name || 'User'));
    return `
      <div class="lb-top-card ${rankClass}">
        <div class="lb-medal">${MEDALS[rank - 1]}</div>
        <div class="lb-top-avatar" style="background:${avatarBg(u.id)}">${initials(u.username || u.display_name)}</div>
        <div class="lb-top-name">${escHtml(name)} ${isMe ? '<span style="color:var(--accent);font-size:.7rem">(You)</span>' : ''}</div>
        <div class="lb-top-streak">${u.current_streak || 0}d streak · ${fmtDur(u.total_study_minutes || 0)}</div>
      </div>`;
  });
  $('lb-podium').innerHTML = podiumSlots.join('');

  $('lb-rows').innerHTML = users.map((u, i) => {
    const rank  = i + 1;
    const isMe  = u.id === uid();
    const uName = isMe ? (u.display_name || 'User') : (u.username ? '@' + u.username : (u.display_name || 'User'));
    const uSub  = u.username ? '@' + u.username : '';
    return `
      <div class="lb-row ${isMe ? 'me' : ''} ${rank <= 3 ? 'top3' : ''}">
        <div class="lb-rank">${rank <= 3 ? MEDALS[rank-1] : rank}</div>
        <div class="lb-user-wrap">
          <div class="lb-av" style="background:${avatarBg(u.id)}">${initials(u.username || u.display_name)}</div>
          <div>
            <div class="lb-name">${escHtml(uName)} ${isMe ? '<span style="color:var(--accent);font-size:.7rem">You</span>' : ''}</div>
            ${uSub && !isMe ? `<div class="lb-username">${escHtml(uSub)}</div>` : ''}
          </div>
        </div>
        <div class="lb-streak-val">${u.current_streak || 0}d</div>
        <div class="lb-time-val">${fmtDur(u.total_study_minutes || 0)}</div>
        <div class="lb-goals-val">${u.total_goals_completed || 0}</div>
      </div>`;
  }).join('');
}

// ── Community ─────────────────────────────────────
async function renderCommunity() {
  if (isOffline) {
    const name     = UP.display_name || 'Local User';
    const studying = !!UP.active_session?.clockIn;
    $('members-list').innerHTML = `
      <div class="member-card" onclick="viewMember('${uid()}', this)">
        <div class="member-av" style="background:${avatarBg(uid())}">
          ${initials(name)}
          ${studying ? '<div class="online-dot"></div>' : ''}
        </div>
        <div class="member-info">
          <div class="member-name">${escHtml(name)} <span class="you-tag">(You)</span></div>
          <div class="member-status ${studying ? 'studying' : ''}">${studying ? 'Studying now' : 'Offline'}</div>
        </div>
      </div>`;
    $('community-detail').innerHTML = `<div class="cd-empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg><div>Sign in to see other users</div></div>`;
    renderCommunityChat();
    return;
  }

  const { data } = await sb.from('profiles').select('*');
  const users = (data || []);
  $('members-list').innerHTML = users.map(u => {
    const isMe     = u.id === uid();
    const studying = !!u.active_session?.clockIn;
    const name     = isMe ? (u.display_name || 'User') : (u.username ? '@' + u.username : (u.display_name || 'User'));
    return `
      <div class="member-card" onclick="viewMember('${u.id}', this)">
        <div class="member-av" style="background:${avatarBg(u.id)}">
          ${initials(u.username || u.display_name)}
          ${studying ? '<div class="online-dot"></div>' : ''}
        </div>
        <div class="member-info">
          <div class="member-name">${escHtml(name)} ${isMe ? '<span class="you-tag">(You)</span>' : ''}</div>
          <div class="member-status ${studying ? 'studying' : ''}">${studying ? 'Studying now' : 'Offline'}</div>
        </div>
      </div>`;
  }).join('');
  $('community-detail').innerHTML = `<div class="cd-empty"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/></svg><div>Select a user to view their progress</div></div>`;
  renderCommunityChat();
  subscribeCommunityChat();
}

async function viewMember(memberUid, cardEl) {
  document.querySelectorAll('.member-card').forEach(c => c.classList.remove('selected'));
  if (cardEl) cardEl.classList.add('selected');

  let u, gList = [];
  const today = todayStr();

  if (isOffline) {
    u = UP; gList = lsGetArr('goals').filter(g => g.user_id === uid() && g.date === today);
  } else {
    const { data: ud } = await sb.from('profiles').select('*').eq('id', memberUid).single();
    u = ud; if (!u) return;
    const { data: gd } = await sb.from('goals').select('*').eq('user_id', memberUid).eq('date', today);
    gList = gd || [];
  }

  const isMe    = memberUid === uid();
  const done    = gList.filter(g => g.completed).length;
  const studying = !!u.active_session?.clockIn;
  const name    = isMe ? (u.display_name || 'User') : (u.username ? '@' + u.username : (u.display_name || 'User'));

  $('community-detail').innerHTML = `
    <div class="cd-header">
      <div class="cd-avatar" style="background:${avatarBg(memberUid)}">${initials(u.username || u.display_name)}</div>
      <div>
        <div class="cd-name">${escHtml(name)} ${isMe ? '<span style="color:var(--accent);font-size:.75rem">(You)</span>' : ''}</div>
        <div class="cd-sub">${studying ? '🟢 Studying now' : 'Offline'}</div>
      </div>
    </div>
    <div class="cd-stats">
      <div class="cd-stat"><div class="cd-stat-val">${u.current_streak || 0}</div><div class="cd-stat-label">Day Streak</div></div>
      <div class="cd-stat"><div class="cd-stat-val">${fmtDur(u.today_study_minutes || 0)}</div><div class="cd-stat-label">Today</div></div>
      <div class="cd-stat"><div class="cd-stat-val">${u.total_goals_completed || 0}</div><div class="cd-stat-label">Goals Done</div></div>
    </div>
    <div class="cd-goals-title">TODAY'S GOALS (${done}/${gList.length})</div>
    ${gList.length
      ? gList.map(g => `
        <div class="cd-goal-item">
          <div class="cd-check ${g.completed ? 'done' : ''}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
          </div>
          <span style="${g.completed ? 'text-decoration:line-through;color:var(--text3)' : ''}">${escHtml(g.text)}</span>
        </div>`).join('')
      : '<div style="color:var(--text3);font-size:.82rem;padding:12px 0">No goals set for today</div>'
    }`;
}

async function getCommunityMessages() {
  if (isOffline) {
    return lsGetArr('community_messages')
      .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
      .slice(-80);
  }
  try {
    const { data, error } = await sb.from('community_messages')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(80);
    if (error) throw error;
    return (data || []).reverse();
  } catch(e) {
    console.warn('Community chat load failed:', e);
    return null;
  }
}

async function renderCommunityChat() {
  const list = $('community-chat-list');
  if (!list) return;
  const mode = $('community-chat-mode');
  if (mode) mode.textContent = isOffline ? 'Local' : 'Live';
  const messages = await getCommunityMessages();
  if (messages === null) {
    list.innerHTML = '<div class="goal-empty">Chat table is not ready. Run the updated Supabase SQL setup first.</div>';
    return;
  }
  if (!messages.length) {
    list.innerHTML = '<div class="goal-empty">No messages yet. Start the study room.</div>';
    return;
  }
  list.innerHTML = messages.map(m => {
    const mine = m.user_id === uid();
    const name = m.username ? '@' + m.username : (m.display_name || 'User');
    return `<div class="chat-msg ${mine ? 'mine' : ''}">
      <div class="chat-msg-avatar" style="background:${avatarBg(m.user_id || name)}">${initials(name)}</div>
      <div class="chat-msg-bubble">
        <div class="chat-msg-meta"><span class="chat-msg-name">${escHtml(mine ? 'You' : name)}</span><span>${fmtTime(m.created_at)}</span></div>
        <div class="chat-msg-text">${escHtml(m.body || '')}</div>
      </div>
    </div>`;
  }).join('');
  list.scrollTop = list.scrollHeight;
}

async function sendCommunityMessage() {
  const input = $('community-chat-input');
  if (!input) return;
  const body = input.value.trim();
  if (!body) return;
  if (body.length > 500) { toast('Message is too long', 'err'); return; }

  const msg = {
    id: genId('chat'),
    user_id: uid(),
    display_name: CU?.displayName || UP?.display_name || 'User',
    username: UP?.username || null,
    body,
    created_at: new Date().toISOString()
  };

  input.value = '';
  if (isOffline) {
    const arr = lsGetArr('community_messages');
    arr.push(msg);
    lsSet('community_messages', arr.slice(-200));
    renderCommunityChat();
    return;
  }

  try {
    const { error } = await sb.from('community_messages').insert(msg);
    if (error) throw error;
    renderCommunityChat();
  } catch(e) {
    console.error('Send chat failed:', e);
    toast('Could not send message. Run the updated SQL setup if needed.', 'err');
    input.value = body;
  }
}

function subscribeCommunityChat() {
  if (isOffline || communityChatChannel || !sb) return;
  try {
    communityChatChannel = sb.channel('community_messages_room')
      .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'community_messages' }, () => renderCommunityChat())
      .subscribe();
  } catch(e) {
    console.warn('Community chat realtime unavailable:', e);
  }
}

// ── Settings ──────────────────────────────────────
function loadSettings() {
  const saved = localStorage.getItem('st-settings');
  if (saved) { try { settings = JSON.parse(saved); } catch(e) {} }
  weeklyGoal = parseInt(localStorage.getItem('st-weekly-goal') || '600', 10) || 600;
  $('set-focus').value = settings.focusMin;
  $('set-short').value = settings.shortMin;
  $('set-long').value  = settings.longMin;
  if ($('set-weekly-goal')) $('set-weekly-goal').value = weeklyGoal;
  applyPomoSettings();
  renderSubjectsCard();
}

function saveSettings() {
  settings.focusMin = Math.max(1, parseInt($('set-focus').value) || 25);
  settings.shortMin = Math.max(1, parseInt($('set-short').value) || 5);
  settings.longMin  = Math.max(1, parseInt($('set-long').value)  || 15);
  weeklyGoal = Math.max(1, parseInt($('set-weekly-goal')?.value, 10) || 600);
  localStorage.setItem('st-settings', JSON.stringify(settings));
  localStorage.setItem('st-weekly-goal', String(weeklyGoal));
  updateWeeklyGoalBanner();
  renderWeeklyGoalRing();
  applyPomoSettings(); toast('Settings saved');
}

function applyPomoSettings() {
  if (!pomoRunning) {
    const modeMap = { focus: settings.focusMin, short: settings.shortMin, long: settings.longMin };
    pomoTotalSeconds = (modeMap[pomoMode] || 25) * 60;
    pomoSecondsLeft  = pomoTotalSeconds;
    updatePomoDisplay();
  }
  // Update mode button labels (text nodes only, keep SVGs)
  document.querySelectorAll('.pomo-mode').forEach(btn => {
    const txt = btn.textContent.trim();
    if (txt.includes('Focus') || txt.includes('focus')) {
      const svgEl = btn.querySelector('svg');
      btn.innerHTML = '';
      if (svgEl) btn.appendChild(svgEl);
      btn.append(` Focus ${settings.focusMin}m`);
    } else if (txt.includes('Short') || txt.includes('short')) {
      const svgEl = btn.querySelector('svg');
      btn.innerHTML = '';
      if (svgEl) btn.appendChild(svgEl);
      btn.append(` Short ${settings.shortMin}m`);
    } else if (txt.includes('Long') || txt.includes('long')) {
      const svgEl = btn.querySelector('svg');
      btn.innerHTML = '';
      if (svgEl) btn.appendChild(svgEl);
      btn.append(` Long ${settings.longMin}m`);
    }
  });
}

// ── Subject Management ────────────────────────────
function renderSubjectsCard() {
  const wrap = $('subjects-chips-wrap');
  if (!wrap) return;
  wrap.innerHTML = subjects.map(s => {
    const cssColor = getComputedStyle(document.documentElement).getPropertyValue(s.color).trim() || '#6366f1';
    return `
      <div class="subject-chip" id="subject-chip-${s.id}" style="border-left:3px solid ${cssColor}">
        <button class="subject-chip-main" onclick="editSubjectName('${s.id}')" title="Edit subject">${s.emoji} ${escHtml(s.name)}</button>
        <button class="subject-chip-edit" onclick="editSubjectDetails('${s.id}')" title="Edit emoji and name">Edit</button>
        <button class="subject-chip-delete" onclick="deleteSubject('${s.id}')" title="Delete subject">×</button>
      </div>`;
  }).join('');
}

function addSubject() {
  const emojiEl = $('subject-emoji');
  const nameEl = $('subject-name');
  const colorEl = $('subject-color-input');
  
  const emoji = (emojiEl.value || '📚').slice(0, 2);
  const name = nameEl.value.trim();
  
  if (!name) { toast('Enter a subject name', 'err'); return; }
  
  if (colorEl.dataset.color) {
    const id = name.toLowerCase().replace(/[^a-z0-9]+/g, '_') || genId('subj');
    if (subjects.find(s => s.id === id)) { toast('Subject already exists', 'err'); return; }
    subjects.push({ id, name, color: colorEl.dataset.color, emoji });
    saveSubjects();
    emojiEl.value = '';
    nameEl.value = '';
    refreshSubjectUi();
    toast(`Added ${name}`);
    return;
  }
  // Determine color from the input element's style
  const bgColor = colorEl.style.backgroundColor || 'rgb(99, 102, 241)';
  const colorMap = {
    'rgb(99, 102, 241)': '--accent',
    'rgb(139, 92, 246)': '--violet',
    'rgb(34, 197, 94)': '--green',
    'rgb(59, 130, 246)': '--blue',
    'rgb(245, 158, 11)': '--amber',
    'rgb(6, 182, 212)': '--cyan',
    'rgb(239, 68, 68)': '--red',
    'rgb(168, 85, 247)': '--purple'
  };
  let color = '--accent';
  for (const [rgb, cssVar] of Object.entries(colorMap)) {
    if (bgColor.includes(rgb.split('(')[1].split(')')[0]) || bgColor === rgb) {
      color = cssVar; break;
    }
  }
  
  const id = name.toLowerCase().replace(/[^a-z0-9]+/g, '_');
  if (subjects.find(s => s.id === id)) {
    toast('Subject already exists', 'err'); return;
  }
  
  subjects.push({ id, name, color, emoji });
  saveSubjects();
  
  emojiEl.value = '';
  nameEl.value = '';
  refreshSubjectUi();
  toast(`Added ${name}`);
}

async function deleteSubject(id) {
  const subj = getSubjectById(id);
  if (!subj) return;
  const refs = lsGetArr('sessions').some(s => s.subject_id === id) || lsGetArr('goals').some(g => g.subject_id === id);
  if (refs && !confirm(`"${subj.name}" is used by sessions or goals. Delete it anyway? They'll be moved to "Other".`)) return;
  if (!refs && !confirm(`Delete "${subj.name}"?`)) return;

  // Remove locally first
  subjects = subjects.filter(s => s.id !== id);
  lsSet('subjects', subjects);

  // Pick a fallback subject for orphaned items
  const fallback = subjects.find(s => s.id === 'other')?.id || subjects[0]?.id || 'other';
  reassignSubjectRefs(id, fallback);

  // Sync deletion + reassignment to the DB
  if (!isOffline && sb && uid()) {
    try {
      const { error: delErr } = await sb.from('subjects').delete().eq('id', id).eq('user_id', uid());
      if (delErr) throw delErr;
      await Promise.all([
        sb.from('goals').update({ subject_id: fallback }).eq('user_id', uid()).eq('subject_id', id),
        sb.from('sessions').update({ subject_id: fallback }).eq('user_id', uid()).eq('subject_id', id),
        sb.from('notes').update({ subject_id: fallback }).eq('user_id', uid()).eq('subject_id', id),
      ]);
    } catch (e) {
      console.error('deleteSubject sync:', e);
      toast('Deleted locally — DB sync failed', 'err');
    }
  }

  refreshSubjectUi();
  toast(`Deleted ${subj.name}`);
}

function editSubjectName(id) {
  const subj = getSubjectById(id);
  if (!subj) return;
  const chip = $(`subject-chip-${id}`);
  if (!chip) return editSubjectDetails(id);
  chip.innerHTML = `
    <input class="subject-chip-input" id="subject-edit-${id}" value="${escHtml(subj.name)}" maxlength="40"/>
    <button class="subject-chip-edit" onclick="saveSubjectInlineEdit('${id}')">Save</button>
    <button class="subject-chip-delete" onclick="renderSubjectsCard()">×</button>
  `;
  const input = $(`subject-edit-${id}`);
  input.focus();
  input.select();
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') saveSubjectInlineEdit(id);
    if (e.key === 'Escape') renderSubjectsCard();
  });
  input.addEventListener('blur', () => setTimeout(() => {
    if (document.activeElement?.onclick?.toString().includes('saveSubjectInlineEdit')) return;
    saveSubjectInlineEdit(id);
  }, 80));
}

function saveSubjectInlineEdit(id) {
  const subj = getSubjectById(id);
  const input = $(`subject-edit-${id}`);
  if (!subj || !input) return;
  const next = input.value.trim();
  if (next) {
    subj.name = next;
    saveSubjects();
    refreshSubjectUi();
  } else {
    renderSubjectsCard();
  }
}

function editSubjectDetails(id) {
  const subj = getSubjectById(id);
  if (!subj) return;
  const nextName = prompt('Subject name', subj.name);
  if (!nextName || !nextName.trim()) return;
  const nextEmoji = prompt('Subject emoji', subj.emoji) || subj.emoji;
  subj.name = nextName.trim();
  subj.emoji = nextEmoji.slice(0, 2) || '📚';
  saveSubjects();
  refreshSubjectUi();
}

function refreshSubjectUi() {
  renderSubjectsCard();
  renderSubjectsPage();
  renderGoalsFilterPills();
  renderSessionsFilterPills();
  renderNotesMoodFilter();
  renderSubjectPicker('sess-subject-picker', selectedSessionSubject || subjects[0]?.id || 'other', subjectId => { selectedSessionSubject = subjectId; });
  renderSubjectPicker('dash-clock-subject-picker', selectedSessionSubject || subjects[0]?.id || 'other', subjectId => { selectedSessionSubject = subjectId; });
  if (document.querySelector('#modal-goal.open')) renderSubjectPicker('goal-subject-picker', selectedGoalSubject || subjects[0]?.id || 'other', subjectId => { selectedGoalSubject = subjectId; });
  if (document.querySelector('#modal-note.open')) renderSubjectPicker('note-subject-picker', selectedNoteSubject || subjects[0]?.id || 'other', subjectId => { selectedNoteSubject = subjectId; });
}

// ── Filter Pills ──────────────────────────────
function renderGoalsFilterPills() {
  const el = $('goals-filter-pills');
  if (!el) return;
  
  let html = `<button class="filter-pill ${goalsFilter === 'all' ? 'active' : ''}" onclick="setGoalsFilter('all')">All Goals</button>`;
  subjects.forEach(s => {
    html += `<button class="filter-pill ${goalsFilter === s.id ? 'active' : ''}" onclick="setGoalsFilter('${s.id}')"><span class="filter-pill-icon">${s.emoji}</span>${s.name}</button>`;
  });
  el.innerHTML = html;
}

function setGoalsFilter(subjectId) {
  goalsFilter = subjectId;
  renderGoalsFilterPills();
  renderGoalsPage();
}

function renderSessionsFilterPills() {
  const el = $('sess-filter-pills');
  if (!el) return;
  
  let html = `<button class="filter-pill ${sessionsFilter === 'all' ? 'active' : ''}" onclick="setSessionsFilter('all')">All Sessions</button>`;
  subjects.forEach(s => {
    html += `<button class="filter-pill ${sessionsFilter === s.id ? 'active' : ''}" onclick="setSessionsFilter('${s.id}')"><span class="filter-pill-icon">${s.emoji}</span>${s.name}</button>`;
  });
  el.innerHTML = html;
}

function setSessionsFilter(subjectId) {
  sessionsFilter = subjectId;
  renderSessionsFilterPills();
  renderSessList();
}

async function exportData() {
  let data;
  if (isOffline) {
    data = { profile: UP, goals: lsGetArr('goals'), sessions: lsGetArr('sessions'), notes: lsGetArr('notes'), events: lsGetArr('events'), exportedAt: new Date().toISOString() };
  } else {
    const [{ data: goals }, { data: sessions }, { data: notes }, { data: events }] = await Promise.all([
      sb.from('goals').select('*').eq('user_id', uid()),
      sb.from('sessions').select('*').eq('user_id', uid()),
      sb.from('notes').select('*').eq('user_id', uid()),
      sb.from('events').select('*').eq('user_id', uid()),
    ]);
    data = { profile: UP, goals, sessions, notes, events, exportedAt: new Date().toISOString() };
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a'); a.href = url; a.download = `studytrack-${todayStr()}.json`; a.click();
  URL.revokeObjectURL(url); toast('Data exported');
}

async function resetData() {
  if (!confirm('Delete ALL your goals, sessions, notes, and events?')) return;
  if (!confirm('This cannot be undone. Are you sure?')) return;

  if (isOffline) {
    lsSet('goals', []); lsSet('sessions', []); lsSet('notes', []); lsSet('events', []); lsSet('activity', []);
    Object.assign(UP, { current_streak: 0, longest_streak: 0, total_study_minutes: 0, today_study_minutes: 0, week_study_minutes: 0, month_study_minutes: 0, total_goals_completed: 0, total_goals_added: 0, active_session: null });
    lsSet('profile', UP);
  } else {
    await Promise.all([
      sb.from('goals').delete().eq('user_id', uid()),
      sb.from('sessions').delete().eq('user_id', uid()),
      sb.from('notes').delete().eq('user_id', uid()),
      sb.from('events').delete().eq('user_id', uid()),
      sb.from('activity').delete().eq('user_id', uid()),
      sb.from('profiles').update({ current_streak: 0, longest_streak: 0, total_study_minutes: 0, today_study_minutes: 0, week_study_minutes: 0, month_study_minutes: 0, total_goals_completed: 0, total_goals_added: 0, active_session: null }).eq('id', uid()),
    ]);
    Object.assign(UP, { current_streak: 0, longest_streak: 0, total_study_minutes: 0, today_study_minutes: 0, week_study_minutes: 0, month_study_minutes: 0, total_goals_completed: 0, total_goals_added: 0, active_session: null });
  }
  clockInTime = null; updateClockUI(); updateDashStats();
  renderDashGoals(); renderActivity(); renderSessList(); renderNotesList(); updateXPBar();
  toast('All data reset');
}

// ── Pomodoro Timer ────────────────────────────────
function setPomoMode(mode, btnEl) {
  pomoMode = mode;
  document.querySelectorAll('.pomo-mode').forEach(b => b.classList.remove('active'));
  // If called from click handler, activate the target button; else find the right one
  if (btnEl) {
    btnEl.classList.add('active');
  } else {
    document.querySelectorAll('.pomo-mode').forEach(b => {
      const t = b.textContent.trim().toLowerCase();
      if ((mode === 'focus' && t.includes('focus')) || (mode === 'short' && t.includes('short')) || (mode === 'long' && t.includes('long')))
        b.classList.add('active');
    });
  }
  if (!pomoRunning) {
    pomoTotalSeconds = (mode === 'focus' ? settings.focusMin : mode === 'short' ? settings.shortMin : settings.longMin) * 60;
    pomoSecondsLeft  = pomoTotalSeconds;
    updatePomoDisplay();
  }
}

function updatePomoDisplay() {
  const min    = Math.floor(pomoSecondsLeft / 60);
  const sec    = pomoSecondsLeft % 60;
  const timeStr = `${p2(min)}:${p2(sec)}`;
  $('pomo-time').textContent  = timeStr;
  $('focus-time').textContent = timeStr;

  const labelMap = { focus: 'Focus', short: 'Short Break', long: 'Long Break' };
  $('pomo-label').textContent  = pomoRunning ? labelMap[pomoMode] : 'Ready';
  $('focus-label').textContent = labelMap[pomoMode];

  // Ring progress — r=96, C=603.19
  const progress     = pomoTotalSeconds > 0 ? (pomoTotalSeconds - pomoSecondsLeft) / pomoTotalSeconds : 0;
  const sessionCirc  = 2 * Math.PI * 96;
  const focusCirc    = 2 * Math.PI * 140;
  const sessRing     = $('pomo-ring-fg');
  const focusRing    = $('focus-ring-fg');
  if (sessRing)  sessRing.setAttribute('stroke-dashoffset',  sessionCirc * (1 - progress));
  if (focusRing) focusRing.setAttribute('stroke-dashoffset', focusCirc   * (1 - progress));

  // Update stroke-dasharray to match actual circumference
  if (sessRing)  sessRing.setAttribute('stroke-dasharray',  sessionCirc);
  if (focusRing) focusRing.setAttribute('stroke-dasharray', focusCirc);

  // Play button icon
  const playBtn = $('pomo-play-btn');
  if (playBtn) playBtn.innerHTML = pomoRunning
    ? '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>';
}

function togglePomo() { if (pomoRunning) pausePomo(); else startPomo(); }

function startPomo() {
  if (pomoRunning) return;
  pomoRunning = true;
  if (pomoMode === 'focus' && !pomoFocusStart) pomoFocusStart = Date.now();

  // Auto clock-in if not already
  if (!clockInTime) {
    clockInLabel = $('sess-label').value.trim() || 'Pomodoro Session';
    clockInTime  = Date.now();
    const today  = todayStr();
    const updates = { active_session: { clockIn: new Date(clockInTime).toISOString(), label: clockInLabel, subject_id: selectedSessionSubject || subjects[0]?.id || 'other' }, last_active_date: today };
    if (isOffline) {
      Object.assign(UP, updates); lsSet('profile', UP);
      pushActivity('clockin', 'Clocked in (Pomodoro)');
    } else {
      sb.from('profiles').update(updates).eq('id', uid());
      sb.from('activity').insert({ id: genId('act'), user_id: uid(), display_name: CU.displayName, username: UP?.username || null, type: 'clockin', detail: 'Clocked in (Pomodoro)' });
    }
    updateClockUI();
  }

  pomoInterval = setInterval(() => {
    pomoSecondsLeft--;
    if (pomoSecondsLeft <= 0) { pomoComplete(); return; }
    updatePomoDisplay();
    renderPomoTally();
  }, 1000);
  updatePomoDisplay(); renderPomoTally();
}

function pausePomo() {
  pomoRunning = false;
  clearInterval(pomoInterval); pomoInterval = null;
  updatePomoDisplay();
}

function stopPomo() { pausePomo(); pomoFocusStart = null; }

function resetPomo() {
  pausePomo(); pomoFocusStart = null;
  pomoTotalSeconds = (pomoMode === 'focus' ? settings.focusMin : pomoMode === 'short' ? settings.shortMin : settings.longMin) * 60;
  pomoSecondsLeft  = pomoTotalSeconds;
  updatePomoDisplay(); renderPomoTally();
}

async function pomoComplete() {
  pausePomo();
  playNotifSound();
  if (Notification.permission === 'granted') {
    new Notification('StudyTrack', { body: `${pomoMode === 'focus' ? 'Focus session' : 'Break'} complete!`, icon: '/favicon.png' });
  }

  if (pomoMode === 'focus') {
    pomoCount++; renderPomoTally();
    if (pomoFocusStart) {
      const dur   = Math.max(1, Math.floor((Date.now() - pomoFocusStart) / 60000));
      const today = todayStr(), yest = dateShift(today, -1);
      let streak = 1;
      if (UP.last_streak_date === yest || UP.last_streak_date === today)
        streak = UP.last_streak_date === today ? UP.current_streak : (UP.current_streak || 0) + 1;
      const longest = Math.max(streak, UP.longest_streak || 0);
      const updates = {
        today_study_minutes:  (UP.today_study_minutes  || 0) + dur,
        total_study_minutes:  (UP.total_study_minutes  || 0) + dur,
        week_study_minutes:   (UP.week_study_minutes   || 0) + dur,
        month_study_minutes:  (UP.month_study_minutes  || 0) + dur,
        current_streak: streak, longest_streak: longest,
        last_streak_date: today, last_active_date: today
      };
      if (isOffline) {
        const sessArr = lsGetArr('sessions');
        sessArr.push({ id: genId('sess'), user_id: uid(), date: today, clock_in: new Date(pomoFocusStart).toISOString(), clock_out: new Date().toISOString(), duration_minutes: dur, label: clockInLabel || 'Pomodoro Focus', subject_id: selectedSessionSubject || 'other' });
        lsSet('sessions', sessArr);
        Object.assign(UP, updates); lsSet('profile', UP);
      } else {
        await sb.from('sessions').insert({ id: genId('sess'), user_id: uid(), date: today, clock_in: new Date(pomoFocusStart).toISOString(), clock_out: new Date().toISOString(), duration_minutes: dur, label: clockInLabel || 'Pomodoro Focus', subject_id: selectedSessionSubject || 'other' });
        await sb.from('profiles').update(updates).eq('id', uid());
        Object.assign(UP, updates);
      }
      pomoFocusStart = null;
    }
    toast(`Focus session complete! +${settings.focusMin} XP`);
    updateDashStats(); renderSessList(); updateXPBar();
    // Auto-switch to break
    if (pomoCount % 4 === 0) { setPomoMode('long'); }
    else { setPomoMode('short'); }
  } else {
    toast('Break over — time to focus!');
    setPomoMode('focus');
  }
}

function renderPomoTally() {
  const el = $('pomo-tally'); if (!el) return;
  const countEl = $('pomo-count-label'); if (countEl) countEl.textContent = pomoCount % 4 || (pomoCount > 0 && pomoCount % 4 === 0 ? 4 : 0);
  let html = '';
  for (let i = 0; i < 4; i++) {
    const done   = i < (pomoCount % 4);
    const active = i === (pomoCount % 4) && pomoRunning && pomoMode === 'focus';
    html += `<div class="pomo-pip ${done ? 'done' : ''} ${active ? 'active' : ''}"></div>`;
  }
  el.innerHTML = html;
}

function playNotifSound() {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.frequency.setValueAtTime(587.33, audioCtx.currentTime);
    osc.frequency.setValueAtTime(783.99, audioCtx.currentTime + 0.15);
    osc.frequency.setValueAtTime(1046.5, audioCtx.currentTime + 0.3);
    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.6);
    osc.start(audioCtx.currentTime); osc.stop(audioCtx.currentTime + 0.65);
  } catch(e) {}
}

// ── Ambient Sounds (WebAudio) ─────────────────────
function initAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

function createNoiseBuffer(type) {
  const size   = 2 * audioCtx.sampleRate;
  const buffer = audioCtx.createBuffer(1, size, audioCtx.sampleRate);
  const data   = buffer.getChannelData(0);
  let last = 0;
  for (let i = 0; i < size; i++) {
    const white = Math.random() * 2 - 1;
    if (type === 'brown') {
      data[i] = (last + 0.02 * white) / 1.02; last = data[i]; data[i] *= 3.5;
    } else if (type === 'rain') {
      data[i] = white * 0.5;
    } else if (type === 'waves') {
      data[i] = (last + 0.01 * white) / 1.01; last = data[i]; data[i] *= 3.5;
      data[i] *= 0.5 + 0.5 * Math.sin(2 * Math.PI * i / audioCtx.sampleRate * 0.15);
    } else if (type === 'wind') {
      data[i] = (last + 0.04 * white) / 1.04; last = data[i]; data[i] *= 2.5;
      data[i] *= 0.3 + 0.7 * Math.sin(2 * Math.PI * i / audioCtx.sampleRate * 0.08);
    }
  }
  return buffer;
}

function toggleAmbient(type, btnEl) {
  initAudioCtx();
  const isActive = !!ambientNodes[type];
  document.querySelectorAll(`.ambient-btn[data-sound="${type}"]`).forEach(b => b.classList.toggle('active', !isActive));
  if (isActive) { stopAmbient(type); return; }
  const source = audioCtx.createBufferSource();
  source.buffer = createNoiseBuffer(type); source.loop = true;
  const gain   = audioCtx.createGain(); gain.gain.value = ambientVolume;
  const filter = audioCtx.createBiquadFilter();
  if      (type === 'rain')  { filter.type = 'bandpass'; filter.frequency.value = 8000; filter.Q.value = 0.5; }
  else if (type === 'waves') { filter.type = 'lowpass';  filter.frequency.value =  400; filter.Q.value = 0.7; }
  else if (type === 'wind')  { filter.type = 'bandpass'; filter.frequency.value =  600; filter.Q.value = 0.3; }
  else                       { filter.type = 'lowpass';  filter.frequency.value =  200; filter.Q.value = 0.5; }
  source.connect(filter); filter.connect(gain); gain.connect(audioCtx.destination);
  source.start(); ambientNodes[type] = { source, gain, filter };
}

function stopAmbient(type) {
  if (ambientNodes[type]) { try { ambientNodes[type].source.stop(); } catch(e) {} delete ambientNodes[type]; }
  document.querySelectorAll(`.ambient-btn[data-sound="${type}"]`).forEach(b => b.classList.remove('active'));
}

function stopAllAmbient() { Object.keys(ambientNodes).forEach(k => stopAmbient(k)); }

function setAmbientVol(val) {
  ambientVolume = val / 100;
  Object.values(ambientNodes).forEach(n => { if (n.gain) n.gain.gain.value = ambientVolume; });
}

// ── Focus Mode Overlay ────────────────────────────
function openFocusMode() {
  $('focus-overlay').classList.add('open');
  updatePomoDisplay();
  spawnFocusParticles();
}

function closeFocusMode() {
  $('focus-overlay').classList.remove('open');
}

function spawnFocusParticles() {
  const container = $('focus-particles'); if (!container) return;
  container.innerHTML = '';
  const count = 18;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.className = 'focus-particle';
    const size = 4 + Math.random() * 12;
    const left = Math.random() * 100;
    const delay = Math.random() * 6;
    const dur   = 6 + Math.random() * 8;
    p.style.cssText = `
      width:${size}px; height:${size}px; border-radius:50%;
      left:${left}vw; background:var(--accent-glow);
      animation-delay:${delay}s; animation-duration:${dur}s;
      animation-name:floatParticle; animation-timing-function:linear; animation-iteration-count:infinite;
    `;
    container.appendChild(p);
  }
}

// ── Confetti ──────────────────────────────────────
function triggerConfetti() {
  const canvas = $('confetti-canvas'); if (!canvas) return;
  canvas.style.display = 'block';
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  const ctx     = canvas.getContext('2d');
  const colors  = ['#6366f1','#8b5cf6','#22c55e','#f59e0b','#3b82f6','#ec4899','#06b6d4'];
  const pieces  = Array.from({ length: 80 }, () => ({
    x: Math.random() * canvas.width, y: -10,
    vx: (Math.random() - 0.5) * 4, vy: 2 + Math.random() * 4,
    color: colors[Math.floor(Math.random() * colors.length)],
    size: 6 + Math.random() * 8, angle: Math.random() * Math.PI * 2,
    spin: (Math.random() - 0.5) * 0.2, alpha: 1
  }));
  let frame;
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let alive = false;
    pieces.forEach(p => {
      if (p.alpha <= 0) return;
      alive = true;
      p.x += p.vx; p.y += p.vy; p.angle += p.spin;
      if (p.y > canvas.height * 0.7) p.alpha -= 0.025;
      ctx.save(); ctx.globalAlpha = Math.max(0, p.alpha);
      ctx.translate(p.x, p.y); ctx.rotate(p.angle);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
      ctx.restore();
    });
    if (alive) { frame = requestAnimationFrame(draw); }
    else { canvas.style.display = 'none'; ctx.clearRect(0, 0, canvas.width, canvas.height); cancelAnimationFrame(frame); }
  }
  cancelAnimationFrame(frame); draw();
  setTimeout(() => { pieces.forEach(p => p.alpha = 0); }, 2500);
}

// ── Command Palette ───────────────────────────────
function buildCommandItems() {
  cmdItems = [
    {
      section: 'Navigation', items: [
        { text: 'Dashboard',   shortcut: 'G D', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>', action: () => showView('dashboard') },
        { text: 'Goals',       shortcut: 'G G', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',                         action: () => showView('goals') },
        { text: 'Subjects',    shortcut: 'G B', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41 11 3H4v7l9.59 9.59a2 2 0 0 0 2.82 0l4.18-4.18a2 2 0 0 0 0-2.82Z"/><circle cx="7.5" cy="7.5" r=".5"/></svg>', action: () => showView('subjects') },
        { text: 'Sessions',    shortcut: 'G S', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',                                                  action: () => showView('sessions') },
        { text: 'Notes',       shortcut: 'G N', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',         action: () => showView('notes') },
        { text: 'Calendar',    shortcut: 'G C', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/></svg>', action: () => showView('calendar') },
        { text: 'Statistics',  shortcut: 'G T', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',  action: () => showView('stats') },
        { text: 'Settings',    shortcut: 'G X', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/></svg>',                                                                                          action: () => showView('settings') },
      ]
    },
    {
      section: 'Quick Actions', items: [
        { text: 'Clock In / Out',    shortcut: '',  icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>', action: () => toggleClockDash() },
        { text: 'Start Pomodoro',    shortcut: '',  icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21"/></svg>', action: () => { showView('sessions'); if (!pomoRunning) startPomo(); } },
        { text: 'Add Goal',          shortcut: 'N', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>', action: () => openModal('modal-goal') },
        { text: 'Add Note',          shortcut: '',  icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/></svg>',                      action: () => openModal('modal-note') },
        { text: 'Add Event',         shortcut: '',  icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2"/></svg>',                          action: () => openModal('modal-event') },
        { text: 'Toggle Theme',      shortcut: 'T', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/></svg>',                                            action: () => toggleTheme() },
        { text: 'Focus Mode',        shortcut: 'F', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="2"/></svg>',            action: () => openFocusMode() },
        { text: 'Export Data',       shortcut: '',  icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/></svg>', action: () => exportData() },
      ]
    }
  ];
}

function openCommandPalette() {
  $('cmd-palette').classList.add('open');
  $('cmd-input').value = ''; $('cmd-input').focus();
  cmdSelectedIdx = 0; renderCmdResults('');
}
function closeCmdPalette() { $('cmd-palette').classList.remove('open'); }
function filterCommands() { cmdSelectedIdx = 0; renderCmdResults($('cmd-input').value.toLowerCase()); }

function renderCmdResults(query) {
  const el  = $('cmd-results');
  let html  = '', idx = 0;
  cmdItems.forEach(section => {
    const filtered = section.items.filter(i => !query || i.text.toLowerCase().includes(query));
    if (!filtered.length) return;
    html += `<div class="cmd-section-label">${section.section}</div>`;
    filtered.forEach(item => {
      html += `<div class="cmd-item ${idx === cmdSelectedIdx ? 'selected' : ''}" onclick="executeCmdItem(${idx})" data-idx="${idx}">
        <span style="color:var(--text3);display:flex;align-items:center">${item.icon}</span>
        <span class="cmd-item-text">${item.text}</span>
        ${item.shortcut ? `<span class="cmd-item-shortcut">${item.shortcut}</span>` : ''}
      </div>`;
      idx++;
    });
  });
  if (!html) html = '<div style="padding:20px;text-align:center;color:var(--text3);font-size:.84rem">No results</div>';
  el.innerHTML = html;
}

function executeCmdItem(idx) {
  let count = 0, item = null;
  for (const section of cmdItems) {
    for (const i of section.items) { if (count === idx) { item = i; break; } count++; }
    if (item) break;
  }
  if (item) { closeCmdPalette(); item.action(); }
}

function cmdKeyDown(e) {
  if (e.key === 'Escape') { closeCmdPalette(); return; }
  if (e.key === 'Enter')  { executeCmdItem(cmdSelectedIdx); return; }
  const allItems = $('cmd-results').querySelectorAll('.cmd-item');
  if (e.key === 'ArrowDown') { e.preventDefault(); cmdSelectedIdx = Math.min(cmdSelectedIdx + 1, allItems.length - 1); }
  if (e.key === 'ArrowUp')   { e.preventDefault(); cmdSelectedIdx = Math.max(cmdSelectedIdx - 1, 0); }
  allItems.forEach((el, i) => el.classList.toggle('selected', i === cmdSelectedIdx));
  if (allItems[cmdSelectedIdx]) allItems[cmdSelectedIdx].scrollIntoView({ block: 'nearest' });
}

// ── Keyboard Shortcuts ────────────────────────────
let keyBuffer = '', keyTimeout = null;

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.ctrlKey || e.metaKey) {
    if (e.key === 'k') { e.preventDefault(); openCommandPalette(); } return;
  }
  if (e.key === 'Escape') {
    if ($('cmd-palette').classList.contains('open'))   { closeCmdPalette(); return; }
    if ($('focus-overlay').classList.contains('open')) { closeFocusMode(); return; }
    document.querySelectorAll('.overlay.open').forEach(o => o.classList.remove('open'));
    return;
  }
  if (e.key === ' ' && $('focus-overlay').classList.contains('open')) { e.preventDefault(); togglePomo(); return; }
  if (e.key === '/' || e.key === 'Slash') { e.preventDefault(); openCommandPalette(); return; }

  clearTimeout(keyTimeout);
  keyBuffer += e.key.toUpperCase();
  keyTimeout = setTimeout(() => { keyBuffer = ''; }, 800);

  if (keyBuffer === 'T') { toggleTheme(); keyBuffer = ''; return; }
  if (keyBuffer === 'F') { openFocusMode(); keyBuffer = ''; return; }
  if (keyBuffer === 'N') { openModal('modal-goal'); keyBuffer = ''; return; }
  if (keyBuffer.startsWith('G') && keyBuffer.length === 2) {
    const map = { D: 'dashboard', G: 'goals', B: 'subjects', S: 'sessions', N: 'notes', C: 'calendar', T: 'stats', L: 'leaderboard', O: 'community', X: 'settings' };
    const view = map[keyBuffer[1]];
    if (view) { showView(view); keyBuffer = ''; }
  }
});

// ── StudyTrack v2 Enhancements ───────────────────
let chartHourly = null;
let chartGoalsDonut = null;
let chartSubjectsBar = null;
let chartRadar = null;
let chart30day = null;
let chartSessions = null;
let reminderInterval = null;
let reminderLastFired = '';
let communityChatChannel = null;

const STUDY_QUOTES = [
  'Small steps every day become big results.',
  'Focus on progress, not perfection.',
  'The expert in anything was once a beginner.',
  'Consistency turns effort into skill.',
  'One focused session is a vote for your future.',
  'Learn deeply. Rest wisely. Return stronger.',
  'Discipline is remembering what you want.',
  'A little better today is still better.',
  'Your future self is taking notes.',
  'Momentum loves a clear next step.'
];

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartGridColor() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  return dark ? 'rgba(255,255,255,0.045)' : 'rgba(15,23,42,0.055)';
}

function subjectColorValue(subject) {
  return cssVar(subject?.color || '--accent') || cssVar('--accent') || '#6366f1';
}

function weekStartStr() {
  const d = new Date();
  const day = d.getDay();
  d.setDate(d.getDate() - (day === 0 ? 6 : day - 1));
  return `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}`;
}

function getWeekMinutesFromProfile() {
  return UP?.week_study_minutes || 0;
}

function goalProgressColor(pct) {
  if (pct < 30) return cssVar('--red') || '#ef4444';
  if (pct < 70) return cssVar('--amber') || '#f59e0b';
  return cssVar('--green') || '#22c55e';
}

function updateWeeklyGoalBanner() {
  const fill = $('weekly-goal-fill');
  if (!fill || !UP) return;
  const mins = getWeekMinutesFromProfile();
  const pct = Math.min(100, Math.round((mins / Math.max(1, weeklyGoal)) * 100));
  $('weekly-goal-label').textContent = `This week: ${fmtDur(mins)} / Goal: ${fmtDur(weeklyGoal)}`;
  $('weekly-goal-pct').textContent = `${pct}%`;
  fill.style.width = `${pct}%`;
  fill.style.background = goalProgressColor(pct);
}

async function getAllSessionsForUser(date) {
  if (isOffline) {
    return lsGetArr('sessions').filter(s => s.user_id === uid() && (!date || s.date === date));
  }
  const q = sb.from('sessions').select('*').eq('user_id', uid());
  if (date) q.eq('date', date);
  const { data } = await q;
  return data || [];
}

async function getAllGoalsForUser(date) {
  if (isOffline) {
    return lsGetArr('goals').filter(g => g.user_id === uid() && (!date || g.date === date));
  }
  const q = sb.from('goals').select('*').eq('user_id', uid());
  if (date) q.eq('date', date);
  const { data } = await q;
  return data || [];
}

async function renderDashboardCharts() {
  const [todaySessions, todayGoals, allSessions] = await Promise.all([
    getAllSessionsForUser(todayStr()),
    getAllGoalsForUser(todayStr()),
    getAllSessionsForUser()
  ]);
  const text = cssVar('--text');
  const text2 = cssVar('--text2');
  const border = chartGridColor();
  const accent = cssVar('--accent');
  const nowHour = new Date().getHours();

  const hourly = Array(24).fill(0);
  todaySessions.forEach(s => {
    const hour = s.clock_in ? new Date(s.clock_in).getHours() : 0;
    hourly[hour] += s.duration_minutes || 0;
  });

  const hourLine = {
    id: 'currentHourLine',
    afterDraw(chart) {
      const xScale = chart.scales.x;
      if (!xScale) return;
      const x = xScale.getPixelForValue(nowHour);
      const { top, bottom } = chart.chartArea;
      const ctx = chart.ctx;
      ctx.save();
      ctx.strokeStyle = accent;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.restore();
    }
  };

  if ($('chart-hourly')) {
    if (chartHourly) chartHourly.destroy();
    chartHourly = new Chart($('chart-hourly'), {
      type: 'line',
      data: { labels: Array.from({ length: 24 }, (_, i) => `${i}h`), datasets: [{ data: hourly, borderColor: accent, backgroundColor: `${accent}22`, fill: true, tension: .35, pointRadius: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: text2, maxTicksLimit: 8 } }, y: { beginAtZero: true, grid: { color: border }, ticks: { color: text2 } } } },
      plugins: [hourLine]
    });
  }

  if ($('chart-goals-donut')) {
    if (chartGoalsDonut) chartGoalsDonut.destroy();
    const done = todayGoals.filter(g => g.completed).length;
    const total = todayGoals.length;
    const pct = total ? Math.round(done / total * 100) : 0;
    const center = { id: 'centerGoalPct', afterDraw(chart) {
      const meta = chart.getDatasetMeta(0).data[0]; if (!meta) return;
      const ctx = chart.ctx;
      ctx.save(); ctx.fillStyle = text; ctx.font = '700 22px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`${pct}%`, meta.x, meta.y); ctx.restore();
    }};
    chartGoalsDonut = new Chart($('chart-goals-donut'), {
      type: 'doughnut',
      data: { labels: ['Completed', 'Remaining'], datasets: [{ data: [done, Math.max(0, total - done)], backgroundColor: [cssVar('--green'), cssVar('--border2')], borderWidth: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { display: false } } },
      plugins: [center]
    });
  }

  const totals = {};
  allSessions.forEach(s => { totals[s.subject_id || 'other'] = (totals[s.subject_id || 'other'] || 0) + (s.duration_minutes || 0); });
  const activeSubjects = subjects.filter(s => totals[s.id] > 0);
  const empty = $('subject-time-empty');
  if (empty) empty.style.display = activeSubjects.length ? 'none' : 'block';
  if ($('chart-subjects-bar')) {
    if (chartSubjectsBar) chartSubjectsBar.destroy();
    chartSubjectsBar = new Chart($('chart-subjects-bar'), {
      type: 'bar',
      data: { labels: activeSubjects.map(s => s.name), datasets: [{ data: activeSubjects.map(s => totals[s.id]), backgroundColor: activeSubjects.map(subjectColorValue), borderRadius: 7 }] },
      options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true, grid: { color: border }, ticks: { color: text2 } }, y: { grid: { display: false }, ticks: { color: text } } } }
    });
  }
}

async function renderStatsCharts() {
  const sessions = await getAllSessionsForUser();
  const text = cssVar('--text');
  const text2 = cssVar('--text2');
  const border = chartGridColor();
  const accent = cssVar('--accent');
  const totals = {};
  sessions.forEach(s => { totals[s.subject_id || 'other'] = (totals[s.subject_id || 'other'] || 0) + (s.duration_minutes || 0); });

  if ($('chart-radar')) {
    if (chartRadar) chartRadar.destroy();
    chartRadar = new Chart($('chart-radar'), {
      type: 'radar',
      data: { labels: subjects.map(s => s.name), datasets: [{ label: 'Minutes', data: subjects.map(s => totals[s.id] || 0), borderColor: accent, backgroundColor: `${accent}30`, pointBackgroundColor: accent }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: text2 } } }, scales: { r: { beginAtZero: true, grid: { color: border }, angleLines: { color: border }, pointLabels: { color: text }, ticks: { color: text2, backdropColor: 'transparent' } } } }
    });
  }

  const days = [];
  const daily = [];
  const streakDates = new Set(sessions.filter(s => (s.duration_minutes || 0) > 0).map(s => s.date));
  for (let i = 29; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    const ds = `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}`;
    days.push(`${d.getMonth()+1}/${d.getDate()}`);
    daily.push(sessions.filter(s => s.date === ds).reduce((a, s) => a + (s.duration_minutes || 0), 0));
  }
  if ($('chart-30day')) {
    if (chart30day) chart30day.destroy();
    chart30day = new Chart($('chart-30day'), {
      type: 'line',
      data: { labels: days, datasets: [{ data: daily, borderColor: accent, backgroundColor: `${accent}22`, fill: true, tension: .35, pointRadius: ctx => daily[ctx.dataIndex] > 0 ? 3 : 1, pointBackgroundColor: ctx => daily[ctx.dataIndex] > 0 ? cssVar('--green') : accent }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: text2, maxTicksLimit: 10 } }, y: { beginAtZero: true, grid: { color: border }, ticks: { color: text2 } } } }
    });
  }
  renderWeeklyGoalRing();
}

function renderWeeklyGoalRing() {
  const el = $('weekly-ring-wrap'); if (!el || !UP) return;
  const mins = getWeekMinutesFromProfile();
  const pct = Math.min(100, Math.round(mins / Math.max(1, weeklyGoal) * 100));
  const r = 72, c = 2 * Math.PI * r;
  const color = goalProgressColor(pct);
  el.innerHTML = `<div class="weekly-ring-shell">
    <svg class="weekly-ring-svg" viewBox="0 0 180 180">
      <circle class="weekly-ring-bg" cx="90" cy="90" r="${r}"></circle>
      <circle class="weekly-ring-fg" cx="90" cy="90" r="${r}" stroke="${color}" stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - pct / 100)}"></circle>
    </svg>
    <div class="weekly-ring-center"><div class="weekly-ring-percent">${pct}%</div></div>
  </div>
  <div class="weekly-ring-caption">${fmtDur(mins)} / Goal: ${fmtDur(weeklyGoal)}</div>`;
}

async function renderSessionsChart() {
  let sessions = await getAllSessionsForUser(sessDate);
  if (sessionsFilter !== 'all') sessions = sessions.filter(s => s.subject_id === sessionsFilter);
  sessions = sessions.sort((a, b) => new Date(a.clock_in) - new Date(b.clock_in));
  if (!$('chart-sessions')) return;
  if (chartSessions) chartSessions.destroy();
  chartSessions = new Chart($('chart-sessions'), {
    type: 'bar',
    data: { labels: sessions.map(s => fmtTime(s.clock_in)), datasets: [{ data: sessions.map(s => s.duration_minutes || 0), backgroundColor: sessions.map(s => subjectColorValue(getSubjectById(s.subject_id))), borderRadius: 7 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: cssVar('--text2') } }, y: { beginAtZero: true, grid: { color: chartGridColor() }, ticks: { color: cssVar('--text2') } } } }
  });
}

async function openSessionFeedback(startMs, endMs, label, subjectId) {
  const duration = Math.max(1, Math.floor((endMs - startMs) / 60000));
  await saveCompletedSession(startMs, endMs, duration, label, subjectId || selectedSessionSubject || 'other');
  const subj = getSubjectById(subjectId || selectedSessionSubject) || getSubjectById('other') || subjects[0];
  const xp = duration;
  $('session-summary-grid').innerHTML = [
    ['Duration', fmtDur(duration)],
    ['Subject', `${subj?.emoji || ''} ${subj?.name || 'Other'}`],
    ['XP Earned', `+${xp} XP`],
    ['Current Streak', `${UP.current_streak || 0} days`]
  ].map(([label, value]) => `<div class="session-summary-item"><div class="session-summary-label">${label}</div><div class="session-summary-value">${escHtml(value)}</div></div>`).join('');
  $('session-summary-quote').textContent = STUDY_QUOTES[Math.floor(Math.random() * STUDY_QUOTES.length)];
  openModal('modal-session-feedback');
}

async function saveCompletedSession(startMs, endMs, duration, label, subjectId) {
  const today = todayStr(), yest = dateShift(today, -1);
  let streak = 1;
  if (UP.last_streak_date === yest || UP.last_streak_date === today) streak = UP.last_streak_date === today ? (UP.current_streak || 1) : (UP.current_streak || 0) + 1;
  const updates = {
    active_session: null,
    today_study_minutes: (UP.today_study_minutes || 0) + duration,
    total_study_minutes: (UP.total_study_minutes || 0) + duration,
    week_study_minutes: (UP.week_study_minutes || 0) + duration,
    month_study_minutes: (UP.month_study_minutes || 0) + duration,
    current_streak: streak,
    longest_streak: Math.max(streak, UP.longest_streak || 0),
    last_streak_date: today,
    last_active_date: today
  };
  const session = { id: genId('sess'), user_id: uid(), date: today, clock_in: new Date(startMs).toISOString(), clock_out: new Date(endMs).toISOString(), duration_minutes: duration, label: label || 'Study session', subject_id: subjectId };
  if (isOffline) {
    Object.assign(UP, updates); lsSet('profile', UP);
    const arr = lsGetArr('sessions'); arr.push(session); lsSet('sessions', arr);
    pushActivity('clockout', `Clocked out after ${fmtDur(duration)}`);
  } else {
    await sb.from('profiles').update(updates).eq('id', uid());
    await sb.from('sessions').insert(session);
    await sb.from('activity').insert({ id: genId('act'), user_id: uid(), display_name: CU.displayName, username: UP?.username || null, type: 'clockout', detail: `Clocked out after ${fmtDur(duration)}` });
    Object.assign(UP, updates);
  }
  clockInTime = null; clockInLabel = ''; $('sess-label').value = '';
  updateClockUI(); updateDashStats(); renderSessList(); renderDashboardCharts(); renderActivity(); updateXPBar();
}

function logAnotherSession() {
  closeModal('modal-session-feedback');
  clockIn();
}

function openSubjectModal() {
  $('modal-subject-emoji').value = '';
  $('modal-subject-name').value = '';
  $('modal-subject-color').dataset.color = '--accent';
  $('modal-subject-color').style.background = cssVar('--accent');
  openModal('modal-subject');
}

function initSubjectColorPickers() {
  const palette = ['--accent','--violet','--green','--blue','--amber','--red','--cyan','--purple'];
  ['subject-color-input','modal-subject-color'].forEach(id => {
    const el = $(id);
    if (!el || el.dataset.ready) return;
    el.dataset.ready = 'true';
    el.dataset.color = el.dataset.color || palette[0];
    el.style.background = cssVar(el.dataset.color);
    el.addEventListener('click', () => {
      const cur = palette.indexOf(el.dataset.color);
      el.dataset.color = palette[(cur + 1) % palette.length];
      el.style.background = cssVar(el.dataset.color);
    });
  });
}

function addSubjectFromModal() {
  const oldEmoji = $('subject-emoji'), oldName = $('subject-name'), oldColor = $('subject-color-input');
  if (oldEmoji && oldName && oldColor) {
    oldEmoji.value = $('modal-subject-emoji').value || '📚';
    oldName.value = $('modal-subject-name').value;
    oldColor.dataset.color = $('modal-subject-color').dataset.color || '--accent';
    addSubject();
    closeModal('modal-subject');
  }
}

function loadReminderSettings() {
  reminderSettings = lsGet('reminder') || { enabled: false, time: '20:00' };
  if ($('reminder-time')) $('reminder-time').value = reminderSettings.time || '20:00';
  updateReminderUI();
}

function saveReminderSettings() {
  reminderSettings.time = $('reminder-time')?.value || '20:00';
  lsSet('reminder', reminderSettings);
  updateReminderUI();
}

async function toggleReminderSetting() {
  if (!reminderSettings.enabled && 'Notification' in window && Notification.permission !== 'granted') {
    const perm = await Notification.requestPermission();
    if (perm !== 'granted') { toast('Notifications were not enabled', 'err'); return; }
  }
  reminderSettings.enabled = !reminderSettings.enabled;
  saveReminderSettings();
}

function updateReminderUI() {
  const toggle = $('reminder-toggle');
  if (toggle) toggle.classList.toggle('on', !!reminderSettings.enabled);
}

function startReminderCheck() {
  if (reminderInterval) clearInterval(reminderInterval);
  reminderInterval = setInterval(checkDailyReminder, 60000);
  checkDailyReminder();
}

function checkDailyReminder() {
  if (!reminderSettings.enabled || !('Notification' in window) || Notification.permission !== 'granted') return;
  const now = new Date();
  const hhmm = `${p2(now.getHours())}:${p2(now.getMinutes())}`;
  const key = `${todayStr()}-${hhmm}`;
  if (hhmm === reminderSettings.time && reminderLastFired !== key) {
    reminderLastFired = key;
    const first = (CU?.displayName || 'there').split(' ')[0];
    new Notification('📚 Time to study, ' + first + '!', { body: `Your streak is at ${UP?.current_streak || 0} days.`, icon: 'favicon.png' });
  }
}

// ── Init ──────────────────────────────────────────
initTheme();

if ('Notification' in window && Notification.permission === 'default') {
  Notification.requestPermission();
}

(async function init() {
  try {
    CU = { id: SINGLE_USER_ID, displayName: SINGLE_USER_NAME, email: '' };
    localStorage.removeItem('st-offline');
    showPage('page-app'); await loadApp(); return;
    // Check offline auto-login
    if (localStorage.getItem('st-offline') === 'true') {
      isOffline = true;
      CU = { id: 'local_user', displayName: 'Local User', email: '' };
      showPage('page-app'); await loadApp(); return;
    }
    if (!sb) { showPage('page-auth'); return; }
    const { data: { session } } = await sb.auth.getSession();
    if (session?.user) {
      CU = { id: session.user.id, displayName: session.user.user_metadata?.display_name || session.user.email, email: session.user.email };
      showPage('page-app'); await loadApp();
    } else {
      showPage('page-auth');
    }
  } catch(e) {
    console.error('Init error:', e);
    showPage('page-auth');
  }
  // Hide loading no matter what after 1.5s fallback
  setTimeout(() => {
    const loading = $('page-loading');
    if (loading) { loading.classList.add('hidden'); setTimeout(() => { try { loading.remove(); } catch(e){} }, 400); }
  }, 1500);
})();


// ── Single-user app state, AI/ML Guide, and Reset tracker ───────────────
async function loadAppState(key, fallback) {
  const localKey = 'st-state-' + key;
  let local = lsGet('state-' + key);
  if (local === null) {
    try { local = JSON.parse(localStorage.getItem(localKey)); } catch(e) { local = null; }
  }
  if (!isOffline && sb && dbReady) {
    try {
      const { data, error } = await sb.from('app_state').select('value').eq('user_id', uid()).eq('key', key).single();
      if (!error && data && data.value !== null) { appStateCache[key] = data.value; return data.value; }
    } catch(e) {}
  }
  appStateCache[key] = local === null || local === undefined ? fallback : local;
  return appStateCache[key];
}

async function saveAppState(key, value) {
  appStateCache[key] = value;
  lsSet('state-' + key, value);
  try { localStorage.setItem('st-state-' + key, JSON.stringify(value)); } catch(e) {}
  if (!isOffline && sb && dbReady) {
    try {
      await sb.from('app_state').upsert({ id: uid() + '_' + key, user_id: uid(), key, value, updated_at: new Date().toISOString() });
    } catch(e) { console.warn('app_state sync failed:', key, e); }
  }
}

const AI_EMPTY = { roadmapDone: [], projects: {}, learningDone: [], expanded: [], projectFilter: 'all', learningFilter: 'all', query: '' };
let aiState = null;
function guideData(){ return window.AIML_GUIDE_DATA || { courseModules: [], projectCategories: [], mlProjects: [], subjectCategories: [], learningSubjects: [], lessons: {} }; }
async function ensureAiState(){ if (!aiState) aiState = Object.assign({}, AI_EMPTY, await loadAppState('aiml_guide', AI_EMPTY)); return aiState; }
function aiSave(){ saveAppState('aiml_guide', aiState); }
function aiProgress(){ const d = guideData(); const total = d.courseModules.reduce((n,m)=>n+m.topics.length,0); return { total, done: aiState?.roadmapDone?.length || 0 }; }
function difficultyBadge(v){ return '<span class="guide-badge">' + escHtml(v || 'item') + '</span>'; }
function setAiQuery(v){ aiState.query = v; aiSave(); renderAiRoadmap(); renderAiProjects(); renderAiLearning(); }
function copyText(text){ navigator.clipboard?.writeText(text).then(()=>toast('Copied')).catch(()=>toast('Copy failed','err')); }
function openLesson(id, title){ const d = guideData(); $('lesson-title').textContent = title; $('lesson-body').textContent = d.lessons[id] || 'No local lesson file was found for this topic.'; openModal('modal-lesson'); }
function toggleRoadmapTopic(key){ const set = new Set(aiState.roadmapDone || []); set.has(key) ? set.delete(key) : set.add(key); aiState.roadmapDone = Array.from(set); aiSave(); renderAiRoadmap(); }
function toggleAiModule(id){ const set = new Set(aiState.expanded || []); set.has(id) ? set.delete(id) : set.add(id); aiState.expanded = Array.from(set); aiSave(); renderAiRoadmap(); }
async function renderAiRoadmap(){
  await ensureAiState(); const d = guideData(); const q = (aiState.query || '').toLowerCase(); const done = new Set(aiState.roadmapDone || []); const prog = aiProgress();
  const root = $('ai-roadmap-root'); if (!root) return;
  root.innerHTML = '<div class="guide-toolbar"><input class="guide-search" placeholder="Search modules and topics..." value="' + escHtml(aiState.query || '') + '" oninput="setAiQuery(this.value)"><span class="guide-chip active">' + prog.done + '/' + prog.total + ' topics complete</span></div>' +
    '<div class="guide-grid">' + d.courseModules.filter(m => !q || m.title.toLowerCase().includes(q) || m.description.toLowerCase().includes(q) || m.topics.some(t => (t.title + t.description).toLowerCase().includes(q))).map(m => {
      const total = m.topics.length, count = m.topics.filter(t => done.has(m.id + '.' + t.id)).length, pct = total ? Math.round(count/total*100) : 0, open = (aiState.expanded || []).includes(m.id) || q;
      return '<div class="guide-card"><div class="guide-card-head"><div><h3>' + m.number + '. ' + escHtml(m.title) + '</h3><p>' + escHtml(m.description) + '</p></div><button class="mini-btn" onclick="toggleAiModule(\'' + m.id + '\')">' + (open ? 'Hide' : 'Open') + '</button></div><div class="guide-progress"><i style="width:' + pct + '%"></i></div><div class="guide-meta">' + difficultyBadge(count + '/' + total) + difficultyBadge(pct + '%') + '</div>' +
        (open ? '<div class="topic-list">' + m.topics.map(t => {
          const key = m.id + '.' + t.id, checked = done.has(key);
          const prompt = d.generateTeachingPrompt(m.id, t.id, t.title, m.title, t.description).replace(/'/g, '&#39;');
          return '<div class="topic-row"><input type="checkbox" ' + (checked?'checked':'') + ' onchange="toggleRoadmapTopic(\'' + key + '\')"><div><div class="topic-title">' + escHtml(t.title) + '</div><div class="topic-desc">' + escHtml(t.description) + '</div><div class="guide-meta">' + difficultyBadge(t.depth) + '</div></div><div class="guide-actions"><button class="mini-btn" onclick="openLesson(\'' + t.id + '\',\'' + escHtml(t.title).replace(/'/g,'&#39;') + '\')">Lesson</button><button class="mini-btn" onclick="copyText(\'' + prompt + '\')">Prompt</button></div></div>';
        }).join('') + '</div>' : '') + '</div>';
    }).join('') + '</div>';
}

function setProjectFilter(id){ aiState.projectFilter = id; aiSave(); renderAiProjects(); }
function cycleProject(id){ const order = ['','planned','active','done']; const cur = aiState.projects[id] || ''; aiState.projects[id] = order[(order.indexOf(cur)+1) % order.length]; aiSave(); renderAiProjects(); }
async function renderAiProjects(){
  await ensureAiState(); const d = guideData(); const q = (aiState.query || '').toLowerCase(); const root = $('ai-projects-root'); if (!root) return;
  const counts = Object.values(aiState.projects || {}).reduce((a,s)=>{ if(s) a[s]=(a[s]||0)+1; return a; }, {});
  const cats = [{id:'all',name:'All Projects'}].concat(d.projectCategories);
  let items = d.mlProjects.filter(p => (aiState.projectFilter === 'all' || p.category === aiState.projectFilter) && (!q || (p.title + p.tags.join(' ') + p.category).toLowerCase().includes(q)));
  root.innerHTML = '<div class="guide-toolbar"><input class="guide-search" placeholder="Search 100+ projects..." value="' + escHtml(aiState.query || '') + '" oninput="setAiQuery(this.value)"><span class="guide-chip active">' + (counts.done || 0) + ' complete</span><span class="guide-chip">' + (counts.active || 0) + ' active</span></div><div class="guide-toolbar">' +
    cats.map(c => '<button class="guide-chip ' + (aiState.projectFilter===c.id?'active':'') + '" onclick="setProjectFilter(\'' + c.id + '\')">' + escHtml(c.name) + '</button>').join('') + '</div>' +
    '<div class="guide-card"><div class="topic-list">' + items.map(p => '<div class="project-row"><button class="mini-btn" onclick="cycleProject(\'' + p.id + '\')">' + escHtml(aiState.projects[p.id] || 'queue') + '</button><div><div class="project-title">' + escHtml(p.title) + '</div><div class="project-tags">' + escHtml(p.tags.join(', ')) + '</div><div class="guide-meta">' + difficultyBadge(p.difficulty) + difficultyBadge(p.category) + '</div></div><a class="mini-btn" target="_blank" rel="noopener" href="' + escHtml(p.url) + '">Open</a></div>').join('') + '</div></div>';
}

function setLearningFilter(id){ aiState.learningFilter = id; aiSave(); renderAiLearning(); }
function toggleLearning(id){ const set = new Set(aiState.learningDone || []); set.has(id) ? set.delete(id) : set.add(id); aiState.learningDone = Array.from(set); aiSave(); renderAiLearning(); }
async function renderAiLearning(){
  await ensureAiState(); const d = guideData(); const q = (aiState.query || '').toLowerCase(); const root = $('ai-learning-root'); if (!root) return;
  const done = new Set(aiState.learningDone || []); const cats = [{id:'all',name:'All'}].concat(d.subjectCategories);
  const prefix = { 'python-fundamentals':['python'], 'math-foundation':['linear','matrices','eigen','calculus','gradient','probability','bayesian','hypothesis'], 'data-wrangling':['numpy','pandas','eda','sql'], 'data-visualization':['matplotlib','seaborn','plotly'], 'ml-algorithms':['ml','supervised','linear-regression','logistic','decision','random','svm','knn','kmeans','pca'], 'deep-learning':['dl','neural','cnn','rnn','lstm','tensorflow','keras','pytorch','gan'], 'nlp':['nlp','text','word','spacy','nltk'], 'genai':['genai','llm','prompt','rag','langchain','vector','agents','diffusion','openai','huggingface'], 'mlops':['docker','deployment','fastapi','mlflow','kubernetes','monitoring','ci-cd','aws'], 'dsa':['arrays','linked','trees','sorting','dynamic','hashing','binary','stacks','ml-interview','system-design'] };
  let items = d.learningSubjects.filter(s => {
    const catOk = aiState.learningFilter === 'all' || (prefix[aiState.learningFilter] || [aiState.learningFilter]).some(p => s.id.includes(p) || s.tags.join(' ').includes(p));
    const qOk = !q || (s.title + s.tags.join(' ')).toLowerCase().includes(q);
    return catOk && qOk;
  });
  root.innerHTML = '<div class="guide-toolbar"><input class="guide-search" placeholder="Search learning links..." value="' + escHtml(aiState.query || '') + '" oninput="setAiQuery(this.value)"><span class="guide-chip active">' + done.size + '/' + d.learningSubjects.length + ' complete</span></div><div class="guide-toolbar">' +
    cats.map(c => '<button class="guide-chip ' + (aiState.learningFilter===c.id?'active':'') + '" onclick="setLearningFilter(\'' + c.id + '\')">' + escHtml(c.name) + '</button>').join('') + '</div>' +
    '<div class="guide-card"><div class="topic-list">' + items.map(s => '<div class="learn-row"><input type="checkbox" ' + (done.has(s.id)?'checked':'') + ' onchange="toggleLearning(\'' + s.id + '\')"><div><div class="learn-title">' + escHtml(s.title) + '</div><div class="learn-tags">' + escHtml(s.tags.join(', ')) + '</div><div class="guide-meta">' + difficultyBadge(s.difficulty) + '</div></div><a class="mini-btn" target="_blank" rel="noopener" href="' + escHtml(s.url) + '">Open</a></div>').join('') + '</div></div>';
}

// Lesson modal is injected once so index stays compact.
document.addEventListener('DOMContentLoaded', () => {
  if (!$('modal-lesson')) {
    const el = document.createElement('div');
    el.className = 'overlay'; el.id = 'modal-lesson'; el.onclick = e => { if (e.target === el) closeModal('modal-lesson'); };
    el.innerHTML = '<div class="modal" style="max-width:900px"><div class="modal-title"><span id="lesson-title">Lesson</span><button class="modal-close" onclick="closeModal(\'modal-lesson\')"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button></div><div id="lesson-body" class="lesson-viewer"></div></div>';
    document.body.appendChild(el);
  }
});

const RESET_BLOCKS = [
 {id:'quant',name:'SSC Quant',time:'11:00-12:15',mins:75},{id:'reasoning',name:'SSC Reasoning',time:'12:30-13:15',mins:45},{id:'mistake',name:'SSC Mistake Review',time:'13:15-14:00',mins:45},{id:'coding',name:'Coding / DSA',time:'15:30-16:45',mins:75},{id:'aiml',name:'AI/ML Project',time:'17:00-18:15',mins:75},{id:'gk',name:'SSC GK',time:'18:30-19:15',mins:45},{id:'english',name:'SSC English',time:'19:15-20:00',mins:45},{id:'job',name:'Job Applications',time:'20:00-20:30',mins:30},{id:'gate',name:'GATE DA/CSE',time:'22:15-23:15',mins:60},{id:'review',name:'Daily Review & Plan',time:'23:15-23:40',mins:25}
];
let resetState = null, resetDate = todayStr(), resetTab = 'today';
function blankResetDay(){ const blocks={}; RESET_BLOCKS.forEach(b=>blocks[b.id]={status:'',quality:0,note:''}); return {wake:'',sleep:'',gym:'',mood:'',focus:'',followed:'',win:'',mistake:'',tomorrow:'',floor:{quant:false,coding:false,gk:false,jobs:false,plan:false},blocks,ssc:{topic:'',attempted:'',correct:'',wrong:'',gk:'',english:'',mock:'',revise:''},job:{apps:[]},code:{topic:'',solved:'',difficulty:'',link:'',concept:'',project:'',github:'',learning:''},gateT:{subject:'',topic:'',concept:'',questions:'',accuracy:'',pyq:'',weak:'',revision:''},dop:{manga:'',gaming:'',social:'',relapse:'',preWork:'',withinLimit:'',trigger:''}}; }
async function ensureReset(){ if (!resetState) resetState = Object.assign({days:{},weekly:{},settings:{sscDate:'2026-08-15',gateDate:'2027-02-07'}}, await loadAppState('reset_tracker_v1', {days:{},weekly:{},settings:{sscDate:'2026-08-15',gateDate:'2027-02-07'}})); return resetState; }
function rd(){ if(!resetState.days[resetDate]) resetState.days[resetDate]=blankResetDay(); return resetState.days[resetDate]; }
function resetSave(){ saveAppState('reset_tracker_v1', resetState); }
function rShift(s,n){ return dateShift(s,n); }
function rScore(day){ const W={done:1,partial:.5,missed:0,'':0}, b=day.blocks; const ssc=['quant','reasoning','mistake','gk','english'].reduce((a,id)=>a+W[b[id].status]*5,0); const job=Math.min((day.job.apps||[]).length/5,1)*15+W[b.job.status]*10; const code=W[b.coding.status]*10+W[b.aiml.status]*10; const gate=W[b.gate.status]*10; const health=(day.gym==='yes'?7:0)+(day.wake && day.wake<='08:30'?3:0); const dop=(day.dop.withinLimit==='yes'?5:0)+(day.dop.preWork==='no'?3:0)+(day.dop.relapse==='no'?2:0); return Math.round(ssc+job+code+gate+health+dop); }
function rTier(s){ return s>=85?'Elite Day':s>=70?'Strong Day':s>=50?'Survival Day':'Reset Day'; }
function rSet(path,val){ const parts=path.split('.'); let o=rd(); for(let i=0;i<parts.length-1;i++) o=o[parts[i]]; o[parts.at(-1)] = val; resetSave(); renderResetTracker(); }
function rTab(t){ resetTab=t; renderResetTracker(); }
function rDate(v){ resetDate=v; renderResetTracker(); }
function rQual(id,n){ const b=rd().blocks[id]; b.quality=b.quality===n?0:n; resetSave(); renderResetTracker(); }
function rApp(action,i,k,v){ const a=rd().job.apps; if(action==='add') a.push({company:'',role:'',link:'',status:'Applied',gap:''}); if(action==='del') a.splice(i,1); if(action==='set') a[i][k]=v; resetSave(); renderResetTracker(); }
function rMarkAll(s){ RESET_BLOCKS.forEach(b=>rd().blocks[b.id].status=s); resetSave(); renderResetTracker(); }
function rFloorMet(day){ const f=day.floor; return f.quant&&f.coding&&f.gk&&f.jobs&&f.plan; }
function rHistory(){ return Object.keys(resetState.days).sort().reverse(); }
function renderResetToday(){ const day=rd(), score=rScore(day); const blockHtml=RESET_BLOCKS.map(b=>{ const x=day.blocks[b.id]; return '<div class="reset-block ' + x.status + '"><div class="wide"><b>' + escHtml(b.name) + '</b><br><small>' + b.time + ' · ' + b.mins + 'm</small></div><select onchange="rSet(\'blocks.'+b.id+'.status\',this.value)"><option value="">status</option><option value="done" '+(x.status==='done'?'selected':'')+'>Done</option><option value="partial" '+(x.status==='partial'?'selected':'')+'>Partial</option><option value="missed" '+(x.status==='missed'?'selected':'')+'>Missed</option></select><div class="reset-stars">' + [1,2,3,4,5].map(n=>'<button class="'+(n<=x.quality?'on':'')+'" onclick="rQual(\''+b.id+'\','+n+')">★</button>').join('') + '</div><input class="reset-input" value="'+escHtml(x.note)+'" placeholder="note" oninput="rSet(\'blocks.'+b.id+'.note\',this.value)"></div>'; }).join('');
  const floor = [['quant','30m Quant'],['coding','30m Coding'],['gk','15m GK'],['jobs','5 applications'],['plan','10m planning']].map(([k,t])=>'<label class="reset-check '+(day.floor[k]?'on':'')+'"><input type="checkbox" '+(day.floor[k]?'checked':'')+' onchange="rSet(\'floor.'+k+'\',this.checked)">'+t+'</label>').join('');
  const apps = (day.job.apps||[]).map((a,i)=>'<div class="reset-block"><input class="reset-input" placeholder="Company" value="'+escHtml(a.company)+'" oninput="rApp(\'set\','+i+',\'company\',this.value)"><input class="reset-input" placeholder="Role" value="'+escHtml(a.role)+'" oninput="rApp(\'set\','+i+',\'role\',this.value)"><select onchange="rApp(\'set\','+i+',\'status\',this.value)">'+['Applied','Screening','Interview','Offer','Rejected'].map(s=>'<option '+(a.status===s?'selected':'')+'>'+s+'</option>').join('')+'</select><button class="mini-btn" onclick="rApp(\'del\','+i+')">Remove</button></div>').join('');
  return '<div class="reset-toolbar"><button class="mini-btn" onclick="rDate(rShift(resetDate,-1))">Prev</button><input class="reset-input" type="date" value="'+resetDate+'" onchange="rDate(this.value)"><button class="mini-btn" onclick="rDate(rShift(resetDate,1))">Next</button><button class="mini-btn" onclick="rDate(todayStr())">Today</button></div><div class="reset-card"><div class="reset-score"><div class="reset-ring" style="--score:'+score+'"><b>'+score+'</b><span>/100</span></div><div><h2>'+rTier(score)+'</h2><p class="sub">Consistency over motivation. A reset day is not a failure.</p><div class="guide-meta">'+difficultyBadge('floor '+(rFloorMet(day)?'cleared':'open'))+difficultyBadge('apps '+(day.job.apps||[]).length)+'</div></div></div></div><div class="reset-card"><h2>Bad-day minimum</h2><p class="sub">Clear this floor to keep the chain alive.</p><div class="reset-checks">'+floor+'</div></div><div class="reset-card"><h2>Daily log</h2><div class="reset-grid">'+
  [['wake','Wake-up','time'],['sleep','Sleep','time'],['mood','Mood 1-10','number'],['focus','Focus 1-10','number']].map(f=>'<label class="reset-field">'+f[1]+'<input type="'+f[2]+'" value="'+escHtml(day[f[0]])+'" oninput="rSet(\''+f[0]+'\',this.value)"></label>').join('')+
  '<label class="reset-field">Gym<select onchange="rSet(\'gym\',this.value)"><option value="">-</option><option value="yes" '+(day.gym==='yes'?'selected':'')+'>Yes</option><option value="no" '+(day.gym==='no'?'selected':'')+'>No</option></select></label><label class="reset-field">Win<textarea oninput="rSet(\'win\',this.value)">'+escHtml(day.win)+'</textarea></label><label class="reset-field">Mistake<textarea oninput="rSet(\'mistake\',this.value)">'+escHtml(day.mistake)+'</textarea></label><label class="reset-field">Tomorrow first task<textarea oninput="rSet(\'tomorrow\',this.value)">'+escHtml(day.tomorrow)+'</textarea></label></div></div><div class="reset-card"><h2>Study blocks</h2><div class="guide-toolbar"><button class="mini-btn" onclick="rMarkAll(\'done\')">Mark all done</button><button class="mini-btn" onclick="rMarkAll(\'missed\')">Mark all missed</button></div>'+blockHtml+'</div>'+
  '<div class="reset-card"><h2>SSC, jobs, coding, GATE, dopamine</h2><div class="reset-grid"><label class="reset-field">SSC topic<input value="'+escHtml(day.ssc.topic)+'" oninput="rSet(\'ssc.topic\',this.value)"></label><label class="reset-field">SSC attempted<input type="number" value="'+escHtml(day.ssc.attempted)+'" oninput="rSet(\'ssc.attempted\',this.value)"></label><label class="reset-field">Correct<input type="number" value="'+escHtml(day.ssc.correct)+'" oninput="rSet(\'ssc.correct\',this.value)"></label><label class="reset-field">AI/ML project<input value="'+escHtml(day.code.project)+'" oninput="rSet(\'code.project\',this.value)"></label><label class="reset-field">DSA solved<input type="number" value="'+escHtml(day.code.solved)+'" oninput="rSet(\'code.solved\',this.value)"></label><label class="reset-field">GATE topic<input value="'+escHtml(day.gateT.topic)+'" oninput="rSet(\'gateT.topic\',this.value)"></label><label class="reset-field">Distraction minutes<input type="number" value="'+escHtml((+day.dop.manga||0)+(+day.dop.gaming||0)+(+day.dop.social||0))+'" disabled></label><label class="reset-field">Within healthy limit<select onchange="rSet(\'dop.withinLimit\',this.value)"><option value="">-</option><option value="yes" '+(day.dop.withinLimit==='yes'?'selected':'')+'>Yes</option><option value="no" '+(day.dop.withinLimit==='no'?'selected':'')+'>No</option></select></label></div><h2 style="margin-top:16px">Job applications</h2>'+apps+'<button class="mini-btn" onclick="rApp(\'add\')">Add application</button></div>';
}
function renderResetDash(){ const ds=rHistory().slice(0,14).reverse(); const scores=ds.map(d=>rScore(resetState.days[d])); const avg=scores.length?Math.round(scores.reduce((a,b)=>a+b,0)/scores.length):0; const best=scores.length?Math.max(...scores):0; const apps=Object.values(resetState.days).reduce((n,d)=>n+(d.job.apps||[]).length,0); return '<div class="reset-grid"><div class="reset-card"><h2>'+avg+'</h2><p class="sub">Average score</p></div><div class="reset-card"><h2>'+best+'</h2><p class="sub">Best score</p></div><div class="reset-card"><h2>'+apps+'</h2><p class="sub">Applications logged</p></div><div class="reset-card"><h2>'+Object.keys(resetState.days).length+'</h2><p class="sub">Days tracked</p></div></div><div class="reset-card"><h2>Last 14 scores</h2><div class="sparkline-row">'+scores.map(s=>'<span title="'+s+'" style="display:inline-block;width:18px;height:'+Math.max(4,s)+'px;max-height:100px;background:var(--accent);border-radius:4px;margin-right:5px;vertical-align:bottom"></span>').join('')+'</div></div>'; }
function renderResetHistory(){ const rows=rHistory().map(d=>'<div class="reset-history-row" onclick="resetDate=\''+d+'\';resetTab=\'today\';renderResetTracker()"><div><b>'+d+'</b><div class="reset-muted">'+fmtDay(d)+' · '+(rFloorMet(resetState.days[d])?'floor cleared':'floor open')+'</div></div><b>'+rScore(resetState.days[d])+'</b><span class="guide-badge">'+rTier(rScore(resetState.days[d]))+'</span></div>').join(''); return '<div class="reset-card"><h2>History</h2>'+(rows||'<p class="sub">No days logged yet.</p>')+'</div>'; }
async function renderResetTracker(){ await ensureReset(); const root=$('reset-tracker-root'); if(!root) return; root.innerHTML='<div class="reset-tabs">'+['today','dash','history'].map(t=>'<button class="reset-chip '+(resetTab===t?'active':'')+'" onclick="rTab(\''+t+'\')">'+(t==='dash'?'Dashboard':t[0].toUpperCase()+t.slice(1))+'</button>').join('')+'</div>'+(resetTab==='today'?renderResetToday():resetTab==='dash'?renderResetDash():renderResetHistory()); }


// ── Enhanced AI and Reset renderers v2 ─────────────────────────────
function aimlBadge(v){ return '<span class="aiml-badge">' + escHtml(v || '') + '</span>'; }
function aimlStatsHtml(kind){
  const d=guideData(); const prog=aiProgress(); const pd=Object.values(aiState?.projects||{}).filter(x=>x==='done').length; const ld=(aiState?.learningDone||[]).length;
  return '<div class="aiml-stats"><div class="aiml-stat"><b>'+prog.done+'/'+prog.total+'</b><span>Roadmap</span></div><div class="aiml-stat"><b>'+pd+'/'+d.mlProjects.length+'</b><span>Projects</span></div><div class="aiml-stat"><b>'+ld+'/'+d.learningSubjects.length+'</b><span>Links</span></div></div>';
}
function aimlHero(title, sub, kicker){ return '<div class="aiml-hero"><div><div class="aiml-kicker">'+escHtml(kicker||'AI ML Guide')+'</div><div class="aiml-title">'+escHtml(title)+'</div><div class="aiml-sub">'+escHtml(sub)+'</div></div>'+aimlStatsHtml()+'</div>'; }
renderAiRoadmap = async function(){
  await ensureAiState(); const d=guideData(); const q=(aiState.query||'').toLowerCase(); const done=new Set(aiState.roadmapDone||[]); const prog=aiProgress(); const root=$('ai-roadmap-root'); if(!root) return;
  const modules=d.courseModules.filter(m=>!q||m.title.toLowerCase().includes(q)||m.description.toLowerCase().includes(q)||m.topics.some(t=>(t.title+' '+t.description).toLowerCase().includes(q)));
  root.innerHTML='<div class="aiml-shell">'+aimlHero('AI/ML Roadmap','The original AI ML Guide curriculum, now merged into StudyTrack with persistent progress and local lesson notes.','Curriculum')+
  '<div class="aiml-toolbar"><input class="aiml-search" value="'+escHtml(aiState.query||'')+'" placeholder="Search modules, topics, math, ML, GenAI..." oninput="setAiQuery(this.value)"><button class="aiml-chip active">'+prog.done+' complete</button><button class="aiml-chip">'+prog.total+' topics</button></div><div class="aiml-grid">'+
  modules.map(m=>{ const total=m.topics.length,count=m.topics.filter(t=>done.has(m.id+'.'+t.id)).length,pct=total?Math.round(count/total*100):0,open=(aiState.expanded||[]).includes(m.id)||q; return '<div class="aiml-card"><div style="display:flex;justify-content:space-between;gap:12px;align-items:start"><div><h3>'+m.number+'. '+escHtml(m.title)+'</h3><p>'+escHtml(m.description)+'</p></div><button class="aiml-btn" onclick="toggleAiModule(\''+m.id+'\')">'+(open?'Collapse':'Open')+'</button></div><div class="aiml-progress"><i style="width:'+pct+'%"></i></div>'+aimlBadge(count+'/'+total)+aimlBadge(pct+'%')+(open?'<div>'+m.topics.map(t=>{ const key=m.id+'.'+t.id, checked=done.has(key); const prompt=d.generateTeachingPrompt(m.id,t.id,t.title,m.title,t.description).replace(/'/g,'&#39;').replace(/\n/g,'\\n'); return '<div class="aiml-topic"><input type="checkbox" '+(checked?'checked':'')+' onchange="toggleRoadmapTopic(\''+key+'\')"><div><div class="aiml-topic-title">'+escHtml(t.title)+'</div><div class="aiml-topic-desc">'+escHtml(t.description)+'</div>'+aimlBadge(t.depth)+'</div><div class="aiml-actions"><button class="aiml-btn" onclick="openLesson(\''+t.id+'\',\''+escHtml(t.title).replace(/'/g,'&#39;')+'\')">Lesson</button><button class="aiml-btn" onclick="copyText(\''+prompt+'\')">Prompt</button></div></div>';}).join('')+'</div>':'')+'</div>'; }).join('')+'</div></div>';
};
renderAiProjects = async function(){
  await ensureAiState(); const d=guideData(); const q=(aiState.query||'').toLowerCase(); const root=$('ai-projects-root'); if(!root) return; const cats=[{id:'all',name:'All Projects'}].concat(d.projectCategories); const items=d.mlProjects.filter(p=>(aiState.projectFilter==='all'||p.category===aiState.projectFilter)&&(!q||(p.title+' '+p.tags.join(' ')+' '+p.category).toLowerCase().includes(q))); const counts=Object.values(aiState.projects||{}).reduce((a,x)=>{if(x)a[x]=(a[x]||0)+1;return a;},{});
  root.innerHTML='<div class="aiml-shell">'+aimlHero('ML Project Tracker','A 101-project project bank from the AI ML Guide. Cycle each item through queue, planned, active, and done.','Practice')+'<div class="aiml-toolbar"><input class="aiml-search" value="'+escHtml(aiState.query||'')+'" placeholder="Search projects, tags, CV, NLP, finance..." oninput="setAiQuery(this.value)"><span class="aiml-chip active">'+(counts.done||0)+' done</span><span class="aiml-chip">'+(counts.active||0)+' active</span><span class="aiml-chip">'+items.length+' shown</span></div><div class="aiml-toolbar">'+cats.map(c=>'<button class="aiml-chip '+(aiState.projectFilter===c.id?'active':'')+'" onclick="setProjectFilter(\''+c.id+'\')">'+escHtml(c.name)+'</button>').join('')+'</div><div class="aiml-grid">'+items.map(p=>'<div class="aiml-card"><div style="display:flex;justify-content:space-between;gap:12px"><h3>'+escHtml(p.title)+'</h3><button class="aiml-btn" onclick="cycleProject(\''+p.id+'\')">'+escHtml(aiState.projects[p.id]||'queue')+'</button></div>'+aimlBadge(p.difficulty)+aimlBadge(p.category)+'<p>'+escHtml(p.tags.join(', '))+'</p><a class="aiml-btn aiml-link" target="_blank" rel="noopener" href="'+escHtml(p.url)+'">Open project</a></div>').join('')+'</div></div>';
};
renderAiLearning = async function(){
  await ensureAiState(); const d=guideData(); const q=(aiState.query||'').toLowerCase(); const root=$('ai-learning-root'); if(!root) return; const done=new Set(aiState.learningDone||[]); const cats=[{id:'all',name:'All'}].concat(d.subjectCategories); const category=d.subjectCategories.find(c=>c.id===aiState.learningFilter); const relevance=category?category.roleRelevance.join(' · '):'Data Scientist · ML Engineer · AI Engineer · GenAI Engineer';
  const items=d.learningSubjects.filter(x=>(aiState.learningFilter==='all'||x.tags.includes(aiState.learningFilter)||x.id.includes(aiState.learningFilter.split('-')[0])||x.tags.some(t=>aiState.learningFilter.includes(t)))&&(!q||(x.title+' '+x.tags.join(' ')).toLowerCase().includes(q)));
  root.innerHTML='<div class="aiml-shell">'+aimlHero('Learning Links','Role-focused tutorials and references for Data Science, ML Engineering, AI Engineering, and GenAI paths. '+relevance,'Reference')+'<div class="aiml-toolbar"><input class="aiml-search" value="'+escHtml(aiState.query||'')+'" placeholder="Search links, tags, roles..." oninput="setAiQuery(this.value)"><span class="aiml-chip active">'+done.size+'/'+d.learningSubjects.length+' complete</span></div><div class="aiml-toolbar">'+cats.map(c=>'<button class="aiml-chip '+(aiState.learningFilter===c.id?'active':'')+'" onclick="setLearningFilter(\''+c.id+'\')">'+escHtml(c.name)+'</button>').join('')+'</div><div class="aiml-grid">'+items.map(x=>'<div class="aiml-card"><div class="aiml-topic" style="grid-template-columns:auto 1fr auto;margin-top:0"><input type="checkbox" '+(done.has(x.id)?'checked':'')+' onchange="toggleLearning(\''+x.id+'\')"><div><div class="aiml-topic-title">'+escHtml(x.title)+'</div><div class="aiml-topic-desc">'+escHtml(x.tags.join(', '))+'</div>'+aimlBadge(x.difficulty)+'</div><a class="aiml-btn aiml-link" target="_blank" rel="noopener" href="'+escHtml(x.url)+'">Open</a></div></div>').join('')+'</div></div>';
};

let resetTimer=null, resetTimerInt=null, resetReminderInt=null, resetReminderOn=false;
function resetToast(m){ toast(m); }
function rExport(){ const blob=new Blob([JSON.stringify(resetState,null,2)],{type:'application/json'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='reset-tracker-'+todayStr()+'.json'; a.click(); resetToast('Reset backup downloaded'); }
function rImportFile(e){ const f=e.target.files[0]; if(!f)return; const r=new FileReader(); r.onload=()=>{ try{ const d=JSON.parse(r.result); if(d.days){ resetState=Object.assign({weekly:{},settings:{}},d); resetSave(); renderResetTracker(); resetToast('Reset backup restored'); } }catch(err){ toast('Could not import reset backup','err'); } }; r.readAsText(f); }
function rCopyYesterday(){ const prev=resetState.days[dateShift(resetDate,-1)]; if(!prev){toast('No previous day to copy','err'); return;} resetState.days[resetDate]=JSON.parse(JSON.stringify(prev)); resetSave(); renderResetTracker(); }
function rResetDay(){ if(!confirm('Clear Reset tracker data for '+resetDate+'?')) return; delete resetState.days[resetDate]; resetSave(); renderResetTracker(); }
function rProductiveHours(day){ const W={done:1,partial:.5,missed:0,'':0}; let m=0; RESET_BLOCKS.forEach(b=>{ if(b.id!=='review') m+=b.mins*W[day.blocks[b.id].status]; }); return Math.round(m/6)/10; }
function rAccuracy(day){ const a=+day.ssc.attempted,c=+day.ssc.correct; return a?Math.round(c/a*100)+'%':'-'; }
function rStartTimer(id){ const b=RESET_BLOCKS.find(x=>x.id===id); if(!b)return; if(resetTimerInt) clearInterval(resetTimerInt); resetTimer={id,remaining:b.mins*60,paused:false}; resetTimerInt=setInterval(()=>{ if(!resetTimer||resetTimer.paused)return; resetTimer.remaining--; if(resetTimer.remaining<=0){ rStopTimer(false); rSet('blocks.'+id+'.status','done'); toast(b.name+' complete'); } else rTimerUI(); },1000); renderResetTracker(); }
function rStopTimer(render=true){ if(resetTimerInt) clearInterval(resetTimerInt); resetTimerInt=null; resetTimer=null; if(render) renderResetTracker(); }
function rPauseTimer(){ if(resetTimer){ resetTimer.paused=!resetTimer.paused; rTimerUI(); } }
function rAddTimer(min){ if(resetTimer){ resetTimer.remaining=Math.max(60, resetTimer.remaining+min*60); rTimerUI(); } }
function rTimerUI(){ const el=$('reset-timer-time'); if(!el||!resetTimer)return; const m=Math.floor(resetTimer.remaining/60),s=resetTimer.remaining%60; el.textContent=String(m).padStart(2,'0')+':'+String(s).padStart(2,'0'); const p=$('reset-timer-pause'); if(p) p.textContent=resetTimer.paused?'Resume':'Pause'; }
function rToggleReminders(){ resetReminderOn=!resetReminderOn; if(resetReminderOn){ if(!resetReminderInt) resetReminderInt=setInterval(rCheckReminder,15000); toast('Reset block reminders on'); } else { if(resetReminderInt) clearInterval(resetReminderInt); resetReminderInt=null; toast('Reset block reminders off'); } renderResetTracker(); }
function rCheckReminder(){ const hm=new Date().toTimeString().slice(0,5); RESET_BLOCKS.forEach(b=>{ const st=b.time.slice(0,5); const key=todayStr()+'-'+b.id; resetState.settings.fired=resetState.settings.fired||{}; if(hm===st&&!resetState.settings.fired[key]){ resetState.settings.fired[key]=true; resetSave(); toast('Time for '+b.name); } }); }
function rField(label,path,value,type='text'){ return '<label class="reset-field">'+label+'<input type="'+type+'" value="'+escHtml(value||'')+'" oninput="rSet(\''+path+'\',this.value)"></label>'; }
function rSelect(label,path,value,opts){ return '<label class="reset-field">'+label+'<select onchange="rSet(\''+path+'\',this.value)"><option value="">-</option>'+opts.map(o=>'<option value="'+o+'" '+(value===o?'selected':'')+'>'+o+'</option>').join('')+'</select></label>'; }
function renderResetToday(){ const day=rd(), score=rScore(day); const apps=day.job.apps||[]; const timerBlock=resetTimer?RESET_BLOCKS.find(b=>b.id===resetTimer.id):null;
 const blocks=RESET_BLOCKS.map(b=>{ const x=day.blocks[b.id]; return '<div class="reset-block '+x.status+'"><div class="wide"><b>'+escHtml(b.name)+'</b><br><small>'+b.time+' · '+b.mins+'m</small></div><button class="mini-btn" onclick="rStartTimer(\''+b.id+'\')">'+(resetTimer&&resetTimer.id===b.id?'Running':'Start')+'</button><select onchange="rSet(\'blocks.'+b.id+'.status\',this.value)"><option value="">status</option><option value="done" '+(x.status==='done'?'selected':'')+'>Done</option><option value="partial" '+(x.status==='partial'?'selected':'')+'>Partial</option><option value="missed" '+(x.status==='missed'?'selected':'')+'>Missed</option></select><div class="reset-stars">'+[1,2,3,4,5].map(n=>'<button class="'+(n<=x.quality?'on':'')+'" onclick="rQual(\''+b.id+'\','+n+')">★</button>').join('')+'</div><input class="reset-input wide" value="'+escHtml(x.note)+'" placeholder="notes / mistakes" oninput="rSet(\'blocks.'+b.id+'.note\',this.value)"></div>'; }).join('');
 const floor=[['quant','30 min Quant'],['coding','30 min Coding'],['gk','15 min GK'],['jobs','5 job applications'],['plan','10 min planning']].map(([k,t])=>'<label class="reset-check '+(day.floor[k]?'on':'')+'"><input type="checkbox" '+(day.floor[k]?'checked':'')+' onchange="rSet(\'floor.'+k+'\',this.checked)">'+t+'</label>').join('');
 const appHtml=apps.map((a,i)=>'<div class="reset-block"><input class="reset-input" placeholder="Company" value="'+escHtml(a.company)+'" oninput="rApp(\'set\','+i+',\'company\',this.value)"><input class="reset-input" placeholder="Role" value="'+escHtml(a.role)+'" oninput="rApp(\'set\','+i+',\'role\',this.value)"><select onchange="rApp(\'set\','+i+',\'status\',this.value)">'+['Applied','Screening','Interview','Offer','Rejected'].map(s=>'<option '+(a.status===s?'selected':'')+'>'+s+'</option>').join('')+'</select><button class="mini-btn" onclick="rApp(\'del\','+i+')">Remove</button><input class="reset-input wide" placeholder="Job link / resume gap" value="'+escHtml((a.link||'')+(a.gap?' · '+a.gap:''))+'" oninput="rApp(\'set\','+i+',\'link\',this.value)"></div>').join('');
 return '<div class="reset-toolbar"><button class="mini-btn" onclick="rDate(dateShift(resetDate,-1))">Prev</button><input class="reset-input" type="date" value="'+resetDate+'" onchange="rDate(this.value)"><button class="mini-btn" onclick="rDate(dateShift(resetDate,1))">Next</button><button class="mini-btn" onclick="rDate(todayStr())">Today</button><button class="mini-btn" onclick="rExport()">Backup</button><button class="mini-btn" onclick="$(\'reset-import\').click()">Restore</button><input id="reset-import" type="file" accept="application/json" style="display:none" onchange="rImportFile(event)"><button class="mini-btn" onclick="rToggleReminders()">Reminders '+(resetReminderOn?'On':'Off')+'</button></div><div class="reset-card"><div class="reset-score"><div class="reset-ring" style="--score:'+score+'"><b>'+score+'</b><span>/100</span></div><div><h2>'+rTier(score)+'</h2><p class="sub">Productive hours: <b>'+rProductiveHours(day)+'h</b> · Quant accuracy: <b>'+rAccuracy(day)+'</b></p><div class="guide-meta">'+difficultyBadge(rFloorMet(day)?'floor cleared':'floor open')+difficultyBadge(apps.length+' apps')+'</div></div></div></div><div class="reset-card"><h2>Bad-day minimum</h2><p class="sub">On low days, clear this floor and keep the streak alive.</p><div class="reset-checks">'+floor+'</div></div><div class="reset-card"><h2>Daily log</h2><div class="reset-grid">'+rField('Wake-up','wake',day.wake,'time')+rField('Sleep','sleep',day.sleep,'time')+rSelect('Gym done?','gym',day.gym,['yes','no'])+rSelect('Followed timetable?','followed',day.followed,['Yes','Partially','No'])+rField('Mood 1-10','mood',day.mood,'number')+rField('Focus 1-10','focus',day.focus,'number')+'<label class="reset-field">Biggest win<textarea oninput="rSet(\'win\',this.value)">'+escHtml(day.win)+'</textarea></label><label class="reset-field">Biggest mistake<textarea oninput="rSet(\'mistake\',this.value)">'+escHtml(day.mistake)+'</textarea></label><label class="reset-field">Tomorrow first task<textarea oninput="rSet(\'tomorrow\',this.value)">'+escHtml(day.tomorrow)+'</textarea></label></div></div><div class="reset-card"><h2>Study blocks</h2><div class="guide-toolbar"><button class="mini-btn" onclick="rCopyYesterday()">Copy yesterday</button><button class="mini-btn" onclick="rMarkAll(\'done\')">Mark all done</button><button class="mini-btn" onclick="rMarkAll(\'missed\')">Mark all missed</button><button class="mini-btn" onclick="rResetDay()">Reset day</button></div><div class="reset-table">'+blocks+'</div></div><div class="reset-card"><h2>SSC / Job / Coding / GATE / Dopamine</h2><div class="reset-section-title">SSC CGL</div><div class="reset-grid">'+rField('Quant topic','ssc.topic',day.ssc.topic)+rField('Attempted','ssc.attempted',day.ssc.attempted,'number')+rField('Correct','ssc.correct',day.ssc.correct,'number')+rField('Wrong','ssc.wrong',day.ssc.wrong,'number')+rField('GK topics','ssc.gk',day.ssc.gk)+rField('English topic','ssc.english',day.ssc.english)+rField('Mock score','ssc.mock',day.ssc.mock)+rField('Weak formulas','ssc.revise',day.ssc.revise)+'</div><div class="reset-section-title">Job applications</div>'+appHtml+'<button class="mini-btn" onclick="rApp(\'add\')">Add application</button><div class="reset-section-title">Coding / DSA / AI-ML</div><div class="reset-grid">'+rField('DSA topic','code.topic',day.code.topic)+rField('Problems solved','code.solved',day.code.solved,'number')+rSelect('Difficulty','code.difficulty',day.code.difficulty,['Easy','Medium','Hard','Mixed'])+rField('Problem link','code.link',day.code.link)+rField('Concept learned','code.concept',day.code.concept)+rField('AI/ML project','code.project',day.code.project)+rField('GitHub commit','code.github',day.code.github)+rField('Practical learning','code.learning',day.code.learning)+'</div><div class="reset-section-title">GATE 2027</div><div class="reset-grid">'+rSelect('Subject','gateT.subject',day.gateT.subject,['DA','CSE'])+rField('Topic','gateT.topic',day.gateT.topic)+rField('Questions','gateT.questions',day.gateT.questions,'number')+rField('Accuracy %','gateT.accuracy',day.gateT.accuracy,'number')+rField('Concept','gateT.concept',day.gateT.concept)+rSelect('PYQ attempted?','gateT.pyq',day.gateT.pyq,['Yes','No'])+rField('Weak area','gateT.weak',day.gateT.weak)+rField('Revision date','gateT.revision',day.gateT.revision,'date')+'</div><div class="reset-section-title">Dopamine / distraction</div><div class="reset-grid">'+rField('Manga/manhwa min','dop.manga',day.dop.manga,'number')+rField('Gaming min','dop.gaming',day.dop.gaming,'number')+rField('Social min','dop.social',day.dop.social,'number')+rSelect('Within healthy limit?','dop.withinLimit',day.dop.withinLimit,['yes','no'])+rSelect('Dopamine before work?','dop.preWork',day.dop.preWork,['yes','no'])+rSelect('Relapse?','dop.relapse',day.dop.relapse,['yes','no'])+rField('Trigger and next action','dop.trigger',day.dop.trigger)+'</div></div>'+(resetTimer?'<div class="reset-timer-panel"><b>'+escHtml(timerBlock?.name||'Timer')+'</b><span class="reset-timer-time" id="reset-timer-time">00:00</span><button class="mini-btn" id="reset-timer-pause" onclick="rPauseTimer()">Pause</button><button class="mini-btn" onclick="rAddTimer(5)">+5</button><button class="mini-btn" onclick="rAddTimer(15)">+15</button><button class="mini-btn" onclick="rStopTimer()">Stop</button></div>':''); }
function mondayOfLocal(s){ const d=new Date(s+'T12:00:00'); const wd=(d.getDay()+6)%7; d.setDate(d.getDate()-wd); return todayStrFromDate(d); }
function todayStrFromDate(d){ return d.getFullYear()+'-'+p2(d.getMonth()+1)+'-'+p2(d.getDate()); }
function renderResetWeek(){ const mon=mondayOfLocal(resetDate); const days=[]; for(let i=0;i<7;i++)days.push(dateShift(mon,i)); const logged=days.filter(d=>resetState.days[d]); const note=resetState.weekly[mon]||{}; const total=logged.reduce((n,d)=>n+rProductiveHours(resetState.days[d]),0); const apps=logged.reduce((n,d)=>n+(resetState.days[d].job.apps||[]).length,0); const scores=logged.map(d=>rScore(resetState.days[d])); return '<div class="reset-print"><h2>Reset Weekly Review</h2><p>'+mon+' to '+dateShift(mon,6)+'</p></div><div class="reset-grid"><div class="reset-card"><h2>'+logged.length+'/7</h2><p class="sub">Days logged</p></div><div class="reset-card"><h2>'+total.toFixed(1)+'h</h2><p class="sub">Productive hours</p></div><div class="reset-card"><h2>'+apps+'</h2><p class="sub">Jobs applied</p></div><div class="reset-card"><h2>'+(scores.length?Math.max(...scores):0)+'</h2><p class="sub">Best score</p></div></div><div class="reset-card"><h2>Reflect and set one improvement</h2><div class="reset-grid">'+rField('Repeated distraction','weekly.'+mon+'.distraction',note.distraction)+rField('Weak topic','weekly.'+mon+'.weak',note.weak)+rField('ONE improvement','weekly.'+mon+'.improve',note.improve)+'</div><button class="mini-btn" onclick="window.print()">Print / Save PDF</button></div>'; }
function renderResetDash(){ const ds=rHistory().slice(0,14).reverse(); const scores=ds.map(d=>rScore(resetState.days[d])); const avg=scores.length?Math.round(scores.reduce((a,b)=>a+b,0)/scores.length):0; const best=scores.length?Math.max(...scores):0; const apps=Object.values(resetState.days).reduce((n,d)=>n+(d.job.apps||[]).length,0); const hours=Object.values(resetState.days).reduce((n,d)=>n+rProductiveHours(d),0); return '<div class="reset-grid"><div class="reset-card"><h2>'+avg+'</h2><p class="sub">Average score</p></div><div class="reset-card"><h2>'+best+'</h2><p class="sub">Best score</p></div><div class="reset-card"><h2>'+hours.toFixed(1)+'h</h2><p class="sub">Total productive hours</p></div><div class="reset-card"><h2>'+apps+'</h2><p class="sub">Applications logged</p></div></div><div class="reset-card"><h2>Last 14 scores</h2><div class="reset-mini-chart">'+scores.map(s=>'<span title="'+s+'" style="height:'+Math.max(4,s)+'px"></span>').join('')+'</div></div>'; }
function rSetWeek(mon,k,v){ if(!resetState.weekly[mon])resetState.weekly[mon]={}; resetState.weekly[mon][k]=v; resetSave(); }
const oldRSet = rSet; rSet = function(path,val){ if(path.startsWith('weekly.')){ const parts=path.split('.'); rSetWeek(parts[1],parts[2],val); return; } oldRSet(path,val); };
renderResetTracker = async function(){ await ensureReset(); const root=$('reset-tracker-root'); if(!root)return; const tabs=['today','dash','week','history']; root.innerHTML='<div class="reset-tabs">'+tabs.map(t=>'<button class="reset-chip '+(resetTab===t?'active':'')+'" onclick="rTab(\''+t+'\')">'+(t==='dash'?'Dashboard':t==='week'?'Weekly Review':t[0].toUpperCase()+t.slice(1))+'</button>').join('')+'</div>'+(resetTab==='today'?renderResetToday():resetTab==='dash'?renderResetDash():resetTab==='week'?renderResetWeek():renderResetHistory()); rTimerUI(); };

function aimlLearningItems(filter,q){
 const d=guideData(); const map={
  'python-fundamentals':['python'], 'math-foundation':['linear','matrices','eigen','calculus','gradient','probability','bayesian','hypothesis','distributions'], 'data-wrangling':['numpy','pandas','eda','sql','missing'], 'data-visualization':['matplotlib','seaborn','plotly','distribution','correlation'], 'ml-algorithms':['ml','supervised','unsupervised','linear-regression','logistic','decision','random','svm','knn','naive','kmeans','pca','reinforcement','apriori'], 'ml-practice':['preprocessing','feature','cross-validation','hyperparameter','metrics','bias','overfitting','regularization','pipeline'], 'deep-learning':['dl','neural','activation','backprop','cnn','rnn','lstm','tensorflow','keras','pytorch','transfer','autoencoder','gan'], 'nlp':['nlp','text','word','named','pos','topic','spacy','nltk','similarity'], 'computer-vision':['cv','opencv','image','edge','object','face','yolo','optical'], 'transformers':['attention','transformer','bert','gpt','t5','vision-transformer','encoder'], 'genai':['genai','llm','prompt','rag','fine-tuning','langchain','llamaindex','vector','agents','diffusion','text-to-image','openai','huggingface','evaluation'], 'mlops':['docker','deployment','fastapi','mlflow','kubernetes','monitoring','ci-cd','aws'], 'data-engineering':['data-engineering','etl','spark','data-lake','streaming','airflow','quality'], 'dsa':['arrays','linked','trees','sorting','dynamic','hashing','binary','stacks','ml-interview','system-design'] };
 return d.learningSubjects.filter(x=>{ const hay=(x.id+' '+x.title+' '+x.tags.join(' ')).toLowerCase(); const catOk=filter==='all'||(map[filter]||[filter]).some(k=>hay.includes(k)); return catOk&&(!q||hay.includes(q)); });
}
renderAiLearning = async function(){
  await ensureAiState(); const d=guideData(); const q=(aiState.query||'').toLowerCase(); const root=$('ai-learning-root'); if(!root) return; const done=new Set(aiState.learningDone||[]); const cats=[{id:'all',name:'All'}].concat(d.subjectCategories); const category=d.subjectCategories.find(c=>c.id===aiState.learningFilter); const relevance=category?category.roleRelevance.join(' · '):'Data Scientist · ML Engineer · AI Engineer · GenAI Engineer'; const items=aimlLearningItems(aiState.learningFilter,q);
  root.innerHTML='<div class="aiml-shell">'+aimlHero('Learning Links','Role-focused tutorials and references for Data Science, ML Engineering, AI Engineering, and GenAI paths. '+relevance,'Reference')+'<div class="aiml-toolbar"><input class="aiml-search" value="'+escHtml(aiState.query||'')+'" placeholder="Search links, tags, roles..." oninput="setAiQuery(this.value)"><span class="aiml-chip active">'+done.size+'/'+d.learningSubjects.length+' complete</span><span class="aiml-chip">'+items.length+' shown</span></div><div class="aiml-toolbar">'+cats.map(c=>'<button class="aiml-chip '+(aiState.learningFilter===c.id?'active':'')+'" onclick="setLearningFilter(\''+c.id+'\')">'+escHtml(c.name)+'</button>').join('')+'</div><div class="aiml-grid">'+items.map(x=>'<div class="aiml-card"><div class="aiml-topic" style="grid-template-columns:auto 1fr auto;margin-top:0"><input type="checkbox" '+(done.has(x.id)?'checked':'')+' onchange="toggleLearning(\''+x.id+'\')"><div><div class="aiml-topic-title">'+escHtml(x.title)+'</div><div class="aiml-topic-desc">'+escHtml(x.tags.join(', '))+'</div>'+aimlBadge(x.difficulty)+'</div><a class="aiml-btn aiml-link" target="_blank" rel="noopener" href="'+escHtml(x.url)+'">Open</a></div></div>').join('')+'</div></div>';
};
