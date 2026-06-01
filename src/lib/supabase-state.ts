"use client";

const SUPABASE_URL = "https://gimzumqhongsksvvjqnr.supabase.co";
const SUPABASE_ANON =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaW16dW1xaG9uZ3Nrc3Z2anFuciIsInJlZiI6ImdpbXp1bXFob25nc2tzdnZqcW5yIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgyMzM4NDgsImV4cCI6MjA5MzgwOTg0OH0.I1MMkElg2ahLi8NQffBf0Du7qjIzzhDAEiilXc2mfOA";

const USER_ID = "studytrack_single_user";

function headers() {
  return {
    apikey: SUPABASE_ANON,
    Authorization: `Bearer ${SUPABASE_ANON}`,
    "Content-Type": "application/json",
    Prefer: "return=minimal",
  };
}

function localKey(key: string) {
  return `aiml-studytrack:${key}`;
}

export async function loadRemoteState<T>(key: string, fallback: T): Promise<T> {
  if (typeof window === "undefined") return fallback;

  const localRaw = window.localStorage.getItem(localKey(key));
  const localValue = localRaw ? (JSON.parse(localRaw) as T) : fallback;

  try {
    const url = `${SUPABASE_URL}/rest/v1/app_state?user_id=eq.${USER_ID}&key=eq.${key}&select=value`;
    const res = await fetch(url, { headers: headers(), cache: "no-store" });
    if (!res.ok) return localValue;
    const rows = (await res.json()) as Array<{ value: T }>;
    return rows[0]?.value ?? localValue;
  } catch {
    return localValue;
  }
}

export async function saveRemoteState<T>(key: string, value: T) {
  if (typeof window !== "undefined") {
    window.localStorage.setItem(localKey(key), JSON.stringify(value));
  }

  try {
    await fetch(`${SUPABASE_URL}/rest/v1/app_state`, {
      method: "POST",
      headers: {
        ...headers(),
        Prefer: "resolution=merge-duplicates,return=minimal",
      },
      body: JSON.stringify({
        id: `${USER_ID}_${key}`,
        user_id: USER_ID,
        key,
        value,
        updated_at: new Date().toISOString(),
      }),
    });
  } catch {
    // Local storage is the reliable fallback when the Supabase table is not set up yet.
  }
}

export const supabaseSetupSql = `
CREATE TABLE IF NOT EXISTS app_state (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value JSONB DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(user_id, key)
);

ALTER TABLE app_state ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY "app_state_all" ON app_state FOR ALL USING (true) WITH CHECK (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS idx_app_state_user_key ON app_state(user_id, key);
`.trim();
