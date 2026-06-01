# Personal Recordbook

Personal Recordbook is a single-user study and learning dashboard built from the AI/ML Guide, StudyTrack, and Reset Tracker. It keeps AI/ML roadmap progress, project progress, learning links, study sessions, goals, notes, subjects, calendar events, and daily discipline records in one app.

The app is designed for one user: `@shubh`, using the app user id `studytrack_single_user`.

## Features

- **Learning Roadmap**: AI/ML curriculum tracker with topic completion, module expansion, search, filters, and lesson viewer.
- **Project Tracker**: 101 machine learning projects with completion tracking, categories, difficulty filters, search, and project links.
- **Subject Tracker**: role-based learning links for Data Scientist, ML Engineer, AI Engineer, and GenAI Engineer paths.
- **StudyTrack**: full study tracker with goals, sessions, Pomodoro/focus tools, notes, subjects, calendar, statistics, settings, and Supabase/local fallback storage.
- **Reset Tracker**: daily discipline tracker with scores, bad-day minimum, study blocks, timers, countdowns, job applications, SSC/GATE/coding logs, dashboard, weekly review, and history.
- **Single-user Supabase storage**: no signup or signin flow required.
- **Responsive layout**: tested at mobile and tablet viewport sizes.
- **Fast StudyTrack tab**: StudyTrack preloads in the background so switching to it is immediate.

## Data Storage

The app uses Supabase as the long-term database and localStorage as a fast/offline cache.

| Area | Storage |
| --- | --- |
| StudyTrack profile/stats | `profiles` |
| StudyTrack goals | `goals` |
| StudyTrack sessions | `sessions` |
| StudyTrack notes | `notes` |
| StudyTrack calendar events | `events` |
| StudyTrack subjects | `subjects` |
| StudyTrack activity | `activity` |
| Reset Tracker | `app_state`, key `reset_tracker_next` |
| AI Roadmap progress | `app_state`, key `roadmap_progress` |
| ML Projects progress | `app_state`, key `ml_projects_progress` |
| Learning/Subject links progress | `app_state`, key `ml_subjects_progress` |

All records are scoped to:

```text
user_id = studytrack_single_user
username = shubh
```

## Supabase Setup

Run the current single-user SQL migration in Supabase SQL Editor before relying on database persistence. The database should contain:

- `profiles`
- `goals`
- `sessions`
- `notes`
- `events`
- `subjects`
- `activity`
- `app_state`

The community table is intentionally removed:

```sql
DROP TABLE IF EXISTS community_messages CASCADE;
```

The app currently has the Supabase URL and anon key inside the frontend code, so Vercel does not require extra environment variables for the current setup. The anon key is public by design in Supabase frontend apps; RLS policies restrict access to the single app user.

## Tech Stack

- **Framework**: Next.js 16 with TypeScript
- **Styling**: Tailwind CSS 4
- **UI**: shadcn/ui and Radix UI
- **Icons**: Lucide React
- **Animation**: Framer Motion
- **Lessons**: Markdown content rendered in the lesson viewer
- **Database**: Supabase REST API

## Getting Started

### Prerequisites

- Node.js 18+
- npm
- Supabase project with the single-user schema applied

### Install

```bash
npm install
```

### Run Locally

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

### Build

```bash
npm run build
```

### Start Production Build

```bash
npm start
```

## Deployment

Deploying to Vercel works with the standard Next.js settings:

- Framework: Next.js
- Build command: `npm run build`
- Output: `.next`

No Vercel environment variables are currently required because the Supabase URL and anon key are already present in the app code.

## Project Structure

```text
aimlguide/
├── public/
│   └── studytrack/                 # Embedded full StudyTrack static app
├── src/
│   ├── app/
│   │   ├── api/                    # Lesson/API routes
│   │   ├── globals.css             # Global styles and tracker utility classes
│   │   ├── layout.tsx              # Metadata and root layout
│   │   └── page.tsx                # Main Personal Recordbook tab shell
│   ├── components/
│   │   ├── learning-tracker.tsx    # Learning/subject link tracker
│   │   ├── lesson-viewer.tsx       # Markdown lesson viewer
│   │   ├── project-tracker.tsx     # ML project tracker
│   │   ├── reset-tracker-app.tsx   # React Reset Tracker
│   │   ├── roadmap-dashboard.tsx   # AI/ML roadmap
│   │   ├── studytrack-app.tsx      # Preloaded StudyTrack iframe wrapper
│   │   └── ui/                     # shadcn/ui components
│   ├── content/                    # Markdown lessons
│   ├── hooks/                      # React hooks
│   └── lib/
│       ├── course-data.ts          # Roadmap data
│       ├── projects-data.ts        # ML project list
│       ├── subjects-data.ts        # Learning link data
│       └── supabase-state.ts       # Compact app_state persistence helper
├── package.json
├── next.config.ts
└── tsconfig.json
```

## Main Tabs

1. **Learning Roadmap**
   Tracks completion across the structured AI/ML curriculum.

2. **Project Tracker**
   Tracks project completion and filters project ideas by category/difficulty.

3. **Subject Tracker**
   Tracks role-specific learning resources and links.

4. **StudyTrack**
   Embeds the full original StudyTrack experience, minus duplicated AI roadmap/project/learning/reset menu entries.

5. **Reset Tracker**
   Tracks daily discipline, score, weekly review, job applications, coding, SSC/GATE work, and distraction control.

## Persistence Behavior

Refreshing the page does not delete data.

The app writes progress to Supabase and also caches it in localStorage. If Supabase is temporarily unavailable, localStorage keeps the app usable. Once Supabase is available, new saves sync back to the database.

## Notes For Future Changes

- Keep `studytrack_single_user` unchanged unless the app code is updated everywhere that depends on it.
- For Supabase free tier, prefer storing compact tab progress in `app_state` instead of creating many tiny rows.
- StudyTrack detailed entities should stay in their dedicated tables because they are dated records.
- Do not re-add community tables or auth/signup UI unless the app becomes multi-user again.

## License

Personal project. Add a license file if you plan to publish it as open source.
