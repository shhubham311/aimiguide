import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export const metadata: Metadata = {
  title: "Personal Recordbook",
  description: "A single personal recordbook for AI/ML learning, projects, study tracking, and daily discipline records.",
  keywords: ["Personal Recordbook", "Study Tracker", "AI", "ML", "Projects", "Learning", "Daily Tracker"],
  authors: [{ name: "Personal Recordbook" }],
  icons: {
    icon: [
      { url: "/favicon-32.png", sizes: "32x32", type: "image/png" },
      { url: "/favicon-16.png", sizes: "16x16", type: "image/png" },
      { url: "/favicon.svg", type: "image/svg+xml" },
    ],
    apple: "/apple-touch-icon.png",
  },
  openGraph: {
    title: "Personal Recordbook",
    description: "Your personal recordbook for learning, projects, study sessions, and daily discipline.",
    url: "https://chat.z.ai",
    siteName: "Personal Recordbook",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Personal Recordbook",
    description: "Your personal recordbook for learning, projects, study sessions, and daily discipline.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground">
        {children}
        <Toaster />
      </body>
    </html>
  );
}
