import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const topicId = searchParams.get("topic");

  if (!topicId) {
    return NextResponse.json(
      { error: "Missing topic parameter" },
      { status: 400 }
    );
  }

  const sanitized = topicId.replace(/[^a-zA-Z0-9_-]/g, "");

  if (sanitized !== topicId) {
    return NextResponse.json(
      { error: "Invalid topic ID" },
      { status: 400 }
    );
  }

  const filePath = path.join(
    process.cwd(),
    "src",
    "content",
    `${sanitized}.md`
  );

  try {
    const content = fs.readFileSync(filePath, "utf-8");
    return NextResponse.json({ content, topicId });
  } catch {
    return NextResponse.json(
      { error: "Topic not found" },
      { status: 404 }
    );
  }
}
