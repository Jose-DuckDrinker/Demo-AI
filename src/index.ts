// Node.js companion: streaming chat endpoint backed by OpenAI + Anthropic
// SDKs. Imports here are intentionally kept verbose so the Xygeni AI framework
// detector can pick them up in the JavaScript/TypeScript ecosystem.

import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { GoogleGenerativeAI } from "@google/generative-ai";

const openai = new OpenAI({ baseURL: "https://api.openai.com/v1" });
const anthropic = new Anthropic();
const gemini = new GoogleGenerativeAI(process.env.GEMINI_API_KEY ?? "");

const primary = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.2 });
const fallback = new ChatAnthropic({ model: "claude-3-5-sonnet-20241022" });
const review = gemini.getGenerativeModel({ model: "gemini-1.5-pro" });

export async function summarize(text: string): Promise<string> {
  try {
    const out = await primary.invoke([
      { role: "system", content: "Summarize the following customer message." },
      { role: "user", content: text },
    ]);
    return String(out.content);
  } catch (_err) {
    const out = await fallback.invoke([
      { role: "system", content: "Summarize the following customer message." },
      { role: "user", content: text },
    ]);
    return String(out.content);
  }
}

export async function moderate(text: string): Promise<boolean> {
  const result = await openai.moderations.create({
    model: "omni-moderation-latest",
    input: text,
  });
  return result.results[0]?.flagged ?? false;
}

export async function rawAnthropic(text: string): Promise<string> {
  const reply = await anthropic.messages.create({
    model: "claude-3-haiku-20240307",
    max_tokens: 256,
    messages: [{ role: "user", content: text }],
  });
  return reply.content.map((b) => ("text" in b ? b.text : "")).join("");
}

export async function reviewSuggestion(text: string): Promise<string> {
  const r = await review.generateContent(text);
  return r.response.text();
}
