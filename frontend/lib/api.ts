import type { Result } from "./types";
import { mockAnalyze } from "./mock";

const API = process.env.NEXT_PUBLIC_API_URL;
export const usingMock = !API;

async function post(path: string, body: object): Promise<Result> {
  const t0 = performance.now();
  const res = await fetch(`${API}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`The server responded with ${res.status}. Try again in a few seconds.`);
  const data = await res.json();
  return { elapsed_ms: Math.round(performance.now() - t0), ...data };
}

export const search = (text: string) => (API ? post("/search", { text }) : mockAnalyze(text));
export const compare = (text_a: string, text_b: string) =>
  API ? post("/compare", { text_a, text_b }) : mockAnalyze(text_a, text_b);
