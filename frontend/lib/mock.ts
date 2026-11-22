import type { Match, Result, Span } from "./types";

// Demo corpus (replaced by the backend's real corpus).
const CORPUS = [
  { label: "Corpus A", text: "A man in a blue jacket is walking his dog along the beach at sunset." },
  { label: "Corpus A", text: "Two children are playing soccer in a large green field near the school." },
  { label: "Corpus B", text: "The committee approved the new budget after a long discussion on Tuesday." },
  { label: "Corpus B", text: "A woman is riding a red bicycle down a crowded street in the city." },
  { label: "Corpus C", text: "Researchers found that the new method improves accuracy on every benchmark." },
];

const split = (t: string) => t.match(/[^.!?]+[.!?]*/g)?.map((s) => s.trim()).filter(Boolean) ?? [];
const tokens = (s: string) => [...s.matchAll(/[A-Za-z']+/g)].map((m) => ({ w: m[0].toLowerCase(), a: m.index!, b: m.index! + m[0].length }));

function shared(a: string, b: string): { spansA: Span[]; spansB: Span[]; score: number } {
  const ta = tokens(a), tb = tokens(b);
  const sa = new Set(ta.map((t) => t.w)), sb = new Set(tb.map((t) => t.w));
  const runs = (tk: typeof ta, other: Set<string>): Span[] => {
    const out: Span[] = []; let s = -1;
    tk.forEach((t, i) => {
      if (other.has(t.w)) { if (s < 0) s = i; }
      if ((!other.has(t.w) || i === tk.length - 1) && s >= 0) {
        const e = other.has(t.w) ? i : i - 1;
        if (e - s + 1 >= 3) out.push([tk[s].a, tk[e].b]);
        s = -1;
      }
    });
    return out;
  };
  const inter = [...sa].filter((w) => sb.has(w)).length;
  return { spansA: runs(ta, sb), spansB: runs(tb, sa), score: inter / new Set([...sa, ...sb]).size };
}

export async function mockAnalyze(text: string, other?: string): Promise<Result> {
  const t0 = performance.now();
  await new Promise((r) => setTimeout(r, 700));
  const pool = other ? split(other).map((t) => ({ label: "Text B", text: t })) : CORPUS;
  const sentences = split(text);
  const matches: Match[] = [];
  for (const sentence of sentences) {
    let best: { c: (typeof pool)[number]; r: ReturnType<typeof shared> } | null = null;
    for (const c of pool) {
      const r = shared(sentence, c.text);
      if (!best || r.score > best.r.score) best = { c, r };
    }
    if (best && best.r.score > 0.25)
      matches.push({
        sentence, source: best.c.text, source_label: best.c.label,
        probability: Math.min(0.99, best.r.score * 1.25),
        spans: best.r.spansA, source_spans: best.r.spansB,
      });
  }
  const probability = sentences.length
    ? 1 - matches.reduce((p, m) => p * (1 - m.probability * 0.9), 1) * (matches.length ? 1 : 1)
    : 0;
  return {
    probability: matches.length ? probability : 0.03,
    total_sentences: sentences.length,
    matches: matches.sort((a, b) => b.probability - a.probability),
    elapsed_ms: Math.round(performance.now() - t0),
  };
}
