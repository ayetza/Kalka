"use client";

import { useEffect, useRef, useState } from "react";
import { compare, search, usingMock } from "@/lib/api";
import type { Match, Result, Span } from "@/lib/types";

type Mode = "search" | "compare";

const SAMPLE =
  "A man in a blue jacket is walking his dog along the beach at sunset. The committee approved the new budget after a long discussion on Tuesday. Nobody expected the rain to stop before midnight.";

function Marked({ text, spans }: { text: string; spans: Span[] }) {
  const parts: React.ReactNode[] = [];
  let i = 0;
  [...spans].sort((a, b) => a[0] - b[0]).forEach(([a, b], k) => {
    if (a > i) parts.push(text.slice(i, a));
    parts.push(<mark key={k}>{text.slice(a, b)}</mark>);
    i = b;
  });
  parts.push(text.slice(i));
  return <>{parts}</>;
}

const DEMO_A = "A man in a blue jacket is walking his dog along the beach.";
const DEMO_B = "Last week, a man in a blue jacket is walking his dog on the sand.";
const DEMO_SHARED = "man in a blue jacket is walking his dog";
const spanOf = (t: string, p: string): Span => [t.indexOf(p), t.indexOf(p) + p.length];

function level(p: number) {
  return p >= 0.7 ? { cls: "high", label: "High probability of copying" }
    : p >= 0.4 ? { cls: "mid", label: "Partial matches" }
    : { cls: "low", label: "No clear copying" };
}

function Verdict({ r }: { r: Result }) {
  const l = level(r.probability);
  const pct = Math.round(r.probability * 100);
  return (
    <section className={`verdict ${l.cls}`} aria-live="polite">
      <p className="pct">{pct}<span>%</span></p>
      <div>
        <h2>{l.label}</h2>
        <div className="gauge" role="img" aria-label={`Infringement probability ${pct}%`}>
          <i style={{ width: `${pct}%` }} />
        </div>
        <p className="meta">
          {r.matches.length} of {r.total_sentences} sentences matched a source, in {r.elapsed_ms} ms
        </p>
      </div>
    </section>
  );
}

function MatchRow({ m }: { m: Match }) {
  const pct = Math.round(m.probability * 100);
  return (
    <li className="match">
      <div className="sheet top">
        <p className="tag">Your text</p>
        <p className="prose"><Marked text={m.sentence} spans={m.spans} /></p>
      </div>
      <div className="sheet bottom">
        <p className="tag">Source{m.source_label ? `: ${m.source_label}` : ""}</p>
        <p className="prose"><Marked text={m.source} spans={m.source_spans} /></p>
      </div>
      <p className="score">{pct}%</p>
    </li>
  );
}

function Blob({ busy }: { busy: boolean }) {
  const ref = useRef<HTMLDivElement>(null);

  // Eyes follow the pointer (mouse or finger). With no input for a moment, as on
  // phones and tablets, they glance around on their own, smoothly.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let last = 0;
    let timer: ReturnType<typeof setTimeout>;
    const look = (x: number, y: number) => {
      el.style.setProperty("--lx", String(x));
      el.style.setProperty("--ly", String(y));
    };
    const follow = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      const dx = (e.clientX - (r.left + r.width / 2)) / (window.innerWidth / 2);
      const dy = (e.clientY - (r.top + r.height / 2)) / (window.innerHeight / 2);
      last = Date.now();
      el.classList.remove("idle");
      look(Math.max(-1, Math.min(1, dx)), Math.max(-1, Math.min(1, dy)));
    };
    let side = 1;
    const wander = () => {
      if (Date.now() - last > 2500) {
        el.classList.add("idle");
        if (Math.random() < 0.3) look(0, 0);
        else { side = -side; look(side * (0.5 + Math.random() * 0.5), (Math.random() - 0.5) * 0.7); }
      }
      timer = setTimeout(wander, 1400 + Math.random() * 1800);
    };
    window.addEventListener("pointermove", follow);
    window.addEventListener("pointerdown", follow);
    if (!reduced) timer = setTimeout(wander, 1200);
    return () => {
      window.removeEventListener("pointermove", follow);
      window.removeEventListener("pointerdown", follow);
      clearTimeout(timer);
    };
  }, []);

  return (
    <div ref={ref} className={`blob ${busy ? "busy" : ""}`} aria-hidden="true">
      <div className="orb">
        <div className="face"><span className="eye l" /><span className="eye r" /><span className="smile" /></div>
      </div>
    </div>
  );
}

function Petals({ size = 120 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 120 120" aria-hidden="true">
      <path d="M60 60 V8 A52 52 0 0 0 8 60Z" fill="#f08d74" transform="translate(-2 -2)" />
      <path d="M60 60 H112 A52 52 0 0 0 60 8Z" fill="#a994e4" transform="translate(2 -2)" />
      <path d="M60 60 V112 A52 52 0 0 0 112 60Z" fill="#f08d74" transform="translate(2 2)" />
      <path d="M60 60 H8 A52 52 0 0 0 60 112Z" fill="#a994e4" transform="translate(-2 2)" />
    </svg>
  );
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("search");
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [res, setRes] = useState<Result | null>(null);
  const results = useRef<HTMLElement>(null);

  const ready = a.trim().length > 0 && (mode === "search" || b.trim().length > 0);

  useEffect(() => {
    if (res || err) results.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [res, err]);

  useEffect(() => {
    document.querySelectorAll<HTMLTextAreaElement>(".pill textarea").forEach((t) => {
      t.style.height = "auto";
      t.style.height = `${Math.min(t.scrollHeight, 220)}px`;
    });
  }, [a, b, mode]);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!ready || busy) return;
    setBusy(true); setErr(""); setRes(null);
    try { setRes(mode === "search" ? await search(a) : await compare(a, b)); }
    catch (x) { setErr(x instanceof Error ? x.message : "Could not analyze the text."); }
    finally { setBusy(false); }
  }

  return (
    <>
      <section className="hero" id="top">
        <nav className="nav">
          <a className="brand" href="#top">Kalka</a>
          <div className="links">
            <a href="#how">How it works</a>
            <a href="#baseline">Baseline</a>
          </div>
          <a className="cta" href="#top" onClick={() => document.getElementById("a")?.focus()}>Try it</a>
        </nav>

        <div className="hero-body">
          <Blob busy={busy} />
          <h1>Find the copy.<br />Find the source.</h1>
          <p className="sub">Paste a text and see which sentences copy another one, and where from.</p>

          <form onSubmit={run} className="ask">
            <div className="pill">
              <textarea id="a" rows={1} value={a} onChange={(e) => setA(e.target.value)}
                aria-label={mode === "search" ? "Text to check" : "Suspect text"}
                placeholder={mode === "search" ? "Paste the text you want to check…" : "Paste the suspect text…"} />
              <button className="go" disabled={!ready || busy} aria-label="Analyze text">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
              </button>
            </div>
            {mode === "compare" && (
              <div className="pill second">
                <textarea id="b" rows={1} value={b} onChange={(e) => setB(e.target.value)}
                  aria-label="Original text" placeholder="Paste the original text to compare against…" />
              </div>
            )}
            <div className="chips">
              <button type="button" aria-pressed={mode === "search"} onClick={() => { setMode("search"); setRes(null); }}>Search the corpus</button>
              <button type="button" aria-pressed={mode === "compare"} onClick={() => { setMode("compare"); setRes(null); }}>Compare two texts</button>
              {!a && <button type="button" className="plain" onClick={() => setA(SAMPLE)}>Use sample text</button>}
            </div>
            {usingMock && <p className="note">Demo mode: results are sample data and don&apos;t use the real model yet.</p>}
          </form>
        </div>
      </section>

      <main className="below">
        <section ref={results} className="results" aria-live="polite">
          {err && <p className="error" role="alert">{err}</p>}
          {busy && <div className="scan" aria-label="Analyzing"><i /><i /><i /></div>}
          {res && (
            <>
              <Verdict r={res} />
              {res.matches.length === 0 ? (
                <p className="none">No similar sentences found in {mode === "search" ? "the corpus" : "the original text"}.</p>
              ) : (
                <ul className="matches">{res.matches.map((m, i) => <MatchRow key={i} m={m} />)}</ul>
              )}
            </>
          )}
        </section>

        <div className="bento" id="how">
          <article className="card c-big">
            <h2>Sentence by sentence</h2>
            <p>Long texts are split into sentences, and each one is traced to its closest source.</p>
            <ul className="sents" aria-label="Example sentences">
              <li className="hit">A man in a blue jacket is walking his dog.</li>
              <li>Nobody expected the rain to stop before midnight.</li>
              <li className="hit">The committee approved the new budget.</li>
            </ul>
          </article>

          <article className="card c-chart" id="baseline">
            <p className="small">Baseline to beat · F1 score</p>
            <p className="big">82.69%</p>
            <ul className="bars">
              <li><span>TF-IDF + BERT ensemble</span><i style={{ width: "82.69%" }} /><b>82.69</b></li>
              <li><span>BERT only</span><i style={{ width: "81.31%" }} /><b>81.31</b></li>
              <li><span>TF-IDF only</span><i style={{ width: "60.57%" }} /><b>60.57</b></li>
            </ul>
            <p className="small">Mutsaddi &amp; Choudhary, LoResLM 2025</p>
          </article>

          <article className="card c-quote">
            <div className="mini" aria-hidden="true">
              <div className="sheet top">
                <p className="tag">Your text</p>
                <p className="prose"><Marked text={DEMO_A} spans={[spanOf(DEMO_A, DEMO_SHARED)]} /></p>
              </div>
              <div className="sheet bottom">
                <p className="tag">Source</p>
                <p className="prose"><Marked text={DEMO_B} spans={[spanOf(DEMO_B, DEMO_SHARED)]} /></p>
              </div>
            </div>
            <p className="small">Copied fragments are highlighted in both texts.</p>
          </article>

          <article className="card c-dark"><Petals /></article>

          <article className="card c-plan">
            <div className="glass">
              <h3>How Kalka checks a text</h3>
              <ul>
                <li className="done">Split the text into sentences</li>
                <li className="done">Find candidate sources in the corpus</li>
                <li>Score each pair with a calibrated probability</li>
              </ul>
            </div>
          </article>
        </div>
      </main>

      <footer className="foot">
        <div className="foot-links">
          <a href="#top">Check a text</a>
          <a href="#how">How it works</a>
          <a href="#baseline">Baseline</a>
        </div>
        <p className="mark" aria-hidden="true">Kalka</p>
      </footer>
    </>
  );
}
