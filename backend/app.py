"""API de Kalka (FastAPI). Contrato definido en Kalka/frontend/lib/types.ts.

    POST /search  {text}            -> busca cada oracion en el corpus (data/train.csv)
    POST /compare {text_a, text_b}  -> compara las oraciones de A contra las de B

Flujo:
  1. El usuario pega texto.
  2. La API parte el texto en oraciones.
  3. Cada oracion se compara rapido contra todas las oraciones del corpus con similitud
     coseno de embeddings (all-MiniLM-L6-v2) y se quedan las 10 mas parecidas.
  4. Se corre predict(candidata, oracion) (la red RoBERTa entrenada) sobre esas 10 y se
     devuelve la de mayor probabilidad de copia.

Ejecutar:  uvicorn app:app --port 8000
"""
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("KALKA_MODEL", ROOT / "model"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = ROOT / "cache"
TOP_K = 10           # candidatas por oracion que evalua la red
THRESHOLD = 0.5      # probabilidad minima para reportar una coincidencia
MAX_SENTENCES = 200

# Oraciones de ejemplo del frontend, para que "Use sample text" funcione.
DEMO = [
    "A man in a blue jacket is walking his dog along the beach at sunset.",
    "Two children are playing soccer in a large green field near the school.",
    "The committee approved the new budget after a long discussion on Tuesday.",
    "A woman is riding a red bicycle down a crowded street in the city.",
    "Researchers found that the new method improves accuracy on every benchmark.",
]

STOP = set("a an the is are was were be of in on at to and or for with by it its this that his her their "
           "he she they we you i as from".split())

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ---------------------------------------------------------------- red de copia (cross-encoder)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device, dtype).eval()
temp_file = MODEL_DIR / "temperature.json"
TEMPERATURE = json.loads(temp_file.read_text())["temperature"] if temp_file.exists() else 1.0


@torch.no_grad()
def predict(candidatas, oraciones):
    """P(copia) calibrada para cada par (candidata = reference, oracion = input)."""
    probs = []
    for i in range(0, len(candidatas), 128):
        enc = tok(candidatas[i:i + 128], oraciones[i:i + 128], truncation=True, max_length=128,
                  padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits.float() / TEMPERATURE
        probs.extend(logits.softmax(-1)[:, 1].cpu().tolist())
    return probs


# ---------------------------------------------------------------- embeddings (busqueda rapida)
emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
emb_model = AutoModel.from_pretrained(EMBED_MODEL).to(device, dtype).eval()


@torch.no_grad()
def embed(texts, bs=512):
    """Embeddings normalizados (mean pooling), asi el producto punto = similitud coseno."""
    out = []
    for i in range(0, len(texts), bs):
        enc = emb_tok(texts[i:i + bs], truncation=True, max_length=128, padding=True,
                      return_tensors="pt").to(device)
        h = emb_model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(F.normalize((h * mask).sum(1) / mask.sum(1), dim=-1))
    return torch.cat(out) if out else torch.empty(0, emb_model.config.hidden_size, device=device, dtype=dtype)


class Index:
    def __init__(self, texts, labels, emb=None):
        self.texts, self.labels = texts, labels
        self.emb = embed(texts) if emb is None else emb

    def candidates(self, sentences, k=TOP_K):
        """Indices de las k oraciones del indice con mayor similitud coseno a cada oracion."""
        sims = embed(sentences) @ self.emb.T
        return sims.topk(min(k, len(self.texts)), dim=-1).indices.cpu().tolist()


def build_corpus():
    df = pd.read_csv(ROOT.parent / "data" / "train.csv", usecols=["id", "reference"], dtype=str, keep_default_na=False)
    df = df.drop_duplicates("reference")
    texts = DEMO + df["reference"].tolist()
    labels = ["Demo"] * len(DEMO) + [f"Corpus SNLI #{i}" for i in df["id"]]

    # los embeddings del corpus se calculan una vez y se guardan en cache/
    key = hashlib.md5((EMBED_MODEL + "\n".join(texts)).encode("utf-8")).hexdigest()[:12]
    cache = CACHE_DIR / f"corpus_emb_{key}.pt"
    if cache.exists():
        emb = torch.load(cache).to(device, dtype)
    else:
        print("Calculando embeddings del corpus (solo la primera vez)...")
        emb = embed(texts)
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(emb.cpu(), cache)
    return Index(texts, labels, emb)


print("Construyendo indice del corpus...")
CORPUS = build_corpus()
print(f"Corpus listo: {len(CORPUS.texts)} oraciones. Modelos en {device}, T={TEMPERATURE:.3f}")


# ---------------------------------------------------------------- utilidades
def split_sentences(t):
    return [s.strip() for s in re.findall(r"[^.!?]+[.!?]*", t) if s.strip()][:MAX_SENTENCES]


def tokens(s):
    return [(m.group().lower(), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9']+", s)]


def shared_spans(a, b):
    """Fragmentos de palabras consecutivas compartidas (resaltado en el frontend)."""
    ta, tb = tokens(a), tokens(b)

    def runs(tk, other):
        out, start = [], None
        for i, (w, _, _) in enumerate(tk + [("\0", 0, 0)]):
            if i < len(tk) and w in other:
                start = i if start is None else start
                continue
            if start is not None:
                seg = tk[start:i]
                if len(seg) >= 2 or any(x[0] not in STOP for x in seg):
                    out.append([seg[0][1], seg[-1][2]])
                start = None
        return out

    return runs(ta, {t[0] for t in tb}), runs(tb, {t[0] for t in ta})


def analyze(text, index):
    sentences = split_sentences(text)                       # 2. partir en oraciones
    if not sentences:
        return {"probability": 0.0, "total_sentences": 0, "matches": []}
    cands = index.candidates(sentences)                     # 3. top-10 por similitud coseno
    pairs = [(si, ci) for si, cs in enumerate(cands) for ci in cs]
    probs = predict([index.texts[c] for _, c in pairs],     # 4. predict(candidata, oracion)
                    [sentences[s] for s, _ in pairs])

    best = {}
    for (si, ci), p in zip(pairs, probs):
        if si not in best or p > best[si][1]:
            best[si] = (ci, p)

    matches = []
    for si, (ci, p) in best.items():
        if p < THRESHOLD:
            continue
        src = index.texts[ci]
        sa, sb = shared_spans(sentences[si], src)
        matches.append({"sentence": sentences[si], "source": src, "source_label": index.labels[ci],
                        "probability": round(p, 4), "spans": sa, "source_spans": sb})
    matches.sort(key=lambda m: -m["probability"])
    overall = max((m["probability"] for m in matches), default=max(probs, default=0.0))
    return {"probability": round(float(overall), 4), "total_sentences": len(sentences), "matches": matches}


# ---------------------------------------------------------------- API
app = FastAPI(title="Kalka API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"])


class SearchIn(BaseModel):
    text: str


class CompareIn(BaseModel):
    text_a: str
    text_b: str


@app.get("/health")
def health():
    return {"ok": True, "device": device, "corpus": len(CORPUS.texts), "temperature": TEMPERATURE,
            "retrieval": f"cosine top-{TOP_K} ({EMBED_MODEL})"}


@app.post("/search")
def search(body: SearchIn):
    return analyze(body.text, CORPUS)


@app.post("/compare")
def compare(body: CompareIn):
    b = split_sentences(body.text_b)
    if not b:
        return {"probability": 0.0, "total_sentences": len(split_sentences(body.text_a)), "matches": []}
    return analyze(body.text_a, Index(b, ["Text B"] * len(b)))
