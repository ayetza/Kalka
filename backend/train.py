"""Entrena la red neuronal de Kalka: un cross-encoder (RoBERTa) que recibe el par
(reference, input) y predice 1 = copia / 0 = no copia.

Uso:
    python train.py --epochs 3            # entrenamiento completo (data/data_por_referencia)
    python train.py --limit 2000 --epochs 1   # prueba rapida

Salida en backend/model/: pesos + tokenizer, temperature.json (calibracion) y metrics.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent
# Division agrupada por oracion `reference` (ninguna se repite entre train/val/test)
DATA = ROOT.parent / "data" / "data_por_referencia"
SEED = 42


def load(name, limit, data=DATA):
    df = pd.read_csv(Path(data) / f"{name}.csv", dtype={"reference": str, "input": str}, keep_default_na=False)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=SEED)
    return df


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, df, tok, max_len):
        enc = tok(df["reference"].tolist(), df["input"].tolist(), truncation=True, max_length=max_len)
        self.enc = enc
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    out, ys = [], []
    for batch in loader:
        ys.append(batch.pop("labels"))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model(**batch).logits.float().cpu())
    return torch.cat(out), torch.cat(ys)


def metrics(logits, y):
    pred = logits.argmax(-1).numpy()
    y = y.numpy()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def fit_temperature(logits, y):
    """Temperature scaling sobre validacion -> probabilidades calibradas."""
    t = torch.zeros(1, requires_grad=True)  # log T
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def step():
        opt.zero_grad()
        loss = nll(logits / t.exp(), y)
        loss.backward()
        return loss

    opt.step(step)
    return float(t.detach().exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="usar solo N filas por split (prueba)")
    ap.add_argument("--out", default=str(ROOT / "model"))
    ap.add_argument("--data", default=str(DATA), help="carpeta con train/val/test.csv")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}")

    tok = AutoTokenizer.from_pretrained(args.model)
    lim_eval = args.limit // 4 if args.limit else 0
    train_df, val_df, test_df = (load("train", args.limit, args.data), load("val", lim_eval, args.data),
                                 load("test", lim_eval, args.data))
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    t0 = time.time()
    ds_train, ds_val, ds_test = (PairDataset(d, tok, args.max_len) for d in (train_df, val_df, test_df))
    print(f"Tokenizado en {time.time() - t0:.0f}s")

    collate = DataCollatorWithPadding(tok)
    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=args.bs * 4, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=args.bs * 4, collate_fn=collate)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: "no_copia", 1: "copia"}, label2id={"no_copia": 0, "copia": 1}
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl_train) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for step, batch in enumerate(dl_train, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item()
            if step % 200 == 0 or step == len(dl_train):
                el = time.time() - t0
                eta = el / step * (len(dl_train) - step)
                print(f"  epoca {epoch} paso {step}/{len(dl_train)} loss={run_loss / step:.4f} "
                      f"({el / 60:.1f} min, faltan {eta / 60:.1f} min)", flush=True)

        logits, y = predict_logits(model, dl_val, device)
        m = metrics(logits, y)
        history.append({"epoch": epoch, **m})
        print(f"Epoca {epoch} validacion: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            model.save_pretrained(out)
            tok.save_pretrained(out)
            print(f"  -> mejor modelo guardado en {out}")

    # recarga el mejor, calibra y evalua en test
    model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    val_logits, val_y = predict_logits(model, dl_val, device)
    temp = fit_temperature(val_logits, val_y)
    (out / "temperature.json").write_text(json.dumps({"temperature": temp}))
    print(f"Temperatura de calibracion: {temp:.3f}")

    test_logits, test_y = predict_logits(model, dl_test, device)
    test_m = metrics(test_logits, test_y)
    print("TEST: " + " ".join(f"{k}={v:.4f}" for k, v in test_m.items()))
    (out / "metrics.json").write_text(json.dumps(
        {"base_model": args.model, "args": vars(args), "history": history, "test": test_m,
         "temperature": temp}, indent=2))


if __name__ == "__main__":
    main()
