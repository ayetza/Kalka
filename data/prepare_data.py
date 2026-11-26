"""Limpia el MIT Plagiarism Detection Dataset y genera train.csv / val.csv / test.csv.

Uso:
    python prepare_data.py --input train_snli.txt --no-header --outdir data
    # si las columnas tienen otros nombres:
    python prepare_data.py --input mit.csv --ref-col sentence1 --inp-col sentence2 --label-col label

Salida (UTF-8, con encabezado): id, reference, input, label
  - id: entero unico por par (se asigna despues de limpiar; no cambia entre archivos)
  - label: 1 = copia, 0 = no copia

Reglas:
  - sin filas vacias ni duplicadas (par reference+input repetido)
  - etiquetas solo 0 o 1
  - 80/20 estratificado (random_state=42); del 80% sale 10% para validacion
    (tambien estratificado, random_state=42)
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# Nombres de columnas comunes; se usan solo si no se pasan --ref-col/--inp-col/--label-col
REF_CANDIDATES = ["reference", "original", "sentence1", "sentence_1", "source", "text_a", "premise", "s1"]
INP_CANDIDATES = ["input", "suspicious", "sentence2", "sentence_2", "suspect", "text_b", "hypothesis", "s2"]
LABEL_CANDIDATES = ["label", "plagiarized", "is_plagiarism", "is_copy", "class", "target"]

LABEL_MAP = {
    "1": 1, "1.0": 1, "true": 1, "yes": 1, "copy": 1, "plagiarism": 1, "plagiarized": 1,
    "0": 0, "0.0": 0, "false": 0, "no": 0, "not_copy": 0, "non_plagiarism": 0, "non-plagiarized": 0,
}


def pick(df, given, candidates, what):
    if given:
        if given not in df.columns:
            sys.exit(f"No existe la columna '{given}' ({what}). Columnas: {list(df.columns)}")
        return given
    lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    sys.exit(f"No pude detectar la columna de {what}. Columnas: {list(df.columns)}. "
             f"Usa --ref-col / --inp-col / --label-col.")


def clean(df, ref_col, inp_col, label_col, log):
    df = df[[ref_col, inp_col, label_col]].copy()
    df.columns = ["reference", "input", "label"]
    log["filas_originales"] = len(df)

    # texto vacio / nulo
    for c in ("reference", "input"):
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    vacias = df["reference"].isna() | df["input"].isna() | (df["reference"] == "") | (df["input"] == "")
    log["filas_vacias_eliminadas"] = int(vacias.sum())
    df = df[~vacias]

    # etiquetas solo 0/1
    lab = df["label"].astype("string").str.strip().str.lower().map(LABEL_MAP)
    malas = lab.isna()
    log["etiquetas_invalidas_eliminadas"] = int(malas.sum())
    df = df[~malas].copy()
    df["label"] = lab[~malas].astype(int)

    # pares conflictivos: mismo (reference, input) con etiquetas distintas -> se quitan todos
    n_labels = df.groupby(["reference", "input"])["label"].transform("nunique")
    conflicto = n_labels > 1
    log["filas_con_etiqueta_conflictiva_eliminadas"] = int(conflicto.sum())
    df = df[~conflicto]

    # duplicados exactos
    antes = len(df)
    df = df.drop_duplicates(subset=["reference", "input"], keep="first")
    log["duplicados_eliminados"] = antes - len(df)

    df = df.reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    log["filas_finales"] = len(df)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/TSV/JSONL del dataset original")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--ref-col")
    ap.add_argument("--inp-col")
    ap.add_argument("--label-col")
    ap.add_argument("--no-header", action="store_true",
                    help="el archivo no trae encabezado: columnas = reference, input, label (ej. train_snli.txt)")
    args = ap.parse_args()

    p = Path(args.input)
    if p.suffix in (".tsv", ".txt"):
        # QUOTE_NONE: las comillas dentro de las oraciones no deben interpretarse como delimitadores
        raw = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8",
                          quoting=csv.QUOTE_NONE, header=None if args.no_header else "infer",
                          names=["reference", "input", "label"] if args.no_header else None)
    elif p.suffix in (".jsonl", ".json"):
        raw = pd.read_json(p, lines=p.suffix == ".jsonl", dtype=str)
    else:
        raw = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")

    ref = pick(raw, args.ref_col, REF_CANDIDATES, "reference (oracion original)")
    inp = pick(raw, args.inp_col, INP_CANDIDATES, "input (oracion sospechosa)")
    lab = pick(raw, args.label_col, LABEL_CANDIDATES, "label")
    print(f"Columnas usadas -> reference: {ref} | input: {inp} | label: {lab}")

    log = {}
    df = clean(raw, ref, inp, lab, log)

    # 80/20 estratificado; del 80% se saca 10% para validacion
    trainval, test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(trainval, test_size=0.10, stratify=trainval["label"], random_state=SEED)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    rows = []
    for name, part in splits.items():
        part = part.sort_values("id")
        part.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
        vc = part["label"].value_counts()
        rows.append({"archivo": f"{name}.csv", "total": len(part),
                     "clase_0_no_copia": int(vc.get(0, 0)), "clase_1_copia": int(vc.get(1, 0))})
    resumen = pd.DataFrame(rows)
    resumen.loc[len(resumen)] = {"archivo": "TOTAL", "total": len(df),
                                 "clase_0_no_copia": int((df.label == 0).sum()),
                                 "clase_1_copia": int((df.label == 1).sum())}
    resumen["pct_copia"] = (resumen["clase_1_copia"] / resumen["total"] * 100).round(2)
    resumen.to_csv(out / "conteo_clases.csv", index=False)

    print("\n== Limpieza ==")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print("\n== Pares por clase ==")
    print(resumen.to_string(index=False))

    # chequeos finales
    assert set(df["label"].unique()) <= {0, 1}
    assert df["id"].is_unique
    ids = [set(s["id"]) for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "ids repetidos entre archivos"
    print("\nChequeos OK (ids unicos, sin traslape entre archivos, etiquetas 0/1).")


if __name__ == "__main__":
    main()
