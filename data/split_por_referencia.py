"""Rehace la division train/val/test agrupando por oracion `reference`.

Asi ninguna oracion original aparece en dos archivos distintos, y la evaluacion en
test mide si el modelo generaliza a oraciones que nunca vio.

Lee train.csv, val.csv y test.csv (ya limpios) y escribe en data_por_referencia/:
train.csv, val.csv, test.csv y conteo_clases.csv. Mismas proporciones que antes:
80/20 y luego 10 % del 80 % para validacion, estratificado por etiqueta (seed 42).
Los archivos originales no se modifican.

Uso:  python split_por_referencia.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_por_referencia"


def take_fold(df, n_splits):
    """Devuelve (resto, fold) con grupos = reference, estratificado por label."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest_idx, fold_idx = next(sgkf.split(df, df["label"], groups=df["reference"]))
    return df.iloc[rest_idx], df.iloc[fold_idx]


def main():
    df = pd.concat([pd.read_csv(ROOT / f"{n}.csv", keep_default_na=False) for n in ("train", "val", "test")])
    df = df.sort_values("id").reset_index(drop=True)

    trainval, test = take_fold(df, 5)        # ~20 % test
    train, val = take_fold(trainval, 10)     # ~10 % del resto para validacion
    splits = {"train": train, "val": val, "test": test}

    OUT.mkdir(exist_ok=True)
    rows = []
    for name, part in splits.items():
        part.sort_values("id").to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
        vc = part["label"].value_counts()
        rows.append({"archivo": f"{name}.csv", "total": len(part),
                     "clase_0_no_copia": int(vc.get(0, 0)), "clase_1_copia": int(vc.get(1, 0))})
    res = pd.DataFrame(rows)
    res.loc[len(res)] = {"archivo": "TOTAL", "total": len(df), "clase_0_no_copia": int((df.label == 0).sum()),
                         "clase_1_copia": int((df.label == 1).sum())}
    res["pct_copia"] = (res["clase_1_copia"] / res["total"] * 100).round(2)
    res.to_csv(OUT / "conteo_clases.csv", index=False)
    print(res.to_string(index=False))

    refs = {k: set(v["reference"]) for k, v in splits.items()}
    ids = {k: set(v["id"]) for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not refs[a] & refs[b], f"reference repetida entre {a} y {b}"
        assert not ids[a] & ids[b], f"id repetido entre {a} y {b}"
    assert sum(len(s) for s in splits.values()) == len(df)
    print("\nChequeos OK: ninguna reference ni id se repite entre archivos.")


if __name__ == "__main__":
    main()
