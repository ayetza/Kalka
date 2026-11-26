# Kalka: detección de copia en textos

Pegas un texto y Kalka te dice qué oraciones son copia de otra y de dónde vienen.
Una red neuronal (RoBERTa con fine-tuning) decide si un par de oraciones es copia;
una búsqueda por embeddings encuentra las candidatas en el corpus.

**Resultado en test** (oraciones de referencia nunca vistas en entrenamiento):
F1 **97.24 %**, accuracy 97.25 %, precision 97.14 %, recall 97.34 %.
El valor de referencia a superar es 82.69 % (TF-IDF + BERT, Mutsaddi & Choudhary, LoResLM 2025).

## Correr en local (Windows)

Requisitos: Python 3.12+, Node.js 20+ y Git con **Git LFS** (el modelo entrenado viene por LFS).
Si no los tienes:
```powershell
winget install Python.Python.3.14 OpenJS.NodeJS.LTS Git.Git
```

1. Clonar la branch (Git LFS baja el modelo, ~240 MB):
   ```powershell
   git clone -b prueba-1 https://github.com/ayetza/Kalka.git
   cd Kalka
   ```
2. Instalar todo (una sola vez; detecta si hay GPU NVIDIA):
   ```powershell
   .\setup.ps1
   ```
3. Arrancar, cada uno en su propia terminal:
   ```powershell
   .\iniciar_backend.ps1
   ```
   ```powershell
   .\iniciar_frontend.ps1
   ```
4. Abrir http://localhost:3000

La primera vez que arranca, la API calcula los embeddings del corpus y los guarda en
`backend/cache/`: tarda unos segundos con GPU y unos minutos con CPU. Las siguientes veces arranca rápido.

Si PowerShell no deja correr los scripts, ejecuta antes:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Estructura
```
frontend/                 página Next.js; habla con la API por NEXT_PUBLIC_API_URL (.env.local)
backend/
  app.py                  API FastAPI: /search, /compare, /health
  train.py                entrena la red neuronal
  model/                  modelo entrenado (pesos fp16 por Git LFS), temperature.json, metrics.json
  requirements.txt
  train_log*.txt          logs de los entrenamientos
data/
  raw/train_snli.zip      dataset original
  prepare_data.py         limpieza -> train.csv / val.csv / test.csv (división por par)
  split_por_referencia.py rehace la división agrupando por oración `reference`
  data_por_referencia/    división con la que se entrenó el modelo
setup.ps1, iniciar_backend.ps1, iniciar_frontend.ps1
```

## La red neuronal
Cross-encoder **RoBERTa-base** con fine-tuning para clasificar el par `(reference, input)`:
1 = copia, 0 = no copia. Ambas oraciones entran juntas al transformer y la cabeza de
clasificación da 2 logits.
- AdamW, lr 2e-5, warmup 6 %, 3 épocas, batch 64, bf16 en GPU (RTX 3080: unos 20 minutos).
- Se guarda la época con mejor F1 en validación.
- **Calibración**: temperature scaling sobre `val.csv`, para que el porcentaje sea una probabilidad confiable.
- Datos: `data/data_por_referencia/`. Ninguna oración `reference` aparece en dos archivos, así que el F1
  en test mide oraciones nunca vistas. Con la división por par (78 % de referencias repetidas) daba
  97.28 %, prácticamente igual.
- El dataset viene de SNLI: "copia" significa que la oración sospechosa se deduce de la original
  (incluye paráfrasis), no solo copia literal.

## Cómo responde la API
1. El usuario pega texto.
2. La API parte el texto en oraciones.
3. Cada oración se compara contra todas las oraciones del corpus con similitud coseno de
   embeddings (`sentence-transformers/all-MiniLM-L6-v2`) y se quedan las 10 más parecidas.
   El corpus son las `reference` únicas de `data/train.csv` (unas 139 mil). En `/compare`,
   el corpus son las oraciones del texto B.
4. Se corre `predict(candidata, oración)` (la red RoBERTa) sobre esas 10 y se queda la de
   mayor probabilidad. Si P(copia) ≥ 0.5, se reporta como coincidencia.
5. La probabilidad global es la máxima entre las oraciones.

## Reentrenar
```powershell
& "$env:USERPROFILE\.venvs\kalka\Scripts\python.exe" backend\train.py --epochs 3
```
Prueba rápida: `--limit 4000 --epochs 1 --out $env:TEMP\kalka_prueba`.
Reentrenar sobrescribe `backend/model/` con pesos fp32 (~475 MB).
