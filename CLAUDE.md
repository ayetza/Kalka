# Kalka: notas para Claude

Detector de copia/plagio entre oraciones. Lee `README.md` para instalación y arquitectura.

## Cómo correrlo
- Instalación única: `.\setup.ps1`. Crea el venv en `%USERPROFILE%\.venvs\kalka` (fuera del repo a propósito),
  instala PyTorch (CUDA si hay `nvidia-smi`) y `npm install`.
- API: `.\iniciar_backend.ps1` → http://localhost:8000 (`GET /health` para comprobar).
- Página: `.\iniciar_frontend.ps1` → http://localhost:3000. Necesita `frontend/.env.local` con
  `NEXT_PUBLIC_API_URL=http://localhost:8000`; si falta, la página usa datos de demo (`lib/mock.ts`).
- Python del proyecto: `& "$env:USERPROFILE\.venvs\kalka\Scripts\python.exe"`.

## Contratos que no se deben romper
- El frontend espera exactamente los tipos de `frontend/lib/types.ts` (`Result`, `Match`, `Span`).
  Si cambias la respuesta de `/search` o `/compare` en `backend/app.py`, actualiza ambos lados.
- `spans` son offsets `[inicio, fin)` de caracteres dentro de la oración.
- La red recibe el par en este orden: `(reference/candidata, input/oración del usuario)`. Así se entrenó;
  invertirlo empeora los resultados.
- `backend/model/temperature.json` calibra las probabilidades. Si reentrenas, `train.py` lo regenera.

## Datos y modelo
- Entrenamiento y evaluación: `data/data_por_referencia/` (sin oraciones `reference` compartidas
  entre splits). `data/*.csv` en la raíz es la división original por par; `app.py` usa
  `data/train.csv` solo como corpus de búsqueda.
- `backend/model/model.safetensors` va por Git LFS. Si pesa unos bytes, falta `git lfs pull`.
- `backend/cache/` (embeddings del corpus) no se versiona; se regenera al arrancar la API.
- Métricas actuales en `backend/model/metrics.json` (F1 test 0.9724). Si cambias el modelo, reporta
  el F1 en `data/data_por_referencia/test.csv` para que sea comparable.

## Convenciones
- Código y comentarios en español, sin acentos en los `.py` y `.ps1` (PowerShell 5.1 lee mal UTF-8 sin BOM).
- Los scripts `.ps1` deben funcionar en Windows PowerShell 5.1 (sin `&&`, `??`, ternarios).
