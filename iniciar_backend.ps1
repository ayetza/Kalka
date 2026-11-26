# Inicia la API de Kalka (red neuronal) en http://localhost:8000
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
Set-Location "$PSScriptRoot\backend"
& "$env:USERPROFILE\.venvs\kalka\Scripts\python.exe" -m uvicorn app:app --port 8000
