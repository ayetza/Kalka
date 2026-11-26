# Instala todo lo necesario para correr Kalka en Windows (una sola vez).
# Requisitos previos: Python 3.12+ (comando `py`), Node.js 20+ y Git con Git LFS.
#   winget install Python.Python.3.14 OpenJS.NodeJS.LTS Git.Git
$ErrorActionPreference = "Stop"
$venv = "$env:USERPROFILE\.venvs\kalka"   # fuera del repo/OneDrive para no sincronizar PyTorch
$py = "$venv\Scripts\python.exe"

# 1. Modelo entrenado (Git LFS): si solo hay un puntero de texto, descarga el archivo real
git lfs install
git lfs pull
if ((Get-Item "$PSScriptRoot\backend\model\model.safetensors").Length -lt 1MB) {
    throw "backend\model\model.safetensors no se descargo. Instala Git LFS y corre: git lfs pull"
}

# 2. Entorno de Python
if (-not (Test-Path $py)) { py -m venv $venv }
& $py -m pip install --upgrade pip
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "GPU NVIDIA detectada: instalando PyTorch con CUDA"
    & $py -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
} else {
    Write-Host "Sin GPU NVIDIA: instalando PyTorch para CPU (funciona, pero mas lento)"
    & $py -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
}
& $py -m pip install -r "$PSScriptRoot\backend\requirements.txt"

# 3. Frontend
$envLocal = "$PSScriptRoot\frontend\.env.local"
if (-not (Test-Path $envLocal)) { "NEXT_PUBLIC_API_URL=http://localhost:8000" | Out-File -Encoding ascii $envLocal }
Push-Location "$PSScriptRoot\frontend"
npm install
Pop-Location

Write-Host "`nListo. Ahora abre dos terminales y corre .\iniciar_backend.ps1 y .\iniciar_frontend.ps1"
