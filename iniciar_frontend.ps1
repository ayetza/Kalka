# Inicia la pagina de Kalka en http://localhost:3000 (usa la API de frontend\.env.local)
$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
$envLocal = "$PSScriptRoot\frontend\.env.local"
if (-not (Test-Path $envLocal)) { "NEXT_PUBLIC_API_URL=http://localhost:8000" | Out-File -Encoding ascii $envLocal }
Set-Location "$PSScriptRoot\frontend"
npm run dev
