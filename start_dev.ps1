# VeritasMed - developer one-click: Qdrant + FastAPI + Vite (conda env medrag)
#
# Usage:
#   .\start_dev.ps1
#   .\start_dev.ps1 -BackendPort 8000 -FrontendPort 5173
#
# Safe when conda is on (base) or (medrag) — always uses the medrag env python.

param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

. "$PSScriptRoot\_start_common.ps1"

Write-Host ""
Write-Host "=== VeritasMed - Dev ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-StepError "Docker not found. Install Docker Desktop for Qdrant."
    exit 1
}
if (-not (Test-DockerDaemon)) {
    Write-StepError "Docker Desktop is not running. Start it and retry."
    exit 1
}

if (-not (Test-QdrantRunning)) {
    if (-not (Start-QdrantDocker)) { exit 1 }
} else {
    Write-Host "[OK] Qdrant already running at $(Get-QdrantBaseUrl)" -ForegroundColor Green
}

$py = Assert-MedragPython
if (-not $py) { exit 1 }

$count = Get-QdrantPointCount -PythonExe $py -OnError {
    param($msg)
    Write-Host "[WARN] Could not read Qdrant status: $msg" -ForegroundColor Yellow
}
if ($count -le 0) {
    Write-Host "[WARN] Qdrant collection empty or missing. Run: .\start_setup.ps1" -ForegroundColor Yellow
    Write-Host "       Or: python scripts/04_build_index.py --phase=index" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Qdrant indexed: $count points" -ForegroundColor Green
}

if (-not (Test-Path (Join-Path $PSScriptRoot ".env"))) {
    Write-Host "[WARN] .env missing - copy .env.example and add OPENAI_API_KEY" -ForegroundColor Yellow
} else {
    Write-Host "[OK] .env found" -ForegroundColor Green
}

Ensure-FrontendEnvLocal -BackendPort $BackendPort

Write-Host "[*] Starting backend (new window) ..." -ForegroundColor Cyan
Start-BackendWindow -PythonExe $py -Port $BackendPort
Start-Sleep -Seconds 2

Write-Host "[*] Starting frontend (new window) ..." -ForegroundColor Cyan
if (-not (Start-FrontendWindow -Port $FrontendPort)) { exit 1 }

Show-DevBanner -BackendPort $BackendPort -FrontendPort $FrontendPort
Write-Host "Close the backend/frontend PowerShell windows to stop servers." -ForegroundColor DarkGray
