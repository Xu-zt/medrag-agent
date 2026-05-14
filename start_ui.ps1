# VeritasMed — One-click Web UI (React + FastAPI)
#
# Usage:
#   .\start_ui.ps1                          # dev mode: FastAPI :8000 + Vite HMR :5173
#   .\start_ui.ps1 -ApiPort 9000            # custom API port
#   .\start_ui.ps1 -ApiOnly                 # backend only (no npm)
#
# Prerequisites:
#   1. conda env "medrag" exists  (conda env create -f environment.yml)
#   2. .env file in project root  (copy .env.example → .env and fill in keys)
#   3. Qdrant running             (docker run -d -p 6333:6333 qdrant/qdrant:latest)
#   4. Node ≥ 18 installed        (for Vite frontend)
param(
    [int]$ApiPort      = 8000,
    [int]$FrontendPort = 5173,
    [switch]$ApiOnly
)

$Root = $PSScriptRoot

# ── Helpers ───────────────────────────────────────────────────────────────────

function Find-Python {
    $candidates = @(
        # Active conda env
        "$env:CONDA_PREFIX\python.exe",
        # Common conda install locations
        "$env:USERPROFILE\.conda\envs\medrag\python.exe",
        "$env:USERPROFILE\miniconda3\envs\medrag\python.exe",
        "$env:USERPROFILE\anaconda3\envs\medrag\python.exe",
        "C:\ProgramData\miniconda3\envs\medrag\python.exe",
        "D:\Anaconda\envs\medrag\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    # Fall back to whatever is in PATH
    $inPath = Get-Command python -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }
    return $null
}

function Check-Env {
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) {
        Write-Host "[WARN] .env not found. Copy .env.example → .env and fill in your API keys." -ForegroundColor Yellow
        Write-Host "       The backend will start but LLM calls will fail without valid credentials." -ForegroundColor Yellow
    } else {
        Write-Host "[OK] .env found" -ForegroundColor Green
    }
}

function Check-Qdrant {
    try {
        $r = Invoke-WebRequest "http://localhost:6333/healthz" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        Write-Host "[OK] Qdrant running at localhost:6333" -ForegroundColor Green
    } catch {
        Write-Host "[WARN] Qdrant not detected at localhost:6333" -ForegroundColor Yellow
        Write-Host "       Start it with:" -ForegroundColor Yellow
        Write-Host "         docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest" -ForegroundColor Yellow
    }
}

# ── Main ──────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "=== VeritasMed Web UI ===" -ForegroundColor Cyan
Write-Host ""

$py = Find-Python
if (-not $py) {
    Write-Host "[ERROR] Python not found. Activate the medrag conda env and re-run." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python: $py" -ForegroundColor Green

Check-Env
Check-Qdrant
Write-Host ""

# ── Start FastAPI backend ──────────────────────────────────────────────────────
Write-Host "Starting FastAPI backend on http://localhost:$ApiPort ..." -ForegroundColor Cyan
Write-Host "  API docs: http://localhost:$ApiPort/docs" -ForegroundColor DarkGray

$backendCmd = "Set-Location '$Root'; $py -m uvicorn medrag.api.app:app --reload --port $ApiPort"
$backendEnv  = "PYTHONPATH=$Root\src; $backendCmd"
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "`$env:PYTHONPATH='$Root\src'; Set-Location '$Root'; & '$py' -m uvicorn medrag.api.app:app --reload --port $ApiPort"

if ($ApiOnly) {
    Write-Host ""
    Write-Host "Backend started (--ApiOnly). Browse http://localhost:$ApiPort/docs" -ForegroundColor Green
    exit 0
}

# ── Start Vite frontend ────────────────────────────────────────────────────────
$frontendDir = Join-Path $Root "frontend"
if (-not (Test-Path "$frontendDir\node_modules")) {
    Write-Host ""
    Write-Host "[INFO] node_modules missing — running npm install first..." -ForegroundColor Yellow
    Push-Location $frontendDir
    npm install
    Pop-Location
}

Start-Sleep -Seconds 2

Write-Host "Starting Vite frontend on http://localhost:$FrontendPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$frontendDir'; npm run dev -- --port $FrontendPort"

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== Ready ===" -ForegroundColor Green
Write-Host "  Web UI : http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "  API    : http://localhost:$ApiPort/docs" -ForegroundColor Green
Write-Host ""
Write-Host "To stop: close the two terminal windows that opened." -ForegroundColor DarkGray
