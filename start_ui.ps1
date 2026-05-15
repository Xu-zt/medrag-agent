# VeritasMed — One-click Web UI
#
# Usage:
#   .\start_ui.ps1              # starts backend; open http://localhost:8000
#   .\start_ui.ps1 -Port 9000   # custom port
#
# Prerequisites:
#   1. conda env "medrag" exists  (conda env create -f environment.yml)
#   2. .env file in project root  (copy .env.example .env  then fill in keys)
#   3. Qdrant running             (docker run -d -p 6333:6333 qdrant/qdrant:latest)
#
# For frontend hot-reload (HMR) during development, also run in a second terminal:
#   cd frontend; npm run dev    (then open http://localhost:5173 instead)
param([int]$Port = 8000)

$Root = $PSScriptRoot

function Find-Python {
    $candidates = @(
        "$env:CONDA_PREFIX\python.exe",
        "$env:USERPROFILE\.conda\envs\medrag\python.exe",
        "$env:USERPROFILE\miniconda3\envs\medrag\python.exe",
        "$env:USERPROFILE\anaconda3\envs\medrag\python.exe",
        "C:\ProgramData\miniconda3\envs\medrag\python.exe",
        "D:\Anaconda\envs\medrag\python.exe"
    )
    foreach ($p in $candidates) { if (Test-Path $p) { return $p } }
    $inPath = Get-Command python -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }
    return $null
}

Write-Host ""
Write-Host "=== VeritasMed Web UI ===" -ForegroundColor Cyan
Write-Host ""

# ── Python ────────────────────────────────────────────────────────────────────
$py = Find-Python
if (-not $py) {
    Write-Host "[ERROR] Python not found. Activate the medrag conda env and re-run." -ForegroundColor Red
    Write-Host "  conda activate medrag" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Python: $py" -ForegroundColor Green

# ── .env ─────────────────────────────────────────────────────────────────────
if (-not (Test-Path "$Root\.env")) {
    Write-Host "[WARN] .env not found — LLM calls will fail without API keys." -ForegroundColor Yellow
    Write-Host "  copy .env.example .env   then edit .env with your OPENAI_API_KEY" -ForegroundColor Yellow
} else {
    Write-Host "[OK] .env found" -ForegroundColor Green
}

# ── Qdrant ────────────────────────────────────────────────────────────────────
try {
    Invoke-WebRequest "http://localhost:6333/healthz" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop | Out-Null
    Write-Host "[OK] Qdrant running" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Qdrant not detected — start it with:" -ForegroundColor Yellow
    Write-Host "  docker run -d -p 6333:6333 qdrant/qdrant:latest" -ForegroundColor Yellow
}

# ── Frontend dist ─────────────────────────────────────────────────────────────
if (-not (Test-Path "$Root\frontend\dist\index.html")) {
    Write-Host "[INFO] Building frontend (first run)..." -ForegroundColor Cyan
    Push-Location "$Root\frontend"
    npm install --silent
    npm run build
    Pop-Location
}

Write-Host ""
Write-Host "Starting server on http://localhost:$Port ..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

$env:PYTHONPATH = "$Root\src"
$env:TOKENIZERS_PARALLELISM = "false"
& $py -m uvicorn medrag.api.app:app --host 0.0.0.0 --port $Port
