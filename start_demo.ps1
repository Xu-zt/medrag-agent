# VeritasMed — Start demo servers
# Usage: .\start_demo.ps1
#
# Requires:
#   - conda env "medrag" activated environment
#   - Qdrant running (docker run qdrant/qdrant)
#   - Ollama running (ollama serve) with qwen3:8b pulled

param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

Write-Host "`n=== VeritasMed Demo ===" -ForegroundColor Cyan

# Check Qdrant
try {
    $r = Invoke-WebRequest "http://localhost:6333/healthz" -TimeoutSec 2 -UseBasicParsing
    Write-Host "[OK] Qdrant is running" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Qdrant not detected at localhost:6333. Start it with:" -ForegroundColor Yellow
    Write-Host "  docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest" -ForegroundColor Yellow
}

# Check Ollama
try {
    $r = Invoke-WebRequest "http://localhost:11434/api/tags" -TimeoutSec 2 -UseBasicParsing
    Write-Host "[OK] Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Ollama not detected at localhost:11434. Start it with: ollama serve" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting FastAPI backend on http://localhost:$BackendPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "conda activate medrag; Set-Location '$PSScriptRoot'; uvicorn medrag.api.app:app --reload --port $BackendPort"

Start-Sleep -Seconds 2

Write-Host "Starting Vite frontend on http://localhost:$FrontendPort ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "cd '$PSScriptRoot\frontend'; npm run dev -- --port $FrontendPort"

Write-Host ""
Write-Host "Open http://localhost:$FrontendPort in your browser." -ForegroundColor Green
Write-Host "API docs: http://localhost:$BackendPort/docs" -ForegroundColor Green
Write-Host ""
