# Run the Streamlit demo UI (lightweight alternative to the full React UI)
#
# Usage:
#   .\scripts\run_ui.ps1
#   .\scripts\run_ui.ps1 -Port 8502
#
# For the full React + FastAPI web UI use: .\start_ui.ps1
param([int]$Port = 8501)

$Root = Split-Path $PSScriptRoot -Parent

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

$py = Find-Python
if (-not $py) {
    Write-Host "[ERROR] Python not found. Activate the medrag conda env first." -ForegroundColor Red
    exit 1
}

$streamlit = Join-Path (Split-Path $py) "streamlit.exe"
if (-not (Test-Path $streamlit)) {
    $streamlit = Join-Path (Split-Path $py) "Scripts\streamlit.exe"
}
if (-not (Test-Path $streamlit)) {
    Write-Host "[ERROR] streamlit not found. Install with: pip install streamlit" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH        = "$Root\src"
$env:PYTHONIOENCODING  = "utf-8"

Write-Host "Starting MedRAG-Agent Streamlit UI at http://localhost:$Port" -ForegroundColor Cyan
& $streamlit run "$Root\src\medrag\ui\app.py" --server.port $Port --server.headless false
