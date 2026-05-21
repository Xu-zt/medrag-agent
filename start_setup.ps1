# VeritasMed - first-time / beginner setup + start everything
#
# Usage:
#   .\start_setup.ps1
#   .\start_setup.ps1 -SkipIngest

param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173,
    [switch]$SkipIngest
)

. "$PSScriptRoot\_start_common.ps1"

$Root = $PSScriptRoot

function Test-HasRawData {
    $pub = Join-Path $Root "data\raw\pubmed\abstracts.jsonl"
    $pmc = Join-Path $Root "data\raw\pmc\full_texts.jsonl"
    return ((Test-Path $pub) -and (Get-Item $pub).Length -gt 0) -or
           ((Test-Path $pmc) -and (Get-Item $pmc).Length -gt 0)
}

function Test-HasIndexCache {
    $dense = Join-Path $Root "data\index_cache\dense.npy"
    $chunks = Join-Path $Root "data\index_cache\chunks.jsonl"
    return (Test-Path $dense) -and (Test-Path $chunks)
}

Write-Host ""
Write-Host "=== VeritasMed - Setup & Start ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-StepError "Install Docker Desktop first: https://www.docker.com/products/docker-desktop/"
    exit 1
}
if (-not (Test-DockerDaemon)) {
    Write-StepError "Docker Desktop is not running. Start it and retry."
    exit 1
}
Write-Host "[OK] Docker ready" -ForegroundColor Green

if (-not (Test-QdrantRunning)) {
    if (-not (Start-QdrantDocker)) { exit 1 }
}

$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    Write-StepError "conda not in PATH. Install Miniconda/Anaconda and reopen the terminal."
    exit 1
}

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$envList = & conda env list 2>&1 | Out-String
$ErrorActionPreference = $prevEap
$hasMedrag = $envList -match "(?m)^$([regex]::Escape($script:MedragCondaEnv))\s"

if (-not $hasMedrag) {
    Write-Host "[*] Creating conda env '$($script:MedragCondaEnv)' (Python 3.12 only) ..." -ForegroundColor Cyan
    Push-Location $Root
    & conda env create -f environment.yml
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-StepError "conda env create failed"
        exit 1
    }
    Pop-Location
    Write-Host "[OK] conda env created" -ForegroundColor Green
} else {
    Write-Host "[OK] conda env $($script:MedragCondaEnv) exists" -ForegroundColor Green
}

$py = Find-MedragPython
if (-not $py) {
    Write-StepError "Could not locate python.exe for conda env $($script:MedragCondaEnv)"
    exit 1
}
Write-Host "[OK] Python: $py" -ForegroundColor Green

$code = Install-ProjectDependencies -PythonExe $py
if ($code -ne 0) {
    Write-StepError "pip install failed" "Try: conda activate medrag; pip install -e ."
    exit 1
}
if (-not (Test-MedragEnvHealthy -PythonExe $py -OnError { param($m) Write-Host $m -ForegroundColor DarkRed })) {
    Write-StepError "Dependencies installed but health check failed" "See output above; try: pip install -e ."
    exit 1
}
Write-Host "[OK] Python dependencies installed (pyproject.toml)" -ForegroundColor Green

$envFile = Join-Path $Root ".env"
if (-not (Test-Path $envFile)) {
    $example = Join-Path $Root ".env.example"
    if (Test-Path $example) {
        Copy-Item $example $envFile
        Write-Host "[WARN] Created .env from .env.example - add OPENAI_API_KEY before chatting" -ForegroundColor Yellow
    }
}

$count = Get-QdrantPointCount -PythonExe $py
if ($count -gt 0) {
    Write-Host "[OK] Qdrant already has $count points - skipping index build" -ForegroundColor Green
} else {
    Write-Host "[*] Qdrant empty - checking local data ..." -ForegroundColor Cyan
    $hasRaw = Test-HasRawData
    $hasCache = Test-HasIndexCache

    if (-not $hasRaw -and -not $hasCache) {
        Write-StepError "No data/raw/ and no data/index_cache/."
        Write-Host "  Unzip data.zip to project root, or run ingest (NCBI_EMAIL in .env)." -ForegroundColor Yellow
        exit 1
    }

    if ($hasCache) {
        Write-Host "[*] Uploading index_cache to Qdrant (--phase=index) ..." -ForegroundColor Cyan
        $code = Invoke-ProjectPython -PythonExe $py -Args @("scripts/04_build_index.py", "--phase=index")
    } elseif ($hasRaw) {
        if ($SkipIngest) {
            Write-StepError "Have raw/ but no index_cache. Run: python scripts/04_build_index.py"
            exit 1
        }
        Write-Host "[*] Building vectors + index from raw/ (long; GPU recommended) ..." -ForegroundColor Cyan
        $code = Invoke-ProjectPython -PythonExe $py -Args @("scripts/04_build_index.py", "--phase=all")
    }

    if ($code -ne 0) {
        Write-StepError "Index build failed (exit $code)"
        exit 1
    }

    $count = Get-QdrantPointCount -PythonExe $py -OnError { param($m) Write-Host $m -ForegroundColor DarkRed }
    if ($count -le 0) {
        Write-StepError "Index step finished but Qdrant still has 0 points."
        exit 1
    }
    Write-Host "[OK] Indexed $count points into Qdrant" -ForegroundColor Green
}

Ensure-FrontendEnvLocal -BackendPort $BackendPort
$feEnvExample = Join-Path $Root "frontend\.env.example"
if (-not (Test-Path $feEnvExample)) {
    Set-Content -Path $feEnvExample -Value "VITE_API_URL=http://localhost:8000" -Encoding UTF8
}

Write-Host ""
Write-Host "[*] Starting backend and frontend ..." -ForegroundColor Cyan
Start-BackendWindow -PythonExe $py -Port $BackendPort
Start-Sleep -Seconds 2
if (-not (Start-FrontendWindow -Port $FrontendPort)) { exit 1 }

Show-DevBanner -BackendPort $BackendPort -FrontendPort $FrontendPort
Write-Host "Setup complete. Logs are in the two new terminal windows." -ForegroundColor Green
