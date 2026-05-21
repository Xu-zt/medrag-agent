# Shared helpers for start_dev.ps1 / start_setup.ps1 / start_mcp.ps1
# Dot-source:  . "$PSScriptRoot\_start_common.ps1"

param()

$script:ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$script:MedragCondaEnv = "medrag"

function Get-ProjectRoot { $script:ProjectRoot }

function Write-StepError {
    param([string]$Message, [string]$Detail = "")
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    if ($Detail) { Write-Host $Detail -ForegroundColor DarkRed }
}

# ── .env / Qdrant URL ─────────────────────────────────────────────────────────

function Get-DotEnvValue {
    param([string]$Key, [string]$Default = "")
    $fromProcess = [Environment]::GetEnvironmentVariable($Key)
    if ($fromProcess) { return $fromProcess.Trim() }
    $envPath = Join-Path $script:ProjectRoot ".env"
    if (-not (Test-Path $envPath)) { return $Default }
    foreach ($line in Get-Content $envPath -Encoding UTF8) {
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=\s*(.+)\s*$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return $Default
}

function Get-QdrantBaseUrl {
    return (Get-DotEnvValue -Key "QDRANT_URL" -Default "http://localhost:6333").TrimEnd('/')
}

function Get-QdrantHealthUrl {
    return "$(Get-QdrantBaseUrl)/healthz"
}

function Get-QdrantHostPort {
    $base = Get-QdrantBaseUrl
    if ($base -match '^https?://[^:]+:(\d+)$') { return [int]$Matches[1] }
    if ($base -match '^https?://([^:/]+)$') { return 6333 }
    return 6333
}

function Test-QdrantRunning {
    try {
        Invoke-WebRequest (Get-QdrantHealthUrl) -TimeoutSec 4 -UseBasicParsing -ErrorAction Stop | Out-Null
        return $true
    } catch { return $false }
}

function Test-DockerDaemon {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $false }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        docker info 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    } finally { $ErrorActionPreference = $prev }
}

function Start-QdrantDocker {
    param([string]$ContainerName = "medrag-qdrant")

    if (-not (Test-DockerDaemon)) {
        Write-StepError "Docker is installed but not running. Start Docker Desktop first."
        return $false
    }

    $port = Get-QdrantHostPort
    $base = Get-QdrantBaseUrl

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $existing = docker ps -a --filter "name=^${ContainerName}$" --format "{{.Names}}" 2>$null
        if ($existing -eq $ContainerName) {
            $running = docker ps --filter "name=^${ContainerName}$" --format "{{.Names}}" 2>$null
            if ($running -ne $ContainerName) {
                Write-Host "[*] Starting container $ContainerName ..." -ForegroundColor Cyan
                docker start $ContainerName 2>&1 | Out-Null
            }
        } else {
            Write-Host "[*] Creating Qdrant container $ContainerName (host port $port) ..." -ForegroundColor Cyan
            docker run -d --name $ContainerName -p "${port}:6333" qdrant/qdrant:latest 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-StepError "docker run failed (port $port in use?)"
                return $false
            }
        }
    } finally { $ErrorActionPreference = $prev }

    foreach ($i in 1..20) {
        Start-Sleep -Seconds 1
        if (Test-QdrantRunning) {
            Write-Host "[OK] Qdrant ready at $base" -ForegroundColor Green
            return $true
        }
    }
    Write-StepError "Qdrant did not become healthy at $base"
    return $false
}

# ── Python / conda (medrag env only) ───────────────────────────────────────────

function Get-MedragPythonFromCondaList {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) { return $null }

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Reliable: resolve executable via conda run (works for ~/.conda, D:\Anaconda\envs, etc.)
        $raw = & conda run -n $script:MedragCondaEnv python -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $raw) {
            $lines = @($raw | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
            $exe = $lines[-1]
            if ($exe -and (Test-Path -LiteralPath $exe)) { return $exe }
        }

        # Fallback: parse `conda env list` (handles non-default install locations)
        $list = & conda env list 2>&1 | Out-String
        if ($list -match "(?m)^$([regex]::Escape($script:MedragCondaEnv))\s+\*?\s+(\S+)") {
            $exe = Join-Path $Matches[2] "python.exe"
            if (Test-Path $exe) { return $exe }
        }
    } finally { $ErrorActionPreference = $prev }

    return $null
}

function Find-MedragPython {
    # 1) conda (authoritative — works for ~/.conda, D:\Anaconda\envs, etc.)
    $fromConda = Get-MedragPythonFromCondaList
    if ($fromConda) { return $fromConda }

    # 2) common install paths
    $candidates = @(
        "$env:USERPROFILE\.conda\envs\$($script:MedragCondaEnv)\python.exe",
        "$env:USERPROFILE\miniconda3\envs\$($script:MedragCondaEnv)\python.exe",
        "$env:USERPROFILE\anaconda3\envs\$($script:MedragCondaEnv)\python.exe",
        "C:\ProgramData\miniconda3\envs\$($script:MedragCondaEnv)\python.exe",
        "C:\ProgramData\Anaconda3\envs\$($script:MedragCondaEnv)\python.exe",
        "D:\Anaconda\envs\$($script:MedragCondaEnv)\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }

    # 3) only if user already activated medrag
    if ($env:CONDA_DEFAULT_ENV -eq $script:MedragCondaEnv -and $env:CONDA_PREFIX) {
        $active = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $active) { return $active }
    }

    return $null
}

function Test-MedragEnvHealthy {
    param(
        [string]$PythonExe,
        [scriptblock]$OnError,
        [switch]$RequireFastMcp
    )
    $extra = if ($RequireFastMcp) { "import fastmcp" } else { "" }
    $code = @"
import sys
sys.path.insert(0, r'$($script:ProjectRoot)\src')
import medrag.api.app
import qdrant_client
$extra
print('ok')
"@
    $prevPath = $env:PYTHONPATH
    $prevEap = $ErrorActionPreference
    $env:PYTHONPATH = "$($script:ProjectRoot)\src"
    $ErrorActionPreference = "Continue"
    try {
        $out = & $PythonExe -c $code 2>&1 | Out-String
        $lines = @($out -split "[\r\n]+" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        $ok = ($LASTEXITCODE -eq 0) -and ($lines.Count -gt 0) -and ($lines[-1] -eq "ok")
        if (-not $ok -and $OnError) { & $OnError $out.Trim() }
        return $ok
    } finally {
        $env:PYTHONPATH = $prevPath
        $ErrorActionPreference = $prevEap
    }
}

function Invoke-ProjectPython {
    param(
        [string]$PythonExe,
        [string[]]$Args
    )
    $prevPath = $env:PYTHONPATH
    $prevEap = $ErrorActionPreference
    $prevLoc = Get-Location
    $env:PYTHONPATH = "$($script:ProjectRoot)\src"
    $ErrorActionPreference = "Continue"
    try {
        Set-Location $script:ProjectRoot
        & $PythonExe @Args
        return $LASTEXITCODE
    } finally {
        Set-Location $prevLoc
        $env:PYTHONPATH = $prevPath
        $ErrorActionPreference = $prevEap
    }
}

# Scheme A: conda env is Python-only; all libraries come from pyproject.toml via pip.
function Install-ProjectDependencies {
    param(
        [string]$PythonExe,
        [switch]$WithDev
    )
    $torchIndex = "https://download.pytorch.org/whl/cu124"
    Write-Host "[*] Installing PyTorch (CUDA 12.4 wheel index) ..." -ForegroundColor Cyan
    $code = Invoke-ProjectPython -PythonExe $PythonExe -Args @(
        "-m", "pip", "install", "torch>=2.6", "--index-url", $torchIndex
    )
    if ($code -ne 0) {
        Write-Host "[WARN] CUDA PyTorch install failed; trying CPU build from PyPI ..." -ForegroundColor Yellow
        $code = Invoke-ProjectPython -PythonExe $PythonExe -Args @("-m", "pip", "install", "torch>=2.6")
        if ($code -ne 0) { return $code }
    }

    $editable = if ($WithDev) { ".[dev]" } else { "." }
    Write-Host "[*] Installing project (pip install -e $editable) from pyproject.toml ..." -ForegroundColor Cyan
    return (Invoke-ProjectPython -PythonExe $PythonExe -Args @("-m", "pip", "install", "-e", $editable))
}

function Get-QdrantPointCount {
    param(
        [string]$PythonExe,
        [scriptblock]$OnError
    )
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $prevPath = $env:PYTHONPATH
        $env:PYTHONPATH = "$($script:ProjectRoot)\src"
        $out = & $PythonExe (Join-Path $script:ProjectRoot "scripts\qdrant_status.py") 2>&1 | Out-String
        $env:PYTHONPATH = $prevPath
        if ($out -match '(?m)^ok:(\d+)\s*$') { return [int]$Matches[1] }
        if ($OnError) { & $OnError $out.Trim() }
        return 0
    } catch {
        if ($OnError) { & $OnError $_.Exception.Message }
        return 0
    } finally { $ErrorActionPreference = $prevEap }
}

function Test-NodeJs {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { return $false }
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { return $false }
    return $true
}

function Ensure-FrontendEnvLocal {
    param([int]$BackendPort = 8000)
    $path = Join-Path $script:ProjectRoot "frontend\.env.local"
    $expected = "VITE_API_URL=http://localhost:$BackendPort"
    if (-not (Test-Path $path)) {
        Set-Content -Path $path -Value $expected -Encoding UTF8
        Write-Host "[OK] Created frontend/.env.local" -ForegroundColor Green
        return
    }
    $content = Get-Content $path -Raw -Encoding UTF8
    if ($content -notmatch [regex]::Escape("localhost:$BackendPort")) {
        Set-Content -Path $path -Value $expected -Encoding UTF8
        Write-Host "[OK] Updated frontend/.env.local for port $BackendPort" -ForegroundColor Green
    }
}

function Escape-SingleQuotedPath {
    param([string]$Path)
    return $Path -replace "'", "''"
}

function Start-BackendWindow {
    param(
        [string]$PythonExe,
        [int]$Port = 8000
    )
    $root = Escape-SingleQuotedPath $script:ProjectRoot
    $py = Escape-SingleQuotedPath $PythonExe
    $cmd = @"
`$Host.UI.RawUI.WindowTitle = 'VeritasMed Backend :$Port'
Set-Location '$root'
`$env:PYTHONPATH = '$root\src'
`$env:TOKENIZERS_PARALLELISM = 'false'
& '$py' -m uvicorn medrag.api.app:app --host 0.0.0.0 --port $Port --reload
"@
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd | Out-Null
}

function Clear-PortListeners {
    param([int]$Port)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
        foreach ($l in $listeners) {
            $owningPid = $l.OwningProcess
            if (-not $owningPid) { continue }
            $proc = Get-Process -Id $owningPid -ErrorAction SilentlyContinue
            $name = if ($proc) { $proc.ProcessName } else { "process" }
            Write-Host "[*] Freeing port ${Port} (stopping $name, PID $owningPid) ..." -ForegroundColor Yellow
            Stop-Process -Id $owningPid -Force -ErrorAction SilentlyContinue
        }
        if ($listeners.Count -gt 0) { Start-Sleep -Seconds 1 }
    } finally { $ErrorActionPreference = $prev }
}

function Start-FrontendWindow {
    param([int]$Port = 5173)
    if (-not (Test-NodeJs)) {
        Write-StepError "Node.js / npm not found. Install Node 18+ from https://nodejs.org/"
        return $false
    }
    Clear-PortListeners -Port $Port
    $feDir = Join-Path $script:ProjectRoot "frontend"
    $fe = Escape-SingleQuotedPath $feDir
    $nodeModules = Join-Path $feDir "node_modules"
    if (-not (Test-Path $nodeModules)) {
        Write-Host "[*] npm install (first time, may take a minute) ..." -ForegroundColor Cyan
        Push-Location $feDir
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        npm install 2>&1 | Out-Host
        $ErrorActionPreference = $prev
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            Write-StepError "npm install failed"
            return $false
        }
        Pop-Location
    }
    # Do NOT pass "--port" via npm here — PowerShell parses "--" as decrement, so
    # "npm run dev -- --port 5173" becomes "vite 5173" (wrong root dir → HTTP 404).
    # Port/host/strictPort are set in frontend/vite.config.ts and package.json "dev".
    if ($Port -ne 5173) {
        Write-Host "[WARN] Frontend port is fixed at 5173 in vite.config.ts (requested $Port ignored)" -ForegroundColor Yellow
    }
    $cmd = @"
`$Host.UI.RawUI.WindowTitle = 'VeritasMed Frontend :5173'
Set-Location '$fe'
npm run dev
"@
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd | Out-Null
    return $true
}

function Show-DevBanner {
    param([int]$BackendPort = 8000, [int]$FrontendPort = 5173)
    Write-Host ""
    Write-Host "  UI:       http://localhost:$FrontendPort" -ForegroundColor Green
    Write-Host "  API:      http://localhost:$BackendPort" -ForegroundColor Green
    Write-Host "  API docs: http://localhost:$BackendPort/docs" -ForegroundColor DarkGray
    Write-Host "  Qdrant:   $(Get-QdrantBaseUrl)" -ForegroundColor DarkGray
    Write-Host ""
}

function Assert-MedragPython {
    param(
        [switch]$RequireFastMcp
    )
    $py = Find-MedragPython
    if (-not $py) {
        Write-StepError "conda env '$($script:MedragCondaEnv)' not found." "Run: .\start_setup.ps1"
        return $null
    }
    Write-Host "[OK] Python: $py" -ForegroundColor Green

    if ($env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne $script:MedragCondaEnv) {
        Write-Host "[WARN] Active conda env is '$($env:CONDA_DEFAULT_ENV)' - using medrag env above anyway." -ForegroundColor Yellow
    }

    $healthy = Test-MedragEnvHealthy -PythonExe $py -RequireFastMcp:$RequireFastMcp -OnError {
        param($msg)
        Write-StepError "Python dependency check failed:" $msg
    }
    if (-not $healthy) {
        Write-Host "  Fix: .\start_setup.ps1" -ForegroundColor Yellow
        return $null
    }
    return $py
}
