# VeritasMed — MCP Server launcher for Claude Desktop / Claude Code
#
# Usage:
#   .\start_mcp.ps1              # print config snippets (safe, read-only)
#   .\start_mcp.ps1 -Dev         # start in dev mode (MCP Inspector, hot-reload)
#   .\start_mcp.ps1 -Install     # install permanently into Claude Desktop
#
# After -Install (or manual config), restart Claude Desktop / Claude Code.
# The MCP tool list will include: search_literature, ask_agent, evaluate_query, search_visual.
#
# Prerequisites:
#   1. conda env "medrag" with fastmcp installed
#   2. .env in project root with OPENAI_API_KEY + OPENAI_BASE_URL
#   3. Qdrant running on localhost:6333
param(
    [switch]$Dev,
    [switch]$Install
)

$Root       = $PSScriptRoot
$ServerPath = Join-Path $Root "src\medrag\mcp_server\server.py"
$EnvFile    = Join-Path $Root ".env"

# ── Find Python ───────────────────────────────────────────────────────────────

function Find-Python {
    $candidates = @(
        "$env:CONDA_PREFIX\python.exe",
        "$env:USERPROFILE\.conda\envs\medrag\python.exe",
        "$env:USERPROFILE\miniconda3\envs\medrag\python.exe",
        "$env:USERPROFILE\anaconda3\envs\medrag\python.exe",
        "C:\ProgramData\miniconda3\envs\medrag\python.exe",
        "D:\Anaconda\envs\medrag\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    $inPath = Get-Command python -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }
    return $null
}

function Find-Mcp {
    param([string]$PyExe)
    # fastmcp installs an 'mcp' console script alongside python
    $mcpExe = Join-Path (Split-Path $PyExe) "mcp.exe"
    if (Test-Path $mcpExe) { return $mcpExe }
    # Also try Scripts sub-directory
    $mcpScripts = Join-Path (Split-Path $PyExe) "Scripts\mcp.exe"
    if (Test-Path $mcpScripts) { return $mcpScripts }
    $inPath = Get-Command mcp -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }
    return $null
}

# ── Prereq checks ─────────────────────────────────────────────────────────────

function Check-Prereqs {
    Write-Host ""
    Write-Host "=== VeritasMed MCP Server ===" -ForegroundColor Cyan
    Write-Host ""

    if (-not (Test-Path $EnvFile)) {
        Write-Host "[WARN] .env not found — copy .env.example → .env and fill in API keys." -ForegroundColor Yellow
    } else {
        Write-Host "[OK] .env found" -ForegroundColor Green
    }

    try {
        Invoke-WebRequest "http://localhost:6333/healthz" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop | Out-Null
        Write-Host "[OK] Qdrant running" -ForegroundColor Green
    } catch {
        Write-Host "[WARN] Qdrant not detected — start with:" -ForegroundColor Yellow
        Write-Host "         docker run -d -p 6333:6333 qdrant/qdrant:latest" -ForegroundColor Yellow
    }
}

# ── Config snippets ───────────────────────────────────────────────────────────

function Print-Config {
    param([string]$PyExe)

    $pyEscaped  = $PyExe  -replace '\\', '\\\\'
    $rootEscaped = $ServerPath -replace '\\', '\\\\'

    Write-Host ""
    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host " Option A — Claude Code (one-time command)" -ForegroundColor White
    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  claude mcp add medrag-agent -- \`"$PyExe\`" \`"$ServerPath\`"" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Or add to your project's .claude/settings.json:" -ForegroundColor DarkGray
    Write-Host @"
  {
    "mcpServers": {
      "medrag-agent": {
        "command": "$pyEscaped",
        "args": ["$rootEscaped"],
        "env": {
          "PYTHONPATH": "$($Root -replace '\\','\\\\')\\src"
        }
      }
    }
  }
"@ -ForegroundColor Yellow
    Write-Host ""
    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host " Option B — Claude Desktop (%APPDATA%\Claude\claude_desktop_config.json)" -ForegroundColor White
    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host @"
  {
    "mcpServers": {
      "medrag-agent": {
        "command": "$pyEscaped",
        "args": ["$rootEscaped"],
        "env": {
          "PYTHONPATH": "$($Root -replace '\\','\\\\')\\src",
          "OPENAI_API_KEY": "<your-key>",
          "OPENAI_BASE_URL": "<your-base-url>"
        }
      }
    }
  }
"@ -ForegroundColor Yellow
    Write-Host ""
    Write-Host " After editing the config, fully restart Claude Desktop." -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host " Option C — Dev mode (test before installing)" -ForegroundColor White
    Write-Host "─────────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  .\start_mcp.ps1 -Dev" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Opens MCP Inspector in browser. Connect to it from Claude Code:" -ForegroundColor DarkGray
    Write-Host "  /mcp" -ForegroundColor Yellow
    Write-Host ""
}

# ── Entry point ───────────────────────────────────────────────────────────────

Check-Prereqs

$py = Find-Python
if (-not $py) {
    Write-Host "[ERROR] Python not found. Activate the medrag conda env first." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python: $py" -ForegroundColor Green

$mcp = Find-Mcp -PyExe $py
if (-not $mcp) {
    Write-Host "[WARN] 'mcp' executable not found. Run: pip install fastmcp" -ForegroundColor Yellow
}

if ($Dev) {
    # ── Dev mode ──────────────────────────────────────────────────────────────
    if (-not $mcp) {
        Write-Host "[ERROR] Cannot start dev mode without 'mcp' executable." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "Starting MCP dev server (hot-reload + Inspector)..." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
    Write-Host ""
    $env:PYTHONPATH = "$Root\src"
    & $mcp dev $ServerPath

} elseif ($Install) {
    # ── Install into Claude Desktop ───────────────────────────────────────────
    if (-not $mcp) {
        Write-Host "[ERROR] Cannot install without 'mcp' executable." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
    Write-Host "Installing MedRAG-Agent into Claude Desktop..." -ForegroundColor Cyan
    $env:PYTHONPATH = "$Root\src"
    & $mcp install $ServerPath --name "MedRAG-Agent"
    Write-Host ""
    Write-Host "[DONE] Restart Claude Desktop to activate the MedRAG-Agent tools." -ForegroundColor Green

} else {
    # ── Default: print config only ────────────────────────────────────────────
    Print-Config -PyExe $py
}
