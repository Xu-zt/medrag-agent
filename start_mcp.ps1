# VeritasMed - MCP Server launcher for Claude Desktop / Claude Code
#
# Usage:
#   .\start_mcp.ps1              # print config snippets
#   .\start_mcp.ps1 -Dev         # MCP Inspector (hot-reload)
#   .\start_mcp.ps1 -Install     # install into Claude Desktop
#
# Uses the same medrag conda env resolution as start_dev.ps1 (never base by mistake).

param(
    [switch]$Dev,
    [switch]$Install
)

$Root = $PSScriptRoot
. "$Root\_start_common.ps1"

$ServerPath = Join-Path $Root "src\medrag\mcp_server\server.py"
$EnvFile    = Join-Path $Root ".env"

function Find-Mcp {
    param([string]$PyExe)
    $dir = Split-Path $PyExe
    foreach ($name in @("mcp.exe", "Scripts\mcp.exe")) {
        $p = Join-Path $dir $name
        if (Test-Path $p) { return $p }
    }
    $inPath = Get-Command mcp -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }
    return $null
}

Write-Host ""
Write-Host "=== VeritasMed MCP Server ===" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $EnvFile)) {
    Write-Host "[WARN] .env not found - copy .env.example and add API keys." -ForegroundColor Yellow
} else {
    Write-Host "[OK] .env found" -ForegroundColor Green
}

if (Test-QdrantRunning) {
    $n = 0
    $pyProbe = Find-MedragPython
    if ($pyProbe) { $n = Get-QdrantPointCount -PythonExe $pyProbe }
    if ($n -gt 0) {
        Write-Host "[OK] Qdrant at $(Get-QdrantBaseUrl) ($n points)" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Qdrant up but collection empty. Run .\start_setup.ps1 first." -ForegroundColor Yellow
    }
} else {
    Write-Host "[WARN] Qdrant not at $(Get-QdrantBaseUrl). Run .\start_dev.ps1 or start Docker Qdrant." -ForegroundColor Yellow
}

$py = Assert-MedragPython -RequireFastMcp
if (-not $py) {
    Write-Host "  Install: pip install fastmcp  (inside conda env medrag)" -ForegroundColor Yellow
    exit 1
}

$mcp = Find-Mcp -PyExe $py
if (-not $mcp) {
    Write-Host "[WARN] mcp CLI not found. Run: pip install fastmcp" -ForegroundColor Yellow
}

function Print-Config {
    param([string]$PyExe)
    $pyEscaped   = $PyExe -replace '\\', '\\\\'
    $rootEscaped = $ServerPath -replace '\\', '\\\\'
    $srcPath     = ($Root -replace '\\', '\\\\') + '\\src'

    Write-Host ""
    Write-Host " Option A - Claude Code:" -ForegroundColor White
    Write-Host "  claude mcp add medrag-agent -- `"$PyExe`" `"$ServerPath`"" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Option B - .claude/settings.json:" -ForegroundColor White
    Write-Host @"
  {
    "mcpServers": {
      "medrag-agent": {
        "command": "$pyEscaped",
        "args": ["$rootEscaped"],
        "env": { "PYTHONPATH": "$srcPath" }
      }
    }
  }
"@ -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Option C - test: .\start_mcp.ps1 -Dev" -ForegroundColor White
    Write-Host ""
}

if ($Dev) {
    if (-not $mcp) {
        Write-StepError "Cannot start dev mode without mcp executable."
        exit 1
    }
    Write-Host "Starting MCP dev server ..." -ForegroundColor Cyan
    $env:PYTHONPATH = "$Root\src"
    & $mcp dev $ServerPath
} elseif ($Install) {
    if (-not $mcp) {
        Write-StepError "Cannot install without mcp executable."
        exit 1
    }
    $env:PYTHONPATH = "$Root\src"
    & $mcp install $ServerPath --name "MedRAG-Agent"
    Write-Host "[DONE] Restart Claude Desktop." -ForegroundColor Green
} else {
    Print-Config -PyExe $py
}
