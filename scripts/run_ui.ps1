# Run the Streamlit demo UI
# Usage: .\scripts\run_ui.ps1

$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\lijingshan\.conda\envs\medrag\python.exe"
$streamlit = "C:\Users\lijingshan\.conda\envs\medrag\Scripts\streamlit.exe"

Set-Location "D:\Desktop\Agent\medrag-agent"

Write-Host "Starting MedRAG-Agent UI at http://localhost:8501" -ForegroundColor Cyan
& $streamlit run src/medrag/ui/app.py --server.port 8501 --server.headless false
