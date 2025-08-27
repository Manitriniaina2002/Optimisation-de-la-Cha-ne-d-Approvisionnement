# Activates the project's venv (PowerShell) and runs the run.ps1 launcher
# Usage: .\start.ps1

# Verify venv exists
$venv = Join-Path (Get-Location) '.venv\Scripts\Activate.ps1'
if (-Not (Test-Path $venv)) {
    Write-Host "Virtual environment activate script not found at $venv" -ForegroundColor Yellow
    Write-Host "Create a virtualenv first: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Activate
. $venv

# Run launcher (which sets PYTHONPATH to src and launches uvicorn)
.\run.ps1
