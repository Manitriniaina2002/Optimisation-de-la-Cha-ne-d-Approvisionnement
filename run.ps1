# Safe launcher for Windows PowerShell: runs the app as a module (preserves package relative imports)
# Usage (PowerShell):
#   ./run.ps1

# Ensure the project's `src` directory is in PYTHONPATH so imports like `from config...` resolve
# Set PYTHONPATH to the repository root so `src` is importable and relative imports inside packages work
$repoRoot = (Get-Location).Path
# Add only the `src` directory to PYTHONPATH so installed site-packages (e.g. kafka-python)
# are not shadowed by same-named top-level folders in the repository.
$srcPath = Join-Path $repoRoot 'src'
$Env:PYTHONPATH = $srcPath

# Prefer uvicorn module to point at the package module: src.main:app
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
	Write-Error "Virtualenv python not found at $venvPython. Activate your venv or adjust the path."
	exit 1
}

# Run uvicorn via the venv python so the reloader's worker uses the same interpreter
& $venvPython -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
