#!/usr/bin/env bash
# Safe launcher for POSIX shells: runs the app as a module (preserves package relative imports)
# Usage: ./run.sh

# Ensure the project's `src` directory is in PYTHONPATH so imports like `from config...` resolve
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Prefer uvicorn module to point at the package module: src.main:app
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
