#!/usr/bin/env bash
set -e
python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "→ http://localhost:8000   (set HF_TOKEN in .env first)"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
