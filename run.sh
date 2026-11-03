#!/usr/bin/env bash
# Start the app:  ./run.sh   ->  http://localhost:8000   (API docs: /docs)
set -e
cd "$(dirname "$0")"
[ -f models/pricing_pipeline.joblib ] || python backend/train.py --demo --imputer group_median
uvicorn main:app --app-dir backend --host 0.0.0.0 --port "${PORT:-8000}"
