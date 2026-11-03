# Adaptive Pricing Engine

Real-time price recommendations from **competitor signals, demand forecasts and margin targets**.
Backend: FastAPI. Frontend: one HTML file served by the backend. Model + optimiser: from your notebook (v6).

## Run
```bash
pip install -r requirements-dev.txt
./run.sh                       # http://localhost:8000   ·  API docs: http://localhost:8000/docs
pytest -q tests                # 8 API tests
```

## Deploy to Vercel
Import the repository with its root directory set to the repository root. Vercel detects `app.py` as the FastAPI
entry point, installs `requirements.txt`, and includes the model, data, backend, and frontend from `vercel.json`.
No custom build command or always-running server is needed. Push commits to the connected Git branch to deploy.

The Python version and scikit-learn version are pinned to match the bundled model; keep them in sync if you retrain it.

## ⚠️ The bundled model is a DEMO
`models/pricing_pipeline.joblib` is trained on **synthetic data with the same schema** (the UI shows a "DEMO MODEL"
badge). To use your real data:
```bash
cp /path/to/sales_data.csv data/sales_data.csv
python backend/train.py --data data/sales_data.csv          # imputer=iterative as in the notebook; refits & overwrites the model
./run.sh
```
The trainer uses scikit-learn's `HistGradientBoostingRegressor` to avoid LightGBM's separate system `libgomp.so.1`
runtime dependency. Keep the CSV in `data/`; the server prefers `data/sales_data.csv` over the demo file.

## How it maps to the notebook
| Notebook | Here |
|---|---|
| §12.4 pipeline (`PricingCleaner → PricingFeatureBuilder → HistoryAwareRegressor`) | `backend/pricing_pipeline_components.py` (verbatim) |
| §9 demand model, 6 features | `backend/train.py` |
| §12.2 `recommend_prices_constrained` | `backend/notebook_engine.py` (verbatim; `features` made a parameter) |
| §12.4d `pipeline_recommend` | `backend/engine.py` + `main.py` |

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/meta` | stores, products, defaults, model info |
| GET | `/api/series?store_id&product_id` | latest signals for a product |
| POST | `/api/recommend` | one product: recommendation + profit curve + 30-day history |
| POST | `/api/batch` | whole catalog (filter by store/category), summary + rows |

`/api/recommend` example:
```json
{"store_id":"S001","product_id":"P0004","cost_share":0.6,"competitor_price":190,"target_margin":0.3,"max_competitor_premium":0.1}
```
Pass `unit_cost` (real cost) instead of `cost_share` for real decisions; `/api/batch` also accepts `unit_costs: {"P0004": 123.4, ...}`.

## Limits (carried over from the notebook — please read)
* **No cost column in the data.** "Illustrative" cost = share of current price; it is only for demos.
* **Elasticities are assumed placeholders** (`engine.DEFAULT_ELASTICITY`), not estimated. In this dataset price moves *with* demand,
  so the demand model's response to price is not causal. With mild elasticities the optimiser tends to push prices to the +15 % cap;
  validate with price experiments before trusting the profit numbers. Use the *Elasticity scale* input to see sensitivity.
* Needs ≥ 29 days of history per store-product; the "today" row is the day after the last date in the history file.
