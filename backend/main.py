"""Adaptive Pricing Engine - FastAPI backend.   Run:  uvicorn main:app --app-dir backend --reload"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))      # the joblib pickles `pricing_pipeline_components` by name

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

import engine

MODEL_PATH = ROOT / "models" / "pricing_pipeline.joblib"
DATA_CANDIDATES = [ROOT / "data" / "sales_data.csv"]
STATE: dict = {}


def clean(o):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.date().isoformat()
    return o


def _rebuild_model_bundle(data_path: Path | None = None):
    """Rebuild the saved model when the pickle was trained with a different sklearn stack."""
    if data_path is None:
        data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        return False

    backend_dir = ROOT / "backend"
    sys.path.insert(0, str(backend_dir))
    try:
        import train  # noqa: PLC0415
    except ModuleNotFoundError:
        return False

    argv = sys.argv[:]
    try:
        sys.argv = [str(backend_dir / "train.py"), "--data", str(data_path)] if data_path.name == "sales_data.csv" \
            else [str(backend_dir / "train.py"), "--demo"]
        train.main()
        return True
    finally:
        sys.argv = argv


def load_state():
    """Load model + history once per process (idempotent; works on serverless cold starts too)."""
    if STATE:
        return
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH} not found. Run `python backend/train.py --demo` or train on your data.")

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:
        path = next((p for p in DATA_CANDIDATES if p.exists()), None)
        if not _rebuild_model_bundle(path):
            raise RuntimeError(
                "Saved model is incompatible with the installed scikit-learn stack and could not be rebuilt. "
                "Run `python backend/train.py --demo` or train on your data."
            ) from exc
        bundle = joblib.load(MODEL_PATH)

    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        raise RuntimeError("No history file: put sales_data.csv (or demo_sales_data.csv) in ./data")
    hist = pd.read_csv(path)
    hist.columns = hist.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist["Store_ID"], hist["Product_ID"] = hist["Store_ID"].astype(str), hist["Product_ID"].astype(str)
    STATE.update(bundle=bundle, pipe=bundle["pipeline"], features=bundle.get("features") or
                 ["Price", "Competitor_Pricing", "Discount", "Category", "Store_ID", "Product_ID"],
                 history=hist.sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True),
                 data_file=path.name)


def ready():
    load_state()


app = FastAPI(title="Adaptive Pricing Engine", version="1.0.0", dependencies=[Depends(ready)],
              description="Real-time price recommendations from competitor signals, demand forecasts and margin targets.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ schemas
class Rules(BaseModel):
    target_margin: float = Field(0.30, ge=0, lt=1, description="minimum net margin, 0.30 = 30 %")
    max_competitor_premium: float = Field(0.10, ge=0, le=2, description="max price above competitor, 0.10 = +10 %")
    max_change: float = Field(0.15, ge=0, le=0.5, description="max relative move per update")
    min_profit_gain: float = Field(0.01, ge=0, le=1, description="hysteresis: ignore gains smaller than this")
    elasticity_scale: float = Field(1.0, gt=0, le=5)
    elasticity: dict[str, float] | None = Field(None, description="override per-category elasticity (must be < 0)")

    @model_validator(mode="after")
    def _check(self):
        if self.elasticity and any(v >= 0 for v in self.elasticity.values()):
            raise ValueError("elasticity values must be negative")
        return self


class CostSpec(BaseModel):
    unit_cost: float | None = Field(None, gt=0, description="actual cost per unit")
    cost_share: float | None = Field(None, gt=0, lt=1, description="ILLUSTRATIVE cost as share of current price")


class RecommendRequest(Rules, CostSpec):
    store_id: str
    product_id: str
    current_price: float | None = Field(None, gt=0)
    competitor_price: float | None = Field(None, gt=0)
    discount: float | None = Field(None, ge=0, lt=100)
    inventory: float | None = Field(None, ge=0)
    promotion: int | None = Field(None, ge=0, le=1)
    epidemic: int | None = Field(None, ge=0, le=1)
    weather: str | None = None

    @model_validator(mode="after")
    def _cost(self):
        if self.unit_cost is None and self.cost_share is None:
            raise ValueError("provide unit_cost (actual) or cost_share (illustrative)")
        return self


class BatchRequest(Rules, CostSpec):
    store_id: str | None = None
    category: str | None = None
    unit_costs: dict[str, float] | None = Field(None, description="Product_ID -> actual unit cost")

    @model_validator(mode="after")
    def _cost(self):
        if self.unit_cost is None and self.cost_share is None and not self.unit_costs:
            raise ValueError("provide unit_costs, unit_cost or cost_share")
        return self


# ------------------------------------------------------------------ helpers
def _elasticity(rules: Rules):
    e = dict(engine.DEFAULT_ELASTICITY)
    e.update(rules.elasticity or {})
    return e


def _engine_kw(r: Rules):
    return dict(target_margin=r.target_margin, max_competitor_premium=r.max_competitor_premium,
                max_change=r.max_change, min_profit_gain=r.min_profit_gain, elasticity_scale=r.elasticity_scale)


def _series(store, product):
    h = STATE["history"]
    s = h[(h.Store_ID == store) & (h.Product_ID == product)]
    if s.empty:
        raise HTTPException(404, f"No history for store {store} / product {product}")
    return s


def _assumptions(spec: CostSpec, batch_costs=False):
    notes = ["Price elasticities are assumed placeholders, not estimated from data (the demand model is not causal in price)."]
    if spec.unit_cost is None and not batch_costs:
        notes.append(f"Unit cost is ILLUSTRATIVE ({spec.cost_share:.0%} of current price). Supply real unit costs for real decisions.")
    return notes


# ------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    return {"status": "ok", "model": STATE["bundle"]["model_name"], "data_file": STATE["data_file"]}


@app.get("/api/meta")
def meta():
    h, b = STATE["history"], STATE["bundle"]
    prods = h.drop_duplicates("Product_ID").set_index("Product_ID")["Category"].sort_index()
    return clean(dict(
        stores=sorted(h.Store_ID.unique()), products=[{"id": p, "category": c} for p, c in prods.items()],
        categories=sorted(h.Category.unique()), weather=sorted(h.Weather_Condition.dropna().unique()),
        latest_date=h.Date.max(), decision_date=h.Date.max() + pd.Timedelta(days=1),
        default_elasticity=engine.DEFAULT_ELASTICITY,
        defaults=Rules().model_dump(exclude={"elasticity"}),
        model=dict(name=b["model_name"], trained_through=b["trained_through"], source=b.get("source", "real"),
                   holdout=b["holdout_metrics"], features=STATE["features"], data_file=STATE["data_file"], rows=len(h))))


@app.get("/api/series")
def series(store_id: str, product_id: str):
    """Latest observed signals for one store/product (used to pre-fill the form)."""
    last = _series(store_id, product_id).iloc[-1]
    return clean(dict(date=last.Date, category=last.Category, price=last.Price, competitor_price=last.Competitor_Pricing,
                      discount=last.Discount, inventory=last.Inventory_Level, promotion=int(last.Promotion),
                      epidemic=int(last.Epidemic), weather=last.Weather_Condition))


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    s = _series(req.store_id, req.product_id)
    raw = engine.build_today(s, dict(Price=req.current_price, Competitor_Pricing=req.competitor_price,
                                     Discount=req.discount, Inventory_Level=req.inventory, Promotion=req.promotion,
                                     Epidemic=req.epidemic, Weather_Condition=req.weather))
    try:
        row = engine.feature_rows(STATE["pipe"], raw)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if row.empty:
        raise HTTPException(422, "Not enough history (>= 29 days) for this store/product.")
    price = float(row["Price"].iloc[0])
    cost = req.unit_cost if req.unit_cost is not None else req.cost_share * price
    el, kw = _elasticity(req), _engine_kw(req)
    try:
        res = engine.recommend(STATE["pipe"], STATE["features"], row, cost, el, **kw)
        base, curve = engine.price_curve(STATE["pipe"], STATE["features"], row, cost, el, req.target_margin,
                                         req.max_competitor_premium, req.max_change)
    except ValueError as e:
        raise HTTPException(422, str(e))
    r = res.iloc[0].to_dict()
    r["Profit_Change_pct"] = (100 * (r["Scenario_Profit"] / r["Estimated_Current_Profit"] - 1)
                              if r["Estimated_Current_Profit"] > 0 else None)
    r["Price_vs_Competitor_pct"] = 100 * (r["Recommended_Price"] / r["Competitor_Price"] - 1)
    r["Inventory"] = float(row["Inventory_Level"].iloc[0]); r["Discount"] = float(row["Discount"].iloc[0])
    hist = s.tail(30)
    return clean(dict(
        store_id=req.store_id, product_id=req.product_id, decision_date=raw["Date"].max(), recommendation=r,
        curve=curve, history=dict(date=hist.Date.dt.date.astype(str).tolist(), demand=hist.Demand.tolist(),
                                  price=hist.Price.tolist(), competitor=hist.Competitor_Pricing.tolist()),
        assumptions=_assumptions(req), cost_used=cost))


@app.post("/api/batch")
def batch(req: BatchRequest):
    h = STATE["history"]
    if req.store_id:
        h = h[h.Store_ID == req.store_id]
    if req.category:
        h = h[h.Category == req.category]
    if h.empty:
        raise HTTPException(404, "No rows match that filter")
    raw = engine.build_today(h)
    try:
        rows = engine.feature_rows(STATE["pipe"], raw)
    except ValueError as e:
        raise HTTPException(422, str(e))
    n_series = h.groupby(["Store_ID", "Product_ID"]).ngroups
    if rows.empty:
        raise HTTPException(422, "No series has enough history.")
    if req.unit_costs:
        cost = rows["Product_ID"].map(req.unit_costs)
        if cost.isna().any():
            missing = sorted(rows.loc[cost.isna(), "Product_ID"].unique())[:10]
            raise HTTPException(422, f"unit_costs missing for products: {missing}")
    elif req.unit_cost is not None:
        cost = req.unit_cost
    else:
        cost = req.cost_share * rows["Price"]
    try:
        res = engine.recommend(STATE["pipe"], STATE["features"], rows, cost, _elasticity(req), **_engine_kw(req))
    except ValueError as e:
        raise HTTPException(422, str(e))
    res.insert(0, "Store_ID", rows["Store_ID"].to_numpy()); res.insert(1, "Product_ID", rows["Product_ID"].to_numpy())
    ok = res[res.Decision_Status != "NO_FEASIBLE_PRICE"]
    cur, new = ok.Estimated_Current_Profit.sum(), ok.Scenario_Profit.sum()
    summary = dict(series_total=n_series, series_priced=len(res), series_skipped=n_series - len(res),
                   status_counts=res.Decision_Status.value_counts().to_dict(),
                   changed=int((res.Price_Change_pct.abs() > 1e-9).sum()),
                   avg_change_pct=float(res.Price_Change_pct.mean()),
                   profit_current=float(cur), profit_scenario=float(new),
                   profit_change_pct=float(100 * (new / cur - 1)) if cur > 0 else None)
    return clean(dict(decision_date=raw["Date"].max(), summary=summary,
                      rows=res.sort_values("Price_Change_pct", key=abs, ascending=False).round(4).to_dict("records"),
                      assumptions=_assumptions(req, batch_costs=bool(req.unit_costs))))


FRONTEND = ROOT / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")
