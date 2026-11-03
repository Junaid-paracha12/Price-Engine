"""Thin service layer around the notebook's pricing logic (notebook_engine.py)."""
import numpy as np
import pandas as pd

import pricing_pipeline_components as ppc
from notebook_engine import recommend_prices_constrained

# Placeholder elasticities from notebook section 12.3 - ASSUMPTIONS, not estimates from data.
DEFAULT_ELASTICITY = {"Groceries": -0.8, "Clothing": -1.4, "Toys": -1.5, "Electronics": -1.8, "Furniture": -1.2}
MULTIPLIERS = np.round(np.arange(0.85, 1.1501, 0.025), 3)
HISTORY_DAYS = 40            # pipeline needs >= 29 days of history per series


def build_today(history: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    """Raw frame = recent history + one 'today' row per series (last date + 1, Demand blank)."""
    last = history["Date"].max()
    recent = history[history["Date"] > last - pd.Timedelta(days=HISTORY_DAYS)].copy()
    today = recent[recent["Date"] == last].copy()
    today["Date"] = last + pd.Timedelta(days=1)
    today["Demand"] = np.nan
    for c in ("Units_Sold", "Units_Ordered"):
        if c in today:
            today[c] = np.nan
    for k, v in (overrides or {}).items():
        if v is not None:
            today[k] = v
    raw = pd.concat([recent, today], ignore_index=True)
    return raw.reset_index(drop=True)


def feature_rows(pipe, raw: pd.DataFrame) -> pd.DataFrame:
    feats = pipe[:-1].transform(raw)
    today = feats[feats["Date"] == raw["Date"].max()]
    return today[today[ppc.HIST_REQUIRED].notna().all(axis=1)]


def recommend(pipe, features, frame, unit_cost, elasticity, **kw):
    return recommend_prices_constrained(frame, pipe[-1].estimator_, unit_cost=unit_cost,
                                        elasticity_by_category=elasticity, features=features, **kw)


def price_curve(pipe, features, row: pd.DataFrame, unit_cost, elasticity, target_margin,
                max_competitor_premium, max_change):
    """Profit/demand at every candidate price for ONE row (same formulas as the optimiser)."""
    base = float(max(0.0, pipe[-1].estimator_.predict(row[list(features)])[0]))
    p0 = float(row["Price"].iloc[0]); comp = float(row["Competitor_Pricing"].iloc[0])
    disc = float(row["Discount"].iloc[0]) / 100.0
    inv = max(0.0, float(row["Inventory_Level"].iloc[0]))
    eps = float(elasticity[str(row["Category"].iloc[0])])
    mult = np.unique(np.append(MULTIPLIERS[np.abs(MULTIPLIERS - 1) <= max_change + 1e-9], 1.0))
    out = []
    for m in mult:
        price = p0 * m; net = price * (1 - disc)
        dem = max(0.0, base * m ** eps); units = min(dem, inv)
        margin = (net - unit_cost) / net if net > 0 else float("-inf")
        out.append(dict(multiplier=float(m), price=price, net_price=net, demand=dem, units=units,
                        profit=(net - unit_cost) * units, margin=margin,
                        feasible=bool(margin >= target_margin and price <= comp * (1 + max_competitor_premium))))
    return base, out
