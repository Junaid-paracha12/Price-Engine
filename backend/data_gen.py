"""Synthetic retail dataset with the SAME schema as sales_data.csv in the notebook.
Used only when the real file is not available, so the app can run end-to-end as a demo."""
import numpy as np
import pandas as pd

CATS = {"Groceries": (30, 90), "Clothing": (25, 80), "Toys": (15, 60), "Electronics": (60, 220), "Furniture": (70, 200)}
WEATHER = ["Sunny", "Rainy", "Cloudy", "Snowy"]
REGION = {"S001": "North", "S002": "South", "S003": "East", "S004": "West", "S005": "North"}


def make_demo_data(days=730, n_stores=5, n_products=20, seed=42, end="2026-10-31"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days, freq="D")
    cats = list(CATS)
    prod_cat = {f"P{i+1:04d}": cats[i % 5] for i in range(n_products)}
    prod_base = {p: rng.uniform(*CATS[c]) for p, c in prod_cat.items()}
    prod_vol = {p: rng.uniform(70, 220) for p in prod_cat}
    rows = []
    t = np.arange(days)
    for s in range(n_stores):
        sid = f"S{s+1:03d}"
        store_f = rng.uniform(0.85, 1.2)
        weather = rng.choice(WEATHER, size=days, p=[.4, .25, .25, .1])
        for p, c in prod_cat.items():
            season = 1 + 0.25 * np.sin(2 * np.pi * (t + rng.uniform(0, 365)) / 365)
            weekly = 1 + 0.08 * np.isin(dates.dayofweek, [5, 6])
            promo = (rng.random(days) < 0.2).astype(int)
            epi = (rng.random(days) < 0.03).astype(int)
            noise = rng.normal(0, 0.05, days)
            idx = season * weekly * (1 + 0.25 * promo - 0.15 * epi) * (1 + noise)      # demand index
            demand = np.clip(prod_vol[p] * store_f * idx, 1, None).round()
            # as in the real data, our price drifts up with demand; competitors track our price
            price = prod_base[p] * (1 + 0.35 * (idx - 1) + rng.normal(0, 0.03, days))
            comp = price * (1 + rng.normal(0.0, 0.07, days))
            disc = rng.choice([0, 5, 10, 15, 20], size=days, p=[.5, .15, .15, .1, .1])
            inv = np.clip(demand * rng.uniform(1.2, 3.5, days), 20, None).round()
            sold = np.minimum(demand, inv).round()
            rows.append(pd.DataFrame({
                "Date": dates, "Store_ID": sid, "Product_ID": p, "Category": c, "Region": REGION[sid],
                "Inventory_Level": inv, "Units_Sold": sold, "Units_Ordered": (demand * 0.9).round(),
                "Demand": demand, "Price": price.round(2), "Competitor_Pricing": comp.round(2),
                "Discount": disc, "Weather_Condition": weather, "Promotion": promo, "Epidemic": epi,
                "Seasonality": np.where(season > 1.1, "Peak", "Normal")}))
    return pd.concat(rows, ignore_index=True)
