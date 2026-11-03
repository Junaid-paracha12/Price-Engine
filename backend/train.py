"""Train / retrain the pricing pipeline exactly as in the notebook's final configuration.

    python train.py --data ../data/sales_data.csv      # your real dataset
    python train.py --demo                              # synthetic data (demo only)

Final model from the notebook: LightGBM on 6 features (Price, Competitor_Pricing, Discount,
Category, Store_ID, Product_ID), raw target, wrapped in the end-to-end pricing pipeline.
"""
import argparse, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pricing_pipeline_components as ppc  # noqa: E402
from data_gen import make_demo_data  # noqa: E402

NUM_FEATS = ["Price", "Competitor_Pricing", "Discount"]
CAT_FEATS = ["Category", "Store_ID", "Product_ID"]
FEATURES = NUM_FEATS + CAT_FEATS
# refined hyper-parameters reported in notebook section 9
BEST_PARAMS = dict(n_estimators=800, learning_rate=0.0835, num_leaves=105, min_child_samples=7,
                   subsample=0.9453, colsample_bytree=0.6651, reg_lambda=0.0426, reg_alpha=0.0027)


def make_estimator(params=BEST_PARAMS, seed=42):
    prep = ColumnTransformer([("num", StandardScaler(), NUM_FEATS),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS)])
    reg = LGBMRegressor(subsample_freq=1, n_jobs=-1, random_state=seed, verbose=-1, **params)
    return TransformedTargetRegressor(regressor=Pipeline([("prep", prep), ("model", reg)]), transformer=None)


def scores(y, p, y_range):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    err = y - p
    return dict(MAE=float(np.abs(err).mean() / y_range), RMSE=float(np.sqrt((err ** 2).mean()) / y_range),
                MAPE=float((np.abs(err) / np.maximum(y, 1)).mean()), WAPE=float(np.abs(err).sum() / y.sum()),
                R2=float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                MAE_units=float(np.abs(err).mean()), RMSE_units=float(np.sqrt((err ** 2).mean())), n_rows=int(m.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--imputer", default="iterative", choices=["mean", "median", "group_median", "knn", "iterative"])
    ap.add_argument("--out", default=str(HERE.parent / "models" / "pricing_pipeline.joblib"))
    a = ap.parse_args()

    if a.demo or not a.data:
        raw, source = make_demo_data(), "demo"
        raw.to_csv(HERE.parent / "data" / "demo_sales_data.csv", index=False)
    else:
        raw, source = pd.read_csv(a.data), "real"
    raw.columns = raw.columns.str.strip().str.replace(r"\s+", "_", regex=True)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)

    # chronological hold-out: the last 20 % of dates
    dates = np.sort(raw["Date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * 0.8)])
    train = raw[raw["Date"] < cut]
    y_range = float(train["Demand"].max() - train["Demand"].min())
    pipe = ppc.make_pricing_pipeline(make_estimator(), imputer=a.imputer).fit(train, train["Demand"])
    pred = pipe.predict(raw)                              # history comes from the full table; model only saw < cut
    mask = (raw["Date"] >= cut).to_numpy()
    hold = scores(raw.loc[mask, "Demand"], pred[mask], y_range)
    print(f"hold-out ({cut.date()} ->): R2={hold['R2']:.4f}  MAE={hold['MAE_units']:.2f} units  WAPE={hold['WAPE']:.3%}")

    # production fit on ALL history
    pipe = ppc.make_pricing_pipeline(make_estimator(), imputer=a.imputer).fit(raw, raw["Demand"])
    Path(a.out).parent.mkdir(exist_ok=True)
    joblib.dump(dict(pipeline=pipe, model_name="LightGBM", trained_through=str(raw["Date"].max().date()),
                     holdout_metrics=hold, y_range=y_range, features=FEATURES, source=source,
                     required_input_columns=ppc.REQUIRED_INPUT, imputer=a.imputer), a.out)
    print("saved", a.out, "| data source:", source)


if __name__ == "__main__":
    main()
