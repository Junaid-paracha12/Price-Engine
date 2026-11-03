"""
pricing_pipeline_components.py
Reusable building blocks of the Adaptive Pricing Engine (v3).

    raw sales table  ->  PricingCleaner  ->  PricingFeatureBuilder  ->  HistoryAwareRegressor
                         (schema, types,     (past-only lag / rolling    (drops rows without history
                          missing, outliers)  / price-vs-own-history      when fitting, returns NaN for
                                              features)                   them when predicting)

Every step keeps ALL input rows in the SAME order, so `pipe.fit(raw, raw["Demand"])` and
`pipe.predict(raw)` line up row by row. Only information available at decision time is used:
every history feature is built with shift(1), so today's Demand is never needed (it may be blank).
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline

TARGET = "Demand"
KEY = ["Date", "Store_ID", "Product_ID"]
SERIES = ["Store_ID", "Product_ID"]
PLACEHOLDERS = {"", "?", "na", "n/a", "nan", "null", "none", "-", "--", "unknown"}
POSITIVE_COLS = ["Price", "Competitor_Pricing"]
NONNEG_COLS = ["Inventory_Level", "Units_Sold", "Units_Ordered", "Discount", "Demand"]
NUM_IMPUTE = ["Inventory_Level", "Price", "Competitor_Pricing", "Discount"]      # target is never imputed
COUNTS = ["Inventory_Level", "Units_Sold", "Units_Ordered", "Demand"]
BINARY = ["Promotion", "Epidemic"]
CAP_COLS = ["Inventory_Level", "Price", "Competitor_Pricing"]                     # Demand is never capped
CAT_COLS = ["Store_ID", "Product_ID", "Category", "Weather_Condition"]
REQUIRED_INPUT = ["Date", "Store_ID", "Product_ID", "Category", "Weather_Condition", "Inventory_Level",
                  "Price", "Competitor_Pricing", "Discount", "Promotion", "Epidemic"]
# Features that need history; a row is only forecastable when all of them exist.
HIST_REQUIRED = ["Demand_Lag7", "Demand_Roll7", "Price_Change_1d", "Comp_Change_1d",
                 "Demand_Lag14", "Demand_Roll14", "Demand_Roll28", "Demand_Std7", "Price_Roll28"]

def _is_text(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))

class PricingCleaner(BaseEstimator, TransformerMixin):
    """Schema -> dtypes -> disguised / impossible values to NaN -> imputation -> per-Category outlier caps.
    Everything data-dependent (imputation statistics, outlier fences) is learned in fit() from the rows it is given."""

    def __init__(self, imputer="group_median", random_state=42):
        self.imputer = imputer
        self.random_state = random_state

    # ---- stateless part
    def _standardise(self, X):
        d = pd.DataFrame(X).copy()
        d.columns = d.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
        missing = [c for c in REQUIRED_INPUT if c not in d.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for c in d.columns:
            if c != "Date" and _is_text(d[c]):
                txt = d[c].astype(str).str.strip().str.lower()
                d[c] = d[c].where(~(txt.isin(PLACEHOLDERS) | d[c].isna()), np.nan)
        for c in NUM_IMPUTE + BINARY + [TARGET, "Units_Sold", "Units_Ordered"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        for c in POSITIVE_COLS:
            d.loc[d[c] <= 0, c] = np.nan
        for c in NONNEG_COLS:
            if c in d.columns:
                d.loc[d[c] < 0, c] = np.nan
        if d["Date"].isna().any() or d[["Store_ID", "Product_ID"]].isna().any().any():
            raise ValueError("Date, Store_ID and Product_ID identify the row and cannot be missing.")
        for c in ["Store_ID", "Product_ID"]:
            d[c] = d[c].astype(str)
        return d

    def _context(self, d):
        ctx = pd.get_dummies(d[["Category", "Weather_Condition"]].astype(object).fillna("Unknown").astype(str), dtype=float)
        return pd.concat([d[BINARY].fillna(0).astype(float), ctx], axis=1).reindex(columns=self.ctx_cols_, fill_value=0.0)

    # ---- learn
    def fit(self, X, y=None):
        d = self._standardise(X)
        self.num_median_ = d[NUM_IMPUTE].median()
        self.num_mean_ = d[NUM_IMPUTE].mean()
        self.group_median_ = d.groupby("Category")[NUM_IMPUTE].median()
        self.binary_mode_ = {c: (d[c].mode().iloc[0] if d[c].notna().any() else 0) for c in BINARY}
        self.cat_mode_ = {c: (d[c].mode().iloc[0] if d[c].notna().any() else "Unknown") for c in ["Category", "Weather_Condition"]}
        self.product_category_ = d.dropna(subset=["Category"]).groupby("Product_ID")["Category"].agg(lambda s: s.mode().iloc[0]).to_dict()
        # model-based imputers (knn / iterative) need a context matrix
        d = self._fill_categoricals(d)
        self.ctx_cols_ = list(pd.concat([d[BINARY].fillna(0).astype(float),
                                         pd.get_dummies(d[["Category", "Weather_Condition"]].astype(str), dtype=float)], axis=1).columns)
        self.model_imputer_ = None
        if self.imputer in ("knn", "iterative"):
            from sklearn.preprocessing import StandardScaler
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import KNNImputer, IterativeImputer
            from sklearn.linear_model import BayesianRidge
            Z = pd.concat([d[NUM_IMPUTE], self._context(d)], axis=1)
            sc = StandardScaler().fit(Z)
            imp = (KNNImputer(n_neighbors=5, weights="distance") if self.imputer == "knn"
                   else IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=self.random_state))
            imp.fit(sc.transform(Z))
            self.model_imputer_ = (sc, imp)
        d = self._impute(d)
        # outlier fences per Category, learned on the rows given to fit (the training period)
        skew = d[CAP_COLS].skew()
        self.k_by_col_ = {c: (3.0 if abs(skew[c]) > 1 else 1.5) for c in CAP_COLS}
        self.fences_ = {}
        for c in CAP_COLS:
            q = d.groupby("Category")[c].quantile([0.25, 0.75]).unstack()
            iqr = q[0.75] - q[0.25]
            lo = (q[0.25] - self.k_by_col_[c] * iqr).clip(lower=0)
            if c in POSITIVE_COLS:
                lo = lo.clip(lower=0.01)
            self.fences_[c] = (lo.to_dict(), (q[0.75] + self.k_by_col_[c] * iqr).to_dict())
        return self

    # ---- apply
    def _fill_categoricals(self, d):
        d = d.copy()
        cat = d["Category"].where(d["Category"].notna(), d["Product_ID"].map(self.product_category_))
        d["Category"] = cat.fillna(self.cat_mode_["Category"]).astype(str)
        d["Weather_Condition"] = d["Weather_Condition"].fillna(self.cat_mode_["Weather_Condition"]).astype(str)
        return d

    def _impute(self, d):
        d = d.copy()
        for c in BINARY:
            d[c] = d[c].fillna(self.binary_mode_[c])
        num = d[NUM_IMPUTE]
        if num.isna().any().any():
            if self.imputer == "mean":
                num = num.fillna(self.num_mean_)
            elif self.imputer == "median":
                num = num.fillna(self.num_median_)
            elif self.imputer in ("knn", "iterative") and self.model_imputer_ is not None:
                sc, imp = self.model_imputer_
                Z = pd.concat([num, self._context(d)], axis=1)
                num = pd.DataFrame(sc.inverse_transform(imp.transform(sc.transform(Z))), columns=Z.columns, index=d.index)[NUM_IMPUTE]
            else:                                                        # group_median (also the safe fallback)
                med = self.group_median_.reindex(d["Category"]).set_index(d.index)
                num = num.fillna(med).fillna(self.num_median_)
            for c in NUM_IMPUTE:
                d[c] = num[c]
        for c in COUNTS:
            if c in d.columns and c != TARGET:
                d[c] = d[c].round()
        d["Discount"] = d["Discount"].round()
        return d

    def transform(self, X):
        d = self._fill_categoricals(self._standardise(X))
        d = self._impute(d)
        for c in CAP_COLS:                                              # winsorise, never delete rows
            lo_map, hi_map = self.fences_[c]
            lo, hi = d["Category"].map(lo_map).astype(float), d["Category"].map(hi_map).astype(float)
            d[c] = d[c].clip(lower=lo, upper=hi)                         # unseen category -> NaN fence -> unchanged
            if c in COUNTS:
                d[c] = d[c].round()
        return d

class PricingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds the engineered features. Row-preserving and order-preserving. Rows without enough history keep NaN."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):          # stateless step: always "fitted" (lets pipe[:-1].transform(...) work)
        return True

    def transform(self, X):
        d = pd.DataFrame(X).copy()
        if not d.index.is_unique:
            raise ValueError("Input index must be unique.")
        original_index = d.index
        s = d.sort_values(SERIES + ["Date"], kind="mergesort")
        g = s.groupby(SERIES, observed=True, sort=False)
        gaps = g["Date"].diff().dropna().dt.days
        if len(gaps) and not (gaps == 1).all():
            warnings.warn(f"{int((gaps != 1).sum())} non-consecutive day gaps: lag features are not exact calendar lags there.")

        s["Price_Ratio"] = s.Price / s.Competitor_Pricing
        s["Price_Gap"] = s.Price - s.Competitor_Pricing
        s["Effective_Price"] = s.Price * (1 - s.Discount / 100)
        s["Eff_Ratio"] = s.Effective_Price / s.Competitor_Pricing

        has_target = TARGET in s.columns
        dem = g[TARGET] if has_target else None
        if not has_target:
            raise ValueError("History needs the Demand column (it may be blank only for the rows to be forecast).")
        s["Demand_Lag1"] = dem.shift(1)
        s["Demand_Lag7"] = dem.shift(7)
        s["Demand_Roll7"] = dem.transform(lambda x: x.shift(1).rolling(7).mean())
        s["Price_Lag1"] = g["Price"].shift(1)
        s["Price_Change_1d"] = (s.Price - s.Price_Lag1) / s.Price_Lag1
        s["Comp_Change_1d"] = g["Competitor_Pricing"].transform(lambda x: x.pct_change())
        # v3 additions: longer demand memory ...
        s["Demand_Lag14"] = dem.shift(14)
        s["Demand_Roll14"] = dem.transform(lambda x: x.shift(1).rolling(14).mean())
        s["Demand_Roll28"] = dem.transform(lambda x: x.shift(1).rolling(28).mean())
        s["Demand_Std7"] = dem.transform(lambda x: x.shift(1).rolling(7).std())
        s["Demand_ExpMean"] = dem.transform(lambda x: x.shift(1).expanding().mean())
        s["Demand_EWM"] = dem.transform(lambda x: x.shift(1).ewm(alpha=0.3).mean())
        # ... and today's price relative to this series' OWN past price level
        s["Price_ExpMean"] = g["Price"].transform(lambda x: x.shift(1).expanding().mean())
        s["Price_Roll7"] = g["Price"].transform(lambda x: x.shift(1).rolling(7).mean())
        s["Price_Roll28"] = g["Price"].transform(lambda x: x.shift(1).rolling(28).mean())
        s["Price_Idx_Exp"] = s.Price / s.Price_ExpMean
        s["Price_Idx_7"] = s.Price / s.Price_Roll7
        s["Price_Idx_28"] = s.Price / s.Price_Roll28

        s["Month"] = s.Date.dt.month
        s["DayOfWeek"] = s.Date.dt.dayofweek
        s["Is_Weekend"] = (s.DayOfWeek >= 5).astype(int)
        for c in CAT_COLS:
            s[c] = s[c].astype(str)
        return s.loc[original_index]

class HistoryAwareRegressor(BaseEstimator, RegressorMixin):
    """Wraps the tuned model. fit(): uses only rows whose history features exist and whose target is known.
    predict(): NaN for rows that lack history (cold start) instead of silently guessing."""

    def __init__(self, estimator, required=None):
        self.estimator = estimator
        self.required = required

    def _ok(self, X):
        req = list(self.required) if self.required is not None else HIST_REQUIRED
        return X[req].notna().all(axis=1).to_numpy()

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        ok = self._ok(X) & np.isfinite(y)
        self.n_fit_rows_, self.n_dropped_rows_ = int(ok.sum()), int((~ok).sum())
        self.estimator_ = clone(self.estimator).fit(X[ok], y[ok])
        return self

    def predict(self, X):
        out = np.full(len(X), np.nan)
        ok = self._ok(X)
        if ok.any():
            out[ok] = self.estimator_.predict(X[ok])
        return out

def make_pricing_pipeline(final_estimator, imputer="group_median", random_state=42):
    """raw table -> clean -> features -> model.  Fit with pipe.fit(raw_history, raw_history['Demand'])."""
    return Pipeline([("clean", PricingCleaner(imputer=imputer, random_state=random_state)),
                     ("features", PricingFeatureBuilder()),
                     ("model", HistoryAwareRegressor(final_estimator))])
