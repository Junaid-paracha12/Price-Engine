"""Pricing optimiser copied verbatim from notebook section 12.2 (only change: `features` is a parameter)."""
import numpy as np
import pandas as pd

def recommend_prices_constrained(
    frame,
    model,
    unit_cost,
    elasticity_by_category,
    target_margin=0.30,
    max_competitor_premium=0.10,
    price_multipliers=None,
    elasticity_scale=1.0,
    max_change=0.15,
    min_profit_gain=0.01,
    features=None,
):
    """Return price scenarios subject to explicit business constraints.

    Required frame columns:
      Price, Competitor_Pricing, Discount, Inventory_Level, Category, plus FEATURES.
    unit_cost:
      scalar or Series aligned to frame rows; must be actual per-unit cost.
    elasticity_scale:
      multiplies every elasticity (use for sensitivity analysis).
    max_change:
      largest allowed relative price move per update (0.15 = +/-15 %).
    min_profit_gain:
      the price only moves if scenario profit beats the current price by more than this relative amount.
    elasticity_by_category:
      category -> price elasticity. Treat values as assumptions unless validated experimentally.
    """
    required = {"Price", "Competitor_Pricing", "Discount", "Inventory_Level", "Category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required pricing columns: {missing}")
    if frame.empty:
        return pd.DataFrame()

    current_price = pd.to_numeric(frame["Price"], errors="coerce").to_numpy(dtype=float)
    competitor = pd.to_numeric(frame["Competitor_Pricing"], errors="coerce").to_numpy(dtype=float)
    discount = pd.to_numeric(frame["Discount"], errors="coerce").fillna(0).to_numpy(dtype=float) / 100.0
    inventory = pd.to_numeric(frame["Inventory_Level"], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)

    if np.any(~np.isfinite(current_price)) or np.any(current_price <= 0):
        raise ValueError("Price must contain finite positive values.")
    if np.any(~np.isfinite(competitor)) or np.any(competitor <= 0):
        raise ValueError("Competitor_Pricing must contain finite positive values.")
    if np.any((discount < 0) | (discount >= 1)):
        raise ValueError("Discount must be between 0% and less than 100%.")
    if not 0 <= target_margin < 1:
        raise ValueError("target_margin must be in [0, 1).")
    if max_competitor_premium < 0:
        raise ValueError("max_competitor_premium cannot be negative.")
    if max_change < 0 or min_profit_gain < 0:
        raise ValueError("max_change and min_profit_gain cannot be negative.")

    if np.isscalar(unit_cost):
        costs = np.full(len(frame), float(unit_cost), dtype=float)
    else:
        costs = pd.Series(unit_cost, index=frame.index).to_numpy(dtype=float)
    if len(costs) != len(frame) or np.any(~np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("unit_cost must be finite, non-negative, and aligned with frame rows.")

    multipliers = np.asarray(
        price_multipliers if price_multipliers is not None
        else np.round(np.arange(0.85, 1.1501, 0.025), 3),
        dtype=float
    )
    if multipliers.ndim != 1 or len(multipliers) == 0 or np.any(multipliers <= 0):
        raise ValueError("price_multipliers must be a non-empty 1D array of positive values.")
    multipliers = np.unique(np.append(multipliers[np.abs(multipliers - 1.0) <= max_change + 1e-9], 1.0))   # enforce the move limit, always allow "keep"
    keep_idx = int(np.argmin(np.abs(multipliers - 1.0)))

    base_demand = np.maximum(0.0, np.asarray(model.predict(frame[list(features) if features is not None else FEATURES]), dtype=float))
    elasticity = frame["Category"].map(elasticity_by_category)
    if elasticity.isna().any():
        unknown = sorted(frame.loc[elasticity.isna(), "Category"].astype(str).unique())
        raise ValueError(f"Missing elasticity assumption for categories: {unknown}")
    elasticity = elasticity.to_numpy(dtype=float) * elasticity_scale

    candidate_prices = current_price[:, None] * multipliers[None, :]
    net_prices = candidate_prices * (1.0 - discount[:, None])
    expected_demand = base_demand[:, None] * np.power(multipliers[None, :], elasticity[:, None])
    expected_demand = np.maximum(0.0, expected_demand)
    fulfilled_units = np.minimum(expected_demand, inventory[:, None])
    profit = (net_prices - costs[:, None]) * fulfilled_units

    margin = np.divide(
        net_prices - costs[:, None], net_prices,
        out=np.full_like(net_prices, -np.inf), where=net_prices > 0
    )
    feasible = (
        (margin >= target_margin)
        & (candidate_prices <= competitor[:, None] * (1 + max_competitor_premium))
    )

    chosen_idx = np.zeros(len(frame), dtype=int)
    statuses = []
    for i in range(len(frame)):
        valid = np.flatnonzero(feasible[i])
        if len(valid):
            best = valid[np.argmax(profit[i, valid])]
            # hysteresis: stay at the current price when it is feasible and the gain is too small to be worth a price change
            if feasible[i, keep_idx] and best != keep_idx and \
               profit[i, best] - profit[i, keep_idx] <= min_profit_gain * abs(profit[i, keep_idx]):
                chosen_idx[i] = keep_idx
                statuses.append("KEEP_CURRENT_SMALL_GAIN")
            else:
                chosen_idx[i] = best
                statuses.append("OK")
        else:
            # Keep current price as a fallback, but clearly mark it as not constraint-feasible.
            chosen_idx[i] = keep_idx
            statuses.append("NO_FEASIBLE_PRICE")

    row_idx = np.arange(len(frame))
    chosen_price = candidate_prices[row_idx, chosen_idx]
    chosen_net = net_prices[row_idx, chosen_idx]
    chosen_demand = expected_demand[row_idx, chosen_idx]
    chosen_units = fulfilled_units[row_idx, chosen_idx]
    chosen_profit = profit[row_idx, chosen_idx]
    chosen_margin = margin[row_idx, chosen_idx]

    # Baseline is evaluated at current price using the same cost, discount, inventory logic.
    current_net = current_price * (1 - discount)
    current_units = np.minimum(base_demand, inventory)
    current_profit = (current_net - costs) * current_units
    current_margin = np.divide(current_net - costs, current_net,
                               out=np.full_like(current_net, -np.inf), where=current_net > 0)

    result = pd.DataFrame({
        "Category": frame["Category"].astype(str).to_numpy(),
        "Current_Price": current_price,
        "Competitor_Price": competitor,
        "Unit_Cost": costs,
        "Recommended_Price": chosen_price,
        "Price_Change_pct": 100 * (chosen_price / current_price - 1),
        "Baseline_Demand_Forecast": base_demand,
        "Expected_Demand_Scenario": chosen_demand,
        "Expected_Fulfilled_Units": chosen_units,
        "Current_Net_Margin": current_margin,
        "Recommended_Net_Margin": chosen_margin,
        "Estimated_Current_Profit": current_profit,
        "Scenario_Profit": chosen_profit,
        "Decision_Status": statuses,
        "Elasticity_Assumption": elasticity,
    }, index=frame.index)
    return result
