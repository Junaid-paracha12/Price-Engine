import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def c():
    with TestClient(app) as client:
        yield client


BODY = dict(store_id="S001", product_id="P0004", cost_share=0.6)


def test_health_and_meta(c):
    assert c.get("/api/health").json()["status"] == "ok"
    m = c.get("/api/meta").json()
    assert m["stores"] and m["products"] and 0 < m["model"]["holdout"]["R2"] <= 1


def test_recommend_respects_rules(c):
    r = c.post("/api/recommend", json=BODY).json()["recommendation"]
    assert abs(r["Price_Change_pct"]) <= 15 + 1e-6                       # max_change
    if r["Decision_Status"] == "OK":
        assert r["Recommended_Net_Margin"] >= 0.30 - 1e-9                # margin floor
        assert r["Recommended_Price"] <= r["Competitor_Price"] * 1.10 + 1e-6   # competitor ceiling


def test_recommended_price_is_curve_optimum(c):
    d = c.post("/api/recommend", json={**BODY, "min_profit_gain": 0}).json()
    feas = [p for p in d["curve"] if p["feasible"]]
    if feas:
        best = max(feas, key=lambda p: p["profit"])
        assert d["recommendation"]["Recommended_Price"] == pytest.approx(best["price"], rel=1e-9)


def test_competitor_signal_changes_decision(c):
    low = c.post("/api/recommend", json={**BODY, "competitor_price": 100}).json()["recommendation"]
    assert low["Decision_Status"] == "NO_FEASIBLE_PRICE" or low["Recommended_Price"] <= 110 + 1e-6


def test_tighter_margin_can_make_infeasible(c):
    r = c.post("/api/recommend", json={**BODY, "target_margin": 0.9}).json()["recommendation"]
    assert r["Decision_Status"] == "NO_FEASIBLE_PRICE"


def test_validation(c):
    assert c.post("/api/recommend", json={"store_id": "S001", "product_id": "P0004"}).status_code == 422   # no cost
    assert c.post("/api/recommend", json={**BODY, "product_id": "NOPE"}).status_code == 404
    assert c.post("/api/recommend", json={**BODY, "elasticity": {"Toys": 1.0}}).status_code == 422


def test_batch(c):
    d = c.post("/api/batch", json={"cost_share": 0.6, "category": "Toys"}).json()
    assert d["summary"]["series_priced"] == len(d["rows"]) > 0
    assert all(r["Category"] == "Toys" for r in d["rows"])
    assert c.post("/api/batch", json={"unit_costs": {"P0003": 30}}).status_code == 422  # costs missing for other products


def test_load_state_rebuilds_on_pickle_mismatch(monkeypatch, tmp_path):
    import main

    model_path = tmp_path / "pricing_pipeline.joblib"
    model_path.write_text("not-a-real-model")
    data_path = tmp_path / "sales_data.csv"
    data_path.write_text(
        "Date,Store_ID,Product_ID,Category,Price,Competitor_Pricing,Discount,Inventory_Level,Promotion,Epidemic,Weather_Condition,Demand\n"
        "2024-01-01,S001,P0001,Toys,30,32,0,20,0,0,Sunny,120\n"
        "2024-01-02,S001,P0001,Toys,31,33,0,20,0,0,Sunny,118\n"
        "2024-01-03,S001,P0001,Toys,32,34,0,20,0,0,Sunny,115\n"
    )

    calls = {"rebuilt": 0}
    bundle = {"model_name": "LightGBM", "pipeline": object(), "features": ["Price"],
              "holdout_metrics": {"R2": 0.8}, "trained_through": "2024-01-03",
              "source": "demo"}

    def fake_load(_):
        if calls["rebuilt"] == 0:
            raise AttributeError("sklearn.compose._column_transformer has no attribute _RemainderColsList")
        return bundle

    def fake_rebuild(path=None):
        calls["rebuilt"] += 1
        return True

    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    monkeypatch.setattr(main, "DATA_CANDIDATES", [data_path])
    monkeypatch.setattr(main.joblib, "load", fake_load)
    monkeypatch.setattr(main, "_rebuild_model_bundle", fake_rebuild)
    main.STATE.clear()

    main.load_state()

    assert calls["rebuilt"] == 1
    assert main.STATE["bundle"] == bundle
