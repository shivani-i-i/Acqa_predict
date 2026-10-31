"""Integration addition: predict() accepts explain=False to skip SHAP (default behaviour unchanged)."""
from aquapredict.forecasting.predict import predict


def test_explain_false_skips_explanation_but_still_forecasts(good_request):
    base = predict(dict(good_request))
    off = predict({**good_request, "explain": False})
    assert off["forecast_days"] == base["forecast_days"] and off["interval_days"] == base["interval_days"]
    assert off["drivers"] == [] and off["explanation"]["available"] is False
    assert "explain=false" in off["explanation"]["reason"]
    assert set(off) == set(base)


def test_explain_defaults_to_on(good_request):
    assert "explanation" in predict(dict(good_request))
