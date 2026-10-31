"""Check that the REAL shap library works with the trained model (run after `pip install shap`).

For each checked row it prints every explainer that was tried (TreeExplainer, PermutationExplainer,
KernelExplainer), whether its output reconstructed the model prediction, and the additivity gap, so a
failure is diagnosable. It passes when at least one explainer is accepted for every checked row.

    python examples/train_demo.py        # first: creates the model artifact
    python examples/verify_shap.py

Exit code 0  = SHAP produced additive explanations for every checked row.
Exit code 1  = shap missing or failing (the forecast itself is unaffected; explanations stay "unavailable").
Nothing here is mocked: it calls models/explain.py exactly as predict() does.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from aquapredict.forecasting.examples.generate_data import demo_splits, rows_to_history
from aquapredict.forecasting.features.engineering import build_features
from aquapredict.forecasting.models.explain import _import_shap, explain_prediction
from aquapredict.forecasting.models.persistence import DEFAULT_ARTIFACT, load_bundle
from aquapredict.forecasting.predict import predict


def main() -> int:
    shap = _import_shap()
    if shap is None:
        print("shap is NOT installed. Install it with:  pip install shap\n"
              "The forecasting core works without it; explanations are reported as unavailable.")
        return 1
    print("shap version:", getattr(shap, "__version__", "unknown"))
    if not DEFAULT_ARTIFACT.exists():
        print(f"No trained model at {DEFAULT_ARTIFACT}. Run first:  python examples/train_demo.py")
        return 1
    model, _ = load_bundle(DEFAULT_ARTIFACT)
    _, splits = demo_splits()
    test = splits["test"]
    sizes = test.groupby(["household_id", "filter_stage"]).size()
    ok_all = True
    for (hh, stage) in list(sizes[sizes >= 40].index[:3]):
        g = test[(test.household_id == hh) & (test.filter_stage == stage)]
        hist = rows_to_history(g, 40)
        X = build_features(g.iloc[:40].assign())
        res = explain_prediction(model, X, top_k=None)
        print(f"\n{hh}/{stage}: available={res['available']} method={res['method']}")
        for a in res.get("attempts", []):
            note = f"gap {a['additivity_gap_days']:.4f} d" if a.get("additivity_gap_days") is not None else ""
            print(f"  tried {a['method']:28s} accepted={a['accepted']!s:5s} {note} {a.get('problem', '')}")
        if not res["available"]:
            print("  reason:", res["reason"])
            ok_all = False
            continue
        print(f"  additivity gap = {res['additivity_gap_days']:.4f} days; base value = {res['base_value_days']:.1f}")
        for d in res["drivers"][:6]:
            print(f"  {d['feature']:26s} value={d['value']!s:>10.10}  contribution {d['contribution_days']:+7.2f} days")
        print(" ", res["summary"])
        full = predict({"filter_stage": stage, "household_id": hh, "history": hist})
        print(f"  predict(): status={full['status']} forecast={full['forecast_days']} drivers_returned={len(full['drivers'])}")
        ok_all &= bool(full["drivers"]) if full["forecast_days"] is not None else True
    print("\nSHAP verification:", "PASSED" if ok_all else "FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
