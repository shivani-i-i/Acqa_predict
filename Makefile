# Windows without `make`: run the python -m commands directly (see README).
.PHONY: api ui models test-app test test-p1 test-p2 data eval eval-calibrated all reference ablation p2-train p2-demo integrate
test:            ; python -m pytest -q
test-p1:         ; python -m pytest -q tests --ignore=tests/forecasting
test-p2:         ; python -m pytest -q tests/forecasting
data:            ; python -m aquapredict.simulator.generate --n 200 --seed 1 --law A --calibrated --out data/synth_A
eval:            ; python -m aquapredict.eval.run_baselines --n 150 --seed 0 --law A --out results
eval-calibrated: ; python -m aquapredict.eval.run_baselines --n 150 --seed 0 --law A --calibrated --out results_calibrated
all:             ; python -m aquapredict.eval.make_all --seeds 5 --n 150 --out results
reference:       ; python -m aquapredict.reference_model.run_reference --n 250 --seed 0 --out results_reference
ablation:        ; python -m aquapredict.reference_model.ablation --n 250 --seed 0
p2-train:        ; python -m aquapredict.forecasting.examples.train_demo
p2-demo:         ; python -m aquapredict.forecasting.examples.run_demo
integrate:       ; python -m aquapredict.integration.evaluate --n 200 --seed 21 --out results_integration

# ---- Person 3: application ----
api:
	uvicorn aquapredict.app.api:app --reload --port 8000
ui:
	streamlit run aquapredict/app/ui/streamlit_app.py
models:
	python -c "from aquapredict.app.service import AquaService; s=AquaService(); print(s.registry.train())"
test-app:
	python -m pytest -q tests/app
