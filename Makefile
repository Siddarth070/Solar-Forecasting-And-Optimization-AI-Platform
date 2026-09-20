.PHONY: train benchmark test

# Rebuilds the served model artifact end to end on a clean clone, with no
# manual steps (roadmap P0.6). Writes src/models/xgboost_solar_v2.json and
# updates src/models/model_card.json.
train:
	python -m src.models.train

# Honest evaluation harness (roadmap P0.4/P0.5): rolling-origin backtest,
# baselines, and the model-skill/deliverable-skill split. Appends its
# results to the same model_card.json `train` writes.
benchmark:
	python benchmark.py

test:
	pytest tests/ -v
