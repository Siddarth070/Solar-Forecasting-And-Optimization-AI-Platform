# Archived exploration — not part of the product

Everything in this directory is exploratory work, kept as an appendix.
None of it runs in the served product (`src/`, `dashboard/`, `benchmark.py`).

Per roadmap P1.8: **the product path is one model family — XGBoost.**
LSTM and Prophet were explored here and are retired from the product:

- `TensorFlow` and `prophet` are not installed by `requirements.txt`
  (both lines are commented out) and are not part of the Docker image —
  keeping them out is a deliberate, permanent decision, not an oversight.
  Reinstall them yourself (`pip install tensorflow prophet`) only if you
  want to re-run these notebooks.
- No external material (README, pitch deck, etc.) may describe this
  project as a multi-model ensemble. It is not one.
- Every metric these notebooks print predates the P0.1 target-leakage
  fix and is invalid — each notebook has its own caveat cell explaining
  why. The current, honest numbers are `benchmark.py` /
  `src/models/model_card.json`.

| Notebook | What it explored |
|---|---|
| `EDA_py.ipynb` | Exploratory data analysis on the simulated Jaipur dataset |
| `feature_engineering.ipynb` | Feature engineering (superseded by `src/features/pipeline.py`) |
| `models_XgBoost.ipynb` | XGBoost training experiment (superseded by `src/models/train.py`) |
| `Models_LSTM.ipynb` | LSTM training experiment — not served |
| `Models_prophet.ipynb` | Prophet training experiment — not served |
| `Ensemble.ipynb` | XGBoost + LSTM + Prophet ensemble experiment — not served |
| `optimization.ipynb` | Battery optimizer experiment |
