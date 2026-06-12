# Drift Detective

Most ML models don't fail loudly. They degrade quietly — the data shifts, the world changes, and the model keeps predicting like nothing happened. Drift Detective monitors your deployed model's input distributions against its training baseline and tells you when something is wrong, before your metrics do.

## What it does

- Compares live feature distributions against training data using KS Test, PSI, and KL Divergence
- Separates covariate shift (input features changed) from concept drift (relationship between features and target changed)
- Scores each feature independently and ranks the top contributors to drift
- Visualizes model health over time with configurable alert thresholds
- Recommends retraining vs fine-tuning based on drift severity and pattern

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** Streamlit
- **ML / Stats:** Scikit-learn, SciPy, NumPy
- **Drift Detection:** KS Test, PSI, KL Divergence
- **Storage:** CSV-based log ingestion (SQLite support planned)

## Project Structure

```text
drift-detective/
├── app/
│   ├── api/          # FastAPI routes
│   ├── core/         # drift detection logic
│   │   ├── ks_test.py
│   │   ├── psi.py
│   │   └── kl_divergence.py
│   ├── models/       # data models
│   └── utils/        # preprocessing, windowing
├── dashboard/        # Streamlit UI
├── data/
│   ├── baseline/     # training distribution snapshots
│   └── logs/         # incoming prediction logs
├── tests/
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/Aadya-Jha/Drift-Detective
cd Drift-Detective
pip install -r requirements.txt

# start the API
uvicorn app.main:app --reload

# start the dashboard
streamlit run dashboard/app.py
```

## How it works

You provide two things: a snapshot of your training data distributions (the baseline) and a stream of incoming prediction logs. Drift Detective runs statistical tests on each feature at configurable intervals, produces a drift score per feature, and aggregates them into an overall model health score. When any feature crosses your alert threshold, you get a notification with an explanation of what drifted and by how much.

## Drift Detection Methods

| Method | Best for | What it measures |
|---|---|---|
| KS Test | Numerical features | Max distance between two CDFs |
| PSI | Numerical + categorical | Population stability over time |
| KL Divergence | Probability distributions | Information loss between distributions |

## Limitations

- Concept drift detection requires ground truth labels on incoming data, which are often delayed in production
- Works best with tabular data; image/text drift detection is not in scope yet
- Baseline window size significantly affects sensitivity — tuning required per use case