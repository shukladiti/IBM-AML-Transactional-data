# IBM AML Fraud Detection (Snowflake / Snowpark)

I’m building an AML fraud-detection pipeline specifically for Snowflake because it’s the platform where the data already lives and where I want the workflow to be operational end-to-end. My goal is to generate transaction-level features from Snowflake tables using Snowpark SQL, validate the resulting dataset for issues like nulls/duplicates and potential leakage, and then train and evaluate models with time-based splits and business-aligned metrics (e.g., alert-budget and Top-K investigation capacity) to reflect realistic AML constraints under extreme class imbalance.

Anti–Money Laundering (AML) fraud detection pipeline built on Snowflake using Snowpark:
1. Build transaction-level features in SQL (`feature_engineering.py`)
2. Validate feature integrity and dataset health (`data_validation.py`)
3. Train models with time-based splits and production-like evaluation (`training_models.py`)
4. Write predictions and run metrics back to Snowflake tables

This repo is designed for realistic AML evaluation under extreme class imbalance, with explicit leakage controls and decision-focused metrics.

## Repository Structure

### Snowflake/Snowpark entrypoints (run these in Snowflake)
- `jobs/feature_engineering.py`
  - Creates one row per transaction with rolling and counterparty/entity features
- `jobs/data_validation.py`
  - Generates summary dataset reports (fraud rate, nulls, duplicates, feature distributions)
- `jobs/training_models.py`
  - Loads the feature table, creates a time split, trains models, evaluates, logs runs, writes predictions

### Reusable Python helpers
- `aml_pipeline/`
  - `preprocessing.py`: impute missing values, correlate-drop, backtest batch alignment helper
  - `sampling.py`: Snowpark stratified sampling helper
  - `train_validation.py`: lightweight pre-training checks and sample-only validation prints
  - `evaluation/metrics.py`: KS and Top-K precision/recall helpers
  - `evaluation/thresholds.py`: threshold selection helpers (Youden/F2, alert-budget cutoff)

## Data

### Inputs
- `AML_PROJECT.RAW.RAW_TRANSACTIONS`
- `AML_PROJECT.RAW.RAW_ACCOUNTS`

### Outputs (used by training / written by this pipeline)
Training defaults in `jobs/training_models.py`:
- Feature table: `AML_PROJECT.RAW.TRANSACTION_FEATURES`
- Prediction table: `AML_PROJECT.RAW.FRAUD_PREDICTIONS`
- Run metrics table: `AML_PROJECT.RAW.ML_RUNS`

You may override table names in the scripts if your schema differs.

## End-to-End Workflow

### 1) Build features
Run `jobs/feature_engineering.py`.

What it does (high level):
- Joins transaction rows to account metadata
- Computes time-based rolling features using backward-looking windows (e.g. 1h/24h/7d)
- Computes counterparty novelty/repetition features
- Outputs a transaction-level feature table (one row per transaction)

**Important:** After changing SQL features, you must rebuild the feature table.

### 2) Validate features
Run `jobs/data_validation.py`.

What it does:
- Computes fraud rate, null percentages, duplicates sanity checks
- Writes a validation report table (in your RAW schema)

This step helps catch:
- label/amount issues
- unexpected missingness
- dataset anomalies

### 3) Train + score (Real production-like evaluation)
Run `jobs/training_models.py`.

#### Split strategy
- Uses `EVENT_TIMESTAMP` to create:
  - `train`: older transactions
  - `test`: the holdout window (`TEST_DAYS`)

#### Fraud-rate realism
- For a credible AML evaluation, `training_models.py` can enforce a target fraud rate in the test sample:
  - `USE_NATURAL_TEST_RATE=True`
  - `TARGET_TEST_FRAUD_RATE=0.01` (default 1%)
- This avoids overly optimistic metrics when the holdout period is fraud-heavy.

#### Leakage controls
- Drops known lifetime-style leaky features for strict evaluation:
  - `EXCLUDE_LEAKY_FEATURES = ["NUM_UNIQUE_RECEIVERS", "NUM_UNIQUE_SENDERS", "NUM_UNIQUE_BANKS"]`
- Uses a strict guard (`STRICT_FEATURE_COLUMNS=True`) to fail fast if the feature table is missing expected columns (prevents silent “model trained on an old feature table” mistakes).

#### Models
`training_models.py` trains:
- XGBoost (primary): time split training + optional Optuna tuning
- LightGBM 
- Logistic Regression (baseline)


## Evaluation (Decision-Focused)

In addition to standard metrics, the script reports AML-relevant decision views:

1. **Model quality**
   - AUC-PR
   - ROC AUC
   - KS

2. **Threshold / confusion matrix reporting**
   - Default threshold at `0.5`
   - Threshold search for Youden and F2

3. **Business operating points**
   - **Alert-budget thresholds**: for an allowed alert rate (e.g. 0.1%, 0.2%, 0.5%, 1.0%), it prints:
     - threshold score
     - precision, recall
     - alerts count, TP, FP

4. **Investigation capacity (Top-K)**
   - Precision/recall and fraud capture for:
     - Top-K absolute budgets (e.g. 1000, 5000, 10000)
     - Top-K percentages
   - This answers: “If we review only the top X transactions, how many frauds do we catch?”

5. **Rolling stability**
   - Backtesting across horizons (e.g. 14d / 28d / 56d) with KS and AUC-PR

6. **Stress test (feature dependency)**
   - Neutralizes the strongest behavioral/counterparty features to estimate how fragile the model is under drift.


## Run Logging + Predictions

At the end, `training_models.py`:
- Saves a model artifact (uses a writable fallback directory if needed)
- Appends a run row to:
  - `AML_PROJECT.RAW.ML_RUNS`
- Appends scored rows to:
  - `AML_PROJECT.RAW.FRAUD_PREDICTIONS`

Predictions include:
- transaction identity columns (`FROM_BANK`, `FROM_ACCOUNT`, `TO_BANK`, `TO_ACCOUNT`, `EVENT_TIMESTAMP`)
- `FRAUD_PROBABILITY`
- run metadata (`RUN_ID`, `MODEL_VERSION`)
- `SCORED_AT`


## Requirements (Python Environment)

The entrypoints are intended to run inside Snowflake/Snowpark, and require at least:
- `snowflake-snowpark-python`
- `numpy`, `pandas`
- `scikit-learn`
- `xgboost`

Optional:
- `optuna` (enables Optuna tuning)
- `lightgbm` (enables LightGBM training)


## Notes / Known Limitations

- GNN/GAT training is **not implemented** in this repo yet.
- Strict leakage assumptions depend on how the feature SQL is built and refreshed.
- The production-like test fraud rate enforcement is designed for realistic evaluation, but it will still reduce metrics stability if the enforced test sample has very few frauds.

## Next Improvements (Roadmap)
- Add GNN/GAT edge classification training (`gat`/`GNN`) as a new entrypoint file
- Expand leakage verification tests (spot-check window outputs against a strict “past-only” definition)
- Add calibration (Platt scaling / isotonic) for more stable decision thresholds
- Track stress-test deltas over time in `ML_RUNS` for drift monitoring
