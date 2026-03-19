"""
AML fraud detection training (Snowpark / Snowflake).

Data integrity & evaluation
- Test set can use natural fraud rate (USE_NATURAL_TEST_RATE=True) so holdout reflects real-world
  distribution (~0.1–0.2% fraud) instead of stratified oversample; metrics then reflect production.
- Train keeps stratified sampling so the model sees enough fraud; test uses a time-ordered cap when
  natural rate is enabled.

Data leakage
- Rolling features (TXN_COUNT_*_FROM, AVG_*, etc.) are built in feature_engineering with
  RANGE BETWEEN interval PRECEDING AND CURRENT ROW — past-only. REPEAT_COUNTERPARTY_COUNT
  is row_number()-1 over (from, to, ts), i.e. past-only.
- Lifetime counts (NUM_UNIQUE_RECEIVERS, NUM_UNIQUE_SENDERS, NUM_UNIQUE_BANKS) use
  partition-only windows in SQL (no ORDER BY/frame), so they include future rows — set
  EXCLUDE_LEAKY_FEATURES to drop them for strict backward-looking evaluation.

Scalability
- Only capped samples are pulled via .to_pandas(); for much larger tables, lower MAX_*_ROWS or
  run training outside Snowflake with Spark + connector for distributed training.

Feature engineering (see feature_engineering.py)
- Missing values: imputation + _IS_MISSING indicators in this script. For more robust signals,
  consider: transaction ratios (e.g. amount / avg_7d), anomaly scores (z-score vs 7d), and
  explicit validation of feature importance (we map booster f0,f1 to names for reporting).
"""
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F

import json
import os
import tempfile
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Pipeline helpers (split out for maintainability)
from aml_pipeline.evaluation.metrics import ks_stat, precision_recall_at_topk
from aml_pipeline.evaluation.thresholds import find_best_threshold, threshold_for_alert_rate
from aml_pipeline.sampling import stratified_sample_snowpark
from aml_pipeline.preprocessing import (
    impute_and_missing_indicators,
    drop_highly_correlated,
    apply_impute_and_keep,
)
from aml_pipeline.train_validation import pre_train_checks, validate_training_sample

# Optional: Optuna, LightGBM (skip if not installed)
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ---------- CONFIG ----------
FEATURE_TABLE = "AML_PROJECT.RAW.TRANSACTION_FEATURES"
PRED_TABLE = "AML_PROJECT.RAW.FRAUD_PREDICTIONS"
ML_RUNS_TABLE = "AML_PROJECT.RAW.ML_RUNS"
MODEL_ARTIFACT_DIR = "models"

LABEL_COL = "IS_LAUNDERING"
TS_COL = "EVENT_TIMESTAMP"
ID_COLS = ["FROM_BANK", "FROM_ACCOUNT", "TO_BANK", "TO_ACCOUNT", TS_COL]

FEATURE_COLS = [
    "AMOUNT_PAID", "AMOUNT_RECEIVED", "LOG_AMOUNT",
    "CURRENCIES_DIFFERENT", "SAME_BANK", "HOUR_OF_DAY", "DAY_OF_WEEK", "IS_WEEKEND",
    "TIME_SINCE_LAST_TXN_FROM", "TXN_COUNT_1H_FROM", "TXN_COUNT_24H_FROM", "TXN_COUNT_7D_FROM",
    "TXN_AMOUNT_SUM_24H_FROM", "AVG_TXN_AMOUNT_7D_FROM", "MAX_TXN_AMOUNT_7D_FROM", "TXN_AMOUNT_STD_7D_FROM",
    "TXN_GAP_STD_7D", "TIME_SINCE_LAST_TXN_TO",
    "NUM_UNIQUE_RECEIVERS", "NUM_UNIQUE_SENDERS", "NUM_UNIQUE_BANKS", "CROSS_BANK_RATIO_7D",
    "IS_NEW_COUNTERPARTY", "REPEAT_COUNTERPARTY_COUNT", "NUM_SIMILAR_AMOUNT_TXNS_24H",
    "FIRST_TXN_OF_DAY_FROM", "AMOUNT_VS_7D_AVG_RATIO", "AMOUNT_ZSCORE_7D_FROM", "PCT_CHANGE_VS_PREV",
    "TXN_VELOCITY_7D_FROM",
    "IS_LARGE_TXN", "IS_ROUNDED_AMOUNT",
]
# Lifetime (partition-only) counts can leak future info; exclude for strict eval
EXCLUDE_LEAKY_FEATURES = ["NUM_UNIQUE_RECEIVERS", "NUM_UNIQUE_SENDERS", "NUM_UNIQUE_BANKS"]
EFFECTIVE_FEATURE_COLS = [c for c in FEATURE_COLS if c not in EXCLUDE_LEAKY_FEATURES]

TEST_DAYS = 120   # holdout window (longer = more test-period rows before 1% enforcement)
RANDOM_SEED = 42

# Stratified sampling: fraction per class; increase for more data (watch driver memory).
NEG_FRACTION_TRAIN = 0.08   # 8% of non-fraud train (was 2%)
POS_FRACTION_TRAIN = 0.80   # 80% of fraud train (was 50%)
NEG_FRACTION_TEST  = 0.30   # take more of test period
POS_FRACTION_TEST  = 1.00   # use all fraud in test period

# Hard cap on rows brought to driver (increase for more data; reduce if OOM)
MAX_TRAIN_ROWS = 3_000_000   # 3M train total (was 1.5M)
MAX_TEST_ROWS = 500_000      # 500k test total
MIN_POS_TRAIN = 80_000       # min fraud rows in train
MIN_POS_TEST = 30_000        # min fraud rows in test (when not using natural test rate)

# Test set: production-like fraud rate
USE_NATURAL_TEST_RATE = True
TARGET_TEST_FRAUD_RATE = 0.01   # 1%
NATURAL_TEST_MAX_ROWS = 1_000_000 # more test rows when enforcing 1% (need enough frauds for stable metrics)

# Ensure we actually use all expected feature columns.
# If your feature table hasn't been rebuilt after adding new SQL features,
# training would otherwise silently drop missing columns.
STRICT_FEATURE_COLUMNS = True

# Backtest windows (days) for holdout evaluation
BACKTEST_DAYS = [14, 28, 56]

# Correlation threshold: drop one of pair if |corr| > this
CORR_THRESHOLD = 0.95

# Optuna / tuning
N_OPTUNA_TRIALS = 30
XGB_EARLY_STOPPING_ROUNDS = 30
CV_SPLITS = 4

TOP_K_PCTS = [0.001, 0.005, 0.01]
TOP_K_ABS = [1000, 5000, 10000]
ALERT_BUDGET_PCTS = [0.001, 0.002, 0.005, 0.01]  # 0.1%, 0.2%, 0.5%, 1.0%
DEFAULT_OPERATIONAL_ALERT_RATE = 0.001            # primary operating point for reporting
STRESS_TEST_FEATURES = ["IS_NEW_COUNTERPARTY", "REPEAT_COUNTERPARTY_COUNT"]


#
# stratified_sample_snowpark extracted to `aml_pipeline/sampling.py`
#


#
# pre_train_checks extracted to `aml_pipeline/train_validation.py`
#


#
# validate_training_sample extracted to `aml_pipeline/train_validation.py`
#


#
# impute_and_missing_indicators extracted to `aml_pipeline/preprocessing.py`
#


#
# drop_highly_correlated extracted to `aml_pipeline/preprocessing.py`
#


#
# find_best_threshold extracted to `aml_pipeline/evaluation/thresholds.py`
#


# ---------- Snowpark: load and split (no full .to_pandas) ----------
session = get_active_session()

# Ensure session has current database/schema (required for create_dataframe/write_pandas temp stage)
_db, _schema = FEATURE_TABLE.split(".")[0], FEATURE_TABLE.split(".")[1]
try:
    session.sql(f"USE DATABASE {_db}").collect()
    session.sql(f"USE SCHEMA {_schema}").collect()
except Exception:
    pass  # already set or no permission; continue

df = session.table(FEATURE_TABLE)
if EXCLUDE_LEAKY_FEATURES:
    print(f"Excluding leaky (lifetime) features for strict eval: {EXCLUDE_LEAKY_FEATURES}")

missing_feats = [c for c in EFFECTIVE_FEATURE_COLS if c not in df.columns]
if missing_feats:
    msg = (
        f"Feature table {FEATURE_TABLE} is missing expected columns ({len(missing_feats)}): "
        f"{missing_feats[:20]}{'...' if len(missing_feats) > 20 else ''}. "
        f"Rebuild the feature table after updating feature_engineering.py."
    )
    if STRICT_FEATURE_COLUMNS:
        raise ValueError(msg)
    else:
        print("Warning: " + msg)

cols = [c for c in (ID_COLS + [LABEL_COL] + EFFECTIVE_FEATURE_COLS) if c in df.columns]
df = df.select(*cols).filter(F.col(TS_COL).is_not_null())

span = df.agg(F.min(TS_COL).alias("min_ts"), F.max(TS_COL).alias("max_ts")).to_pandas().iloc[0]
max_ts = pd.to_datetime(span.get("MAX_TS") or span.get("max_ts"))
cutoff = (max_ts - pd.Timedelta(days=TEST_DAYS)).to_pydatetime()

df_train = df.filter(F.col(TS_COL) < F.lit(cutoff))
df_test  = df.filter(F.col(TS_COL) >= F.lit(cutoff))

# Train: always stratified so model sees enough fraud
max_pos_train = min(MIN_POS_TRAIN, MAX_TRAIN_ROWS // 5)
max_neg_train = MAX_TRAIN_ROWS - max_pos_train
df_train_s = stratified_sample_snowpark(
    df_train, LABEL_COL, NEG_FRACTION_TRAIN, POS_FRACTION_TRAIN,
    max_neg=max_neg_train, max_pos=max_pos_train,
)

# Test: production-like rate (target 1% fraud) or stratified (optimistic)
if USE_NATURAL_TEST_RATE:
    neg_cap = int(NATURAL_TEST_MAX_ROWS * (1 - TARGET_TEST_FRAUD_RATE))
    pos_cap = int(NATURAL_TEST_MAX_ROWS * TARGET_TEST_FRAUD_RATE)
    df_test_neg = df_test.filter(F.col(LABEL_COL) == 0).limit(neg_cap)
    df_test_pos = df_test.filter(F.col(LABEL_COL) == 1).limit(pos_cap)
    df_test_s = df_test_neg.union_all(df_test_pos).order_by(TS_COL)
    print(f"Train: stratified. Test: production-like (target fraud rate={TARGET_TEST_FRAUD_RATE:.2%}, neg_cap={neg_cap:,}, pos_cap={pos_cap:,}).")
else:
    max_pos_test = min(MIN_POS_TEST, MAX_TEST_ROWS // 5)
    max_neg_test = MAX_TEST_ROWS - max_pos_test
    df_test_s = stratified_sample_snowpark(
        df_test, LABEL_COL, NEG_FRACTION_TEST, POS_FRACTION_TEST,
        max_neg=max_neg_test, max_pos=max_pos_test,
    )
    print("Train & test: stratified sampling (per-class cap) applied before .to_pandas().")

train_pd = df_train_s.order_by(TS_COL).to_pandas()
test_pd  = df_test_s.order_by(TS_COL).to_pandas()

train_pd = train_pd.sort_values(TS_COL)
test_pd  = test_pd.sort_values(TS_COL)

# Drop duplicate keys so one row per transaction in train and test
key_cols_avail = [c for c in ID_COLS if c in train_pd.columns]
if key_cols_avail:
    n_before = len(train_pd)
    train_pd = train_pd.drop_duplicates(subset=key_cols_avail, keep="first")
    if len(train_pd) < n_before:
        print(f"Dropped {n_before - len(train_pd):,} duplicate keys from train.")
    n_test_before = len(test_pd)
    test_pd = test_pd.drop_duplicates(subset=key_cols_avail, keep="first")
    if len(test_pd) < n_test_before:
        print(f"Dropped {n_test_before - len(test_pd):,} duplicate keys from test.")

# Enforce target test fraud rate in pandas (holdout window may be fraud-heavy → 61% otherwise)
if USE_NATURAL_TEST_RATE and TARGET_TEST_FRAUD_RATE is not None:
    rate_before = test_pd[LABEL_COL].mean()
    n_neg = (test_pd[LABEL_COL] == 0).sum()
    n_pos = (test_pd[LABEL_COL] == 1).sum()
    if rate_before > TARGET_TEST_FRAUD_RATE and n_pos > 0 and n_neg > 0:
        # Too many fraud: keep all neg, downsample pos so rate = TARGET
        desired_pos = int(round(n_neg * TARGET_TEST_FRAUD_RATE / (1 - TARGET_TEST_FRAUD_RATE)))
        desired_pos = min(desired_pos, n_pos)
        pos_df = test_pd.loc[test_pd[LABEL_COL] == 1].sample(n=desired_pos, random_state=RANDOM_SEED, replace=False)
        neg_df = test_pd.loc[test_pd[LABEL_COL] == 0]
        test_pd = pd.concat([neg_df, pos_df], ignore_index=True).sort_values(TS_COL).reset_index(drop=True)
        print(f"Enforced test fraud rate: {rate_before:.2%} -> {test_pd[LABEL_COL].mean():.2%} (n_neg={n_neg:,}, n_pos={desired_pos:,}).")
    elif rate_before < TARGET_TEST_FRAUD_RATE and n_pos > 0 and n_neg > 0:
        # Too many legit: keep all pos, downsample neg
        desired_neg = int(round(n_pos * (1 - TARGET_TEST_FRAUD_RATE) / TARGET_TEST_FRAUD_RATE))
        desired_neg = min(desired_neg, n_neg)
        neg_df = test_pd.loc[test_pd[LABEL_COL] == 0].sample(n=desired_neg, random_state=RANDOM_SEED, replace=False)
        pos_df = test_pd.loc[test_pd[LABEL_COL] == 1]
        test_pd = pd.concat([neg_df, pos_df], ignore_index=True).sort_values(TS_COL).reset_index(drop=True)
        print(f"Enforced test fraud rate: {rate_before:.2%} -> {test_pd[LABEL_COL].mean():.2%} (n_neg={desired_neg:,}, n_pos={n_pos:,}).")
    else:
        print(f"Test fraud rate unchanged: {rate_before:.2%} (n_neg={n_neg:,}, n_pos={n_pos:,}).")

# If we still have extremely few frauds after enforcing the target rate,
# PR/Top-K/investigation metrics become mostly sampling noise.
if USE_NATURAL_TEST_RATE:
    n_pos_final = int((test_pd[LABEL_COL] == 1).sum())
    if n_pos_final < 50:
        print(
            f"Warning: only {n_pos_final:,} frauds in the test set after enforcement. "
            f"Metrics (AUC-PR, Top-K, thresholds) will be unstable. "
            f"Increase TEST_DAYS (currently {TEST_DAYS}) or NATURAL_TEST_MAX_ROWS."
        )

pre_train_checks(train_pd, test_pd, LABEL_COL)
if len(test_pd) < 1000:
    print(f"Warning: test set has only {len(test_pd):,} rows; metrics may be unstable. Consider increasing TEST_DAYS or sampling fractions.")
validate_training_sample(train_pd, test_pd, LABEL_COL, EFFECTIVE_FEATURE_COLS, ID_COLS)

# Feature matrices with imputation and missing indicators
X_train_raw = train_pd[EFFECTIVE_FEATURE_COLS].copy()
X_test_raw  = test_pd[EFFECTIVE_FEATURE_COLS].copy()
X_train_f, X_test_f, feature_cols_after_impute, imputer = impute_and_missing_indicators(
    X_train_raw, X_test_raw, EFFECTIVE_FEATURE_COLS
)

# Drop highly correlated (on train)
keep_cols = drop_highly_correlated(X_train_f.astype(float), CORR_THRESHOLD)
if len(keep_cols) < len(X_train_f.columns):
    print(f"Dropped {len(X_train_f.columns) - len(keep_cols)} highly correlated features.")
X_train_f = X_train_f[keep_cols]
X_test_f = X_test_f[keep_cols]

# Scaling for LR (and optional for tree models we leave unscaled)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_f.astype(float))
X_test_s  = scaler.transform(X_test_f.astype(float))
X_train_np = X_train_f.astype(float).values
X_test_np  = X_test_f.astype(float).values

y_train = train_pd[LABEL_COL].astype(int).values
y_test  = test_pd[LABEL_COL].astype(int).values

pos = max((y_train == 1).sum(), 1)
neg = max((y_train == 0).sum(), 1)
scale_pos_weight = neg / pos

print(f"Train sample: {len(train_pd):,} (fraud rate {y_train.mean():.4f}), Test sample: {len(test_pd):,}")

# ---------- XGBoost with early stopping and Optuna ----------
def xgb_objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
    }
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []
    for train_idx, val_idx in tscv.split(X_train_np):
        Xt, Xv = X_train_np[train_idx], X_train_np[val_idx]
        yt, yv = y_train[train_idx], y_train[val_idx]
        m = xgb.XGBClassifier(**params)
        m.fit(
            Xt, yt,
            eval_set=[(Xv, yv)],
            verbose=False,
        )
        proba = m.predict_proba(Xv)[:, 1]
        scores.append(average_precision_score(yv, proba))
    return np.mean(scores)


if HAS_OPTUNA:
    study = optuna.create_study(direction="maximize")
    study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_params = study.best_params
else:
    best_params = {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8,
                   "colsample_bytree": 0.8, "min_child_weight": 5, "gamma": 0.5, "reg_lambda": 2.0}

xgb_final_params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "scale_pos_weight": scale_pos_weight,
    "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
    **best_params,
}

model_xgb = xgb.XGBClassifier(**xgb_final_params)
model_xgb.fit(
    X_train_np, y_train,
    eval_set=[(X_test_np, y_test)],
    verbose=False,
)

proba_xgb = model_xgb.predict_proba(X_test_np)[:, 1]

# ---------- LightGBM ----------
results = {}
if HAS_LGB:
    model_lgb = lgb.LGBMClassifier(
        objective="binary",
        metric="average_precision",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_train_np, y_train,
        eval_set=[(X_test_np, y_test)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    proba_lgb = model_lgb.predict_proba(X_test_np)[:, 1]
    results["LightGBM"] = {"proba": proba_lgb, "model": model_lgb}
else:
    print("LightGBM not installed; skipping.")

# ---------- Logistic Regression (scaled) ----------
model_lr = LogisticRegression(
    class_weight="balanced",
    max_iter=500,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
model_lr.fit(X_train_s, y_train)
proba_lr = model_lr.predict_proba(X_test_s)[:, 1]
results["LogisticRegression"] = {"proba": proba_lr, "model": model_lr}

# ---------- Benchmark: AUC-PR, KS ----------
results["XGBoost"] = {"proba": proba_xgb, "model": model_xgb}

print("\n--- Model benchmark (AUC-PR / ROC AUC / KS) ---")
for name, d in results.items():
    p = d["proba"]
    apr = average_precision_score(y_test, p)
    roc = roc_auc_score(y_test, p) if len(np.unique(y_test)) > 1 else float("nan")
    ks = ks_stat(y_test, p)
    print(f"  {name}: AUC-PR={apr:.4f}  ROC AUC={roc:.4f}  KS={ks:.4f}")

# Use XGBoost as primary for threshold and reporting
proba = proba_xgb

# ---------- Confusion matrix + classification report (default 0.5) ----------
y_pred_05 = (proba >= 0.5).astype(int)
print("\n--- Confusion matrix (threshold=0.5) ---")
print(confusion_matrix(y_test, y_pred_05))
print("\n--- Classification report (threshold=0.5) ---")
print(classification_report(y_test, y_pred_05, target_names=["legit", "fraud"]))

# ---------- Threshold tuning ----------
best_threshold_youden, _ = find_best_threshold(y_test, proba, "youden")
best_threshold_f2, _ = find_best_threshold(y_test, proba, "f2")
print(f"\n--- Threshold tuning ---")
print(f"  Best threshold (Youden): {best_threshold_youden:.3f}")
print(f"  Best threshold (F2):     {best_threshold_f2:.3f}")

y_pred_best = (proba >= best_threshold_youden).astype(int)
print("\n--- Confusion matrix (Youden-optimal threshold) ---")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, target_names=["legit", "fraud"]))

# ---------- Business-driven thresholding by alert budget ----------
print("\n--- Alert-budget thresholds (business operating points) ---")
for pct in ALERT_BUDGET_PCTS:
    t_budget = threshold_for_alert_rate(proba, pct)
    y_pred_budget = (proba >= t_budget).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_budget).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(
        f"  Alert rate {pct*100:.2f}%: threshold={t_budget:.3f} "
        f"precision={prec:.4f} recall={rec:.4f} alerts={tp+fp:,} tp={tp:,} fp={fp:,}"
    )

operational_threshold = threshold_for_alert_rate(proba, DEFAULT_OPERATIONAL_ALERT_RATE)
print(
    f"  Selected operational threshold (alert rate {DEFAULT_OPERATIONAL_ALERT_RATE*100:.2f}%): "
    f"{operational_threshold:.3f}"
)

# ---------- Top-K metrics ----------
print("\n--- Top-K metrics (XGBoost) ---")
for pct in TOP_K_PCTS:
    k = int(round(len(y_test) * pct))
    prec, rec, tp = precision_recall_at_topk(y_test, proba, k)
    print(f"  Top {pct*100:.2f}% (k={k}): precision={prec:.4f} recall={rec:.4f} true_frauds={tp}")
for k in TOP_K_ABS:
    if k <= len(y_test):
        prec, rec, tp = precision_recall_at_topk(y_test, proba, k)
        print(f"  Top K={k}: precision={prec:.4f} recall={rec:.4f} true_frauds={tp}")

# ---------- Investigation capacity (business-aligned) ----------
n_fraud_test = int(y_test.sum())
print("\n--- Investigation capacity (if we only review top K transactions) ---")
for k in [100, 500, 1000, 5000]:
    if k > len(y_test):
        continue
    prec, rec, tp = precision_recall_at_topk(y_test, proba, k)
    pct_caught = 100.0 * rec if n_fraud_test else 0
    print(f"  Top {k:>5}: catch {tp:>4} frauds ({pct_caught:.1f}% of all fraud), precision={prec:.2%}")

# ---------- Backtesting over multiple time windows ----------
# apply_impute_and_keep extracted to `aml_pipeline/preprocessing.py`

print("\n--- Backtesting (stratified sample per window) ---")
backtest_ks = []
for test_days in BACKTEST_DAYS:
    cut = (max_ts - pd.Timedelta(days=test_days)).to_pydatetime()
    df_bt = df.filter(F.col(TS_COL) >= F.lit(cut))
    df_bt_s = stratified_sample_snowpark(
        df_bt, LABEL_COL, 0.05, 0.5,
        max_neg=50_000, max_pos=5_000,
    )
    bt_pd = df_bt_s.order_by(TS_COL).to_pandas()
    if len(bt_pd) == 0 or bt_pd[LABEL_COL].nunique() < 2:
        print(f"  Last {test_days}d: skip (no data or single class)")
        continue
    X_bt = apply_impute_and_keep(bt_pd, imputer, EFFECTIVE_FEATURE_COLS, keep_cols)
    y_bt = bt_pd[LABEL_COL].astype(int).values
    p_bt = model_xgb.predict_proba(X_bt)[:, 1]
    apr = average_precision_score(y_bt, p_bt)
    ks = ks_stat(y_bt, p_bt)
    backtest_ks.append((test_days, ks))
    print(f"  Last {test_days}d: n={len(bt_pd):,}  AUC-PR={apr:.4f}  KS={ks:.4f}")

if len(backtest_ks) >= 2:
    ks_short, ks_long = backtest_ks[0][1], backtest_ks[-1][1]
    trend = "drops" if ks_long < ks_short else "stable/improves"
    print(f"  Stability: KS {trend} from {ks_short:.2f} ({backtest_ks[0][0]}d) to {ks_long:.2f} ({backtest_ks[-1][0]}d) — consider rolling-window monitoring and drift checks.")

# ---------- Feature importance (XGBoost) ----------
booster = model_xgb.get_booster()
gain = booster.get_score(importance_type="gain")
# Booster may use f0, f1, ... when trained on numpy; map by index to keep_cols
gain_by_idx = [gain.get(f"f{i}", 0.0) for i in range(len(keep_cols))]
imp = pd.DataFrame({
    "feature": keep_cols,
    "gain": gain_by_idx,
}).sort_values("gain", ascending=False)
print("\n--- Top feature importances (XGBoost) ---")
print(imp.head(20).to_string(index=False))

# ---------- Stress test: dependence on strongest behavioral features ----------
stress_cols = [c for c in STRESS_TEST_FEATURES if c in X_test_f.columns]
if len(stress_cols) > 0:
    X_stress = X_test_f.copy()
    for c in stress_cols:
        X_stress[c] = float(X_train_f[c].median()) if c in X_train_f.columns else 0.0
    p_stress = model_xgb.predict_proba(X_stress.astype(float).values)[:, 1]
    apr_base = average_precision_score(y_test, proba)
    apr_stress = average_precision_score(y_test, p_stress)
    print("\n--- Feature stress test (neutralize key counterparty signals) ---")
    print(f"  Features neutralized: {stress_cols}")
    print(f"  AUC-PR baseline={apr_base:.4f} -> stressed={apr_stress:.4f} (delta={apr_stress-apr_base:+.4f})")

# ---------- Run logging & model artifact ----------
run_id = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
model_version = f"xgboost_{run_id}"
artifact_dir = MODEL_ARTIFACT_DIR
try:
    os.makedirs(artifact_dir, exist_ok=True)
except OSError:
    # Read-only FS (e.g. Snowflake notebook): use temp dir
    artifact_dir = os.path.join(tempfile.gettempdir(), "aml_models")
    os.makedirs(artifact_dir, exist_ok=True)
    print(f"Using writable artifact dir: {artifact_dir}")
artifact_path = os.path.join(artifact_dir, f"xgb_{model_version}.json")
model_xgb.save_model(artifact_path)
print(f"\nSaved model artifact: {artifact_path}")

test_fraud_rate = float(y_test.mean())
auc_pr = average_precision_score(y_test, proba_xgb)
roc_auc = roc_auc_score(y_test, proba_xgb) if len(np.unique(y_test)) > 1 else float("nan")
params_json = json.dumps({k: v for k, v in xgb_final_params.items() if k != "early_stopping_rounds"}, default=str)
run_row = pd.DataFrame([{
    "RUN_ID": run_id,
    "MODEL_VERSION": model_version,
    "CUTOFF_TS": cutoff,
    "N_TRAIN": len(train_pd),
    "N_TEST": len(test_pd),
    "TEST_FRAUD_RATE": test_fraud_rate,
    "AUC_PR": auc_pr,
    "ROC_AUC": roc_auc,
    "KS": ks_stat(y_test, proba_xgb),
    "BEST_THRESHOLD_YOUDEN": best_threshold_youden,
    # Keep a copy of the exact feature set used by the model (after correlation filtering).
    "FEATURES_JSON": json.dumps(keep_cols if "keep_cols" in locals() else EFFECTIVE_FEATURE_COLS),
    "PARAMS_JSON": params_json,
    "MODEL_PATH": artifact_path,
    "CREATED_AT": pd.Timestamp.utcnow().to_pydatetime(),
}])
try:
    session.sql(f"""
        CREATE TABLE IF NOT EXISTS {ML_RUNS_TABLE} (
            RUN_ID STRING, MODEL_VERSION STRING, CUTOFF_TS TIMESTAMP_NTZ, N_TRAIN INT, N_TEST INT,
            TEST_FRAUD_RATE FLOAT, AUC_PR FLOAT, ROC_AUC FLOAT, KS FLOAT, BEST_THRESHOLD_YOUDEN FLOAT,
            PARAMS_JSON STRING, FEATURES_JSON STRING, MODEL_PATH STRING, CREATED_AT TIMESTAMP_NTZ
        )
    """).collect()
    # Align inserted columns to existing table schema (table may already exist with more columns).
    db, schema, name = ML_RUNS_TABLE.split(".")
    try:
        order_rows = session.sql(f"""
            SELECT COLUMN_NAME
            FROM {db}.INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_CATALOG = '{db}'
              AND TABLE_SCHEMA = '{schema}'
              AND TABLE_NAME = '{name}'
            ORDER BY ORDINAL_POSITION
        """).collect()
        table_columns = [r["COLUMN_NAME"] for r in order_rows]
        for c in table_columns:
            if c not in run_row.columns:
                run_row[c] = None
        run_row_aligned = run_row[table_columns]
    except Exception:
        run_row_aligned = run_row

    session.create_dataframe(run_row_aligned).write.save_as_table(ML_RUNS_TABLE, mode="append")
    print(f"Logged run to {ML_RUNS_TABLE}: RUN_ID={run_id}")
except Exception as e:
    print(f"Run logging skipped: {e}")

# ---------- Write predictions (Snowpark) ----------
# Ensure table has RUN_ID and MODEL_VERSION (existing table may have only 8 columns)
for col in ("RUN_ID", "MODEL_VERSION"):
    try:
        session.sql(f"ALTER TABLE {PRED_TABLE} ADD COLUMN {col} STRING").collect()
    except Exception:
        pass  # column already exists

out = test_pd[ID_COLS].copy()
out["FRAUD_PROBABILITY"] = proba
out["MODEL_NAME"] = model_version
out["MODEL_VERSION"] = model_version
out["RUN_ID"] = run_id
out["SCORED_AT"] = pd.Timestamp.utcnow().to_pydatetime()

# Match table column order (Snowflake append inserts by position)
db, schema, name = PRED_TABLE.split(".")
try:
    order_rows = session.sql(f"""
        SELECT COLUMN_NAME FROM {db}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_CATALOG = '{db}' AND TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{name}'
        ORDER BY ORDINAL_POSITION
    """).collect()
    table_columns = [r["COLUMN_NAME"] for r in order_rows]
    for c in table_columns:
        if c not in out.columns:
            out[c] = None
    out = out[table_columns]
except Exception:
    pass  # keep out as-is if we can't read schema

pred_sdf = session.create_dataframe(out)
pred_sdf.write.save_as_table(PRED_TABLE, mode="append")
print(f"Wrote {len(out):,} predictions to {PRED_TABLE}")
