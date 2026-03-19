from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import functions as F

# ---------- CONFIG ----------
FEATURE_TABLE = "AML_PROJECT.RAW.TRANSACTION_FEATURES"   
REPORT_TABLE  = "AML_PROJECT.RAW.FEATURE_VALIDATION_REPORT"
LABEL_COL     = "IS_LAUNDERING"
TS_COL        = "EVENT_TIMESTAMP"

KEY_COLS = ["FROM_BANK", "FROM_ACCOUNT", "TO_BANK", "TO_ACCOUNT", TS_COL]
AMOUNT_COLS = ["AMOUNT_PAID", "AMOUNT_RECEIVED", "LOG_AMOUNT"]
FEATURE_COLS = [
    "CURRENCIES_DIFFERENT","SAME_BANK","HOUR_OF_DAY","DAY_OF_WEEK","IS_WEEKEND",
    "TIME_SINCE_LAST_TXN_FROM","TXN_COUNT_1H_FROM","TXN_COUNT_24H_FROM","TXN_AMOUNT_SUM_24H_FROM",
    "AVG_TXN_AMOUNT_7D_FROM","MAX_TXN_AMOUNT_7D_FROM","TXN_AMOUNT_STD_7D_FROM","TXN_GAP_STD_7D",
    "TIME_SINCE_LAST_TXN_TO",
    "NUM_UNIQUE_RECEIVERS","NUM_UNIQUE_SENDERS","NUM_UNIQUE_BANKS","CROSS_BANK_RATIO_7D",
    "IS_NEW_COUNTERPARTY","REPEAT_COUNTERPARTY_COUNT","NUM_SIMILAR_AMOUNT_TXNS_24H",
    "IS_LARGE_TXN","IS_ROUNDED_AMOUNT",
]
# ---------------------------

def _session() -> Session:
    try:
        return get_active_session()
    except Exception:
        raise RuntimeError("No active Snowpark session. In a Python Worksheet, use the built-in session.")

session = _session()
df_features = session.table(FEATURE_TABLE)

# 1) Basic counts / label rate
counts = df_features.agg(
    F.count("*").alias("row_count"),
    F.sum(F.col(LABEL_COL)).alias("fraud_count"),
    (F.sum(F.col(LABEL_COL)) / F.count("*")).alias("fraud_rate")
)

# 2) Time span
time_span = df_features.agg(
    F.min(F.col(TS_COL)).alias("min_ts"),
    F.max(F.col(TS_COL)).alias("max_ts")
)

# 3) Null rates for important cols
check_cols = list(dict.fromkeys(KEY_COLS + [LABEL_COL] + AMOUNT_COLS + FEATURE_COLS))
null_exprs = [
    (F.sum(F.iff(F.col(c).is_null(), 1, 0)) / F.count("*")).alias(f"null_rate__{c}")
    for c in check_cols if c in df_features.columns
]
null_rates = df_features.agg(*null_exprs) if null_exprs else session.create_dataframe([{}])

# 4) Bad value checks (amounts, label)
bad_checks = df_features.agg(
    F.sum(F.iff((F.col("AMOUNT_PAID") <= 0) | F.col("AMOUNT_PAID").is_null(), 1, 0)).alias("bad_amount_paid_cnt"),
    F.sum(F.iff((F.col("AMOUNT_RECEIVED") <= 0) | F.col("AMOUNT_RECEIVED").is_null(), 1, 0)).alias("bad_amount_received_cnt"),
    F.sum(F.iff(~F.col(LABEL_COL).isin([0,1]) | F.col(LABEL_COL).is_null(), 1, 0)).alias("bad_label_cnt"),
)

# 5) Duplicate key-ish rows (helps catch accidental duplication)
dup_cnt = (
    df_features.group_by(*[c for c in KEY_COLS if c in df_features.columns])
      .count()
      .filter(F.col("COUNT") > 1)
      .agg(F.sum(F.col("COUNT") - 1).alias("duplicate_rows_estimate"))
)

# 6) Numeric stats + percentiles (p50/p90/p99) for key numeric columns
num_cols = [c for c in (AMOUNT_COLS + FEATURE_COLS) if c in df_features.columns]
stats_exprs = []
for c in num_cols:
    stats_exprs += [
        F.avg(F.col(c)).alias(f"avg__{c}"),
        F.stddev(F.col(c)).alias(f"std__{c}"),
        F.min(F.col(c)).alias(f"min__{c}"),
        F.max(F.col(c)).alias(f"max__{c}"),
        F.approx_percentile(F.col(c), 0.50).alias(f"p50__{c}"),
        F.approx_percentile(F.col(c), 0.90).alias(f"p90__{c}"),
        F.approx_percentile(F.col(c), 0.99).alias(f"p99__{c}"),
    ]
num_stats = df_features.agg(*stats_exprs) if stats_exprs else session.create_dataframe([{}])

# 7) Build a single-row report and write it
report = (
    counts.cross_join(time_span)
          .cross_join(null_rates)
          .cross_join(bad_checks)
          .cross_join(dup_cnt)
          .with_column("feature_table", F.lit(FEATURE_TABLE))
          .with_column("generated_at", F.current_timestamp())
)

report.write.save_as_table(REPORT_TABLE, mode="overwrite")
report.show()
print(f"Wrote validation report to {REPORT_TABLE}")
