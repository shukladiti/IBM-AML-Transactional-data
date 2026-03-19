import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def impute_and_missing_indicators(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_cols: list):
    """
    Median-impute numeric features.
    Also add an *_IS_MISSING indicator for any column that has nulls in train.
    Returns (X_train_imputed, X_test_imputed, feature_columns_after_impute, fitted_imputer).
    """
    imputer = SimpleImputer(strategy="median")
    X_tr = X_train[feature_cols].copy()
    X_te = X_test[feature_cols].copy()

    missing_tr = X_tr.isna()

    X_tr_imputed = pd.DataFrame(
        imputer.fit_transform(X_tr),
        columns=feature_cols,
        index=X_tr.index,
    )
    X_te_imputed = pd.DataFrame(
        imputer.transform(X_te),
        columns=feature_cols,
        index=X_te.index,
    )

    for c in feature_cols:
        if missing_tr[c].any():
            X_tr_imputed[f"{c}_IS_MISSING"] = missing_tr[c].astype(int).values
            X_te_imputed[f"{c}_IS_MISSING"] = X_te[c].isna().astype(int).values

    return X_tr_imputed, X_te_imputed, list(X_tr_imputed.columns), imputer


def drop_highly_correlated(X: pd.DataFrame, threshold: float):
    """Return list of columns to keep (drop one from any highly correlated pair)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in X.columns if c not in drop]


def apply_impute_and_keep(bt_pd: pd.DataFrame, imputer, feature_cols: list, keep_cols: list):
    """
    Backtest helper:
      - impute features with the training imputer
      - add *_IS_MISSING indicators when the column exists in keep_cols
      - return numpy array in keep_cols order
    """
    X = bt_pd[feature_cols].copy()
    missing = X.isna()
    X_imp = pd.DataFrame(imputer.transform(X), columns=feature_cols, index=X.index)

    for c in feature_cols:
        miss_col = f"{c}_IS_MISSING"
        if miss_col in keep_cols:
            X_imp[miss_col] = missing[c].astype(int).values if c in X.columns else 0

    return X_imp[keep_cols].astype(float).values

