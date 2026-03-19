import numpy as np
import pandas as pd


def pre_train_checks(train_pd: pd.DataFrame, test_pd: pd.DataFrame, label_col: str) -> None:
    """Lightweight checks before training (no full validation duplication)."""
    assert label_col in train_pd.columns and label_col in test_pd.columns
    assert train_pd[label_col].notna().all() and test_pd[label_col].notna().all()
    assert set(train_pd[label_col].unique()) >= {0, 1}, "Train must have both classes"
    assert set(test_pd[label_col].unique()) >= {0, 1}, "Test must have both classes"
    assert len(train_pd) > 0 and len(test_pd) > 0


def validate_training_sample(
    train_pd: pd.DataFrame,
    test_pd: pd.DataFrame,
    label_col: str,
    feature_cols: list,
    key_cols: list,
) -> None:
    """
    Pre-training validation on the sampled data only:
    - null rates
    - duplicates by key
    - label distribution
    - basic numeric feature distribution summary
    """
    print("\n--- Pre-training validation (on sampled train/test) ---")

    null_counts = train_pd[feature_cols + [label_col]].isna().sum()
    null_pct = (null_counts / len(train_pd) * 100).round(2)
    cols_with_nulls = null_pct[null_pct > 0]
    if len(cols_with_nulls) > 0:
        print("  Null % (train):", cols_with_nulls.to_dict())
    else:
        print("  Null % (train): no nulls in features/label")

    key_cols_avail = [c for c in key_cols if c in train_pd.columns]
    dup_train = train_pd.duplicated(subset=key_cols_avail).sum()
    dup_test = test_pd.duplicated(subset=key_cols_avail).sum()
    print(f"  Duplicate rows (by key): train={dup_train:,}, test={dup_test:,}")

    train_rate = train_pd[label_col].mean()
    test_rate = test_pd[label_col].mean()
    print(f"  Fraud rate: train={train_rate:.4f}, test={test_rate:.4f}")

    numeric = train_pd[feature_cols].select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        dist = pd.DataFrame(
            {
                "min": numeric.min(),
                "max": numeric.max(),
                "mean": numeric.mean().round(4),
                "null_pct": (numeric.isna().sum() / len(train_pd) * 100).round(2),
            }
        )
        print("  Feature distributions (train sample):")
        print(dist.head(15).to_string())
        if len(dist) > 15:
            print("  ...")

    print("--- End pre-training validation ---\n")

