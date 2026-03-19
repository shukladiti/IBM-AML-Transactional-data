import numpy as np
from sklearn.metrics import confusion_matrix


def ks_stat(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for binary classification scores."""
    order = np.argsort(y_score)
    y = y_true[order]
    pos = (y == 1).astype(np.int32)
    neg = (y == 0).astype(np.int32)
    pos_cum = np.cumsum(pos) / max(pos.sum(), 1)
    neg_cum = np.cumsum(neg) / max(neg.sum(), 1)
    return float(np.max(np.abs(pos_cum - neg_cum)))


def precision_recall_at_topk(y_true: np.ndarray, y_score: np.ndarray, k: int):
    """Precision/recall computed on the top-k highest scored instances."""
    k = int(min(max(k, 1), len(y_true)))
    idx = np.argsort(-y_score)[:k]
    yk = y_true[idx]
    prec = float(yk.mean())
    rec = float(yk.sum() / max(y_true.sum(), 1))
    return prec, rec, int(yk.sum())

