import numpy as np
from sklearn.metrics import confusion_matrix


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "youden"):
    """
    Optimize a classification threshold over a fixed grid.

    - youden: sens + spec - 1
    - f2: F2 score (precision weighted more heavily)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score, best_t = -1, 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)

        if metric == "youden":
            score = sens + spec - 1
        else:
            prec = tp / max(tp + fp, 1)
            rec = sens
            score = (5 * prec * rec) / (4 * prec + rec) if (prec + rec) > 0 else 0

        if score > best_score:
            best_score, best_t = score, t

    return float(best_t), float(best_score)


def threshold_for_alert_rate(y_score: np.ndarray, alert_rate: float) -> float:
    """
    Cutoff that flags top `alert_rate` fraction of scores.

    This is a capacity/operations view: "if I can only review X alerts, what threshold do I use?"
    """
    n = len(y_score)
    k = int(max(1, round(n * alert_rate)))
    # Partition returns k-th largest element index: n-k
    return float(np.partition(y_score, n - k)[n - k])

