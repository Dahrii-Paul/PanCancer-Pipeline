import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    roc_auc_score
)


def multiclass_specificity(y_true, y_pred, **kwargs):
    """Mean specificity across all classes (used as PyCaret custom metric)."""
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    specificities = []
    for cm in mcm:
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificities.append(spec)
    return np.mean(specificities)


def compute_classwise_metrics(y_true, y_pred, y_prob, class_names, model_label):
    """
    Computes per-class: Sensitivity, Specificity, Precision, F1, MCC, AUC.
    Returns a DataFrame with one row per class.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    rows = []

    for i, cm in enumerate(mcm):
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision   = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) != 0 else 0
        )

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc   = (tp * tn - fp * fn) / denom if denom != 0 else 0

        try:
            y_true_bin = (np.array(y_true) == i).astype(int)
            auc = roc_auc_score(y_true_bin, y_prob[:, i])
        except Exception:
            auc = np.nan

        rows.append({
            'Model':       model_label,
            'Class':       class_names[i],
            'Sensitivity': round(sensitivity, 4),
            'Specificity': round(specificity, 4),
            'Precision':   round(precision, 4),
            'F1_Score':    round(f1, 4),
            'MCC':         round(mcc, 4),
            'AUC':         round(auc, 4) if not np.isnan(auc) else np.nan
        })

    return pd.DataFrame(rows)


def save_confusion_matrix(y_true, y_pred, class_names, model_label, confusion_dir):
    """Saves the confusion matrix as a CSV."""
    cm     = confusion_matrix(y_true, y_pred)
    cm_df  = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{confusion_dir}/{model_label}_confusion_matrix.csv")
    print(f"  Confusion matrix saved: {model_label}")
