"""
metrics.py — Multi-Task Classification & Regression Evaluation Suite
"""
from typing import Any, Dict, List

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

CLASSIFICATION_TASKS = ["bioavailability", "bbb", "herg"]
REGRESSION_TASKS = ["qed", "sascore"]


def evaluate_multitask_predictions(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    mask_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Evaluate all 5 ADMET & physicochemical tasks strictly on observed (masked=True) entries.

    Expected prediction ranges:
      - bioavailability: probability in [0, 1] (or [0, 100]% automatically normalized to [0, 1])
      - bbb, herg: probabilities in [0, 1]
      - qed: scalar in [0, 1]
      - sascore: scalar in [1, 10]
    """
    metrics: Dict[str, Any] = {}
    roc_aucs: List[float] = []
    reg_r2s: List[float] = []

    for task in CLASSIFICATION_TASKS:
        mask = np.asarray(mask_dict[task], dtype=bool)
        y_true = np.asarray(y_true_dict[task], dtype=np.float64)[mask]
        y_prob = np.asarray(y_pred_dict[task], dtype=np.float64)[mask]
        if task == "bioavailability" and len(y_prob) > 0 and y_prob.max() > 1.0 + 1e-5:
            y_prob = y_prob / 100.0
        y_prob = np.clip(y_prob, 0.0, 1.0)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            metrics[task] = {
                "roc_auc": 0.5,
                "pr_auc": 0.5,
                "f1": 0.0,
                "balanced_accuracy": 0.5,
                "n_samples": int(len(y_true)),
            }
            roc_aucs.append(0.5)
            continue

        y_bin_pred = (y_prob >= 0.5).astype(int)
        y_bin_true = (y_true >= 0.5).astype(int)
        roc = float(roc_auc_score(y_bin_true, y_prob))
        pr = float(average_precision_score(y_bin_true, y_prob))
        f1 = float(f1_score(y_bin_true, y_bin_pred, zero_division=0))
        bacc = float(balanced_accuracy_score(y_bin_true, y_bin_pred))

        metrics[task] = {
            "roc_auc": roc,
            "pr_auc": pr,
            "f1": f1,
            "balanced_accuracy": bacc,
            "n_samples": int(len(y_true)),
        }
        roc_aucs.append(roc)

    for task in REGRESSION_TASKS:
        mask = np.asarray(mask_dict[task], dtype=bool)
        y_true = np.asarray(y_true_dict[task], dtype=np.float64)[mask]
        y_pred = np.asarray(y_pred_dict[task], dtype=np.float64)[mask]

        if len(y_true) < 2:
            metrics[task] = {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "pearson_r": 0.0, "n_samples": int(len(y_true))}
            reg_r2s.append(0.0)
            continue

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
            r_val = float(pearsonr(y_true, y_pred)[0])
        else:
            r_val = 0.0

        metrics[task] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": r_val,
            "n_samples": int(len(y_true)),
        }
        reg_r2s.append(r2)

    macro_roc = float(np.mean(roc_aucs)) if roc_aucs else 0.5
    mean_r2 = float(np.mean(reg_r2s)) if reg_r2s else 0.0
    composite = 0.7 * macro_roc + 0.3 * max(0.0, mean_r2)

    metrics["aggregate"] = {
        "macro_roc_auc": macro_roc,
        "mean_reg_r2": mean_r2,
        "composite_val_score": float(composite),
    }
    return metrics


def format_metrics_markdown(all_model_metrics: Dict[str, Dict[str, Any]]) -> str:
    """Format a Markdown benchmark comparison table across models."""
    lines = [
        "| Model | Bioavail. ROC-AUC | BBB ROC-AUC | hERG ROC-AUC | Macro ROC-AUC | QED $R^2$ (MAE) | SAScore $R^2$ (MAE) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for model_name, m in all_model_metrics.items():
        bio = m["bioavailability"]["roc_auc"]
        bbb = m["bbb"]["roc_auc"]
        herg = m["herg"]["roc_auc"]
        macro = m["aggregate"]["macro_roc_auc"]
        qed_r2 = m["qed"]["r2"]
        qed_mae = m["qed"]["mae"]
        sa_r2 = m["sascore"]["r2"]
        sa_mae = m["sascore"]["mae"]
        lines.append(
            f"| **{model_name}** | {bio:.4f} | {bbb:.4f} | {herg:.4f} | **{macro:.4f}** | "
            f"{qed_r2:.4f} ({qed_mae:.3f}) | {sa_r2:.4f} ({sa_mae:.3f}) |"
        )
    return "\n".join(lines)
