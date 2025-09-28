#!/usr/bin/env python3
"""
threshold_eval.py
Post-hoc per-class thresholding for imbalanced multiclass, kept separate from training.

What it does
- Loads the trained LightGBM model from outputs_lightgbm/lgbm_final.pkl
- Loads the same data splits via lightgbm_pipeline.load_and_split_data()
- Learns class-specific thresholds on the validation set (one-vs-rest F1 maximization)
- Applies them to the test set using an argmax over p_i / t_i decision rule
- Saves metrics, confusion matrix, classification report, and thresholds to outputs_lightgbm/thresholded/

Run:
    python threshold_eval.py

Requires:
    - lightgbm_pipeline.py (for load_and_split_data)
    - A trained model at outputs_lightgbm/lgbm_final.pkl (produced by your SMOTE-NC pipeline)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

# Import ONLY the data loader to ensure identical splits/base_dir
from train_lightgbm import load_and_split_data

OUT_DIR = Path("outputs_lightgbm")
MODEL_PATH = OUT_DIR / "lgbm_final.pkl"
THR_DIR = OUT_DIR / "thresholded"


def multiclass_brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Multiclass Brier: mean over samples of sum_k (p_k - o_k)^2."""
    onehot = np.zeros_like(probas)
    onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((probas - onehot) ** 2, axis=1)))


def find_class_thresholds(probs_val: np.ndarray, y_val: np.ndarray, grid=None) -> dict[int, float]:
    """
    One-vs-rest per-class threshold search to maximize F1 on validation.
    For each class c, choose t_c maximizing F1( y==c vs 1[p_c >= t_c] ).
    """
    if grid is None:
        # reasonably wide grid; adjust if needed
        grid = np.linspace(0.2, 0.8, 31)

    n_classes = probs_val.shape[1]
    thresholds: dict[int, float] = {}
    for c in range(n_classes):
        y_bin = (y_val == c).astype(int)
        p_c = probs_val[:, c]
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_bin, (p_c >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = float(best_t)
    return thresholds


def argmax_with_thresholds(probs: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    """
    Single-label decision: pick argmax over p_i / t_i.
    Lower t_i lowers the bar for class i (helpful for minority classes).
    """
    t = np.array([thresholds[i] for i in range(probs.shape[1])], dtype=float)
    t[t <= 1e-9] = 1e-9
    adjusted = probs / t[None, :]
    return adjusted.argmax(axis=1)


def main():
    # 1) Load trained model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}. Run your training pipeline first.")
    model = joblib.load(MODEL_PATH)

    # 2) Load the same splits (no retrain)
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    feats = [c for c in X_train.columns if c != "row_id"]

    # 3) Learn thresholds on validation
    probs_val = model.predict_proba(X_val[feats])
    thresholds = find_class_thresholds(probs_val, y_val.values)

    # 4) Apply to test
    probs_test = model.predict_proba(X_test[feats])
    preds_thr = argmax_with_thresholds(probs_test, thresholds)

    # 5) Metrics: Macro-F1 (thresholded predictions), AUC/Brier (raw probs)
    macro_f1 = f1_score(y_test, preds_thr, average="macro")
    macro_auc = roc_auc_score(pd.get_dummies(y_test).values, probs_test, multi_class="ovr", average="macro")
    brier = multiclass_brier_score(y_test.values, probs_test)
    cm = confusion_matrix(y_test, preds_thr)

    # Named classes (0,1,2) -> ["low","med","high"]; adjust if your label map changes
    report_txt = classification_report(y_test, preds_thr, target_names=["low", "med", "high"], zero_division=0, output_dict=False)

    # 6) Save outputs
    THR_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series({"macro_f1": macro_f1, "macro_auc": macro_auc, "brier": brier}).to_csv(THR_DIR / "metrics_thresholded.csv")
    np.save(THR_DIR / "confusion_matrix_thresholded.npy", cm)
    with open(THR_DIR / "classification_report_thresholded.txt", "w") as fh:
        fh.write(report_txt)
    pd.Series(thresholds).to_csv(THR_DIR / "class_thresholds.csv")

    # 7) Console summary
    pred_counts = pd.Series(preds_thr).value_counts().sort_index().to_dict()
    print(f"[threshold_eval] Saved thresholded outputs to: {THR_DIR}")
    print(f"macro_f1={macro_f1:.4f}  macro_auc={macro_auc:.4f}  brier={brier:.4f}  pred_counts={pred_counts}")
    print(f"class_thresholds: {thresholds}")


if __name__ == "__main__":
    main()
