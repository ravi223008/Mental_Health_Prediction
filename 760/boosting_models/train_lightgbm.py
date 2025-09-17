#!/usr/bin/env python3
"""
lightgbm_pipeline.py
- Hyperparam tuning (RandomizedSearchCV on train)
- Fit final LightGBM with early stopping on val
- Evaluate on test: Macro-F1, Macro-AUC, multiclass Brier score, confusion matrix
- SHAP explanations (global, per-class, per-individual)
- Compare with L2 multinomial logistic regression coefficients
- Simple simulation utilities to measure effect of changing a feature
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import lightgbm as lgb

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
RND = 42

# ---------- Utility / data-loading (adapted from your earlier code) ----------
def load_and_split_data(base_dir: str = "../data/processed") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load clean_numeric.csv and splits_70_15_15_k5.csv, return X_train,y_train,X_val,y_val,X_test,y_test (X includes 'row_id')."""
    data_path = Path(base_dir) / "clean_numeric.csv"
    splits_path = Path(base_dir) / "splits_70_15_15_k5.csv"
    df = pd.read_csv(data_path).reset_index().rename(columns={"index": "row_id"})
    splits = pd.read_csv(splits_path)
    df = df.merge(splits, on="row_id", how="left")
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    drop_cols = [c for c in ["User_ID", "Severity_ord", "split", "cv_fold"] if c in train_df.columns]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["Severity_ord"]
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df["Severity_ord"]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["Severity_ord"]

    logger.info(f"Loaded splits: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# ---------- metrics ----------
def multiclass_brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    """
    Multiclass Brier score: average over samples of sum_k (p_k - o_k)^2
    where o_k is 1 for true class k else 0.
    Returns scalar (lower is better).
    """
    n_samples, n_classes = probas.shape
    onehot = np.zeros_like(probas)
    onehot[np.arange(n_samples), y_true] = 1
    bs = np.mean(np.sum((probas - onehot) ** 2, axis=1))
    return bs

# ---------- hyperparam tuning on training set ----------
def tune_lgbm(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 40, cv_folds: int = 5, random_state: int = RND) -> Dict[str, Any]:
    """Randomized search for useful LightGBM parameters (small n_iter for speed)."""
    feats = [c for c in X_train.columns if c != "row_id"]
    X = X_train[feats].values
    y = y_train.values

    param_dist = {
        "num_leaves": [31, 63, 127],
        "max_depth": [5, 7, 9, -1],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 400],
        "min_child_samples": [5, 10, 20],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 0.5],
    }

    base = LGBMClassifier(objective="multiclass", num_class=int(np.unique(y).size), random_state=random_state, verbosity=-1)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rnd = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter, scoring="f1_macro", cv=skf, n_jobs=-1, verbose=2, random_state=random_state)
    rnd.fit(X, y)
    logger.info("RandomizedSearchCV best score: %s", rnd.best_score_)
    logger.info("RandomizedSearchCV best params: %s", rnd.best_params_)
    return rnd.best_params_

# ---------- training final model (use val for early stopping) ----------
def train_final_lgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: Dict[str, Any], sample_weight: Optional[np.ndarray] = None) -> LGBMClassifier:
    feats = [c for c in X_train.columns if c != "row_id"]
    X_train_f = X_train[feats]
    X_val_f = X_val[feats]

    # Build model with chosen params but increase n_estimators for final training
    model = LGBMClassifier(objective="multiclass", num_class=int(y_train.nunique()), random_state=RND, **params)
    # if params didn't include n_estimators, set a safe default
    if "n_estimators" not in params:
        model.set_params(n_estimators=1000)

    callbacks = [lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
    model.fit(X_train_f, y_train, sample_weight=sample_weight, eval_set=[(X_val_f, y_val)], eval_metric="multi_logloss", callbacks=callbacks)
    logger.info("Trained final LightGBM. Best iter: %s", getattr(model, "best_iteration_", None))
    return model

# ---------- SHAP explainability ----------
def compute_and_save_shap(model: LGBMClassifier, X: pd.DataFrame, feature_names: list, out_dir: Path):
    """Compute SHAP values and produce summary plots per class + save arrays."""
    explainer = shap.TreeExplainer(model, feature_perturbation="auto")
    shap_values_raw = explainer.shap_values(X)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    else:
        raise TypeError(
            f"Unexpected SHAP output type/shape: {type(shap_values_raw)} with "
            f"ndim={getattr(shap_values_raw, 'ndim', 'n/a')}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shap_values.npy", np.array(shap_values, dtype=object), allow_pickle=True)

    for class_idx, class_shap in enumerate(shap_values):
        plt.figure(figsize=(8, 6))
        shap.summary_plot(class_shap, X, feature_names=feature_names, show=False)
        plt.title(f"SHAP summary (class {class_idx})")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_class_{class_idx}.png")
        plt.close()

    importances = {}
    for class_idx, class_shap in enumerate(shap_values):
        mean_abs = np.abs(class_shap).mean(axis=0)
        importances[f"class_{class_idx}"] = mean_abs

    imp_df = pd.DataFrame(importances, index=feature_names)
    imp_df.to_csv(out_dir / "shap_mean_abs_per_class.csv")
    logger.info("Saved SHAP outputs to %s", out_dir)


# ---------- compare with logistic regression ----------
def fit_logistic_and_compare(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, feature_names: list, out_dir: Path):
    feats = [c for c in X_train.columns if c != "row_id"]
    # standardize features for logistic
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[feats])
    clf = LogisticRegression(multi_class="multinomial", solver="saga", penalty="l2", C=1.0, max_iter=5000, random_state=RND)
    clf.fit(X_tr_scaled, y_train)
    coefs = pd.DataFrame(clf.coef_.T, index=feats, columns=[f"class_{i}_coef" for i in range(clf.coef_.shape[0])])
    coefs.to_csv(out_dir / "logistic_coefs.csv")
    logger.info("Saved logistic coefficients to %s", out_dir)
    return clf, scaler, coefs

# ---------- simulation: change feature value and measure delta P(high) ----------
def simulate_feature_shift(model: LGBMClassifier, X: pd.DataFrame, feature: str, delta: float, target_class: int = 2) -> float:
    """Return average change in predicted probability for target_class after adding delta to `feature` in X (works on X copy)."""
    feats = [c for c in X.columns if c != "row_id"]
    X_copy = X.copy()
    if feature not in feats:
        raise ValueError(f"{feature} not a feature")
    # apply delta (could be positive or negative); here we add delta to all rows (simple simulation)
    X_copy[feature] = X_copy[feature] + delta
    prob_before = model.predict_proba(X[feats])[:, target_class]
    prob_after = model.predict_proba(X_copy[feats])[:, target_class]
    return float(np.mean(prob_after - prob_before))

# ---------- evaluation wrapper ----------
def evaluate(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path):
    feats = [c for c in X_test.columns if c != "row_id"]
    X_test_f = X_test[feats]
    probs = model.predict_proba(X_test_f)
    preds = probs.argmax(axis=1)
    f1 = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score(pd.get_dummies(y_test).values, probs, multi_class="ovr", average="macro")
    brier = multiclass_brier_score(y_test.values, probs)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["low","med","high"], output_dict=False)
    # save results
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"macro_f1": f1, "macro_auc": auc, "brier": brier}
    pd.Series(metrics).to_csv(out_dir / "metrics.csv")
    np.save(out_dir / "confusion_matrix.npy", cm)
    with open(out_dir / "classification_report.txt", "w") as fh:
        fh.write(report)
    logger.info("Evaluation saved to %s. Metrics: %s", out_dir, metrics)
    return metrics, cm

# ---------- main flow ----------
def main():
    out_dir = Path("outputs_lightgbm")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()

    # quick sanity
    logger.info("Train label counts: %s", dict(pd.Series(y_train).value_counts()))
    # 1) tune on train with 5-fold stratified CV
    best_params = tune_lgbm(X_train, y_train, n_iter=30, cv_folds=5, random_state=RND)

    # 2) compute combined sample weights (same strategy you used earlier)
    # If you already have get_sample_weights in your project, call that; here we assume class-only normalized.
    counts = y_train.value_counts().to_dict()
    total = len(y_train)
    class_weights_raw = {k: total / v for k, v in counts.items()}
    mean_w = np.mean(list(class_weights_raw.values()))
    class_weights_norm = {k: v / mean_w for k, v in class_weights_raw.items()}
    sample_weights = y_train.map(class_weights_norm).values  # optionally multiply with fairness weights if available
    # normalize to mean 1 and clip to [0.5, 2.0]
    sample_weights = sample_weights / sample_weights.mean()
    sample_weights = np.clip(sample_weights, 0.5, 2.0)

    # 3) train final LGBM with early stopping on val
    model = train_final_lgbm(X_train, y_train, X_val, y_val, best_params, sample_weight=sample_weights)

    # save model
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "lgbm_final.pkl")

    # 4) evaluation
    metrics, cm = evaluate(model, X_test, y_test, out_dir)

    # 5) SHAP explainability and save plots
    feats = [c for c in X_train.columns if c != "row_id"]
    compute_and_save_shap(model, X_test[feats], feature_names=feats, out_dir=out_dir / "shap")

    # 6) Logistic regression baseline compare: train logistic on train and save coefficients
    clf, scaler, coefs = fit_logistic_and_compare(X_train, y_train, X_test, feats, out_dir)
    # compare top features: combine shap mean abs with logistic coef magnitudes
    shap_imp = pd.read_csv(out_dir / "shap" / "shap_mean_abs_per_class.csv", index_col=0)
    # Save a simple compare file
    compare = pd.concat([coefs.abs(), shap_imp.mean(axis=1).rename("shap_mean_abs_avg")], axis=1).sort_values(by="shap_mean_abs_avg", ascending=False)
    compare.to_csv(out_dir / "features_compare_logistic_shap.csv")

    # 7) simulation example: +1 hour sleep (if Sleep_Hours exists)
    if "Sleep_Hours" in X_test.columns:
        delta = simulate_feature_shift(model, X_test, "Sleep_Hours", delta=1.0, target_class=2)
        logger.info("Average change in P(high) after +1 Sleep_Hours: %0.5f", delta)
        with open(out_dir / "simulation_sleep_delta.txt", "w") as fh:
            fh.write(f"avg_delta_P_high_after_+1_sleep: {delta}\n")
    logger.info("Pipeline done. Outputs in %s", out_dir)

if __name__ == "__main__":
    main()
