#!/usr/bin/env python3
"""
lightgbm_pipeline.py — LightGBM (OVA) + inverse-frequency weights + Optuna α-tuning
- Objective: multiclassova to help minority separation.
- Imbalance handling: per-sample inverse-frequency weights (normalized), with α_low/α_high tuned by Optuna.
- Hyperparameter tuning: Optuna TPE + pruning, 5-fold StratifiedKFold.
  * Search objective emphasizes minorities: score = (2*F1_low + 1*F1_med + 2*F1_high)/5.
- Early stopping: on untouched validation set (multi_logloss).
- Outputs: metrics.csv, confusion_matrix.npy, classification_report.txt, shap/, logistic baseline, simulation, lgbm_final.pkl.
Project structure (unchanged):
  ../data/processed/clean_numeric.csv
  ../data/processed/splits_70_15_15_k5.csv
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
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

# -------- logging --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
RND = 42

USE_OVA = True  # True -> objective="multiclassova" ; False -> "multiclass"


# ---------- data loading ----------
def load_and_split_data(base_dir: str = "../data/processed") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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
    y_train = train_df["Severity_ord"].astype(int)
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df["Severity_ord"].astype(int)
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["Severity_ord"].astype(int)

    logger.info(f"Loaded splits: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------- feature helpers ----------
def get_feature_lists(X_df: pd.DataFrame) -> Tuple[List[str], List[str], List[int]]:
    feats = [c for c in X_df.columns if c != "row_id"]
    cat_cols = [c for c in feats if c.endswith(("_lbl", "_bin"))]
    cat_idx = [feats.index(c) for c in cat_cols]
    return feats, cat_cols, cat_idx


# ---------- metrics ----------
def multiclass_brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    onehot = np.zeros_like(probas)
    onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((probas - onehot) ** 2, axis=1)))


# ---------- inverse-frequency weights (+ optional α multipliers) ----------
def inverse_freq_weights(
    y: pd.Series,
    clip: tuple[float, float] = (0.3, 5.0),
    alpha_multipliers: Dict[int, float] | None = None,
) -> np.ndarray:
    """
    Per-sample weights proportional to inverse class frequency, normalized to mean 1, softly clipped.
    Optional α multipliers (e.g., {0: α_low, 2: α_high}) applied AFTER normalization.
    Wider upper clip (5.0) to allow stronger minority emphasis than before.
    """
    counts = y.value_counts().to_dict()
    total = len(y)
    inv = {k: total / v for k, v in counts.items()}
    mean_w = np.mean(list(inv.values()))
    base = {k: v / mean_w for k, v in inv.items()}   # mean ≈ 1

    w = y.map(base).astype(float).values
    if alpha_multipliers:
        w *= y.map(lambda c: alpha_multipliers.get(int(c), 1.0)).astype(float).values

    lo, hi = clip
    w = np.clip(w, lo, hi)
    logger.info(f"Inverse-freq weights with clip={clip}, α={alpha_multipliers}; base={base}")
    return w


# ---------- Optuna tuning (model params + α multipliers) ----------
def tune_lgbm_optuna_alpha(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 120,
    cv_folds: int = 5,
    random_state: int = RND,
) -> Dict[str, Any]:
    """
    Optuna TPE hyperparameter + imbalance tuning:
      - LightGBM params
      - α_low, α_high per-class multipliers applied to inverse-frequency sample weights
    CV scoring emphasizes minorities: (2*F1_low + 1*F1_med + 2*F1_high)/5.
    Returns:
      {
        "lgbm_params": {...},
        "alpha_low": float,
        "alpha_high": float,
        "n_estimators": int
      }
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except Exception as e:
        raise ImportError("Optuna not installed. Install with: pip install optuna") from e

    feats = [c for c in X_train.columns if c != "row_id"]
    X = X_train[feats].reset_index(drop=True)
    y = y_train.reset_index(drop=True)
    n_classes = int(y.nunique())
    objective_name = "multiclassova" if USE_OVA else "multiclass"

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def obj(trial: "optuna.trial.Trial") -> float:
        # ---- LightGBM hyperparameters ----
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 223, step=16),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100, step=5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 8e-3, 0.10, log=True),
            "n_estimators": 3500,  # ES will cut
        }
        # ---- Imbalance hyperparameters ----
        alpha_low  = trial.suggest_float("alpha_low",  1.0, 5.0)
        alpha_high = trial.suggest_float("alpha_high", 1.0, 5.0)

        # Tiny penalty to avoid extreme α
        alpha_penalty = 0.0008 * ((alpha_low - 1.0) + (alpha_high - 1.0))

        fold_scores, best_iters = [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # Per-fold inverse-freq weights + α multipliers for low/high
            w_tr = inverse_freq_weights(
                y_tr, clip=(0.3, 5.0),
                alpha_multipliers={0: alpha_low, 2: alpha_high}
            )

            clf = LGBMClassifier(
                objective=objective_name,
                num_class=n_classes,
                random_state=random_state,
                n_jobs=-1,
                **params,
            )
            callbacks = [lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=0)]
            clf.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="multi_logloss",
                callbacks=callbacks,
            )

            prob_va = clf.predict_proba(X_va)
            pred_va = np.argmax(prob_va, axis=1)

            # Minority-weighted fold score
            f1_per = f1_score(y_va, pred_va, average=None, labels=[0, 1, 2])
            fold_score = (2.0 * f1_per[0] + 1.0 * f1_per[1] + 2.0 * f1_per[2]) / 5.0
            fold_scores.append(float(fold_score) - alpha_penalty)

            best_iters.append(int(getattr(clf, "best_iteration_", params["n_estimators"])))
            trial.report(fold_scores[-1], step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("avg_best_iter", int(np.round(np.mean(best_iters))))
        return mean_score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state, n_startup_trials=16),
        pruner=MedianPruner(n_warmup_steps=2),
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params.copy()
    avg_iter = int(study.best_trial.user_attrs.get("avg_best_iter", 900))
    n_estimators = int(np.clip(int(1.15 * avg_iter), 250, 2500))

    out = {
        "lgbm_params": {
            k: v for k, v in best.items()
            if k in {
                "num_leaves", "max_depth", "min_child_samples", "reg_alpha",
                "reg_lambda", "subsample", "colsample_bytree", "learning_rate"
            }
        },
        "alpha_low": float(best["alpha_low"]),
        "alpha_high": float(best["alpha_high"]),
        "n_estimators": n_estimators,
    }
    logger.info("Optuna best (minority-weighted CV F1): %.5f", study.best_value)
    logger.info("Optuna best params (incl. α): %s", out)
    return out


# ---------- final training ----------
def train_final_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best: Dict[str, Any],
) -> LGBMClassifier:
    feats = [c for c in X_train.columns if c != "row_id"]
    X_tr, X_va = X_train[feats], X_val[feats]

    alpha_low, alpha_high = best["alpha_low"], best["alpha_high"]
    w_tr = inverse_freq_weights(
        y_train, clip=(0.3, 5.0),
        alpha_multipliers={0: alpha_low, 2: alpha_high}
    )

    objective_name = "multiclassova" if USE_OVA else "multiclass"
    params = best["lgbm_params"].copy()
    params["n_estimators"] = best["n_estimators"]

    model = LGBMClassifier(
        objective=objective_name,
        num_class=int(y_train.nunique()),
        random_state=RND,
        n_jobs=-1,
        **params,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=0)]
    model.fit(
        X_tr, y_train,
        sample_weight=w_tr,
        eval_set=[(X_va, y_val)],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )
    logger.info("Trained final LightGBM. Best iter: %s", getattr(model, "best_iteration_", None))
    return model


# ---------- SHAP ----------
def compute_and_save_shap(model: LGBMClassifier, X: pd.DataFrame, feature_names: list, out_dir: Path):
    explainer = shap.TreeExplainer(model, feature_perturbation="auto")
    shap_values_raw = explainer.shap_values(X)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    else:
        raise TypeError(f"Unexpected SHAP output type/shape: {type(shap_values_raw)}")

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
        importances[f"class_{class_idx}"] = np.abs(class_shap).mean(axis=0)
    pd.DataFrame(importances, index=feature_names).to_csv(out_dir / "shap_mean_abs_per_class.csv")
    logger.info("Saved SHAP outputs to %s", out_dir)


# ---------- logistic baseline ----------
def fit_logistic_and_compare(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, feature_names: list, out_dir: Path):
    feats = [c for c in X_train.columns if c != "row_id"]
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[feats])
    clf = LogisticRegression(solver="lbfgs", penalty="l2", C=1.0, max_iter=2000, random_state=RND)
    clf.fit(X_tr_scaled, y_train)
    coefs = pd.DataFrame(clf.coef_.T, index=feats, columns=[f"class_{i}_coef" for i in range(clf.coef_.shape[0])])
    coefs.to_csv(out_dir / "logistic_coefs.csv")
    logger.info("Saved logistic coefficients to %s", out_dir)
    return clf, scaler, coefs


# ---------- simulation ----------
def simulate_feature_shift(model: LGBMClassifier, X: pd.DataFrame, feature: str, delta: float, target_class: int = 2) -> float:
    feats = [c for c in X.columns if c != "row_id"]
    if feature not in feats:
        raise ValueError(f"{feature} not a feature")
    X_copy = X.copy()
    X_copy[feature] = X_copy[feature] + delta
    prob_before = model.predict_proba(X[feats])[:, target_class]
    prob_after = model.predict_proba(X_copy[feats])[:, target_class]
    return float(np.mean(prob_after - prob_before))


# ---------- evaluation ----------
def evaluate(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path):
    feats = [c for c in X_test.columns if c != "row_id"]
    probs = model.predict_proba(X_test[feats])
    preds = probs.argmax(axis=1)

    pred_counts = pd.Series(preds).value_counts().sort_index()
    logger.info(f"Pred counts by class: {pred_counts.to_dict()}")

    f1 = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score(pd.get_dummies(y_test).values, probs, multi_class="ovr", average="macro")
    brier = multiclass_brier_score(y_test.values, probs)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["low","med","high"], output_dict=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series({"macro_f1": f1, "macro_auc": auc, "brier": brier}).to_csv(out_dir / "metrics.csv")
    np.save(out_dir / "confusion_matrix.npy", cm)
    with open(out_dir / "classification_report.txt", "w") as fh:
        fh.write(report)
    logger.info("Evaluation saved to %s. Metrics: macro_f1=%.4f macro_auc=%.4f brier=%.4f", out_dir, f1, auc, brier)
    return {"macro_f1": f1, "macro_auc": auc, "brier": brier}, cm


# ---------- main ----------
def main():
    out_dir = Path("outputs_lightgbm")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()
    logger.info("Train label counts: %s", dict(pd.Series(y_train).value_counts()))

    # 1) Optuna: jointly tune LGBM params + α_low/α_high (inverse-frequency base)
    best = tune_lgbm_optuna_alpha(X_train, y_train, n_trials=120, cv_folds=5, random_state=RND)

    # 2) Final training on full train, early stopping on val
    model = train_final_lgbm(X_train, y_train, X_val, y_val, best)

    # Save model (threshold_eval.py will load this)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "lgbm_final.pkl")

    # 3) Evaluate on untouched test
    evaluate(model, X_test, y_test, out_dir)

    # 4) SHAP explainability
    feats, _, _ = get_feature_lists(X_train)
    compute_and_save_shap(model, X_test[feats], feature_names=feats, out_dir=out_dir / "shap")

    # 5) Logistic baseline
    clf, scaler, coefs = fit_logistic_and_compare(X_train, y_train, X_test, feats, out_dir)
    shap_imp = pd.read_csv(out_dir / "shap" / "shap_mean_abs_per_class.csv", index_col=0)
    pd.concat([coefs.abs(), shap_imp.mean(axis=1).rename("shap_mean_abs_avg")], axis=1) \
      .sort_values(by="shap_mean_abs_avg", ascending=False) \
      .to_csv(out_dir / "features_compare_logistic_shap.csv")

    # 6) Simulation example
    if "Sleep_Hours" in X_test.columns:
        delta = simulate_feature_shift(model, X_test, "Sleep_Hours", delta=1.0, target_class=2)
        logger.info("Average change in P(high) after +1 Sleep_Hours: %0.5f", delta)
        with open(out_dir / "simulation_sleep_delta.txt", "w") as fh:
            fh.write(f"avg_delta_P_high_after_+1_sleep: {delta}\n")

    logger.info("Pipeline done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
