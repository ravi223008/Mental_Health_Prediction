#!/usr/bin/env python3
"""
XGBoost pipeline — Optuna tuning + inverse-frequency weights (with alpha multipliers)
- Multiclass severity (0: low, 1: med, 2: high) via objective="multi:softprob"
- No SMOTE; class imbalance handled via per-sample inverse-frequency weights and tuned α_low/α_high
- Optuna TPE tunes XGBoost hyperparams + α multipliers; CV score emphasizes minorities:
    score = (2*F1_low + 1*F1_med + 2*F1_high) / 5
- Early stopping on untouched validation split
- Test metrics: Macro-F1, Macro-AUC (OvR), Brier, Accuracy
- Artifacts: metrics.csv, classification_report.txt, confusion_matrix.npy, per_class_auc.json, pred_counts.json
- SHAP: global per-class summary plots + mean |SHAP| per class
- Logistic baseline (multinomial L2) coefficients saved

Run: python train_xgboost_optuna.py
"""

import inspect
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# XGBoost
XgbEarlyStopping = None
try:
    from xgboost import XGBClassifier
    try:
        from xgboost.callback import EarlyStopping as XgbEarlyStopping
    except Exception:
        XgbEarlyStopping = None
except Exception as e:
    raise SystemExit("xgboost not installed. Install with: pip install xgboost") from e

try:
    _XGB_FIT_PARAMS = inspect.signature(XGBClassifier().fit).parameters
    _XGB_FIT_ACCEPTS_CALLBACKS = "callbacks" in _XGB_FIT_PARAMS
    _XGB_FIT_ACCEPTS_EARLY_STOPPING = "early_stopping_rounds" in _XGB_FIT_PARAMS
except Exception:
    _XGB_FIT_ACCEPTS_CALLBACKS = True
    _XGB_FIT_ACCEPTS_EARLY_STOPPING = True

# SHAP
try:
    import shap
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit("shap and matplotlib required. Install with: pip install shap matplotlib") from e

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except Exception as e:
    raise SystemExit("optuna not installed. Install with: pip install optuna") from e


# -------------------- paths (edit if needed) --------------------
ENGINEERED_CSV = "data/processed/clean_numeric_engineered_v2.csv"
SPLITS_CSV     = "data/processed/splits_70_15_15_k5.csv"
OUT_DIR        = Path("outputs_xgb_optuna")

# -------------------- logging / seed --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xgb-optuna")
RND = 42
np.random.seed(RND)


# --------------------- data loading ---------------------
def load_with_splits(engineered_csv: str, splits_csv: Optional[str]) -> pd.DataFrame:
    ec = Path(engineered_csv)
    sc = Path(splits_csv) if splits_csv else None

    if not ec.exists():
        raise FileNotFoundError(f"Engineered CSV not found: {ec.resolve()}")
    if sc and not sc.exists():
        raise FileNotFoundError(f"Splits CSV not found: {sc.resolve()}")

    df = pd.read_csv(ec)
    if "row_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_id"})

    if sc:
        splits = pd.read_csv(sc)
        if "row_id" not in splits.columns:
            splits = splits.reset_index().rename(columns={"index": "row_id"})
        if "split" not in splits.columns:
            raise ValueError("Splits CSV must contain a 'split' column (train/val/test).")
        df = df.merge(splits[["row_id", "split"]], on="row_id", how="left")

    if "split" not in df.columns:
        raise ValueError("No 'split' column found after merging.")
    if "Severity_ord" not in df.columns:
        raise ValueError("Target 'Severity_ord' missing in engineered CSV.")

    for part in ("train", "val", "test"):
        if part not in set(df["split"].unique()):
            raise ValueError(f"Missing split '{part}' in data.")

    return df


def make_partitions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series,
                                               pd.DataFrame, pd.Series,
                                               pd.DataFrame, pd.Series]:
    drop_cols = [c for c in ["User_ID", "Severity_ord", "split", "cv_fold"] if c in df.columns]

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df["Severity_ord"].astype(int)

    X_val = val_df.drop(columns=drop_cols, errors="ignore")
    y_val = val_df["Severity_ord"].astype(int)

    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df["Severity_ord"].astype(int)

    uniq = np.sort(pd.unique(df["Severity_ord"].astype(int)))
    if not np.array_equal(uniq, np.array([0, 1, 2])):
        raise ValueError(f"Expected labels {0,1,2}, got {uniq.tolist()}")

    logger.info("Shapes: X_train=%s, X_val=%s, X_test=%s", X_train.shape, X_val.shape, X_test.shape)
    logger.info("Class counts (train): %s", y_train.value_counts().sort_index().to_dict())
    logger.info("Class counts (val)  : %s", y_val.value_counts().sort_index().to_dict())
    logger.info("Class counts (test) : %s", y_test.value_counts().sort_index().to_dict())

    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------- imbalance & metrics utilities ---------------
def inverse_freq_weights(
    y: pd.Series,
    clip=(0.3, 5.0),
    alpha_multipliers: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Per-sample weights proportional to inverse class frequency, normalized to mean ≈1, softly clipped.
    Optional α multipliers (e.g., {0: α_low, 2: α_high}) applied after normalization.
    """
    counts = y.value_counts().to_dict()
    total = len(y)
    inv = {k: total / v for k, v in counts.items()}
    mean_w = np.mean(list(inv.values()))
    base = {k: v / mean_w for k, v in inv.items()}  # mean ≈ 1

    w = y.map(base).astype(float).values
    if alpha_multipliers:
        w *= y.map(lambda c: alpha_multipliers.get(int(c), 1.0)).astype(float).values

    lo, hi = clip
    w = np.clip(w, lo, hi)
    return w


def multiclass_brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    onehot = np.zeros_like(probas)
    onehot[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((probas - onehot) ** 2, axis=1)))


def _fit_with_early_stopping(
    clf: XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray,
    eval_set: List[Tuple[pd.DataFrame, pd.Series]],
    rounds: int = 100,
    verbose: bool = False,
) -> None:
    fit_kwargs: Dict[str, Any] = {
        "sample_weight": sample_weight,
        "eval_set": eval_set,
        "verbose": verbose,
    }
    callbacks_list: Optional[List[Any]] = None
    if XgbEarlyStopping is not None:
        callbacks_list = [XgbEarlyStopping(rounds=rounds, save_best=True)]

    if _XGB_FIT_ACCEPTS_CALLBACKS and callbacks_list is not None:
        fit_kwargs["callbacks"] = callbacks_list
    elif _XGB_FIT_ACCEPTS_EARLY_STOPPING:
        fit_kwargs["early_stopping_rounds"] = rounds
    else:
        if callbacks_list is not None:
            try:
                clf.set_params(callbacks=callbacks_list)
            except (TypeError, ValueError, AttributeError):
                pass
        try:
            clf.set_params(early_stopping_rounds=rounds)
        except (TypeError, ValueError, AttributeError):
            pass

    clf.fit(X, y, **fit_kwargs)


def _extract_best_iteration(clf: XGBClassifier, default: int) -> int:
    for attr in ("best_iteration", "best_iteration_"):
        val = getattr(clf, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    booster_fn = getattr(clf, "get_booster", None)
    if callable(booster_fn):
        booster = booster_fn()
        val = getattr(booster, "best_iteration", None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    return int(default)


# ----------------- Optuna tuning (model + α) -----------------
def tune_xgb_optuna_alpha(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    cv_folds: int = 5,
    random_state: int = RND,
) -> Dict[str, Any]:
    """
    Jointly tune XGBoost hyperparameters and α_low/α_high multipliers (applied to inverse-frequency weights).
    CV objective emphasizes minorities:
        score = (2*F1_low + 1*F1_med + 2*F1_high)/5
    Returns dict with {'xgb_params': {...}, 'alpha_low': float, 'alpha_high': float, 'n_estimators': int}
    """
    feats = list(X_train.columns)
    X = X_train.reset_index(drop=True)
    y = y_train.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def obj(trial: "optuna.trial.Trial") -> float:
        # ---- XGBoost hyperparams ----
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,

            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "n_estimators": 3500,  # big cap; ES cuts
            "eval_metric": "mlogloss",
        }

        # ---- Imbalance hyperparams ----
        alpha_low = trial.suggest_float("alpha_low", 1.0, 6.0)
        alpha_high = trial.suggest_float("alpha_high", 1.0, 6.0)
        alpha_penalty = 0.001 * ((alpha_low - 1.0) + (alpha_high - 1.0))  # discourage extremes

        fold_scores, best_iters = [], []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            w_tr = inverse_freq_weights(y_tr, clip=(0.3, 5.0),
                                        alpha_multipliers={0: alpha_low, 2: alpha_high})

            clf = XGBClassifier(**params)
            _fit_with_early_stopping(
                clf=clf,
                X=X_tr,
                y=y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                rounds=100,
                verbose=False,
            )

            probs_va = clf.predict_proba(X_va)
            preds_va = probs_va.argmax(axis=1)
            f1_per = f1_score(y_va, preds_va, average=None, labels=[0, 1, 2])
            fold_score = (2.0 * f1_per[0] + 1.0 * f1_per[1] + 2.0 * f1_per[2]) / 5.0

            # subtract tiny penalty for aggressive α
            fold_scores.append(float(fold_score) - alpha_penalty)

            best_iters.append(_extract_best_iteration(clf, params["n_estimators"]))
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

    xgb_param_keys = {
        "learning_rate", "max_depth", "min_child_weight", "gamma",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"
    }
    out = {
        "xgb_params": {k: best[k] for k in xgb_param_keys},
        "alpha_low": float(best["alpha_low"]),
        "alpha_high": float(best["alpha_high"]),
        "n_estimators": n_estimators,
        "best_cv_score": float(study.best_value),
    }
    logger.info("Optuna best (minority-weighted CV score): %.5f", study.best_value)
    logger.info("Optuna best params (incl. α): %s", out)
    return out


# --------------------- final training ---------------------
def train_final_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best: Dict[str, Any],
) -> XGBClassifier:
    params = dict(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        random_state=RND,
        n_jobs=-1,
        eval_metric="mlogloss",
        **best["xgb_params"],
        n_estimators=best["n_estimators"],
    )

    w_tr = inverse_freq_weights(
        y_train, clip=(0.3, 5.0),
        alpha_multipliers={0: best["alpha_low"], 2: best["alpha_high"]}
    )

    model = XGBClassifier(**params)
    _fit_with_early_stopping(
        clf=model,
        X=X_train,
        y=y_train,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        rounds=100,
        verbose=False,
    )
    logger.info("Trained final XGBoost. Best iteration: %s", _extract_best_iteration(model, params["n_estimators"]))
    return model


# -------------------------- SHAP --------------------------
def compute_and_save_shap(model: XGBClassifier, X: pd.DataFrame, feature_names: List[str], out_dir: Path):
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X)

    # Normalize to list[class] of arrays [n_samples, n_features]
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
    else:
        raise TypeError(f"Unexpected SHAP output type/shape: {type(shap_values_raw)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shap_values.npy", np.array(shap_values, dtype=object), allow_pickle=True)

    # Per-class SHAP summary plots
    for class_idx, class_shap in enumerate(shap_values):
        plt.figure(figsize=(8, 6))
        shap.summary_plot(class_shap, X, feature_names=feature_names, show=False)
        plt.title(f"XGB SHAP summary (class {class_idx})")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_summary_class_{class_idx}.png")
        plt.close()

    # Mean |SHAP| per class
    importances = {f"class_{i}": np.abs(vals).mean(axis=0) for i, vals in enumerate(shap_values)}
    pd.DataFrame(importances, index=feature_names).to_csv(out_dir / "shap_mean_abs_per_class.csv")
    logger.info("Saved SHAP outputs to %s", out_dir)


# --------------- logistic baseline (sanity) ---------------
def fit_logistic_and_save(X_train: pd.DataFrame, y_train: pd.Series, out_dir: Path) -> None:
    feats = list(X_train.columns)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train[feats])
    clf = LogisticRegression(solver="lbfgs", penalty="l2", C=1.0, max_iter=2000, random_state=RND, multi_class="auto")
    clf.fit(X_tr_scaled, y_train)
    coefs = pd.DataFrame(clf.coef_.T, index=feats, columns=[f"class_{i}_coef" for i in range(clf.coef_.shape[0])])
    coefs.to_csv(out_dir / "logistic_coefs.csv")
    joblib.dump({"scaler": scaler, "clf": clf}, out_dir / "logistic_model.pkl")
    logger.info("Saved logistic baseline to %s", out_dir)


# ---------------------- evaluation ----------------------
def evaluate_and_save(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path,
) -> Dict[str, float]:
    probs = model.predict_proba(X_test)
    preds = probs.argmax(axis=1)

    macro_f1 = f1_score(y_test, preds, average="macro")
    accuracy = accuracy_score(y_test, preds)

    Y_ovr = pd.get_dummies(y_test).reindex(columns=[0, 1, 2], fill_value=0).values
    macro_auc = roc_auc_score(Y_ovr, probs, multi_class="ovr", average="macro")

    brier = multiclass_brier_score(y_test.values, probs)

    print(f"\nMacro-F1 = {macro_f1:.4f}")
    print(f"Macro-AUC (OvR) = {macro_auc:.4f}")
    print(f"Brier score = {brier:.4f}")
    print(f"Accuracy = {accuracy:.4f}\n")

    report = classification_report(y_test, preds, target_names=["low", "med", "high"])
    print("Classification Report (TEST):\n", report)

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    pred_counts = pd.Series(preds).value_counts().sort_index().to_dict()
    print("Prediction counts:", pred_counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(
        {"macro_f1": macro_f1, "macro_auc": macro_auc, "brier": brier, "accuracy": accuracy}
    ).to_csv(out_dir / "metrics.csv")

    np.save(out_dir / "confusion_matrix.npy", cm)
    with open(out_dir / "classification_report.txt", "w") as fh:
        fh.write(report)

    per_class_auc = {int(c): float(roc_auc_score((y_test.values == c).astype(int), probs[:, c])) for c in [0, 1, 2]}
    pd.Series(per_class_auc).to_json(out_dir / "per_class_auc.json")
    pd.Series({int(k): int(v) for k, v in pred_counts.items()}).to_json(out_dir / "pred_counts.json")

    return {"macro_f1": float(macro_f1), "macro_auc": float(macro_auc), "brier": float(brier), "accuracy": float(accuracy)}


# ---------------------------- main ----------------------------
def main():
    print("Loading data…")
    df = load_with_splits(ENGINEERED_CSV, SPLITS_CSV)
    X_train, y_train, X_val, y_val, X_test, y_test = make_partitions(df)

    # Safety: median-impute numeric NaNs
    for X in (X_train, X_val, X_test):
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            med = X_train[num_cols].median()
            X[num_cols] = X[num_cols].fillna(med)

    # 1) Optuna: tune XGB + α_low/α_high with minority-weighted CV objective
    best = tune_xgb_optuna_alpha(X_train, y_train, n_trials=100, cv_folds=5, random_state=RND)

    # 2) Final training with early stopping on val
    print("Training final XGBoost…")
    model = train_final_xgb(X_train, y_train, X_val, y_val, best)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "xgb_final.pkl")

    # 3) Evaluate on test
    print("Evaluating on TEST…")
    _ = evaluate_and_save(model, X_test, y_test, OUT_DIR)

    # 4) SHAP on test
    feats = list(X_train.columns)
    compute_and_save_shap(model, X_test[feats], feature_names=feats, out_dir=OUT_DIR / "shap")

    # 5) Logistic baseline (sanity)
    fit_logistic_and_save(X_train, y_train, OUT_DIR)

    print(f"\nSaved artifacts to: {OUT_DIR.resolve()}")
    print("DONE.")


if __name__ == "__main__":
    main()
