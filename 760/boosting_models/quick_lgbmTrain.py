# diagnostics_and_ablation.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import accuracy_score, log_loss, classification_report
from lightgbm import LGBMClassifier
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("diag")

# load using the same function as your script (or import it)
from train_lightgbm import load_and_split_data, get_class_weights, get_sample_weights

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data()

# Quick data checks
logger.info("Train/Val/Test sizes: %d / %d / %d", len(y_train), len(y_val), len(y_test))
logger.info("Label distribution (train): %s", dict(Counter(y_train)))
logger.info("Label distribution (val): %s", dict(Counter(y_val)))
logger.info("Label distribution (test): %s", dict(Counter(y_test)))

# Feature checks
features = [c for c in X_train.columns if c != "row_id"]
X_train_feat = X_train[features]
logger.info("Feature dtypes:\n%s", X_train_feat.dtypes.value_counts().to_string())
logger.info("Any NaNs in features? %s", X_train_feat.isna().any().any())
logger.info("Number of columns with zero variance: %d", (X_train_feat.nunique() <= 1).sum())
logger.info("Example feature summary:\n%s", X_train_feat.describe().T.head(8).to_string())

# Sample weights (combined class + fairness) from your function
combined_weights = get_sample_weights(X_train, y_train)  # this uses your script's logic

# Create alternative weight vectors
ones = np.ones_like(combined_weights)
# class-only weights (compute from y_train and normalize mean=1)
cw_raw = get_class_weights(y_train)
mean_w = np.mean(list(cw_raw.values()))
cw_norm = {k: v/mean_w for k,v in cw_raw.items()}
class_only = y_train.map(cw_norm).values

logger.info("Sample weights: combined mean=%0.4f class-only mean=%0.4f", combined_weights.mean(), class_only.mean())

# helper to train a small LGBM for quick comparison
def train_quick(X_tr, y_tr, X_v, y_v, X_te, y_te, sample_weight=None, label="model"):
    feats = [c for c in X_tr.columns if c != "row_id"]
    X_tr_f = X_tr[feats]
    X_v_f = X_v[feats]
    X_te_f = X_te[feats]
    model = LGBMClassifier(objective="multiclass", num_class=int(y_tr.nunique()), n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_tr_f, y_tr, sample_weight=sample_weight, eval_set=[(X_v_f, y_v)], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)])
    preds = model.predict(X_te_f)
    probas = model.predict_proba(X_te_f)
    acc = accuracy_score(y_te, preds)
    ll = log_loss(y_te, probas)
    logger.info("%s -> Acc: %.4f, LogLoss: %.4f", label, acc, ll)
    logger.info("%s classification report:\n%s", label, classification_report(y_te, preds, target_names=["low","med","high"]))
    return model

logger.info("TRAINING A: Combined weights (class * fairness)")
mA = train_quick(X_train, y_train, X_val, y_val, X_test, y_test, sample_weight=combined_weights, label="A_combined")

logger.info("TRAINING B: No weights (baseline)")
mB = train_quick(X_train, y_train, X_val, y_val, X_test, y_test, sample_weight=None, label="B_unweighted")

logger.info("TRAINING C: Class-only weights")
mC = train_quick(X_train, y_train, X_val, y_val, X_test, y_test, sample_weight=class_only, label="C_class_only")
