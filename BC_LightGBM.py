# ============================================
# LightGBM Clean Baseline (with callbacks early stopping)
# - sample_weight (train/val)
# - small manual grid search
# - threshold optimization on val, test evaluation
# ============================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import product
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

# --------------------------
# 1) Load & basic cleanup
# --------------------------
df = pd.read_csv("clean_numeric_model.csv")
drop_cols = [
    "User_ID",
    "Age_isna", "Sleep_Hours_isna", "Work_Hours_isna",
    "Physical_Activity_Hours_isna", "Social_Media_Usage_isna","Mental_Health_Condition_lbl"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

splits = pd.read_csv("splits_70_15_15_k5.csv")

y = df["Has_MH_condition_bin"].astype(int)
X = df.drop(columns=["Has_MH_condition_bin"])

train_idx = splits.index[splits["split"] == "train"]
val_idx   = splits.index[splits["split"] == "val"]
test_idx  = splits.index[splits["split"] == "test"]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val,   y_val   = X.loc[val_idx],   y.loc[val_idx]
X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

# --------------------------
# 2) Optional sample weights
# --------------------------
use_weights = False
w_train, w_val = None, None
try:
    weights = pd.read_csv("sample_weights.csv")
    w_train = weights.loc[train_idx, "w_combo"].values
    w_val   = weights.loc[val_idx, "w_combo"].values
    use_weights = True
    print("Using sample weights (w_combo).")
except FileNotFoundError:
    print("sample_weights.csv not found. Proceeding without weights.")

# --------------------------
# 3) LightGBM Datasets
# --------------------------
dtrain = lgb.Dataset(X_train, label=y_train, weight=(w_train if use_weights else None))
dval   = lgb.Dataset(X_val,   label=y_val,   weight=(w_val   if use_weights else None))

# --------------------------
# 4) Base params + small search space
#    (keep it small; expand if needed)
# --------------------------
base_params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "seed": 760,
    "num_threads": -1
}

search_space = {
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "min_data_in_leaf": [20, 50],
    "feature_fraction": [0.9],
    "bagging_fraction": [0.8],
    "bagging_freq": [5],
    "lambda_l2": [0.0, 1.0]
}

def param_combos(space):
    keys = list(space.keys())
    for values in product(*[space[k] for k in keys]):
        yield {k: v for k, v in zip(keys, values)}

# --------------------------
# 5) Train with early stopping (callbacks) over grid
# --------------------------
best_auc = -np.inf
best_params = None
best_bst = None

for p in param_combos(search_space):
    params = base_params.copy()
    params.update(p)
    print(f"\n>> Try params: {p}")

    bst = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),      # <-- version-safe early stopping
            lgb.log_evaluation(period=100)
        ]
    )
    # Get best val AUC
    val_auc = bst.best_score.get("val", {}).get("auc", None)
    if val_auc is None:  # older versions may use "valid_1"
        val_auc = bst.best_score.get("valid_1", {}).get("auc", -np.inf)

    print(f"Val AUC (best_iter={bst.best_iteration}): {val_auc:.5f}")

    if val_auc > best_auc:
        best_auc = val_auc
        best_params = params
        best_bst = bst

print("\nBest params:", best_params)
print("Best Val AUC:", best_auc)
print("Best iteration:", best_bst.best_iteration)

# --------------------------
# 6) Threshold optimization on VAL
# --------------------------
y_val_prob = best_bst.predict(X_val, num_iteration=best_bst.best_iteration)

ths = np.linspace(0.05, 0.95, 181)
f1s = [f1_score(y_val, (y_val_prob > t).astype(int)) for t in ths]
best_t_f1 = ths[int(np.argmax(f1s))]

fpr, tpr, thr_roc = roc_curve(y_val, y_val_prob)
youden_js = tpr - fpr
t_youden = thr_roc[int(np.argmax(youden_js))]

print(f"Best threshold (F1 on val): {best_t_f1:.3f}")
print(f"Best threshold (Youden's J on val): {t_youden:.3f}")

best_threshold = best_t_f1  # choose your metric of interest

# --------------------------
# 7) Final Test Evaluation
# --------------------------
y_test_prob = best_bst.predict(X_test, num_iteration=best_bst.best_iteration)
y_test_pred = (y_test_prob > best_threshold).astype(int)

print(f"\n=== Test Set (LightGBM, threshold={best_threshold:.3f}) ===")
print(classification_report(y_test, y_test_pred, digits=3))
print("ROC-AUC:", f"{roc_auc_score(y_test, y_test_prob):.3f}")

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix (test):\n", cm)

# --------------------------
# 8) Curves
# --------------------------
# ROC
fpr_t, tpr_t, _ = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr_t, tpr_t, label=f"AUC={roc_auc_score(y_test, y_test_prob):.3f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LightGBM")
plt.legend(); plt.grid(True); plt.tight_layout()

# PR
prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
plt.figure(figsize=(6,5))
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve - LightGBM")
plt.grid(True); plt.tight_layout()

plt.show()

# --------------------------
# 9) Feature importance
# --------------------------
imp = best_bst.feature_importance(importance_type="gain")
imp_df = pd.DataFrame({"Feature": X.columns, "Importance": imp}) \
         .sort_values(by="Importance", ascending=False)
print("\nTop features by importance (gain):")
print(imp_df.head(15))
