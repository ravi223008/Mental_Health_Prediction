# ============================================
# XGBoost Baseline for Mental Health Prediction
# Compatible with Zenodo dataset + your preprocessing script
# - Uses train/val split for early stopping
# - Optional sample weights (w_combo)
# - Refit on train+val with best rounds, evaluate on test
# ============================================

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --------------------------
# 1. Load cleaned dataset
# --------------------------
df = pd.read_csv("clean_numeric_model.csv")

# Remove irrelevant / constant columns
drop_cols = [
    "User_ID", "Mental_Health_Condition_lbl",
    "Age_isna", "Sleep_Hours_isna", "Work_Hours_isna",
    "Physical_Activity_Hours_isna", "Social_Media_Usage_isna"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# --------------------------
# 2. Split by existing splits file
# --------------------------
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
# 3. Load sample weights
# --------------------------
use_weights = False
w_train = None
w_val   = None
try:
    weights = pd.read_csv("sample_weights.csv")
    w_train = weights.loc[train_idx, "w_combo"].values
    w_val   = weights.loc[val_idx,   "w_combo"].values
    use_weights = True
    print("Using sample weights (w_combo).")
except Exception:
    print("sample_weights.csv not found or unreadable. Proceeding without weights.")
# --------------------------
# 4. Train XGBoost model (with early stopping on val)
# --------------------------
import xgboost as xgb

# DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, weight=(w_train if use_weights else None))
dval   = xgb.DMatrix(X_val,   label=y_val,   weight=(w_val   if use_weights else None))
dtest  = xgb.DMatrix(X_test)

# 如果没有样本权重，按类别比例设置 scale_pos_weight（负类/正类）
if not use_weights:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0
else:
    scale_pos_weight = 1.0

# 基础参数（稳定、通用）
params = {
    "booster": "gbtree",
    "tree_method": "hist",          # 更快
    "eval_metric": "auc",
    "objective": "binary:logistic",
    "eta": 0.08,                    # learning_rate
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,                  # L2
    "alpha": 0.0,                   # L1
    "random_state": 760,
    "scale_pos_weight": scale_pos_weight
}

watchlist = [(dtrain, "train"), (dval, "val")]
num_boost_round = 2000
early_stopping_rounds = 100

print("Training XGBoost with early stopping...")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=watchlist,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=100
)

if hasattr(bst, "best_iteration") and bst.best_iteration is not None:
    best_iter = int(bst.best_iteration)
else:
    # 某些版本将 best_iteration 存在 attributes() 里
    best_iter = int(bst.attributes().get("best_iteration", num_boost_round - 1))

print(f"Best iteration (val): {best_iter + 1} rounds")

# --------------------------
# 5. Refit on train+val with best rounds, then evaluate on test
# --------------------------
X_trval = pd.concat([X_train, X_val], axis=0)
y_trval = pd.concat([y_train, y_val], axis=0)

if use_weights:
    w_trval = np.concatenate([w_train, w_val], axis=0)
else:
    w_trval = None

dtrval = xgb.DMatrix(X_trval, label=y_trval, weight=w_trval)
bst_final = xgb.train(
    params,
    dtrval,
    num_boost_round=best_iter ,
    evals=[(dtrval, "trval")],
    verbose_eval=False
)

# 预测
y_prob = bst_final.predict(dtest)
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== XGBoost - Test ===")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))
print("ROC-AUC (Test): %.3f" % roc_auc_score(y_test, y_prob))

# --------------------------
# 6. Plot ROC Curve
# --------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 7. Feature importance
# --------------------------
# 使用 'gain' 重要性（更稳健），展示前20个
importance = bst_final.get_score(importance_type="gain")
imp_series = pd.Series(importance).sort_values(ascending=False)
print("\nTop features by gain:")
print(imp_series.head(20))
