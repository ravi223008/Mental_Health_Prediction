from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ==== Read and merge by different keys ====
Xy = pd.read_csv("clean_numeric_model.csv")        # contains User_ID and target column
splits = pd.read_csv("splits_70_15_15_k5.csv")     # contains row_id, split ∈ {train,val,test}

# Align User_ID (left) ↔ row_id (right)
df = Xy.merge(splits[["row_id", "split"]], left_on="User_ID", right_on="row_id", how="inner")

# ==== Target column setting (choose according to your actual case) ====
TARGET = "Severity_ord"       # if your target is severity_level, change it to "severity_level"

# Encode target if it is not integer
if df[TARGET].dtype.kind not in "iu":
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))
    print("Encoded target classes:", dict(enumerate(le.classes_)))

# ==== Assemble features and target ====
# Drop target, split, and ID columns, keep the rest as features
drop_cols = {TARGET, "split", "row_id", "User_ID"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df[TARGET].astype(int)

# Split train/test
X_train, y_train = X[df["split"] == "train"], y[df["split"] == "train"]
X_test,  y_test  = X[df["split"] == "test"],  y[df["split"] == "test"]
print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==== Model (L2 multinomial logistic regression) ====
clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",            # supports multinomial
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# ==== Prediction and evaluation ====
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, zero_division=0, digits=4))

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
macro_auc = np.nan
try:
    # Multiclass macro-average AUC (OvR)
    macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
except Exception as e:
    print("[Warn] AUC failed:", e)

print("\n=== Macro metrics ===")
print(f"Macro-F1       : {macro_f1:.4f}")
print("Macro ROC-AUC  :", "NaN" if np.isnan(macro_auc) else f"{macro_auc:.4f}")
# ==== Assemble features and target (same as above until here) ====
drop_cols = {TARGET, "split", "row_id", "User_ID"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df[TARGET].astype(int)

# ===== Split by 'split' (apply SMOTE/fit only on train) =====
X_train, y_train = X[df["split"] == "train"], y[df["split"] == "train"]
X_test,  y_test  = X[df["split"] == "test"],  y[df["split"] == "test"]
print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ====== SMOTENC settings ======
# 1) Prefer to read categorical_indices / numeric_indices from smote_config.json
# 2) If not available, infer automatically: *_lbl / *_bin as categorical, others as numeric
import json, os
from pathlib import Path

smote_cfg_path = Path("smote_config.json")
if smote_cfg_path.exists():
    with open(smote_cfg_path, "r", encoding="utf-8") as f:
        smote_cfg = json.load(f)
    # indices are based on feature_cols order
    cat_idx = smote_cfg.get("categorical_indices", [])
    num_idx = smote_cfg.get("numeric_indices", [])
    # ensure indices are valid
    cat_idx = [i for i in cat_idx if 0 <= i < len(feature_cols)]
    num_idx = [i for i in num_idx if 0 <= i < len(feature_cols)]
else:
    cat_cols = [c for c in feature_cols if c.endswith("_lbl") or c.endswith("_bin")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    cat_idx = [feature_cols.index(c) for c in cat_cols]
    num_idx = [feature_cols.index(c) for c in num_cols]

# Keep column names for ColumnTransformer
cat_cols = [feature_cols[i] for i in cat_idx]
num_cols = [feature_cols[i] for i in num_idx]

print(f"[Info] Categorical columns: {len(cat_cols)} -> {cat_cols[:8]}{'...' if len(cat_cols)>8 else ''}")
print(f"[Info] Numeric columns    : {len(num_cols)} -> {num_cols[:8]}{'...' if len(num_cols)>8 else ''}")
# ====== Build pipeline: SMOTENC (applied only on train) -> preprocessing -> logistic regression ======
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np
# SMOTENC parameters: use suggestions from smote_config.json if available; otherwise fallback defaults
smote_params = {
    "categorical_features": cat_idx,
    "sampling_strategy": "auto",   # can change to float/dict if needed
    "k_neighbors": 5,
    "random_state": 42,
}
if smote_cfg_path.exists():
    # override common keys if present
    for k in ["sampling_strategy", "k_neighbors", "random_state"]:
        if k in smote_cfg:
            smote_params[k] = smote_cfg[k]

smote = SMOTENC(**smote_params)

# Preprocessing: categorical -> OneHot, numeric -> StandardScaler (helps LR convergence)
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop"
)

# Logistic regression: multinomial, L2 regularization
# (If still imbalanced, try class_weight="balanced"; but do not use with SMOTE at the same time)
clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    n_jobs=-1,
    random_state=42
)

pipe = ImbPipeline(steps=[
    ("smote", smote),          # applied only when fitting on train
    ("prep", preprocess),
    ("clf", clf)
])

# ====== Train (fit only on train; SMOTE will not touch test) ======
pipe.fit(X_train, y_train)

# ====== Predict and evaluate ======
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, zero_division=0, digits=4))

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
try:
    macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
except Exception as e:
    macro_auc = np.nan
    print("[Warn] AUC failed:", e)

print("\n=== Macro metrics ===")
print(f"Macro-F1       : {macro_f1:.4f}")
print("Macro ROC-AUC  :", "NaN" if np.isnan(macro_auc) else f"{macro_auc:.4f}")
