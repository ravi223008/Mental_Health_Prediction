from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ==== Read and align ====
Xy = pd.read_csv("clean_numeric_model.csv")        # contains User_ID and target column
splits = pd.read_csv("splits_70_15_15_k5.csv")     # contains row_id, split

# Align User_ID (left) with row_id (right)
df = Xy.merge(splits[["row_id", "split"]], left_on="User_ID", right_on="row_id", how="inner")

TARGET = "Severity_ord"   # if you switch to severity_level, just change this

# Encode target if it is not integer
if df[TARGET].dtype.kind not in "iu":
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))
    print("Encoded target classes:", dict(enumerate(le.classes_)))

# ==== Features and labels ====
drop_cols = {TARGET, "split", "row_id", "User_ID"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df[TARGET].astype(int)

# Split train/test
X_train, y_train = X[df["split"] == "train"], y[df["split"] == "train"]
X_test,  y_test  = X[df["split"] == "test"],  y[df["split"] == "test"]

print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==== Model (L2 multinomial logistic regression + class_weight) ====
clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    class_weight="balanced",   # key parameter to mitigate class imbalance
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# ==== Predict and evaluate ====
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, zero_division=0, digits=4))

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
macro_auc = np.nan
try:
    macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
except Exception as e:
    print("[Warn] AUC failed:", e)

print("\n=== Macro metrics ===")
print(f"Macro-F1       : {macro_f1:.4f}")
print("Macro ROC-AUC  :", "NaN" if np.isnan(macro_auc) else f"{macro_auc:.4f}")
