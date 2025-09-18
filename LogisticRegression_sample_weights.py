from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ==== Read and align (features ↔ splits) ====
Xy = pd.read_csv("clean_numeric_model.csv")        # contains User_ID and target column
splits = pd.read_csv("splits_70_15_15_k5.csv")     # contains row_id, split
df = Xy.merge(splits[["row_id", "split"]], left_on="User_ID", right_on="row_id", how="inner")

TARGET = "Severity_ord"   # if you want to use severity_level, just change this

# Encode target into 0..K-1 if not integer
if df[TARGET].dtype.kind not in "iu":
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))
    print("Encoded target classes:", dict(enumerate(le.classes_)))

# ==== Features and labels ====
drop_cols = {TARGET, "split", "row_id", "User_ID"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df[TARGET].astype(int)

# Train/test split
train_mask = df["split"] == "train"
test_mask  = df["split"] == "test"
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
rowid_train = df.loc[train_mask, "row_id"].astype(str)
print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==== Read sample_weights (align by row_id) ====
wdf = pd.read_csv("sample_weights.csv")
wdf["row_id"] = wdf["row_id"].astype(str)

# Select weight column (priority: w_combo > w_group > w_label)
weight_col = next((c for c in ["w_combo", "w_group", "w_label"] if c in wdf.columns), None)
if weight_col is None:
    raise ValueError("No w_combo / w_group / w_label column found in sample_weights.csv.")
print(f"Using weight column: {weight_col}")

# Base weights (train set)
w_train = (
    pd.DataFrame({"row_id": rowid_train})
    .merge(wdf[["row_id", weight_col]], on="row_id", how="left")
    [weight_col]
    .fillna(1.0)
    .astype(float)
    .to_numpy()
)

print(f"[Base weights] min: {w_train.min():.3f}, max: {w_train.max():.3f}, mean: {w_train.mean():.3f}")

# ==== Boost minority class weights ====
# Rule: treat the class with most samples as majority, others as minorities
counts = y_train.value_counts().sort_index()
majority_class = counts.idxmax()
minority_classes = [c for c in counts.index if c != majority_class]
AMP = 4

print(f"[Class dist train] {counts.to_dict()}")
print(f"[Majority] {majority_class}, [Minorities] {minority_classes}, AMP={AMP}")

# Build boost vector: one per training sample
boost = np.ones_like(y_train.to_numpy(), dtype=float)
if len(minority_classes) > 0:
    mask_minor = np.isin(y_train.to_numpy(), minority_classes)
    boost[mask_minor] = AMP

# Final weights = base weights * boost factor
w_train_boosted = w_train * boost
print(f"[Boosted weights] min: {w_train_boosted.min():.3f}, max: {w_train_boosted.max():.3f}, "
      f"mean: {w_train_boosted.mean():.3f}")

# ==== Model (L2 multinomial logistic regression + boosted sample_weight) ====
clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train, sample_weight=w_train_boosted)

# ==== Prediction and evaluation ====
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

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
