# ============================================
# Logistic Regression Baseline for Mental Health Prediction
# Compatible with Zenodo dataset + your preprocessing script
# ============================================

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --------------------------
# 1. Load cleaned dataset
# --------------------------
df = pd.read_csv("clean_numeric_model.csv")

# Remove irrelevant / constant columns
drop_cols = [
    "User_ID",
    "Age_isna", "Sleep_Hours_isna", "Work_Hours_isna",
    "Physical_Activity_Hours_isna", "Social_Media_Usage_isna"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# --------------------------
# 2. Split by existing splits file
# --------------------------
splits = pd.read_csv("splits_70_15_15_k5.csv")

y = df["Has_MH_condition_bin"]
X = df.drop(columns=["Has_MH_condition_bin"])

train_idx = splits.index[splits["split"] == "train"]
val_idx   = splits.index[splits["split"] == "val"]
test_idx  = splits.index[splits["split"] == "test"]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val,   y_val   = X.loc[val_idx],   y.loc[val_idx]
X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

# --------------------------
# 3. (Optional) Load sample weights
# --------------------------
try:
    weights = pd.read_csv("sample_weights.csv")
    w_train = weights.loc[train_idx, "w_combo"]
    use_weights = True
    print("Using sample weights (w_combo).")
except FileNotFoundError:
    w_train = None
    use_weights = False
    print("sample_weights.csv not found. Proceeding without weights.")

# --------------------------
# 4. Train Logistic Regression model
# --------------------------
model = LogisticRegression(
    solver="liblinear",     # robust for small/medium data
    penalty="l2",
    max_iter=1000,
    random_state=760
)

if use_weights:
    model.fit(X_train, y_train, sample_weight=w_train)
else:
    model.fit(X_train, y_train)

# --------------------------
# 5. Evaluate on test set
# --------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, digits=3))
print("ROC-AUC: %.3f" % roc_auc_score(y_test, y_prob))

# --------------------------
# 6. Plot ROC Curve
# --------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 7. Feature importance (coefficients)
# --------------------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nTop positive predictors:")
print(coef_df.head(10))
print("\nTop negative predictors:")
print(coef_df.tail(10))
