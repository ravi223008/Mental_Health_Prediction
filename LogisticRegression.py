from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ==== 读取与按不同键合并 ====
Xy = pd.read_csv("clean_numeric_model.csv")        # 含 User_ID 与目标列
splits = pd.read_csv("splits_70_15_15_k5.csv")     # 含 row_id, split ∈ {train,val,test}

# 用 User_ID (左) ↔ row_id (右) 对齐
df = Xy.merge(splits[["row_id", "split"]], left_on="User_ID", right_on="row_id", how="inner")

# ==== 目标列设置（按你们实际情况二选一）====
TARGET = "Severity_ord"       # 如果你们用的是 severity_level，就改成 "severity_level"

# 若目标非整数，做编码（0..K-1）
if df[TARGET].dtype.kind not in "iu":
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))
    print("Encoded target classes:", dict(enumerate(le.classes_)))

# ==== 组装特征与标签 ====
# 去掉目标、split、以及两个ID列，剩余都是特征
drop_cols = {TARGET, "split", "row_id", "User_ID"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df[TARGET].astype(int)

# 划分训练/测试
X_train, y_train = X[df["split"] == "train"], y[df["split"] == "train"]
X_test,  y_test  = X[df["split"] == "test"],  y[df["split"] == "test"]
print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==== 模型（L2 多项逻辑回归） ====
clf = LogisticRegression(
    penalty="l2",
    solver="lbfgs",            # 支持 multinomial
    multi_class="multinomial",
    max_iter=1000,
    C=1.0,
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# ==== 预测与评估 ====
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, zero_division=0, digits=4))

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
macro_auc = np.nan
try:
    # 多分类宏平均 AUC（OvR）
    macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
except Exception as e:
    print("[Warn] AUC failed:", e)

print("\n=== Macro metrics ===")
print(f"Macro-F1       : {macro_f1:.4f}")
print("Macro ROC-AUC  :", "NaN" if np.isnan(macro_auc) else f"{macro_auc:.4f}")
