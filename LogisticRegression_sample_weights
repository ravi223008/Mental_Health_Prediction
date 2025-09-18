from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# ==== 读取与对齐（features ↔ splits）====
Xy = pd.read_csv("clean_numeric_model.csv")        # 含 User_ID 与目标列
splits = pd.read_csv("splits_70_15_15_k5.csv")     # 含 row_id, split
df = Xy.merge(splits[["row_id", "split"]], left_on="User_ID", right_on="row_id", how="inner")

TARGET = "Severity_ord"   # 如用 severity_level 就改这里

# 若目标非整数，编码为 0..K-1
if df[TARGET].dtype.kind not in "iu":
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET].astype(str))
    print("Encoded target classes:", dict(enumerate(le.classes_)))

# ==== 特征与标签 ====
drop_cols = {TARGET, "split", "row_id", "User_ID"}
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df[TARGET].astype(int)

# 划分训练/测试
train_mask = df["split"] == "train"
test_mask  = df["split"] == "test"
X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
rowid_train = df.loc[train_mask, "row_id"].astype(str)
print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ==== 读取 sample_weights（以 row_id 对齐）====
wdf = pd.read_csv("sample_weights.csv")
wdf["row_id"] = wdf["row_id"].astype(str)

# 选择权重列（优先级：w_combo > w_group > w_label）
weight_col = next((c for c in ["w_combo", "w_group", "w_label"] if c in wdf.columns), None)
if weight_col is None:
    raise ValueError("sample_weights.csv 中未找到 w_combo / w_group / w_label 任一列。")
print(f"Using weight column: {weight_col}")

# 基础权重（训练集）
w_train = (
    pd.DataFrame({"row_id": rowid_train})
    .merge(wdf[["row_id", weight_col]], on="row_id", how="left")
    [weight_col]
    .fillna(1.0)
    .astype(float)
    .to_numpy()
)

print(f"[Base weights] min: {w_train.min():.3f}, max: {w_train.max():.3f}, mean: {w_train.mean():.3f}")

# ==== 放大少数类权重 ====
# 规则：以训练集样本数最多的类别为“多数类”，其余皆为“少数类”
counts = y_train.value_counts().sort_index()
majority_class = counts.idxmax()
minority_classes = [c for c in counts.index if c != majority_class]
AMP = 4

print(f"[Class dist train] {counts.to_dict()}")
print(f"[Majority] {majority_class}, [Minorities] {minority_classes}, AMP={AMP}")

# 构建放大向量：对应 y_train 的每个样本
boost = np.ones_like(y_train.to_numpy(), dtype=float)
if len(minority_classes) > 0:
    mask_minor = np.isin(y_train.to_numpy(), minority_classes)
    boost[mask_minor] = AMP

# 复合权重 = 基础权重 * 放大量
w_train_boosted = w_train * boost
print(f"[Boosted weights] min: {w_train_boosted.min():.3f}, max: {w_train_boosted.max():.3f}, "
      f"mean: {w_train_boosted.mean():.3f}")

# ==== 模型（L2 多项逻辑回归 + boosted sample_weight）====
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

# ==== 预测与评估 ====
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
