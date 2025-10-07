import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.random.seed(760)

# --------------------------
# 1) Load cleaned dataset
# --------------------------
df = pd.read_csv("clean_numeric_model.csv")

# Remove irrelevant / leakage / constant columns
drop_cols = [
    "User_ID", "Mental_Health_Condition_lbl",   # 泄漏/无用
    "Age_isna", "Sleep_Hours_isna", "Work_Hours_isna",
    "Physical_Activity_Hours_isna", "Social_Media_Usage_isna"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# --------------------------
# 2) Split by existing splits file
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
# 3) Load sample weights → approximate class_weight
# --------------------------
use_class_weight = False
class_weight = None
try:
    weights = pd.read_csv("sample_weights.csv")
    w_train = weights.loc[train_idx, "w_combo"]
    pos_w = w_train[y_train.values == 1].mean() if (y_train == 1).sum() > 0 else 1.0
    neg_w = w_train[y_train.values == 0].mean() if (y_train == 0).sum() > 0 else 1.0
    s = (pos_w + neg_w)
    class_weight = {0: float(neg_w / s * 2.0), 1: float(pos_w / s * 2.0)}
    use_class_weight = True
    print("Using class_weight (from w_combo approximation):", class_weight)
except Exception:
    print("sample_weights.csv not found or unreadable. Proceeding without class_weight.")

# --------------------------
# 4) Standardize
# --------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# --------------------------
# 5) Validation search: 子样本 + 快速特征选择 + SVC(no prob)
# --------------------------
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# —— 关键加速开关 ——
TUNE_SUBSAMPLE = 8000   # 调参使用的训练/验证子样本量（分层抽样）
K_LIST = [4, 8, 12, 16]  # k 候选精简
C_grid = [0.5, 1, 2, 4]
gamma_grid = ["scale", 0.5, 1.0]

# 分层子样本（train 与 val 各抽）
def stratified_subsample(Xs, ys, n):
    n = min(n, len(ys))
    # 近似分层：按标签分别抽样
    ix0 = np.where(ys.values == 0)[0]
    ix1 = np.where(ys.values == 1)[0]
    n0 = min(len(ix0), n // 2)
    n1 = min(len(ix1), n - n0)
    rng = np.random.default_rng(760)
    sub = np.concatenate([rng.choice(ix0, n0, replace=False),
                          rng.choice(ix1, n1, replace=False)])
    return Xs[sub], ys.iloc[sub]

Xtr_tune, ytr_tune = stratified_subsample(X_train_s, y_train, TUNE_SUBSAMPLE)
Xv_tune,  yv_tune  = stratified_subsample(X_val_s,   y_val,   TUNE_SUBSAMPLE)

best = {"auc": -1.0, "k": None, "C": None, "gamma": None}

for k in K_LIST:
    # 用 f_classif（极快）做筛选；拟合只在 train 子样本上
    fs = SelectKBest(score_func=f_classif, k=k)
    fs.fit(Xtr_tune, ytr_tune)

    Xtr_fs = fs.transform(Xtr_tune)
    Xv_fs  = fs.transform(Xv_tune)

    for C in C_grid:
        for gamma in gamma_grid:
            # 调参阶段关闭 probability=True，速度提升巨大
            clf = SVC(kernel="rbf", C=C, gamma=gamma, probability=False, random_state=760)
            clf.fit(Xtr_fs, ytr_tune)

            # 用 decision_function 计算 AUC（不需要概率）
            vscore = clf.decision_function(Xv_fs)
            val_auc = roc_auc_score(yv_tune, vscore)

            if val_auc > best["auc"]:
                best = {"auc": val_auc, "k": k, "C": C, "gamma": gamma}

print("\n[VAL-FAST] Best (subsample) -> k={}, C={}, gamma={} | val AUC={:.3f}"
      .format(best["k"], best["C"], best["gamma"], best["auc"]))

# —— 用最佳参数在 full train+val 上重训（这一步较慢，但只做一次） ——
# 重新在 full train+val 上做相同的标准化/特征选择流程
from sklearn.pipeline import make_pipeline

# 合并 train+val
X_trval = np.vstack([X_train_s, X_val_s])
y_trval = pd.concat([y_train, y_val], axis=0)

fs_final = SelectKBest(score_func=f_classif, k=best["k"])
clf_final = SVC(kernel="rbf", C=best["C"], gamma=best["gamma"], probability=True, random_state=760)

# 拟合特征选择器 + 模型
X_trval_fs = fs_final.fit_transform(X_trval, y_trval)
X_test_fs  = fs_final.transform(X_test_s)

clf_final.fit(X_trval_fs, y_trval)

# 测试集评估
y_pred = clf_final.predict(X_test_fs)
y_prob = clf_final.predict_proba(X_test_fs)[:, 1]

print("\n=== SVM (RBF, fast-tuned) - Test ===")
print("Classification Report:\n", classification_report(y_test, y_pred, digits=3))
print("ROC-AUC (Test): %.3f" % roc_auc_score(y_test, y_prob))

# ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM (RBF, fast-tuned)")
plt.legend()
plt.grid(True)
plt.show()
