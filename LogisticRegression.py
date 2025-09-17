from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# 读取数据
X = pd.read_csv("clean_numeric_model.csv")
splits = pd.read_csv("splits_70_15_15_k5.csv")

# 假设标签列叫 'Mental_Health_Condition'
y = X['Mental_Health_Condition']
X = X.drop(columns=['Mental_Health_Condition'])

# 划分训练和测试
train_idx = splits[splits['split']=="train"].index
test_idx  = splits[splits['split']=="test"].index

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]

# 多项逻辑回归，带L2正则化
clf = LogisticRegression(
    penalty='l2',
    solver='lbfgs',      # 支持多分类
    multi_class='multinomial',
    max_iter=1000,
    C=1.0                # L2 正则化强度 (越小正则越强)
)

clf.fit(X_train, y_train)

# 预测与评估
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print(classification_report(y_test, y_pred))
print("Macro AUC:", roc_auc_score(y_test, y_prob, multi_class="ovr"))
