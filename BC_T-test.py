import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# --------------------------
# 1) Load dataset
# --------------------------
df = pd.read_csv("clean_numeric_model.csv")

target = "Has_MH_condition_bin"
features_to_check = ["Sleep_Hours", "Work_Hours", "Social_Media_Usage", 
                     "Physical_Activity_Hours", "Age"]

# --------------------------
# 2) Group stats
# --------------------------
print("=== Group Mean/Std by Target ===")
stats = df.groupby(target)[features_to_check].agg(["mean","std"])
print(stats)

# --------------------------
# 3) T-tests
# --------------------------
print("\n=== T-tests (0 vs 1) ===")
for feat in features_to_check:
    group0 = df.loc[df[target]==0, feat].dropna()
    group1 = df.loc[df[target]==1, feat].dropna()
    stat, pval = ttest_ind(group0, group1, equal_var=False)
    print(f"{feat:25s} | p={pval:.4e}")

# --------------------------
# 4) Boxplots for visualization
# --------------------------
plt.figure(figsize=(12,8))
for i, feat in enumerate(features_to_check, 1):
    plt.subplot(2,3,i)
    sns.boxplot(x=target, y=feat, data=df, palette="Set2")
    plt.title(feat)
plt.tight_layout()
plt.show()
