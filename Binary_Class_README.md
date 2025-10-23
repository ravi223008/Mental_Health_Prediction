# 🧠 Mental Health Binary Classification (Baseline Screening)

This directory contains the **binary classification experiments** for the *Mental Health Risk Screening from Lifestyle Signals* project (CS760 Final Report).  
The goal of this stage is to test whether simple demographic and lifestyle features can distinguish individuals **with vs. without** reported mental-health conditions.

---

## 📁 File Overview

| File | Description |
|------|--------------|
| `BC_Logistic.py` | Implements the baseline multinomial logistic regression (L2 regularization). |
| `BC_LightGBM.py` | Gradient boosting classifier with cost-sensitive weighting and early stopping. |
| `BC_XGBoost.py` | XGBoost binary classifier with the same preprocessing and evaluation protocol. |
| `BC_SVC.py` | RBF-kernel Support Vector Classifier with cross-validated tuning (C, γ). |
| `BC_T-test.py` | Statistical comparison of feature distributions (t-tests) between mental-health vs. control groups. |

---

## ⚙️ Experimental Setup

- **Data:** Cross-sectional survey data with ~50,000 rows and lifestyle features (sleep hours, work hours, physical activity, social media use, age, etc.).
- **Task:** Binary classification — predict whether an individual reported any mental-health condition (`Has_MH_condition_bin ∈ {0,1}`).
- **Split:** Stratified 70/15/15 (train/val/test).
- **Evaluation Metrics:** Accuracy, Macro-F1, ROC-AUC (on test set).
- **Note:** All pipelines are leakage-free. No feature transformations were fit on validation/test data.

---

## 🧩 Results Summary

| Model | Accuracy | ROC-AUC | Macro-F1 | Key Observation |
|:------|:----------:|:----------:|:----------:|:---------------|
| **Logistic Regression** | 0.492 | 0.487 | 0.492 | Predicts both classes evenly but performs at chance. |
| **Polynomial Logistic** | 0.500 | 0.494 | 0.500 | No measurable gain from polynomial expansion. |
| **RBF SVC** | 0.508 | 0.509 | 0.508 | Slight noise fitting; no significant improvement. |
| **LightGBM** | 0.500 | 0.496 | 0.333 | Collapses to majority class; threshold = 0.05 causes degenerate predictions. |
| **XGBoost** | 0.504 | 0.500 | 0.504 | Balanced but near-random performance. |

---

## 🧮 Statistical Feature Analysis (T-test)

Performed two-sample t-tests on selected lifestyle features across the two classes.

| Feature | p-value | Significant? |
|:---------|:---------:|:-------------:|
| Sleep Hours | 0.918 | ❌ No |
| Work Hours | 0.946 | ❌ No |
| Social Media Usage | 0.953 | ❌ No |
| Physical Activity Hours | 0.195 | ❌ No |
| Age | 0.205 | ❌ No |

**Interpretation:**  
No lifestyle feature shows statistically significant difference (p < 0.05) between individuals with and without mental-health conditions.  
This suggests **weak signal** in the available cross-sectional data.

---

## 🔍 Key Takeaways

1. **All models perform at chance level** — ROC-AUC ≈ 0.49–0.51, indicating the dataset lacks discriminative signal for binary screening.  
2. **Linear and nonlinear models behave similarly**, confirming that model capacity is not the limiting factor.  
3. **Lifestyle indicators alone are insufficient** for binary mental-health prediction; this validates the need for finer-grained **tri-class severity modeling** (low/medium/high).  
4. **Feature-level t-tests** reinforce the same conclusion — no significant differences across groups.  
5. **Pipeline validated:** consistent preprocessing and evaluation protocols confirmed data integrity and no leakage.

---

## 🧠 Next Steps

- Transition to **multi-class severity classification** (Low / Medium / High) for finer granularity.  
- Introduce **cost-sensitive weighting** and **explainability (SHAP)** analysis to interpret model behavior.  
- Evaluate whether non-linear boosting models can capture subtle behavioral patterns absent in the binary formulation.

---
