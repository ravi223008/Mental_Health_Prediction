# Baseline Model Report

## 1. Branch and File Overview
The `baseline_model` branch contains the following files, each corresponding to one strategy for handling class imbalance:  

- **`LogisticRegression.py`**  
  Implements the plain multinomial logistic regression (L2 regularization), without any balancing.  

- **`LogisticRegression_balanced.py`**  
  Adds `class_weight="balanced"` to automatically give higher weights to minority classes.  

- **`LogisticRegression_sample_weights.py`**  
  Loads `sample_weights.csv` and further amplifies minority class weights (AMP=4), combining fairness weights with imbalance adjustment.  

- **`LogisticRegression-SMOTE.py`**  
  Applies **SMOTENC** oversampling on the training set to synthetically generate minority samples and balance class distribution.  

---

## 2. Experiments and Results

### 1. Plain Logistic Regression
- **Setup**: Standard logistic regression, no weights.  
- **Results**:  
  - Accuracy = **0.6665**  
  - Macro-F1 = **0.2666**  
  - Macro-AUC = **0.4989**  
- **Analysis**: The model predicts almost exclusively the majority class (class 1), while minority classes (0 and 2) are completely ignored.  

---

### 2. Class Weight = "balanced"
- **Setup**: Built-in weight balancing in sklearn.  
- **Results**:  
  - Accuracy = **0.3252**  
  - Macro-F1 = **0.2934**  
  - Macro-AUC = **0.4988**  
- **Analysis**: Minority class recall improved slightly, but overall accuracy dropped significantly. This indicates better fairness but weaker overall performance.  

---

### 3. Sample Weights (AMP=4)
- **Setup**: Based on `w_combo` weights, minority samples are amplified by a factor of 4.  
- **Results**:  
  - Accuracy = **0.3724**  
  - Macro-F1 = **0.3108**  
  - Macro-AUC = **0.4982**  
- **Analysis**: Outperforms the balanced model in Macro-F1. Accuracy is higher than the balanced approach but still lower than the plain model. This shows that customized sample-level weighting helps improve minority metrics.  

---

### 4. SMOTE Oversampling
- **Setup**: SMOTENC oversampling applied to the training set only (no leakage to validation/test).  
- **Results**:  
  - Accuracy = **0.4972**  
  - Macro-F1 = **0.3245**  
  - Macro-AUC = **0.4997**  
- **Analysis**: Achieved the **highest Macro-F1** among all methods, proving that oversampling improves minority recall. However, accuracy is still lower than the plain model.  

---

## 3. Overall Comparison

| Method                     | Accuracy | Macro-F1 | Macro-AUC | Notes |
|-----------------------------|----------|----------|-----------|-------|
| Plain Logistic Regression   | 0.6665   | 0.2666   | 0.4989    | High accuracy but predicts only majority class |
| Class Weight = "balanced"   | 0.3252   | 0.2934   | 0.4988    | Improves minority recall, hurts accuracy |
| Sample Weights (AMP=4)      | 0.3724   | 0.3108   | 0.4982    | Best trade-off among weighting methods |
| SMOTE Oversampling          | 0.4972   | 0.3245   | 0.4997    | Best Macro-F1, improved minority recall |

---

## 4. Conclusion
- **Conclusions**:  
  - Logistic regression, in its plain form, is highly biased toward the majority class.  
  - Weighting strategies (balanced, sample weights) improve fairness but reduce accuracy.  
  - SMOTE oversampling yields the best Macro-F1 and enhances minority recall, but accuracy remains limited.  
