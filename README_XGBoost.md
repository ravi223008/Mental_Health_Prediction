# Mental Health Risk Screening from Lifestyle Signals

This repository contains the code and analysis for a project focused on mental health risk screening using self-reported lifestyle data. The primary challenge addressed is the heavy class imbalance in the dataset.

This project implements a leakage-safe preprocessing pipeline, engineers key features (e.g., Sleep Score, Activity Index), and compares cost-sensitive machine learning models (Logistic Regression, LightGBM, XGBoost). The approach avoids synthetic sampling artifacts by using inverse-frequency instance weighting. Hyperparameter optimization is managed by Optuna, and model interpretability is assessed using SHAP.

# Methodology

Data Preprocessing: A robust, leakage-safe pipeline to harmonize heterogeneous self-reported data.

Feature Engineering: Creation of interpretable features like Sleep Score, Activity Index, and workload proxies.

Imbalance Handling: Employs cost-sensitive learning (inverse-frequency instance weighting) to manage heavy class imbalance without using synthetic sampling methods like SMOTE.

Modeling: Compares a LogisticRegression baseline with LightGBM and XGBoost.

Optimization: Uses Optuna for efficient hyperparameter optimization (including class-weight multipliers) within a stratified K-fold cross-validation framework.

Interpretability: Applies SHAP to explain model predictions.

# Dataset

The analysis is based on self-reported lifestyle signals, likely from a source like the National Health and Nutrition Examination Survey (NHANES) as referenced in the accompanying report.

# Note: The raw dataset is not included in this repository. The notebook (XGBoost_Second_Version_Data_Final_Multi.ipynb) expects the pre-processed data to be available in a local path.

# Key Technologies

Data Analysis: pandas, numpy

Machine Learning: scikit-learn, xgboost, lightgbm

Hyperparameter Tuning: optuna

Interpretability: shap

Environment: jupyter

# Setup and Installation

# Clone the repository:

git clone https://github.com/ravi223008/Mental_Health_Prediction.git
cd your-repository-name


# Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


# Install the required dependencies (the requirements.txt file should be in your repository):

pip install -r requirements.txt


# Usage

Ensure you have the necessary dataset and have updated the file paths in the notebook to point to it.

# Start the Jupyter server:

jupyter lab


Open the XGBoost_Second_Version_Data_Final_Multi.ipynb notebook and run the cells sequentially.
