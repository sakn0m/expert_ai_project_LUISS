# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Italian Financial Challenge - Starter Template
#
# **Student Name:** [Your Name]
#
# **Challenge:** [Choose: Bankruptcy Prediction / Financial Health Classification / Revenue Forecasting]
#
# **Date:** [Date]
#
# ---
#
# ## Instructions
#
# This template provides a suggested structure for your challenge solution. You can adapt it to your needs, but make sure to cover all required sections:
#
# 1. Problem Statement and Objectives
# 2. Data Loading and Exploration
# 3. Data Preprocessing
# 4. Feature Engineering
# 5. Model Development
# 6. Model Evaluation
# 7. Interpretation and Business Insights
# 8. Conclusions and Future Work
#
# **Remember:**
# - Document your thought process with markdown cells
# - Comment your code clearly
# - Create visualizations to support your insights
# - Interpret results in business terms
# - Check the evaluation rubric to ensure you meet all criteria

# %% [markdown]
# ## 1. Problem Statement and Objectives
#
# **Challenge:** [Describe which challenge you chose and why]
#
# **Objective:** [State the prediction goal clearly]
#
# **Success Criteria:** [Define what "good" performance looks like for your chosen challenge]
#
# **Approach:** [Briefly outline your planned approach]

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# %% [markdown]
# ## 2. Data Loading and Exploration
#
# Load the training data and perform initial exploration.

# %%
# Load training data
train_df = pd.read_csv('../data/processed/train_data.csv')

print(f"Dataset shape: {train_df.shape}")
print(f"\nColumns: {train_df.columns.tolist()}")
print(f"\nFirst few rows:")
train_df.head()

# %%
# Basic information
train_df.info()

# %%
# Summary statistics
train_df.describe()

# %%
# Check for missing values
missing_values = train_df.isnull().sum()
missing_pct = (missing_values / len(train_df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

print("Missing Values:")
print(missing_df)

# %%
# Target variable analysis
# TODO: Uncomment the relevant section for your challenge

# For Challenge 1: Bankruptcy Prediction
# print("\nBankruptcy Distribution:")
# print(train_df['bankruptcy_next_year'].value_counts())
# print(f"\nBankruptcy rate: {train_df['bankruptcy_next_year'].mean():.2%}")

# For Challenge 2: Financial Health Classification
# print("\nFinancial Health Distribution:")
# print(train_df['financial_health_class'].value_counts().sort_index())
# print("\nPercentages:")
# print(train_df['financial_health_class'].value_counts(normalize=True).sort_index())

# For Challenge 3: Revenue Forecasting
# print("\nRevenue Change Statistics:")
# print(train_df['revenue_change'].describe())
# print(f"\nMissing revenue_change: {train_df['revenue_change'].isnull().sum()}")

# %% [markdown]
# ### Exploratory Data Analysis (EDA)
#
# Create visualizations to understand the data better.

# %%
# TODO: Create visualizations
# - Distribution plots for key financial ratios
# - Correlation heatmap
# - Target variable by sector, region, year, etc.
# - Box plots for outlier detection

# Example: Correlation heatmap
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
correlation_matrix = train_df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# %% [markdown]
# **Key Insights from EDA:**
#
# TODO: Document your key findings
# - 
# - 
# - 

# %% [markdown]
# ## 3. Data Preprocessing
#
# Handle missing values, outliers, and prepare data for modeling.

# %%
# Create a copy for preprocessing
df_processed = train_df.copy()

# TODO: Handle missing values
# - Decide on imputation strategy (median, mean, by group, etc.)
# - Document your rationale

# TODO: Handle outliers
# - Identify outliers (IQR, winsorization, etc.)
# - Decide on treatment strategy

# TODO: Encode categorical variables
# - One-hot encoding, label encoding, etc.

print("Preprocessing complete!")

# %%
# Train/Validation split
# IMPORTANT: Use temporal split, not random!

# For Challenges 1 & 2:
# train_years = [2018, 2019, 2020]
# val_year = 2021

# For Challenge 3 (time series):
# train_years = [2018, 2019]
# val_year = 2020
# test_year = 2021

# TODO: Implement temporal split
# df_train = df_processed[df_processed['fiscal_year'].isin(train_years)]
# df_val = df_processed[df_processed['fiscal_year'] == val_year]

print("Train/validation split complete!")

# %%
# Feature scaling
# IMPORTANT: Fit scaler on training data only!

from sklearn.preprocessing import StandardScaler, RobustScaler

# TODO: Choose and apply scaler
# scaler = StandardScaler()  # or RobustScaler() for outlier resistance
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

print("Feature scaling complete!")

# %% [markdown]
# ## 4. Feature Engineering
#
# Create new features based on domain knowledge.

# %%
# TODO: Engineer features
# Ideas:
# - Financial ratios (if not already present)
# - Year-over-year changes (growth rates)
# - Interaction features
# - Altman Z-Score or similar bankruptcy models
# - Sector-relative features (company vs sector average)
# - Temporal features (trends, volatility)

# Example: Year-over-year change
# df_processed = df_processed.sort_values(['company_id', 'fiscal_year'])
# df_processed['roe_yoy_change'] = df_processed.groupby('company_id')['roe'].diff()

print("Feature engineering complete!")
print(f"Total features: {df_processed.shape[1]}")

# %% [markdown]
# **Engineered Features:**
#
# TODO: List and explain your engineered features
# 1. 
# 2. 
# 3. 

# %% [markdown]
# ## 5. Model Development
#
# Train multiple models and compare performance.

# %%
# Import models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, roc_auc_score, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

# For handling class imbalance (Challenges 1 & 2)
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

print("Model libraries imported!")

# %%
# TODO: Prepare X and y
# feature_cols = [...]  # List your feature columns
# target_col = 'bankruptcy_next_year'  # or 'financial_health_class' or 'revenue_change'

# X_train = df_train[feature_cols]
# y_train = df_train[target_col]
# X_val = df_val[feature_cols]
# y_val = df_val[target_col]

print("Data prepared for modeling!")

# %% [markdown]
# ### Baseline Model

# %%
# TODO: Train baseline model
# For classification: Logistic Regression
# For regression: Linear Regression

# Example for Challenge 1 (Bankruptcy):
# baseline_model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
# baseline_model.fit(X_train_scaled, y_train)
# y_pred_baseline = baseline_model.predict(X_val_scaled)

# print("Baseline Model Performance:")
# print(classification_report(y_val, y_pred_baseline))
# print(f"F1-Score: {f1_score(y_val, y_pred_baseline):.4f}")

# %% [markdown]
# ### Advanced Models

# %%
# TODO: Train multiple models
# - Random Forest
# - XGBoost
# - Gradient Boosting
# - etc.

# For classification with imbalance, consider:
# - SMOTE oversampling
# - Class weights
# - Threshold tuning

# Example for Random Forest with SMOTE:
# smote = SMOTE(random_state=RANDOM_STATE)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# rf_model = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=10,
#     random_state=RANDOM_STATE
# )
# rf_model.fit(X_train_resampled, y_train_resampled)

print("Advanced models trained!")

# %%
# TODO: Hyperparameter tuning
# Use GridSearchCV or RandomizedSearchCV

from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Example:
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [5, 10, 15],
#     'min_samples_split': [2, 5, 10]
# }

# grid_search = GridSearchCV(
#     RandomForestClassifier(random_state=RANDOM_STATE),
#     param_grid,
#     cv=StratifiedKFold(n_splits=5),
#     scoring='f1',
#     n_jobs=-1
# )
# grid_search.fit(X_train_scaled, y_train)
# best_model = grid_search.best_estimator_

print("Hyperparameter tuning complete!")

# %% [markdown]
# ## 6. Model Evaluation
#
# Compare models and select the best one.

# %%
# TODO: Create model comparison table
# Compare all models on validation set using appropriate metrics

# For Challenge 1: F1-Score, AUC-ROC, Precision, Recall
# For Challenge 2: Weighted F1, Macro F1, Accuracy, Confusion Matrix
# For Challenge 3: RMSE, MAE, MAPE, R², Directional Accuracy

# Example comparison:
# models_comparison = pd.DataFrame({
#     'Model': ['Baseline', 'Random Forest', 'XGBoost'],
#     'F1-Score': [...],
#     'AUC-ROC': [...],
#     'Precision': [...],
#     'Recall': [...]
# })
# print(models_comparison)

# %%
# TODO: Confusion Matrix (for classification)
# from sklearn.metrics import ConfusionMatrixDisplay

# ConfusionMatrixDisplay.from_estimator(best_model, X_val_scaled, y_val)
# plt.title('Confusion Matrix - Best Model')
# plt.show()

# %%
# TODO: ROC Curve (for binary classification)
# from sklearn.metrics import RocCurveDisplay

# RocCurveDisplay.from_estimator(best_model, X_val_scaled, y_val)
# plt.title('ROC Curve - Best Model')
# plt.show()

# %% [markdown]
# **Model Selection:**
#
# TODO: Justify your final model choice
# - 
# - 

# %% [markdown]
# ## 7. Interpretation and Business Insights
#
# Explain the model and extract business value.

# %%
# TODO: Feature Importance
# For tree-based models, use built-in feature importance
# For other models, consider SHAP values

# Example:
# feature_importance = pd.DataFrame({
#     'feature': feature_cols,
#     'importance': best_model.feature_importances_
# }).sort_values('importance', ascending=False)

# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
# plt.xlabel('Importance')
# plt.title('Top 15 Most Important Features')
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

# %%
# TODO: Error Analysis
# - Which cases does the model misclassify/mispredicts?
# - Are there patterns in the errors?
# - How do errors vary by sector, size, region, etc.?

# Example:
# errors_df = df_val.copy()
# errors_df['prediction'] = y_pred
# errors_df['error'] = (errors_df['prediction'] != errors_df[target_col])

# print("Error rate by sector:")
# print(errors_df.groupby('ateco_sector')['error'].mean().sort_values(ascending=False))

# %% [markdown]
# **Business Insights:**
#
# TODO: Translate technical findings into business language
#
# **Key Risk Factors** (for bankruptcy/health challenges):
# 1. 
# 2. 
# 3. 
#
# **Revenue Drivers** (for forecasting challenge):
# 1. 
# 2. 
# 3. 
#
# **Actionable Recommendations:**
# - 
# - 
# - 

# %% [markdown]
# ## 8. Conclusions and Future Work
#
# Summarize findings and discuss limitations.

# %% [markdown]
# **Summary:**
#
# TODO: Summarize your work
# - 
# - 
#
# **Performance vs Targets:**
#
# TODO: Compare your results to challenge targets
# - My F1-Score: [X.XX] vs Target: [0.55-0.70]
# - ...
#
# **Model Limitations:**
#
# TODO: Discuss limitations honestly
# 1. 
# 2. 
# 3. 
#
# **Future Improvements:**
#
# TODO: What would you do with more time?
# 1. 
# 2. 
# 3. 
#
# **Lessons Learned:**
#
# TODO: Reflect on the experience
# - 
# - 

# %% [markdown]
# ---
#
# ## Final Checklist
#
# Before submitting, verify:
#
# - [ ] All code cells execute without errors
# - [ ] Markdown cells explain each step clearly
# - [ ] Visualizations are clear and labeled
# - [ ] Feature importance is analyzed
# - [ ] Error analysis is performed
# - [ ] Business insights are provided
# - [ ] Model limitations are discussed
# - [ ] Code is well-commented
# - [ ] Results meet or exceed minimum performance targets
# - [ ] No data leakage (temporal split, proper scaling, etc.)
# - [ ] Citations for any external code/resources
#
# **Good luck with your challenge!** 🚀
