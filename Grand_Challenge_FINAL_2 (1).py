# Generated from: Grand_Challenge_FINAL_2 (1).ipynb
# Converted at: 2026-04-13T10:51:31.824Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # **Italian Financial Challenge - Project Notebook**
# 
# **Group:** 18
# 
# **Group Members:** Sofia Capriolo [309371], Andrea Cipolla [319211], Giorgio Vanini [311581], and Arianna Cambi [317971]
# 
# **Company:** Expert.ai
# 
# **Challenge:** Challenge 1 — Bankruptcy Prediction
# 


# ## **1. Problem Statement and Objectives**
# 


# **Challenge:** Bankruptcy Prediction
# 
# **Business context:** We chose the Bankruptcy Prediction challenge because, having witnessed major macroeconomic shocks such as the COVID-19 crisis, the energy price spike and the subsequent tightening of monetary policy, we wanted to use firm-level financial statements to see how these events translate into concrete stress on companies’ balance sheets and into real bankruptcy risk. By predicting whether a company will go bankrupt within the next 12 months based on its financial statements and business characteristics, we can analyse who survives and who defaults, which accounting signals anticipate this, and how to build robust models that cope with rare events, class imbalance and temporal drift rather than treating macro shocks as abstract stories
# 
# **Objective:** Build a model to predict bankruptcy_next_year (0 = healthy, 1 = bankruptcy).
# 
# **Success Criteria:** A “good” model in this challenge is not only one with high overall accuracy, but one that reliably identifies bankruptcies despite their rarity. Concretely, we aim for a materially higher recall and F1-score on the bankrupt class than a naive or majority-class baseline, stable performance across fiscal years (no collapse when moving from 2018–2021 to 2022–2023), and interpretable drivers of risk (e.g. leverage, liquidity, profitability) that are economically consistent with the macro shocks observed in Italy over the period.
# 
# **Approach:** We will start with an in-depth EDA on the training set to understand class imbalance, temporal patterns and the impact of macro shocks on key financial ratios, and then design a time-aware train/validation split by fiscal year. Next, we will engineer and select features capturing leverage, liquidity and profitability, and compare several baseline models (logistic regression, tree-based methods, ensemble models) with appropriate class weighting. Finally, we will choose the model that offers the best trade-off between recall/F1 on bankruptcies and robustness over time, and we will interpret its behaviour using feature-importance interpretation to link the predictions back to the underlying accounting signals.
# 
# **Key technical constraints (from the project guide):**
# Severe class imbalance (bankruptcies are rare)
# Avoid temporal data leakage: never random split; validate on future years
# Interpretability matters: identify key risk drivers for stakeholders


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")

# ##**2. Data Loading and Exploration**


# ###**2.1 Data Loading**
# 
# We start by **loading the two main CSV files** of the challenge:
# 
# - `train_data.csv` → contains both the **input features** and the **target
#   variables** (`revenue_change`, `financial_health_class`, and `bankruptcy_next_year`).
# - `test_features.csv` → contains only the **input features**, for which we
#   will later generate predictions.
# 
# After reading the files with `pd.read_csv`, we print the `.shape` of each
# DataFrame. This gives a quick sanity check:
# 
# - how many **rows** (firm-year observations) we have in train and test;
# - how many **columns** (features + targets) each dataset contains.
# 
# It’s a simple but important step to verify that the dataset size matches the
# description of the challenge and that the files were loaded correctly.


path_train = "train_data.csv"
path_test = "test_features.csv"

train = pd.read_csv(path_train)
test = pd.read_csv(path_test)

development_years = [2018, 2019, 2020]
validation_year = 2021

train_fit_raw = train[train["fiscal_year"].isin(development_years)].copy()
val_2021_raw = train[train["fiscal_year"] == validation_year].copy()
test_raw = test.copy()

sort_cols = [c for c in ["company_id", "fiscal_year"] if c in train.columns]
if len(sort_cols) > 0:
    train_fit_raw = train_fit_raw.sort_values(sort_cols).reset_index(drop=True)
    val_2021_raw = val_2021_raw.sort_values(sort_cols).reset_index(drop=True)
    test_raw = test_raw.sort_values(sort_cols).reset_index(drop=True)

print("Train_data shape:", train.shape)
print("Test_features shape :", test.shape)
print(f"Development rows (2018-2020): {train_fit_raw.shape[0]}")
print(f"Validation rows (2021):       {val_2021_raw.shape[0]}")
print(f"Official test rows:           {test_raw.shape[0]}")


# **Interpretation of the results**
# 
# - `Train_data shape: (11828, 30)`  
#   The training set has **11,828 observations** and **30 columns**.
# 
#   These 30 columns include:
#   - company identifiers and basic characteristics,
#   - financial variables (assets, debt, margins, etc.),
#   - and the **three target variables**.
# 
# 
# - `Test_features shape : (5811, 27)`  
#   The test set has **5,811 observations** and **27 columns**.  
#   The number of rows is smaller because the test period is shorter
#   (2022–2023), and the number of columns is lower because **the targets are
#   not present** in this file.
# 
# This confirms that:
# 
# - the data was loaded correctly,
# - the train set is roughly **twice as large** as the test set (11,828 vs 5,811
#   rows), which is reasonable for a temporal split,
# - and the structure is consistent with the competition setup:
#   **train = features + targets**, **test = features only**.


# ###**2.2 Data Exploration**
# After loading the data, we use `DataFrame.info()` for both the training and
# test sets. This command is a compact summary that tells us:
# 
# - how many **rows** we have;
# - for each column:
#   - the **number of non-null values** (so we can see where we have missing
#     data);
#   - the **data type** (`float64`, `int64`, `object`, …);
# - the **overall memory usage** of the dataset.
# 
# This step is important because:
# 
# - it shows which variables are **numeric** (ready for statistics and
#   modelling) and which are **categorical** (`object`: e.g. region, legal form);
# - it highlights the **first missing-value patterns** that we will need to
#   handle later;
# - it confirms that the train and test sets have a similar structure.


print("Train info:\n")
train.info()

print("\n\nTest info:\n")
test.info()


# **Interpretation of the results**
# 
# 
# In the **training set**:
# - Again, the training set has **11,828 entries** and **30 columns**.
# - Most financial variables (`total_assets`, `total_debt`, `production_value`,
#   `production_costs`, `current_ratio`, `profit_margin`, etc.) are `float64`,
#   so they are ready for numerical analysis.
# - We see a few categorical (`object`) columns such as `company_id`, `province`,
#   `region`, `legal_form` and `financial_health_class`.
# - Some variables already show missing values:
#   - `roe` and `leverage` have **11,783 non-null values** each, meaning there
#     are **45 missing values** for these ratios;
#   - `revenue_change` has **8,829 non-null values** out of 11,828, so about
#     **2,999 observations are missing** this target.
# - Memory usage is about **2.7 MB**, which is small enough to easily handle all
#   EDA and modelling steps in memory.
# 
# This tells us that the train set is mostly complete but we will have to deal
# with **missing values in a few key ratios and in `revenue_change`** during the
# preprocessing phase.
# 
# 
# 


# Instead, in the **test set**:
# - Again, it has **5,811 entries** and **27 columns**.
# - The structure is very similar: financial variables are `float64`, while
#   `company_id`, `province`, `region` and `legal_form` are `object`.
# - Again, we see some missing data:
#   - `province` has **5,360 non-null values**, so **451 firms** in the test set
#     do not have a recorded province;
#   - `roe` and `leverage` have **5,797 non-null values**, meaning **14 missing
#     values** each.
# - Memory usage is about **1.2 MB**, roughly half of the train set, which fits
#   with the smaller number of rows.
# 
# 


# Overall, `train.info()` and `test.info()` confirm that:
# 
# - train and test have a **consistent schema** (same types, similar columns);
# - a few variables, especially `province`, `roe`, `leverage` and
#   `revenue_change`, require **missing-value handling**;
# - most features are numeric and suitable for the statistical analysis and
#   machine-learning models we will build later in the notebook.
# 


# ## **3. Data Integrity Check**


# Before doing any modelling,  we want to make sure that the **structure of the
# training set and the test set is consistent**. If the two datasets had
# different columns, the model would break later when we try to apply it to the
# test data.
# 
# This small integrity check does two things:
# 
# 1. It loops over **all columns in the test set** and verifies that **each of
#    them also exists in the train set**.  
#    - If a column appears in the test but not in the train, it is an error
#      (the model would see a feature at prediction time that it never saw
#      during training).
# 
# 2. It then finds columns that are **present in the train set but not in the
#    test set**.  
#    - In our project we *expect* this to happen only for the **target
#      variables** (the labels we are trying to predict), which must not be
#      included in `test_features`.


train_cols = set(train.columns)
test_cols = set(test.columns)

# Check if there are columns in Test that do not exist in Train
test_not_in_train = test_cols - train_cols

if len(test_not_in_train) == 0:
    print("All good! All Test columns also exist in the Train set.")
else:
    print("ERROR: These columns are in the Test set but missing in the Train set:")
    for col in sorted(test_not_in_train):
        print(f" - {col}")
    print("There are discrepancies between Train and Test columns.")

# Check which columns are present in Train but missing in Test
missing_cols = train_cols - test_cols

print(f"\nThere are {len(missing_cols)} columns that are only in the Train set (and not in the Test set):")
for col in sorted(missing_cols):
    print(f" - {col}")

# Expected missing columns: target variables
expected_missing = {'bankruptcy_next_year', 'financial_health_class', 'revenue_change'}

if missing_cols == expected_missing:
    print("\nReason: These are the target variables that we want to predict!")
    print("It is correct that they are not included in the Test set.")
else:
    print("\nWarning: The columns missing from the Test set are not exactly the expected target variables.")
    print("Please check Train/Test consistency carefully.")
    print(f"Expected missing columns: {sorted(expected_missing)}")

# 
# **Interpretation of the results**
# 
# - The first message says:
# 
#     All good! All Test columns also exist in the Train set.
# 
#   This is exactly what we want: there is **no column that appears only in the test set**,
#   so the model will not face **unknown features at prediction time**.
# 
# - Then we get:
# 
#     There are 3 columns that are only in the Train set (and not in the Test set):
# 
#     - revenue_change
#     - financial_health_class
#     - bankruptcy_next_year
# 
#   So there are **exactly three columns present only in `train_data` and missing in `test_features`**,
#   and they **match the expected target variables**.
# 
# - Finally, the cell prints:
# 
#     Reason: These are the target variables that we want to predict!
#     It is correct that they are not included in the Test set.
# 
#   This confirms that the difference between **train** and **test** is exactly what we expect:  
#   the training set contains the **target labels** (`revenue_change`, `financial_health_class`, `bankruptcy_next_year`),  
#   while the test set contains **only the input features used for prediction**.
# 
# - In summary, this **integrity check** confirms that the dataset is **well structured for our bankruptcy prediction task**:
# 
#     - the **same feature columns** appear in both train and test,
#     - the test set **does not contain any unseen features**,
#     - the **only columns missing from the test set are exactly the target variables**,
#       which should exist **only in the training data**.
# 


# ## **4. Data Quality Check**


# Before exploring the features in detail, we first check some basic data quality issues
# on the training set:
# 
# 1. **Duplicate rows by (`company_id`, `fiscal_year`)**  
#    Each company-year combination should appear at most once. If we had duplicates,
#    we might be double-counting the same observation when training the model.
# 
# 2. **Negative `production_value` (revenues)**  
#    Revenues should normally be non-negative. A large number of negative values could
#    indicate data entry errors or sign mistakes that would distort the analysis.
# 
# 3. **Accounting identity: `total_assets = shareholders_equity + total_debt`**  
#    In a standard balance sheet, assets are financed by equity and debt.  
#    We check whether this identity roughly holds and count how many rows show a large
#    inconsistency. Big mismatches could signal unreliable financial statements.
# 
# The following code performs these three checks and prints the number of problematic
# rows for each case.
# 


duplicates = train.duplicated(subset=['company_id', 'fiscal_year']).sum()
print(f"Duplicate entries: {duplicates}")


neg_production_value = (train['production_value'] < 0).sum()
print(f"\nNegative Production Value entries: {neg_production_value}")


identity_diff = train['total_assets'] - (train['shareholders_equity'] + train['total_debt'])
identity_errors = (identity_diff.abs() > 1.0).sum()
print(f"\nAccounting Identity Errors: {identity_errors}")

# **Interpretation of the results**
# 
# The results we obtain are:
# 
# - `Duplicate entries: 0`
#   → There are no duplicated company-year rows in the training set.  
#   Each firm in each fiscal year is counted only once when we train the model.
# 
# - `Negative Production Value entries: 0`  
#   → No company has a negative `production_value`.  
#   This suggests that revenues are recorded with the expected sign and we do not need
#   to correct obvious sign errors.
# 
# - `Accounting Identity Errors: 0`  
#   → For all rows, `total_assets` is consistent (within a small tolerance) with  
#   `shareholders_equity + total_debt`.  
#   This means the basic balance-sheet identity holds and the accounting data look
#   internally coherent.
# 
# Overall, these checks suggest that the training data have good basic quality:
# we do not need to drop rows or fix obvious structural errors before continuing
# with the EDA and modelling.
# 


# ## **5. Missing Values Analysis**


# In this section we analyze more carefully how many missing values we have in each column, both in the
# training set and in the test set, and we visualise their pattern with a heatmap.
# 
# From the initial data exploration (`train.info()` and `test.info()`) we already
# saw that **several variables have missing entries** (for example `province`,
# `roe`, `leverage`, `revenue_change`). This is a clear signal that we should
# not ignore the problem, but study it in detail before building any model.
# 
# We do this for two reasons:
# 
# 1. **Quantify the problem** – for every variable we compute the number of missing
#    entries and the corresponding percentage of rows. This tells us which features
#    will need special treatment (e.g. imputation or removal) before modelling.
# 
# 2. **Understand the pattern of missingness** – the heatmaps show where the missing
#    values appear (which rows and which columns). If missing values are concentrated
#    in specific variables or in blocks of rows, this may indicate systematic issues
#    (e.g. some firms or years not reporting a given indicator) rather than random
#    noise.
# 
# The code below:
# - calculates missing counts and percentages for `train` and `test`,
# - prints the variables with at least one missing value,
# - and draws a heatmap using only those columns.
# 


missing_values_train = train.isnull().sum()
missing_pct_train = (missing_values_train / len(train)) * 100

missing_df_train = pd.DataFrame({
    'Missing Count': missing_values_train,
    'Missing %': missing_pct_train.round(2)
})

missing_df_train = missing_df_train[missing_df_train['Missing Count'] > 0] \
                                   .sort_values('Missing Count', ascending=False)

print("Missing values in train_data:")
print(missing_df_train)

missing_values_test = test.isnull().sum()
missing_pct_test = (missing_values_test / len(test)) * 100

missing_df_test = pd.DataFrame({
    'Missing Count': missing_values_test,
    'Missing %': missing_pct_test.round(2)
})

missing_df_test = missing_df_test[missing_df_test['Missing Count'] > 0] \
                                 .sort_values('Missing Count', ascending=False)

print("\n\nMissing values in test_features:")
print(missing_df_test)


# **Interpretation of the results**
# 
# From the tables we see that, in the **training set**:
# 
# - `revenue_change` has 2,999 missing values, which is about **25.36%** of the rows.
#   This is a substantial amount and means that this variable cannot be used
#   without some form of imputation or a careful decision on whether to keep it.
# - `province` is missing in around **7.77%** of the observations. This is a
#   moderate level of missingness on a categorical feature and suggests we may
#   need a simple imputation strategy (e.g. "Unknown" category).
# - `roe` and `leverage` have very few missing values (around **0.38%**), which is
#   small enough that median/mean imputation will have little impact on the data.
# 
# In the **test set** the situation is similar but slightly better:
# 
# - `province` is missing in about **7.76%** of rows.
# - `roe` and `leverage` again have only a very small fraction of missing entries
#   (around **0.24%** each).
# - `revenue_change` does not appear in the test missing table, which means either
#   there are no missing values for this variable in `test` or it is not included
#   in that dataset.


cols_with_na_train = missing_df_train.index.tolist()

if len(cols_with_na_train) > 0:
  fig, ax = plt.subplots(figsize=(10,4))
  ax.set_facecolor("white")
  fig.set_facecolor("white")
  sns.heatmap(train[cols_with_na_train].isnull(),
              cbar=False,
              cmap="Blues",
              ax=ax)
  plt.title("Missing values heatmap – train_data")
  plt.xlabel("Features")
  plt.ylabel("Rows")
  plt.show()



cols_with_na_test = missing_df_test.index.tolist()

if len(cols_with_na_test) > 0:
  fig, ax = plt.subplots(figsize=(10,4))
  ax.set_facecolor("white")
  fig.set_facecolor("white")
  sns.heatmap(test[cols_with_na_test].isnull(),
              cbar=False,
              cmap="Blues",
              ax=ax)
  plt.title("Missing values heatmap – test_features")
  plt.xlabel("Features")
  plt.ylabel("Rows")
  plt.show()





# The heatmaps confirm these patterns visually:
# 
# - In the training heatmap, the vertical band for `revenue_change` shows many
#   dark lines spread across the rows, indicating that missing values are
#   scattered across different firms and years rather than concentrated in a
#   single block. This suggests a general reporting issue for this indicator.
# - The columns `province`, `roe` and `leverage` show much fewer dark lines,
#   consistent with their lower missing percentages.
# - In the test heatmap, only `province`, `roe` and `leverage` display missing
#   values, and again the amount is limited.


# Overall, these checks tell us that:
# - `revenue_change` is the only feature with a large missingness problem and will
#   require a clear decision (e.g. keep with imputation, or drop it if it does not
#   add much information).
# - The other variables (`province`, `roe`, `leverage`) can be safely handled with
#   simple imputation techniques without heavily distorting the dataset.
# 
# This understanding of missing values will guide the preprocessing choices we
# make later.


# ## **6. Target variable `bankruptcy_next_year` analysis**
# 
# 


# In this step we analyse the target variable to understand **how rare bankruptcy is** in our
# dataset. In a bankruptcy prediction problem we usually have a strong class imbalance:
# most firms survive, only a very small fraction actually goes bankrupt.
# 
# Understanding this imbalance is crucial because:
# 
# - it affects which metrics we should use (F1, recall, precision) and not only accuracy;
# - it shows how “hard” it is for a model to detect bankrupt firms;
# - it helps us interpret all the following distributions, always remembering that the
#   bankrupt class is a small minority.
# 
# The code below prints the number of healthy vs bankrupt firms, the overall bankruptcy
# rate, and a bar plot of the class distribution.
# 


fig, ax = plt.subplots()

print("Bankruptcy Distribution:")
print(train["bankruptcy_next_year"].value_counts())
print(f"\nBankruptcy rate: {train["bankruptcy_next_year"].mean():.2%}")

target_counts = train["bankruptcy_next_year"].value_counts()
target_counts.index = target_counts.index.map({0: "Healthy", 1: "Bankrupt"})

target_counts.plot(kind='bar', ax=ax)

ax.set_ylabel("Number of observations")
ax.set_title("\nClass distribution of `bankruptcy_next_year`")
plt.xticks(rotation=0)

for p in ax.patches:
    value = int(p.get_height())
    ax.annotate(f"{value}",
                (p.get_x() + p.get_width()/2., value),
                ha='center', va='bottom', fontsize=9)

plt.show()


# **Interpretation of the results**
# 
# The printed counts and the bar plot show that:
# 
# - there are **11,744 healthy firms** (`bankruptcy_next_year = 0`);
# - there are only **84 bankrupt firms** (`bankruptcy_next_year = 1`);
# - the overall bankruptcy rate is about **0.71%** of the observations.
# 
# This confirms a **very strong class imbalance**: the event we want to predict
# (bankruptcy within 12 months) is rare compared to the number of non-bankrupt firms.
# As a consequence, a naive model that always predicts “Healthy” would already have
# very high accuracy, but would be useless for detecting real bankruptcies. This is
# why later we will focus on metrics such as F1-score and recall for the bankrupt class,
# and consider techniques like class weights or threshold tuning.


# **Link to macroeconomic shocks**
# 
# The very low bankruptcy rate (0.71%) is also informative when we read it in the
# context of the macroeconomic shocks that hit Italian firms between 2018 and 2021.
# During this period, companies were exposed to a sequence of adverse events like
# the COVID-19 pandemic.
# 
# However, the data show that **most firms did not formally default** within the
# 12-month horizon covered by our target. This is consistent with the idea that:
# 
# 1. **Shocks affect all firms, but not all in the same way.**  
#    Revenue losses, higher production costs and more expensive debt put pressure
#    on balance sheets across the board, but only the most fragile firms  
#    (high leverage, low liquidity, weak profitability) are pushed over the edge
#    into bankruptcy. Our 84 bankrupt observations are precisely this small group.
# 
# 2. **Policy support and financial flexibility slow down default.**  
#    In the COVID years, government support schemes, loan guarantees and temporary
#    moratoria helped many distressed firms to avoid immediate default. Similarly,
#    banks may have restructured or rolled over loans instead of forcing formal
#    bankruptcy. As a result, the impact of the shocks appears more clearly in
#    deteriorated financial ratios than in a large wave of legal bankruptcies.
# 
# 3. **Bankruptcy is the “tip of the iceberg” of financial distress.**  
#    The macro shocks leave a strong footprint in the distributions of leverage,
#    liquidity and profitability (which we analyse later), while the target
#    variable only captures the extreme outcome where distress becomes unsustainable.
#    This explains why the bankrupt class remains small even after several crisis
#    episodes: many firms are weakened, but only a few actually fail.
# 
# In summary, the rare-event nature of `bankruptcy_next_year = 1` is not a sign that
# macroeconomic shocks were unimportant; rather, it tells us that these shocks act
# as a **stress test on already vulnerable firms**. The goal of our model is to use
# the information in financial statements to detect exactly those companies that are
# unable to absorb the shocks and end up in bankruptcy.
# 


# ##**7. Firm-Level Patterns Across Macroeconomic Regimes**


# The following exploratory analysis is not interpreted as purely cross-sectional, because firm-level financial distributions are also shaped by the macroeconomic environment in which firms operate. For this reason, we explicitly account for three macroeconomic regimes affecting the Italian economy during the observation window:
# 
# - **2018–2019: normal risk environment**  
#   This period represents the baseline context, in which bankruptcy risk mainly reflects ordinary firm-specific weaknesses rather than a major systemic shock.
# 
# - **2020: COVID-19 with strong support measures**  
#   This year captures the pandemic shock combined with extraordinary public intervention, including guarantees, moratoria, and other emergency measures that may have delayed formal bankruptcies despite significant financial stress.
# 
# - **2021: partial normalisation and lagged effects**  
#   This phase reflects the gradual withdrawal of emergency support and the emergence of delayed distress, as some vulnerable firms could no longer avoid default.
# 
# The cell below creates a new categorical variable that assigns each observation to one of these three macroeconomic shock regimes based on `fiscal_year`. This allows the subsequent distributions to be visualised not only in aggregate form, but also by macroeconomic context.
# 
# As a result, the following plots will show whether variables such as size, leverage, liquidity, profitability, sector composition, and bankruptcy incidence remain stable across years or instead shift under different macroeconomic conditions. In this way, the EDA highlights that the same financial indicator may carry a different interpretation depending on the shock environment in which firms are observed.


def assign_macro_shock(year):
    if year in [2018, 2019]:
        return "2018–2019: normal risk environment"
    elif year == 2020:
        return "2020: COVID-19 with strong support measures"
    elif year == 2021:
        return "2021: partial normalisation and lagged effects"
    else:
        return "Other"

train_eda = train.copy()
train_eda["macro_shock"] = train_eda["fiscal_year"].apply(assign_macro_shock)

shock_order = [
    "2018–2019: normal risk environment",
    "2020: COVID-19 with strong support measures",
    "2021: partial normalisation and lagged effects"
]

# Keep only the years used in the EDA narrative
train_eda = train_eda[train_eda["macro_shock"].isin(shock_order)].copy()

print(train_eda["macro_shock"].value_counts())


# The output confirms that the macroeconomic-regime variable has been created correctly and that each observation has been assigned to one of the three shock periods used in the analysis.
# 
# In particular, the largest group is **2018–2019: normal risk environment** with **5,940 observations**, which represents the baseline macroeconomic context before the pandemic shock. The other two groups are **2020: COVID-19 with strong support measures** with **2,956 observations** and **2021: partial normalisation and lagged effects** with **2,932 observations**.
# 
# This distribution is coherent with the structure of the dataset: the pre-shock period covers two fiscal years and therefore contains more observations, while 2020 and 2021 are each represented by a single year. As a result, the following visualisations can compare firm-level distributions across distinct macroeconomic regimes, while keeping in mind that the baseline period is larger than each of the two individual shock years.


# ## **8. Exploratory Data Analysis (EDA)**


# ### **Bankruptcy rate by `fiscal_year`**
# 


# Here we look at **how the bankruptcy rate changes over time**, by computing the
# share of firms with `bankruptcy_next_year = 1` in each fiscal year (2018–2021).
# 
# This is important because:
# 
# - it tells us whether some years are *riskier* than others;
# - it helps us see if there is any visible effect of major macroeconomic shocks
#   (e.g. COVID-19) on the probability of default;
# - it prepares the ground for a **temporal validation strategy**, where we train
#   on earlier years and validate on a later year.
# 
# The code below groups the training data by `fiscal_year`, computes the mean of
# `bankruptcy_next_year` (i.e. the default rate), and plots the result as a line chart.
# It also prints the exact value of the bankruptcy rate for each year.


import matplotlib.ticker as mtick

yearly_bankruptcy = (
    train_eda.groupby("fiscal_year")["bankruptcy_next_year"]
    .mean()
    .reset_index(name="bankruptcy_rate")
)

shock_palette = {
    "2018–2019: normal risk environment": "#4C72B0",
    "2020: COVID-19 with strong support measures": "#DD8452",
    "2021: partial normalisation and lagged effects": "#55A868"
}

fig, ax = plt.subplots(figsize=(11, 5))

# Segment 2018-2019
seg_1 = yearly_bankruptcy[yearly_bankruptcy["fiscal_year"].isin([2018, 2019])]
ax.plot(
    seg_1["fiscal_year"],
    seg_1["bankruptcy_rate"],
    color=shock_palette["2018–2019: normal risk environment"],
    marker="o",
    linewidth=2.5,
    markersize=8,
    label="2018–2019: normal risk environment"
)

# Segment 2019-2020 to keep continuity into 2020
seg_2 = yearly_bankruptcy[yearly_bankruptcy["fiscal_year"].isin([2019, 2020])]
ax.plot(
    seg_2["fiscal_year"],
    seg_2["bankruptcy_rate"],
    color=shock_palette["2020: COVID-19 with strong support measures"],
    marker="o",
    linewidth=2.5,
    markersize=8,
    label="2020: COVID-19 with strong support measures"
)

# Segment 2020-2021 to keep continuity into 2021
seg_3 = yearly_bankruptcy[yearly_bankruptcy["fiscal_year"].isin([2020, 2021])]
ax.plot(
    seg_3["fiscal_year"],
    seg_3["bankruptcy_rate"],
    color=shock_palette["2021: partial normalisation and lagged effects"],
    marker="o",
    linewidth=2.5,
    markersize=8,
    label="2021: partial normalisation and lagged effects"
)

# Highlight points
ax.scatter(
    yearly_bankruptcy["fiscal_year"],
    yearly_bankruptcy["bankruptcy_rate"],
    s=65,
    color="white",
    edgecolor="black",
    linewidth=1.2,
    zorder=3
)

# Labels
for _, row in yearly_bankruptcy.iterrows():
    ax.annotate(
        f"{row['bankruptcy_rate']:.2%}",
        xy=(row["fiscal_year"], row["bankruptcy_rate"]),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

ax.set_title("Bankruptcy Rate by Fiscal Year and Macroeconomic Regime", fontsize=12, pad=12)
ax.set_xlabel("Fiscal year")
ax.set_ylabel("Bankruptcy rate")
ax.set_xticks(sorted(yearly_bankruptcy["fiscal_year"].unique()))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend outside plot
ax.legend(
    title="Macroeconomic regime",
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    frameon=False
)


plt.tight_layout()
plt.show()


import plotly.graph_objects as go

yearly_bankruptcy = (
    train_eda.groupby("fiscal_year")["bankruptcy_next_year"]
    .mean()
    .reset_index(name="bankruptcy_rate")
)

shock_palette = {
    "2018–2019: normal risk environment": "#4C72B0",
    "2020: COVID-19 with strong support measures": "#DD8452",
    "2021: partial normalisation and lagged effects": "#55A868"
}

fig = go.Figure()

# Segment 2018-2019
seg_1 = yearly_bankruptcy[yearly_bankruptcy["fiscal_year"].isin([2018, 2019])]
fig.add_trace(go.Scatter(
    x=seg_1["fiscal_year"],
    y=seg_1["bankruptcy_rate"],
    mode="lines+markers+text",
    name="2018–2019: normal risk environment",
    line=dict(color=shock_palette["2018–2019: normal risk environment"], width=3),
    marker=dict(size=9, color=shock_palette["2018–2019: normal risk environment"]),
    text=[f"{v:.2%}" for v in seg_1["bankruptcy_rate"]],
    textposition="top center",
    hovertemplate="Fiscal year: %{x}<br>Bankruptcy rate: %{y:.2%}<extra></extra>"
))

# Segment 2019-2020
seg_2 = yearly_bankruptcy[yearly_bankruptcy["fiscal_year"].isin([2019, 2020])]
fig.add_trace(go.Scatter(
    x=seg_2["fiscal_year"],
    y=seg_2["bankruptcy_rate"],
    mode="lines+markers+text",
    name="2020: COVID-19 with strong support measures",
    line=dict(color=shock_palette["2020: COVID-19 with strong support measures"], width=3),
    marker=dict(size=9, color=shock_palette["2020: COVID-19 with strong support measures"]),
    text=[f"{v:.2%}" for v in seg_2["bankruptcy_rate"]],
    textposition="top center",
    hovertemplate="Fiscal year: %{x}<br>Bankruptcy rate: %{y:.2%}<extra></extra>"
))

# Segment 2020-2021
seg_3 = yearly_bankruptcy[yearly_bankruptcy["fiscal_year"].isin([2020, 2021])]
fig.add_trace(go.Scatter(
    x=seg_3["fiscal_year"],
    y=seg_3["bankruptcy_rate"],
    mode="lines+markers+text",
    name="2021: partial normalisation and lagged effects",
    line=dict(color=shock_palette["2021: partial normalisation and lagged effects"], width=3),
    marker=dict(size=9, color=shock_palette["2021: partial normalisation and lagged effects"]),
    text=[f"{v:.2%}" for v in seg_3["bankruptcy_rate"]],
    textposition="top center",
    hovertemplate="Fiscal year: %{x}<br>Bankruptcy rate: %{y:.2%}<extra></extra>"
))

fig.update_layout(
    title="Bankruptcy Rate by Fiscal Year and Macroeconomic Regime",
    xaxis_title="Fiscal year",
    yaxis_title="Bankruptcy rate",
    template="plotly_white",
    legend_title="Macroeconomic regime",
    width=950,
    height=500
)

fig.update_xaxes(
    tickmode="array",
    tickvals=sorted(yearly_bankruptcy["fiscal_year"].unique())
)

fig.update_yaxes(
    tickformat=".2%"
)

fig.show()


# **Interpretation of the results**
# 
# The table below the plot shows that the bankruptcy rate is always around **0.5–0.8%**,
# with some small variation across years:
# 
# - 2018: **0.78%**
# - 2019: **0.81%** (slightly higher)
# - 2020: **0.54%** (noticeable drop)
# - 2021: **0.72%** (partial rebound)
# 
# So, even at its “peak”, the default rate remains below 1%, but it does not stay
# perfectly constant over time.


# **Link to macroeconomic shocks**
# 
# The small changes in the bankruptcy rate over time can be read in light of the
# shocks that hit the Italian economy:
# 
# - **2018–2019 – normal risk environment**  
#   Default rates around 0.8% are consistent with a “baseline” level of failures:
#   firms go bankrupt mainly because of structural weaknesses (business model,
#   leverage, low profitability), not because of a systemic shock.
# 
# - **2020 – COVID-19 with strong support measures**  
#   Despite the pandemic, the bankruptcy rate **drops to 0.54%**. This reflects
#   the effect of emergency policies (grants, guarantees, moratoria) and more
#   accommodating banks, which **delay many formal bankruptcies** even though
#   balance sheets are under heavy stress.
# 
# - **2021 – partial normalisation and lagged effects**  
#   The rate rises again to about **0.72%**, closer to pre-COVID levels. As support
#   measures are gradually reduced and new pressures emerge (supply-chain and
#   energy issues), some firms that were kept alive in 2020 can no longer avoid
#   default.
# 
# In summary, the shocks mainly show up as **deterioration in financial ratios**
# and only with a delay in the observed default rate. This motivates our focus on
# balance-sheet indicators (leverage, liquidity, profitability) as early warning
# signals of bankruptcy.
# 


# ###**Sectoral and geographic coverage**


# Before looking at bankruptcy risk, we first need to understand **which firms we
# are analysing**. In this section we describe the composition of the training set
# along two key dimensions:
# 
# - the **business sector** of each firm (`ateco_sector`);
# - the **Italian region** where the firm is located (`region`).
# 
# Knowing how observations are distributed across sectors and regions is crucial
# for two reasons. First, it tells us where the dataset puts most of its “weight”:
# if a few sectors or regions dominate, they will largely drive the patterns we
# see in the financial ratios and in bankruptcy outcomes. Second, it helps us
# anticipate **where macroeconomic shocks are likely to leave the strongest
# footprint** in the data (for example, in industrial or tourism-intensive areas).
# 
# The code below plots the top 10 ATECO sectors and the top 10 regions by number
# of observations in the training set.
# 
# 
# 


sorted(train["ateco_sector"].unique())

# ATECO legend
ateco_labels = {
    45: "Motor trade & repair",
    46: "Wholesale trade (excl. vehicles)",
    47: "Retail trade (excl. vehicles)",
    41: "Building construction",
    43: "Specialised construction work",
    25: "Metal products (non-machinery)",
    10: "Food manufacturing",
    62: "Software & IT consulting",
    56: "Restaurants & catering",
    71: "Architecture & engineering",
    68: "Real estate activities",
    77: "Rental & leasing",
    82: "Business support services"
}

train["ateco_sector_label"] = train["ateco_sector"].map(ateco_labels)


# [ISTAT ATECO 2007 CLASSIFICATION ](https://f-digital.co/wp-content/uploads/2024/03/CODICI-ATECO-2007-ELENCO-RIEPILOGATIVO-DI-TUTTI-I-COMPARTI.pdf)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 10 ATECO sectors by number of firms
train["ateco_sector_label"].value_counts().head(10).plot(
    kind="bar",
    ax=axes[0]
)
axes[0].set_title("Top 10 ATECO Sectors (train)")
axes[0].set_xlabel("ateco_sector")
axes[0].tick_params(axis="x", rotation=45)

# Top 10 regions by number of firms
train["region"].value_counts().head(10).plot(
    kind="bar",
    ax=axes[1]
)
axes[1].set_title("Top 10 Regions (train)")
axes[1].set_xlabel("region")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# **Interpretation of the results**
# 
# - On the **sector** side, a small number of ATECO sectors accounts for most of
#   the observations. Wholesale trade (code 46), retail trade (47) and building
#   construction (41) each exceed 1,400 firm-years in the training set. They are
#   followed by specialised construction activities (43), metal-product
#   manufacturing (25) and the food industry (10), with almost 1,000 observations
#   each. IT and software services (62), restaurants and catering (56), motor
#   trade and repair (45) and architectural/engineering services (71) close the
#   top-10 with between 500 and 900 firm-years. This means that our dataset is
#   heavily concentrated in classic **trade and construction** activities, plus a
#   few important service sectors.
# 
# - On the **regional** side, a similar concentration appears:
#   - **Lombardia** clearly dominates, followed by **Lazio** and **Emilia-Romagna**;
#   - other regions such as **Veneto**, **Campania**, **Sicilia** and **Piemonte**
#     still have several hundred observations, but noticeably fewer than Lombardia;
#   - the tail of the distribution contributes
#     relatively few firms.
# 
# In short, the dataset mainly reflects firms located in the economically stronger
# regions of Northern and Central Italy and operating in a limited set of ATECO
# sectors.
# 
# 


# ### **Bankruptcy risk by business sector and geographic region**
# 


# After looking at how many observations we have in each sector and region, we now
# focus on **how the bankruptcy rate itself varies across these groups**.
# 
# Concretely, we:
# 
# - compute the bankruptcy rate (`share of bankruptcy_next_year = 1`) for the
#   **top 10 ATECO sectors** in terms of number of observations;
# - compute the same rate for **each region**, and sort regions from highest to
#   lowest risk.
# 
# This allows us to see whether some sectors or regions are systematically more
# fragile than others, which is essential for interpreting the impact of
# macroeconomic shocks and for deciding how to use `ateco_sector` and `region`
# as features in the model.


ateco_labels = {
    45: "Motor trade & repair",
    46: "Wholesale trade (excl. vehicles)",
    47: "Retail trade (excl. vehicles)",
    41: "Building construction",
    43: "Specialised construction work",
    25: "Metal products (non-machinery)",
    10: "Food manufacturing",
    62: "Software & IT consulting",
    56: "Restaurants & catering",
    71: "Architecture & engineering",
    68: "Real estate activities",
    77: "Rental & leasing",
    82: "Business support services"
}

top_sectors = train_eda["ateco_sector"].value_counts().head(10).index.tolist()
sector_df = train_eda[train_eda["ateco_sector"].isin(top_sectors)].copy()

fig, ax = plt.subplots(figsize=(14, 5.5))

sns.barplot(
    x="ateco_sector",
    y="bankruptcy_next_year",
    hue="macro_shock",
    hue_order=shock_order,
    data=sector_df,
    order=top_sectors,
    errorbar=None,
    ax=ax
)

ax.set_title("Bankruptcy Rate by Sector (Top 10 - train) and Macroeconomic Regime")
ax.set_ylabel("Bankruptcy Rate")
ax.set_xlabel("")
ax.set_xticklabels([str(x) for x in top_sectors], rotation=45, ha="right")

ax.legend(
    title="Macroeconomic regime",
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    frameon=False
)


sector_text = "\n".join(
    [f"{code} = {ateco_labels.get(code, 'Unknown')}" for code in top_sectors]
)

fig.text(
    0.66, 0.62,
    sector_text,
    va="top",
    ha="left",
    fontsize=9
)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.show()



top_regions = train_eda["region"].value_counts().head(10).index.tolist()
region_df = train_eda[train_eda["region"].isin(top_regions)].copy()

plt.figure(figsize=(11, 5.5))
sns.barplot(
    data=region_df,
    x="region",
    y="bankruptcy_next_year",
    hue="macro_shock",
    hue_order=shock_order,
    order=top_regions,
    errorbar=None
)
plt.title("Bankruptcy Rate by Region (Top 10 - train) and Macroeconomic Regime")
plt.ylabel("Bankruptcy Rate")
plt.xlabel("")
plt.xticks(rotation=45, ha="right")
plt.legend(
    title="Macroeconomic regime",
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    frameon=False
)
plt.tight_layout()
plt.show()


import plotly.express as px
sector_agg = (
    sector_df.groupby(["ateco_sector", "macro_shock"], as_index=False)["bankruptcy_next_year"]
    .mean()
)

sector_agg["sector_label"] = sector_agg["ateco_sector"].map(
    lambda x: f"{x} - {ateco_labels.get(x, 'Unknown')}"
)

fig_sector = px.bar(
    sector_agg,
    x="ateco_sector",
    y="bankruptcy_next_year",
    color="macro_shock",
    category_orders={
        "macro_shock": shock_order,
        "ateco_sector": top_sectors
    },
    color_discrete_map=shock_palette,
    barmode="group",
    title="Bankruptcy Rate by Sector (Top 10 - train) and Macroeconomic Regime",
    labels={
        "ateco_sector": "",
        "bankruptcy_next_year": "Bankruptcy Rate",
        "macro_shock": "Macroeconomic regime"
    },
    hover_data={
        "sector_label": True,
        "ateco_sector": False,
        "macro_shock": True,
        "bankruptcy_next_year": ':.2%'
    }
)

fig_sector.update_layout(template="plotly_white", width=1100, height=550)
fig_sector.update_xaxes(
    tickmode="array",
    tickvals=top_sectors,
    ticktext=[str(x) for x in top_sectors],
    tickangle=45
)
fig_sector.update_yaxes(tickformat=".2%")

fig_sector.show()


import plotly.express as px
region_agg = (
    region_df.groupby(["region", "macro_shock"], as_index=False)["bankruptcy_next_year"]
    .mean()
)

fig_region = px.bar(
    region_agg,
    x="region",
    y="bankruptcy_next_year",
    color="macro_shock",
    category_orders={
        "macro_shock": shock_order,
        "region": top_regions
    },
    color_discrete_map=shock_palette,
    barmode="group",
    title="Bankruptcy Rate by Region (Top 10 - train) and Macroeconomic Regime",
    labels={
        "region": "",
        "bankruptcy_next_year": "Bankruptcy Rate",
        "macro_shock": "Macroeconomic regime"
    },
    hover_data={
        "region": True,
        "macro_shock": True,
        "bankruptcy_next_year": ':.2%'
    }
)

fig_region.update_layout(template="plotly_white", width=1000, height=550)
fig_region.update_xaxes(tickangle=45)
fig_region.update_yaxes(tickformat=".2%")

fig_region.show()


# **Interpretation of the sector results**
# 
# The bar plot for the top 10 ATECO sectors shows that the bankruptcy rate ranges
# roughly between **0.4% and 1.2%**:
# 
# - **Motor trade, retail and wholesale** stand out as the riskiest activities:
#   firms in the motor trade and repair sector (code **45**) and in retail trade
#   (47) have the highest default rates, around **1.1–1.2%**, while wholesale
#   trade (46) is also in the upper part of the distribution.
# - The **food industry** (10) is relatively risky as well, with a rate just
#   below **1%**, reflecting thin margins and high exposure to demand and cost
#   shocks.
# - **Building construction** (41) and **specialised construction activities**
#   (43) sit in the middle, with default rates around **0.4–0.5%**.
# - **Restaurants and catering** (56), **software and IT services** (62) and
#   **architectural and engineering services** (71) are at the lower end, with
#   default rates clearly below **0.6%**.
# 
# Even within the same country and period, some sectors have almost **three
# times** the bankruptcy rate of others. This confirms that sector risk is not
# just noise: the ATECO code carries systematic information about how fragile
# firms are.
# 
# **Link to macroeconomic shocks**
# 
# These sector differences are consistent with how the 2018–2023 shocks hit the
# Italian economy:
# 
# - **Motor trade and retail/wholesale trade** (45, 47, 46) combine high exposure
#   to **COVID demand drops and mobility restrictions** with sensitivity to
#   **energy-price increases** along the supply chain. Firms in these sectors
#   have to manage both lower revenues and higher operating costs, which makes
#   them more likely to slide into default.
# - The **food industry** (10) is vulnerable to higher input and energy costs and
#   to changes in consumption patterns, which can quickly compress margins.
# - Sectors such as **IT and software services** (62), **architecture and
#   engineering** (71) and, to a lesser extent, **restaurants and catering**
#   (56) appear more resilient in our data, either because they could adapt more
#   easily to remote work and new business models or because they benefited from
#   the rebound after COVID restrictions.
# 
# From a modelling perspective, this justifies including `ateco_sector` (via
# one-hot encoding) so that the model can assign **higher predicted risk** to
# firms operating in historically more fragile sectors like trade and motor
# services, and lower risk to structurally more resilient sectors.
# 


# **Interpretation of the regional results**
# 
# The regional bar chart shows a clear ranking of bankruptcy rates across Italian
# regions, with values roughly between **0.4% and 1.1%**:
# 
# - **Emilia-Romagna** has the highest rate, around **1.1%**, followed closely by
#   **Liguria** (~**1.0%**);
# - **Campania** and **Sardegna** follow, with rates around **0.8–0.9%**;
# - **Veneto**, **Lazio** and **Sicilia** are in the middle, around **0.7–0.8%**;
# - **Lombardia** and **Puglia** are slightly below this, around **0.6–0.65%**;
# - **Umbria**, **Toscana**, **Piemonte** and **Friuli-Venezia Giulia** show the
#   **lowest rates**, between **0.35% and 0.5%**.
# 
# Again, the differences are not huge in absolute terms, but some regions have
# **almost three times the default rate of others**, which is economically
# meaningful.
# 
# **Link to macroeconomic shocks**
# 
# These regional patterns can be related to how shocks propagate through local
# economies:
# 
# - Regions such as **Emilia-Romagna and Liguria**, with strong **industrial and
#   logistics/port activity**, are more sensitive to **energy-price shocks and
#   supply-chain disruptions**, which can explain their higher bankruptcy rates.
# - Regions with a higher weight of **tourism and services** (e.g. **Campania,
#   Sardegna, Liguria, Lazio, Sicilia**) were more exposed to **COVID-19
#   restrictions and mobility changes**, which hit many contact-intensive
#   businesses.
# - Regions like **Piemonte, Friuli-Venezia Giulia, Toscana or Umbria** show
#   lower default rates, possibly reflecting more diversified economies, less
#   energy-intensive production or a better adaptation of local firms to the new
#   environment.
# 
# 


# Taken together, the regional analysis confirms that the impact of macro shocks
# is **not spatially neutral**: certain combinations of **sector × region** are
# systematically more fragile. This supports the decision to keep both `region`
# and `ateco_sector` as features in the predictive model.


# ### **Firm size: `total_assets` and `production_value`**
# 


# In this section we look at **how big the firms in our dataset are**, using two
# basic measures of size:
# 
# - the balance sheet side: `total_assets`;
# - the income/production side: `production_value` (turnover / value of production).
# 
# Firm size matters for bankruptcy risk because:
# 
# - **small and medium enterprises (SMEs)** usually have less cash, weaker access
#   to credit and fewer buffers, so they are more vulnerable to shocks;
# - **large firms** can absorb one or two bad years more easily, but if they fail
#   the macro impact is much larger.
# 
# The code below plots, for each variable, the histogram in **levels** and the
# histogram after a **log(1 + x) transformation**, to see both the raw scale and
# a more “regular” version suitable for modelling.
# 


for col in ['total_assets', 'production_value']:
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))

    sns.histplot(
        data=train_eda,
        x=col,
        hue="macro_shock",
        hue_order=shock_order,
        palette=shock_palette,
        bins=50,
        stat="density",
        common_norm=False,
        multiple="stack",
        alpha=1,
        edgecolor="white",
        linewidth=0.4,
        ax=ax[0]
    )
    ax[0].set_title(f"{col} (level) by macroeconomic shock")
    ax[0].set_xlabel(col)
    ax[0].set_ylabel("Density")

    mask = train_eda[col] >= 0
    sns.histplot(
        data=train_eda.loc[mask].assign(log_value=np.log1p(train_eda.loc[mask, col])),
        x="log_value",
        hue="macro_shock",
        hue_order=shock_order,
        palette=shock_palette,
        bins=50,
        stat="density",
        common_norm=False,
        multiple="stack",
        alpha=1,
        edgecolor="white",
        linewidth=0.4,
        ax=ax[1]
    )
    ax[1].set_title(f"log(1 + {col}) by macroeconomic shock")
    ax[1].set_xlabel(f"log(1 + {col})")
    ax[1].set_ylabel("Density")

    # Remove axis-level legends to avoid duplicates
    if ax[0].get_legend() is not None:
        ax[0].get_legend().remove()
    if ax[1].get_legend() is not None:
        ax[1].get_legend().remove()

    # Create one figure-level legend
    legend_handles = [
        plt.Line2D([0], [0], color=shock_palette[shock], lw=6)
        for shock in shock_order
    ]

    fig.legend(
        legend_handles,
        shock_order,
        title="Macroeconomic regime",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.show()


import plotly.express as px

shock_palette = {
    "2018–2019: normal risk environment": "#4C72B0",
    "2020: COVID-19 with strong support measures": "#DD8452",
    "2021: partial normalisation and lagged effects": "#55A868"
}

for col in ['total_assets', 'production_value']:
    # Level distribution
    fig_level = px.histogram(
        train_eda,
        x=col,
        color="macro_shock",
        category_orders={"macro_shock": shock_order},
        color_discrete_map=shock_palette,
        nbins=50,
        opacity=0.75,
        barmode="overlay",
        title=f"{col} (level) by macroeconomic shock"
    )

    fig_level.update_layout(
        legend_title_text="Macro shock",
        bargap=0.05,
        template="plotly_white"
    )

    fig_level.show()

    # Log distribution
    plot_df = train_eda.loc[train_eda[col] >= 0, [col, "macro_shock"]].copy()
    plot_df[f"log_{col}"] = np.log1p(plot_df[col])

    fig_log = px.histogram(
        plot_df,
        x=f"log_{col}",
        color="macro_shock",
        category_orders={"macro_shock": shock_order},
        color_discrete_map=shock_palette,
        nbins=50,
        opacity=0.75,
        barmode="overlay",
        title=f"log(1 + {col}) by macroeconomic shock"
    )

    fig_log.update_layout(
        legend_title_text="Macro shock",
        bargap=0.05,
        template="plotly_white"
    )

    fig_log.show()


# **Interpretation of the results**
# 
# - In **level** (left-hand plots) both `total_assets` and `production_value` are
#   **strongly right-skewed**: almost all firms are concentrated very close to
#   zero, while a small number of very large firms extend the scale far to the
#   right. This is exactly what we expect in a real economy with many SMEs and
#   a few giants.
# 
# - After applying `log(1 + x)` (right-hand plots), the distributions become
#   much more symmetric and “bell-shaped”. The log transform compresses the
#   differences between very large firms and very small ones, creating a scale
#   where most observations are spread more evenly.
# 
# - We do not see obvious separate peaks or gaps, which suggests a **continuum
#   of firm sizes** rather than distinct “classes” with empty ranges in between.
# 
# Overall, the EDA indicates that our dataset is dominated by relatively small
# firms, with a long tail of larger companies.
# 
# 
# 


# **Link to macroeconomic shocks**
# 
# Firm size is a key lens to interpret the shocks of 2018–2021:
# 
# - **Smaller firms**, with lower assets and production volumes, have thinner
#   financial buffers and weaker bargaining power with banks. When revenues fall
#   or costs rise, they can run out of liquidity quickly and become prime
#   candidates for bankruptcy.
# 
# - **Larger firms** usually have more diversified activities and easier access
#   to funding, so they withstand stress for longer. However, their global supply
#   chains and energy needs mean that severe and persistent shocks can still
#   compress margins and increase leverage.
# 
# Given that the empirical distribution is heavily tilted towards **SMEs**, it is
# plausible that later we will see fat left tails in profitability and liquidity:
# macro shocks push an already large group of small and fragile firms further
# into distress.
# 


# ### **Debt structure and leverage: `total_debt`, `debt_to_assets`, `leverage`**
# 
# 


# In this section we focus on **how much firms rely on debt** and how risky their
# capital structure looks. We use three related variables:
# 
# - `total_debt` – absolute amount of debt on the balance sheet;
# - `debt_to_assets` – share of total assets financed with debt;
# - `leverage` – debt-to-equity ratio.
# 
# For bankruptcy prediction, leverage is a key driver: the more debt a firm has
# (relative to its assets and equity), the more sensitive it becomes to shocks
# that hit profits and interest rates. If earnings fall or the cost of debt rises,
# highly leveraged firms are the first to get into trouble.
# 
# The code below plots, for each of these three variables, the histogram in
# **levels** and the histogram after a **log(1 + x) transformation**.


for col in ['total_debt', 'debt_to_assets', 'leverage']:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(
        data=train_eda,
        x=col,
        hue="macro_shock",
        hue_order=shock_order,
        palette=shock_palette,
        bins=50,
        stat="density",
        common_norm=False,
        multiple="stack",
        alpha=1,
        edgecolor="white",
        linewidth=0.4,
        ax=ax[0]
    )
    ax[0].set_title(f"Distribution of {col} by macroeconomic shock")
    ax[0].set_xlabel(col)
    ax[0].set_ylabel("Density")
    mask = train_eda[col] >= 0

    sns.histplot(
        data=train_eda.loc[mask].assign(log_value=np.log1p(train_eda.loc[mask, col])),
        x="log_value",
        hue="macro_shock",
        hue_order=shock_order,
        palette=shock_palette,
        bins=50,
        stat="density",
        common_norm=False,
        multiple="stack",
        alpha=1,
        edgecolor="white",
        linewidth=0.4,
        ax=ax[1]
    )

    ax[1].set_title(f"log(1 + {col}) by macroeconomic shock")
    ax[1].set_xlabel(f"log(1 + {col})")
    ax[1].set_ylabel("Density")

    # Remove axis-level legends to avoid duplicates
    if ax[0].get_legend() is not None:
        ax[0].get_legend().remove()
    if ax[1].get_legend() is not None:
        ax[1].get_legend().remove()

    # Create one figure-level legend
    legend_handles = [
        plt.Line2D([0], [0], color=shock_palette[shock], lw=6)
        for shock in shock_order
    ]

    fig.legend(
        legend_handles,
        shock_order,
        title="Macroeconomic regime",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.show()


import plotly.graph_objects as go
from plotly.subplots import make_subplots

for col in ['total_debt', 'debt_to_assets', 'leverage']:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Distribution of {col} by macroeconomic shock",
            f"log(1 + {col}) by macroeconomic shock"
        )
    )

    # LEFT: level
    for shock in shock_order:
        subset = train_eda.loc[train_eda["macro_shock"] == shock, col].dropna()
        fig.add_trace(
            go.Histogram(
                x=subset,
                name=shock,
                marker_color=shock_palette[shock],
                histnorm="probability density",
                nbinsx=50,
                opacity=1
            ),
            row=1, col=1
        )

    # RIGHT: log
    for shock in shock_order:
        subset = train_eda.loc[
            (train_eda["macro_shock"] == shock) & (train_eda[col] >= 0),
            col
        ].dropna()
        fig.add_trace(
            go.Histogram(
                x=np.log1p(subset),
                name=shock,
                marker_color=shock_palette[shock],
                histnorm="probability density",
                nbinsx=50,
                opacity=1,
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        width=1200,
        height=450,
        legend_title="Macroeconomic regime",
        title_text=f"{col}: level and log distributions by macroeconomic shock"
    )

    fig.update_xaxes(title_text=col, row=1, col=1)
    fig.update_xaxes(title_text=f"log(1 + {col})", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    fig.show()


# **Interpretation of the results**
# 
# - For `total_debt` (top row), the level distribution is **extremely right-skewed**:
#   most firms have relatively small amounts of debt, while a small group carries
#   very large debts. After the log transform, the distribution becomes much more
#   symmetric, indicating that `log(1 + total_debt)` is a more suitable scale for
#   modelling.
# 
# - For `debt_to_assets` (middle row), most firms cluster in a band roughly between
#   **0.4 and 0.8**, meaning that 40–80% of their assets are financed by debt.
#   There is also a **thin right tail** with values close to or above 1, which is
#   unusual (debt as large as or larger than total assets) and signals firms that
#   are very highly geared or have accounting issues. The log transform smooths
#   the shape but the concentration around high debt shares remains visible.
# 
# - For `leverage` (bottom row), the level distribution shows that:
#   - a large majority of firms have leverage in a relatively low range (e.g. 0–5);
#   - a **few extreme outliers** reach very high values (above 20, 50 or even 100),
#     which often corresponds to firms with almost zero equity.
#   In the log scale, we see a long right tail: most firms have moderate leverage,
#   but a non-negligible subset is **heavily over-leveraged**.
# 
# Together, these plots tell us that the dataset contains many firms with a
# standard debt profile, but also a smaller group with **very aggressive debt
# ratios**. These highly indebted firms are natural candidates for bankruptcy.
# 
# 
# 


# **Link to macroeconomic shocks and bankruptcy risk**
# 
# The shape of these debt ratios fits well with the macro environment:
# 
# - During the COVID phase many firms took on **additional debt** to bridge the
#   revenue collapse, pushing some observations further to the right in
#   `total_debt` and increasing both `debt_to_assets` and `leverage`.
# 
# 
# In summary, long right tails and outliers in these ratios identify firms that
# entered or exited the shock period with a **heavy debt burden**—exactly the
# companies we expect to see more often in the bankrupt class.
# 


# ### **Short-term liquidity: `current_ratio` and `quick_ratio`**
# 
# 


# Here we analyze **how well firms can cover their short-term debts with their
# short-term assets**, using two standard liquidity ratios:
# 
# - `current_ratio = current_assets / short_term_debt`
# - `quick_ratio  = (current_assets − inventory) / short_term_debt`
# 
# A value around **1** is often used as a rough threshold:
# - if the ratio is **below 1**, current assets are not enough to repay
#   all short-term liabilities;
# - if it is **well above 1**, the firm has a comfortable liquidity buffer.
# 
# For bankruptcy prediction, liquidity is often the spark that triggers default:
# even a business that is not “rotten” in terms of long-run profitability can go
# bankrupt if it cannot pay its bills on time.
# 
# The code below plots the distribution of `current_ratio` and `quick_ratio`
# and draws a **vertical red line at 1** to highlight the short-term solvency
# threshold.
# 


for col in ['current_ratio', 'quick_ratio']:
    fig, ax = plt.subplots(figsize=(9, 4))

    sns.histplot(
        data=train_eda,
        x=col,
        hue="macro_shock",
        hue_order=shock_order,
        palette=shock_palette,
        bins=50,
        stat="density",
        common_norm=False,
        multiple="stack",
        alpha=1,
        edgecolor="white",
        linewidth=0.4,
        ax=ax
    )
    ax.axvline(1, color='red', linestyle='--', label='threshold 1')
    ax.set_title(f"Distribution of {col} by macroeconomic shock")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    mask = train_eda[col] >= 0

    # Remove axis-level legends to avoid duplicates
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Create one figure-level legend
    legend_handles = [
        plt.Line2D([0], [0], color=shock_palette[shock], lw=6)
        for shock in shock_order
    ]

    fig.legend(
        legend_handles,
        shock_order,
        title="Macroeconomic regime",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.show()



import plotly.express as px
import plotly.graph_objects as go

for col in ['current_ratio', 'quick_ratio']:
    fig = px.histogram(
        train_eda,
        x=col,
        color="macro_shock",
        category_orders={"macro_shock": shock_order},
        color_discrete_map=shock_palette,
        nbins=50,
        histnorm="probability density",
        barmode="stack",
        title=f"Distribution of {col} by macroeconomic shock"
    )

    # Add threshold line at x = 1
    fig.add_vline(
        x=1,
        line_width=2,
        line_dash="dash",
        line_color="red"
    )

    # Optional annotation for threshold
    fig.add_annotation(
        x=1,
        y=1,
        yref="paper",
        text="threshold 1",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="red", size=10)
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Macroeconomic regime",
        width=950,
        height=450,
        bargap=0.02
    )

    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Density")

    fig.show()


# **Interpretation of the results**
# 
# - For both `current_ratio` and `quick_ratio`, most observations lie **to the
#   right of the red line at 1**. This means that, in “normal” times, the
#   typical firm in the dataset has enough current assets to cover its
#   short-term debt.
# 
# - There is still a **non-negligible mass of firms with ratios below 1**:
#   these are companies that already enter the period with tight liquidity.
#   They may survive as long as cash inflows are stable, but they have very
#   little room for adverse shocks.
# 
# - The right tails show some firms with very high ratios (2, 3 or more),
#   indicating large liquidity buffers. These firms are much better positioned
#   to absorb temporary drops in revenue or increases in costs.
# 
# - As expected, the **quick ratio is generally lower than the current ratio**:
#   once inventories are excluded from current assets, some firms move closer to
#   the threshold of 1. This highlights the difference between “accounting”
#   liquidity and more conservative, cash-like liquidity.
# 
# Overall, the distributions suggest a mixed picture: many firms are reasonably
# liquid, but a sizeable group sits close to or below the short-term solvency
# threshold.
# 
# 
# 


# **Link to macroeconomic shocks**
# 
# Short-term liquidity is one of the first places where macro shocks bite:
# 
# - Revenue drops and delayed payments during COVID tend to **push
#   `current_ratio` and `quick_ratio` down**, especially for firms that were
#   already close to 1.
# 
# - Higher production and energy costs further **drain cash buffers**, while
#   rolling over short-term debt becomes more expensive when interest rates rise.
# 
# A large mass of firms with `current_ratio < 1` or only slightly above 1
# indicates a system that enters these shocks **already under pressure**. Any new
# disturbance can easily turn stretched liquidity positions into actual
# bankruptcies.
# 


# ### **Profitability and bottom line: `net_profit_loss`, `profit_margin`, `roi`, `roe`**
# 
# 


# In this section we analyse **how much profit firms are able to generate**, using
# four related indicators:
# 
# - `net_profit_loss` – bottom-line net result in absolute value;
# - `profit_margin` – net profit relative to revenues;
# - `roi` – return on assets (profit relative to total assets);
# - `roe` – return on equity (profit relative to shareholders’ equity).
# 
# For bankruptcy prediction, profitability acts as a **cushion against shocks**:
# a firm that starts from healthy margins can absorb temporary losses, while a
# firm that is already close to zero or in the red can be pushed into default by
# a relatively small macro shock.
# 
# The code below plots, for each variable, the histogram in **levels** and the
# histogram after a **log(1 + x) transformation**, so that we can see both the
# raw distribution (including negative values) and a compressed version that
# highlights the central mass of observations.
# 


for col in ['net_profit_loss', 'profit_margin', 'roi', 'roe']:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(
        data=train_eda,
        x=col,
        hue="macro_shock",
        hue_order=shock_order,
        palette=shock_palette,
        bins=50,
        stat="density",
        common_norm=False,
        multiple="stack",
        alpha=1,
        edgecolor="white",
        linewidth=0.4,
        ax=ax[0]
    )
    ax[0].set_title(f"Distribution of {col} by macroeconomic shock")
    ax[0].set_xlabel(col)
    ax[0].set_ylabel("Density")
    mask = train_eda[col] >= 0

    sns.histplot(
        data=train_eda.loc[mask].assign(log_value=np.log1p(train_eda.loc[mask, col])),
        x="log_value",
        hue="macro_shock",
        hue_order=shock_order,
        palette=shock_palette,
        bins=50,
        stat="density",
        common_norm=False,
        multiple="stack",
        alpha=1,
        edgecolor="white",
        linewidth=0.4,
        ax=ax[1]
    )

    ax[1].set_title(f"log(1 + {col}) by macroeconomic shock")
    ax[1].set_xlabel(f"log(1 + {col})")
    ax[1].set_ylabel("Density")

    # Remove axis-level legends to avoid duplicates
    if ax[0].get_legend() is not None:
        ax[0].get_legend().remove()
    if ax[1].get_legend() is not None:
        ax[1].get_legend().remove()

    # Create one figure-level legend
    legend_handles = [
        plt.Line2D([0], [0], color=shock_palette[shock], lw=6)
        for shock in shock_order
    ]

    fig.legend(
        legend_handles,
        shock_order,
        title="Macroeconomic regime",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.show()


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

shock_palette = {
    "2018–2019: normal risk environment": "#4C72B0",
    "2020: COVID-19 with strong support measures": "#DD8452",
    "2021: partial normalisation and lagged effects": "#55A868"
}

for col in ['net_profit_loss', 'profit_margin', 'roi', 'roe']:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Distribution of {col} by macroeconomic shock",
            f"log(1 + {col}) by macroeconomic shock"
        )
    )

    # LEFT: level distribution
    for shock in shock_order:
        subset = train_eda.loc[train_eda["macro_shock"] == shock, col].dropna()
        fig.add_trace(
            go.Histogram(
                x=subset,
                name=shock,
                marker_color=shock_palette[shock],
                histnorm="probability density",
                nbinsx=50,
                opacity=1
            ),
            row=1, col=1
        )

    # RIGHT: log distribution only for non-negative values
    for shock in shock_order:
        subset = train_eda.loc[
            (train_eda["macro_shock"] == shock) & (train_eda[col] >= 0),
            col
        ].dropna()

        fig.add_trace(
            go.Histogram(
                x=np.log1p(subset),
                name=shock,
                marker_color=shock_palette[shock],
                histnorm="probability density",
                nbinsx=50,
                opacity=1,
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        width=1200,
        height=450,
        legend_title="Macroeconomic regime",
        title_text=f"{col}: level and log distributions by macroeconomic shock"
    )

    fig.update_xaxes(title_text=col, row=1, col=1)
    fig.update_xaxes(title_text=f"log(1 + {col})", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    fig.show()


# **Interpretation of the results**
# 
# - For `net_profit_loss`, the level distribution is strongly asymmetric:
#   - most firms cluster around moderately positive net results;
#   - there is a **left tail of loss-making firms** with negative net income;
#   - a few very profitable firms create a long right tail.
#   The log-transformed version compresses the scale, making the central mass of
#   firms easier to visualise, but the existence of both gain and loss firms
#   remains clear.
# 
# - For `profit_margin`, we see a concentration of observations in a relatively
#   narrow positive range (for example between 0 and 10%), but with a **visible
#   bump of companies with negative margins**. This indicates that a non-trivial
#   share of firms is already operating at a loss even before we consider future
#   shocks. The distribution is quite wide, suggesting heterogeneous business
#   models and exposure to costs.
# 
# - `roi` (return on assets) has a shape closer to a bell curve centred on small
#   positive values, but again with a **left tail** extending into negative
#   returns. Many firms generate modest returns on their asset base, while a
#   minority either performs very well or very poorly.
# 
# - `roe` (return on equity) is the most extreme: the level distribution shows
#   some **very large negative values**. This typically happens when equity is
#   very small or close to zero. While not a direct indicator of bankruptcy, this reflects severe financial fragility. In such cases even a small loss generates a huge
#   negative ROE. After applying the log transform, most “normal” values are
#   compressed into a more regular shape, but the fact that some firms have
#   extremely fragile equity positions is clear.
# 
# Overall, these distributions tell us that:
# 
# - the majority of firms has **positive but relatively thin margins**;
# - a significant minority is already in the **loss-making zone** or very close
#   to it;
# - some firms have **extremely weak equity**, which makes ROE explode on the
#   downside and puts them at high risk if conditions worsen.
# 
# 
# 


# **Link to macroeconomic shocks and other financial variables**
# 
# Macro shocks leave a very direct numerical footprint on profitability:
# 
# - COVID-related revenue losses immediately hit `net_profit_loss` and
#   `profit_margin`, pushing many firms from small profits into outright losses.
# 
# - Supply-chain disruptions and the energy crisis raised `production_costs` and
#   operating expenses, squeezing margins and shifting more observations into the
#   left side of the `roi` and `roe` distributions.
# 
# 
# The unstable shapes and pronounced left tails in these variables can be read as
# the **numeric imprint** of these shocks. Firms that end up in the extreme
# negative part of the distributions are precisely those for which weak
# profitability, high costs and expensive debt make bankruptcy a very plausible
# outcome.
# 


# ### **Comparison between bankrupt and non-bankrupt firms: leverage, liquidity, margins**


# In this section we want to see **how the distributions of key financial
# variables change with the target `bankruptcy_next_year`**.
# 
# The practical question is: compared with healthy firms (`bankruptcy_next_year = 0`),
# do bankrupt firms (`bankruptcy_next_year = 1`) have:
# 
# - higher leverage?
# - lower liquidity?
# - worse profit margins?
# 
# If the answer is yes, these variables are strong candidates for the model and
# they also confirm the economic story we have in mind: firms that enter the
# COVID/energy/interest-rate shocks with weak balance sheets are the ones that
# end up defaulting.
# 
# The boxplots below compare, for bankrupt vs non-bankrupt firms:
# 
# - `leverage` (debt/equity),
# - `current_ratio` (short-term liquidity),
# - `profit_margin` (profitability).
# 


features_to_check = ['leverage', 'debt_to_assets', 'current_ratio', 'profit_margin', 'roi']

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

legend_handles = None
legend_labels = None

for i, col in enumerate(features_to_check):
    sns.boxplot(
        data=train_eda,
        x='macro_shock',
        y=col,
        hue='bankruptcy_next_year',
        order=shock_order,
        ax=axes[i]
    )

    axes[i].set_title(f"{col} by macro shock and bankruptcy")
    axes[i].tick_params(axis='x', rotation=20)
    axes[i].set_xlabel("")

    # Make only leverage easier to read
    if col == 'leverage':
        axes[i].set_yscale('log')
        axes[i].set_ylabel("leverage (log scale)")
    else:
        axes[i].set_ylabel(col)

    # Save legend handles only once
    if legend_handles is None:
        legend_handles, legend_labels = axes[i].get_legend_handles_labels()

    # Remove subplot legend
    if axes[i].get_legend() is not None:
        axes[i].get_legend().remove()

# Remove empty last subplot
fig.delaxes(axes[5])

# Create one common legend outside the plot
fig.legend(
    legend_handles,
    ["Healthy", "Bankrupt"],
    title="bankruptcy_next_year",
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=False
)

plt.tight_layout(rect=[0, 0, 0.86, 1])
plt.show()

# **Interpretation of the results**
# 
# - **Leverage vs bankruptcy**  
#   The median leverage for bankrupt firms is slightly higher than for healthy
#   firms, and the bankrupt group shows **several very high outliers**. This
#   suggests that many failed firms were more heavily indebted, and a subset of
#   them had extremely fragile equity positions (very high debt/equity).
# 
# - **Current ratio vs bankruptcy**  
#   Bankrupt firms tend to have a **lower median current ratio** and a more
#   compressed box: they operate closer to the short-term solvency threshold,
#   with fewer observations in the “comfortable” liquidity range. Healthy firms
#   display a higher median and a wider spread, indicating that many of them have
#   enough current assets to cover short-term debt.
# 
# - **Profit margin vs bankruptcy**  
#   The box for healthy firms is centred on **clearly positive margins**, even if
#   there are some negative outliers. For bankrupt firms, the median profit margin
#   is **lower and closer to zero**, and the entire box is shifted downward. This
#   tells us that failed firms were already less profitable, with less room to
#   absorb additional shocks.
# 
# In short, the boxplots support the idea that, relative to survivors, bankrupt
# firms combine **higher leverage, weaker liquidity and thinner margins**.
# 
# 
# 


# **Link to macroeconomic shocks and other financial variables**
# 
# Macro shocks leave a very direct numerical footprint on profitability:
# 
# - COVID-related revenue losses immediately hit `net_profit_loss` and
#   `profit_margin`, pushing many firms from small profits into outright losses.
# 
# - Supply-chain disruptions and the energy crisis raised `production_costs` and
#   operating expenses, squeezing margins and shifting more observations into the
#   left side of the `roi` and `roe` distributions.
# 
# - Higher interest rates increase `financial_expenses`, eating into the bottom
#   line and worsening `net_profit_loss` and `roe`, especially for highly
#   leveraged firms.
# 
# The unstable shapes and pronounced left tails in these variables can be read as
# the **numeric imprint** of these shocks. Firms that end up in the extreme
# negative part of the distributions are precisely those for which weak
# profitability, high costs and expensive debt make bankruptcy a very plausible
# outcome.
# 


# ### **Train vs Test comparison: temporal drift in key features**
# 


# Here we compare the **distributions of key features between the training period
# (2018–2021) and the test period (2022–2023)**.
# 
# The main question is:  
# _Is the world we want to predict (2022–2023) similar to the world we trained on
# (2018–2021), or have recent shocks (energy crisis, interest-rate hikes) shifted
# the distributions?_
# 
# We focus on a few variables that should react strongly to the macro environment:
# 
# - `production_costs` → directly affected by the 2021–2022 energy crisis;
# - `financial_expenses` → directly affected by the 2022–2023 interest-rate hikes;
# - `profit_margin`, `leverage`, `current_ratio` → summary indicators of how
#   margins, debt and liquidity are stressed.
# 
# The function below overlays the kernel density estimates (KDEs) of each variable
# for the train set (`Train 2018–2021`) and the test set (`Test 2022–2023`).  
# By eye, we can detect **shifts** (curves moving to the right or left), **changes
# in shape** (wider distributions, heavier tails) and therefore possible temporal
# drift that the model will have to handle.
# 


def compare_train_test_dist(col):
    fig, ax = plt.subplots(figsize=(6,3))
    sns.kdeplot(train[col], label='Train (2018-2021)', fill=True, alpha=0.5, ax=ax)
    sns.kdeplot(test[col], label='Test (2022-2023)', fill=True, alpha=0.5, ax=ax)
    ax.set_title(f"Train vs Test distribution of {col}")
    ax.legend(
    title= "Dataset Temporal Split",
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    frameon=False
    )

    plt.show()

for col in ['leverage', 'current_ratio', 'profit_margin', 'production_costs', 'financial_expenses']:
    if col in train.columns and col in test.columns:
        compare_train_test_dist(col)


import plotly.express as px
import pandas as pd

def compare_train_test_dist_plotly(col):
    plot_df = pd.DataFrame({
        col: pd.concat([train[col], test[col]], ignore_index=True),
        "Dataset Temporal Split": (
            ["Train (2018-2021)"] * len(train) +
            ["Test (2022-2023)"] * len(test)
        )
    }).dropna()

    fig = px.histogram(
        plot_df,
        x=col,
        color="Dataset Temporal Split",
        nbins=50,
        histnorm="probability density",
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={
            "Train (2018-2021)": "#4C72B0",
            "Test (2022-2023)": "#DD8452"
        },
        title=f"Train vs Test distribution of {col}"
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Dataset Temporal Split",
        width=900,
        height=450,
        bargap=0.02
    )

    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Density")

    fig.show()

for col in ['leverage', 'current_ratio', 'profit_margin', 'production_costs', 'financial_expenses']:
    if col in train.columns and col in test.columns:
        compare_train_test_dist_plotly(col)


# **Interpretation of the results**
# 
# - For **leverage**, both train and test are concentrated in the range
#   **0–5**, with a clear peak around **1–2**. The train period (2018–2021)
#   shows a few extreme outliers above **40–60**, while in the test period
#   (2022–2023) the density beyond **20** is already very small. So the typical
#   firm in both periods has moderate leverage, but there are some highly
#   leveraged firms mostly in the training years.
# 
# - For **current_ratio**, the bulk of observations lies between **1.0** and
#   **3.0**, with the density peak around **1.5–2.0** in both train and test.
#   The test curve puts slightly more mass just above **1.0**, meaning that in
#   2022–2023 more firms operate closer to the short-term solvency threshold,
#   while very high values above **3.5–4.0** are rare in both periods.
# 
# - For **profit_margin**, values range roughly from **–0.15** to **+0.13**, but
#   most firms are between **0** and **0.10**. The main hump of the distribution
#   is around **3–8%** in both train and test. The test curve is a bit more
#   pronounced in the lower part of this range, suggesting slightly thinner
#   margins on average in 2022–2023.
# 
# - For **production_costs**, the x-axis goes up to about **8×10¹¹**, but almost
#   all the density is concentrated below **1×10¹¹**. In the test period the
#   curve stretches a bit further to the right, with more mass between
#   **2×10¹⁰** and **5×10¹⁰**, which is consistent with higher operating and
#   energy costs in the later years.
# 
# - For **financial_expenses**, most firms in both periods have values well
#   below **2×10⁹**, but the test distribution shows a slightly heavier right
#   tail, with more observations between **5×10⁸** and **1×10⁹**. This points to
#   higher interest expenses in 2022–2023 compared with 2018–2021, even if truly
#   extreme values above **1×10¹⁰** remain very rare.
# 
# Overall, these numbers indicate that the test period is **more costly and
# financially “heavier”** (higher production costs and interest expenses), while
# leverage levels and basic liquidity remain broadly in the same ranges but with
# a bit more pressure near the risky zones (current_ratio just above 1, thinner
# margins).
# 
# 
# 
# 


# **Link to macroeconomic shocks**
# 
# These graphs show **which firms did not withstand the shocks**:
# 
# - Firms with **high leverage** are more exposed to interest-rate increases and
#   to the need for extra debt during crises.
# 
# - Firms with **low liquidity** (current ratio close to or below 1) are more
#   vulnerable to sudden drops in revenue and delayed payments from customers.
# 
# - Firms with **already weak profit margins** cannot absorb extra costs from
#   supply-chain disruptions and higher energy prices.
# 
# The concentration of bankruptcies in the “wrong” tails of these boxplots
# (high leverage, low liquidity, low margins) confirms that the COVID shock, the
# energy crisis and tighter monetary policy hit hardest those companies that
# were **already financially fragile** before the shocks.
# 


# ### **Outlier diagnostics**


# In this section we examine **extreme values and tail behaviour** in firm-level financial variables. The goal is to distinguish between economically meaningful extremes, common in financial data, and potential data anomalies.
# 
# Specifically, we aim to:
# - identify variables with **heavy tails or strong skewness**;
# -	detect observations that may represent **structural financial distress** rather than errors;
# - highlight values that could distort modelling if left untreated.
# 
# Understanding the nature of outliers is essential before applying transformations, since extreme values in financial ratios and size variables often carry important signals of **financial fragility and bankruptcy risk.**


# Select only numeric variables (exclude the target)
numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'bankruptcy_next_year']

#Skewness: detects asymmetric distributions
# High absolute values indicate long tailswh
skewness = train[numeric_cols].skew().sort_values(ascending=False)

print("Skewness of all numeric variables (sorted):")
display(skewness.to_frame("skewness"))

# Extreme percentiles: detect heavy tails
# We inspect 1%, 5%, 95%, 99% percentiles
percentiles = train[numeric_cols].quantile([0.01, 0.05, 0.50, 0.95, 0.99]).T
percentiles.columns = ["1%", "5%", "50%", "95%", "99%"]
percentiles["max"] = train[numeric_cols].max()

# Ratio between extreme tail (99th percentile)
# and the median highlights extreme right tails
percentiles["tail_ratio_99_med"] = (
    percentiles["99%"] / (percentiles["50%"] + 1e-9)
)

print("Variables with strongest extreme tails:")
display(
    percentiles
    .sort_values("tail_ratio_99_med", ascending=False)

)

print("Outlier screening completed.")


# **How to read the distribution diagnostics.**
# 
# The two tables above summarize the shape and tail behaviour of the variables, helping distinguish structural economic heterogeneity from potential data errors.
# 
# **Skewness (distribution asymmetry).**
# Skewness measures how much a variable's distribution is unbalanced:
# - **≈ 0** -> symmetric distribution
# - **> 0** -> long right tail (few extremely large values)
# - **&lt; 0** -> long left tail
# - **skew > 1** -> strong asymmetry
# - **skew >> 10** -> extremely heavy tails
# 
# In firm-level financial data, large positive skewness typically indicates the coexistence of a few very large firms alongside many small ones, reflecting real economic structure rather than measurement error.
# 
# Strong negative skewness (e.g., **ROE**) often signals financial distress dynamics, where small equity bases amplify losses.
# 
# **Extreme tail diagnostics (percentiles & tail ratio).**
# The second table evaluates how extreme the upper tail is using percentiles and dispersion metrics:
# - **median (50%)** represents the typical firm;
# - **95% and 99% percentiles** show the upper range of large observations;
# - **max** highlights the most extreme value;
# - **tail_ratio_99_med** (99th percentile ÷ median) measures how far extreme values lie from the typical firm.
# 
# Interpretation of the `tail_ratio_99_med`:
# - 1-3 -> normal dispersion
# - 5-10 -> strong asymmetry
# - 10-50 -> heavy tails
# - 50 -> extreme heterogeneity
# 
# Very large ratios indicate rare but economically meaningful events such as mergers, restructuring, or severe distress, rather than data errors.
# 
# Together, these diagnostics confirm that extreme values reflect real economic variability and justify the use of robust transformations (log scaling and winsorization) instead of removing observations.


# **Interpretation of the results**
# 
# The distribution diagnostics reveal **substantial heterogeneity** in scale,
# skewness, and tail behaviour across firm-level financial variables.
# 
# - Magnitude variables such as `revenue_change`, `financial_income`,
#   `total_fixed_assets`, `short_term_debt`, `total_assets`, and `total_debt`
#   exhibit extremely high positive skewness. This reflects the coexistence of
#   a small number of very large firms alongside a majority of smaller ones,
#   producing heavy right tails.
# 
# - Cost and financing variables (`financial_expenses`, `production_costs`,
#   `operating_income`) also show strong right-skewed distributions and very
#   large upper percentiles, consistent with firm size heterogeneity and
#   differences in financing burden.
# 
# - Liquidity and leverage indicators (`current_ratio`, `quick_ratio`,
#   `debt_to_assets`, `leverage`) display more moderate skewness and relatively
#   compact percentile ranges, suggesting that extreme values are less frequent
#   but still economically meaningful when present.
# 
# - Profitability measures highlight financial fragility:
#   - `roe` exhibits extremely negative skewness and a very wide lower tail,
#   reflecting how small equity bases can amplify losses;
#   - `profit_margin` and `roi` show moderate dispersion with asymmetric tails.
# 
# - Structural variables such as `years_in_business`, `fiscal_year`, and sector
#   identifiers show limited skewness and tail dispersion, confirming that
#   extreme values are not a concern for these features.
# 
# The tail ratio analysis confirms that extreme values are concentrated in
# size, income, and cost variables, where firm heterogeneity and macroeconomic
# shocks generate substantial scale effects.
# 
# Overall, these patterns indicate that extreme observations carry economic
# meaning rather than measurement error. Preprocessing should therefore focus on
# mitigating scale effects in magnitude variables, limiting the influence of
# extreme tails where appropriate, and preserving distress signals embedded in
# leverage and profitability metrics.
# 
# This supports the use of robust transformations and selective winsorization
# rather than indiscriminate outlier removal.


# ###**Visual inspection of outliers**


# After identifying heavy tails and skewness in the distribution diagnostics, we perform a visual inspection of selected variables to better understand the nature of extreme values.
# 
# Rather than examining every feature, we focus on a **small set of economically meaningful indicators** representing firm size, leverage, liquidity, profitability, and operating costs. This targeted inspection helps distinguish structurally large values and **genuine distress signals from potential anomalies**, while keeping the analysis interpretable and aligned with bankruptcy risk dynamics.


# Variables to show in the EDA (representative groups: size, debt, liquidity, profitability, costs)
eda_outlier_vars = [
    'total_assets',
    'total_debt',
    'debt_to_assets',
    'leverage',
    'current_ratio',
    'quick_ratio',
    'profit_margin',
    'roi',
    'roe',
    'financial_expenses',
    'production_costs'
]

# Safety: keep only variables that exist in the dataset
selected_vars = [c for c in eda_outlier_vars if c in train.columns]

print("Selected variables for boxplot inspection:")
print(selected_vars)

# Plot boxplots (single-variable distributions)
def boxplots_for_outliers(df, cols, n_cols=3, figsize_per_col=5, figsize_per_row=4):
    n = len(cols)
    n_rows = math.ceil(n / n_cols)
    plt.figure(figsize=(figsize_per_col * n_cols, figsize_per_row * n_rows))

    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(y=df[col])
        plt.title(col)
        plt.xlabel("")

    plt.tight_layout()
    plt.show()

boxplots_for_outliers(train, selected_vars, n_cols=3)

# If some plots look "flat" because of extreme scale, re-plot on log scale
# NOTE: we only apply log1p to strictly non-negative variables; otherwise we skip.
nonneg_vars = [c for c in selected_vars if (train[c].min() >= 0)]
if len(nonneg_vars) > 0:
    train_log = train.copy()
    for c in nonneg_vars:
        train_log[c] = np.log1p(train_log[c])

    print("\nRe-plotting non-negative selected variables with log1p scale:")
    boxplots_for_outliers(train_log, nonneg_vars, n_cols=3)


# **Why focus on a subset of variables**
# 
# The distribution diagnostics identified heavy tails across a large number of
# magnitude variables. However, visual inspection focuses on a selected subset
# based on two criteria:
# 
# - variables exhibiting the most extreme tail behaviour, indicating strong scale
#   effects and potential **influence on model stability**;
# 
# - financially meaningful indicators directly linked to **bankruptcy risk**,
#   including leverage, liquidity, and profitability ratios.
# 
# This selection balances **statistical relevance with economic interpretability**,
# allowing inspection of the most informative variables while avoiding redundancy.
# 
# **Interpretation of the results**
# 
# The boxplots confirm that extreme values are an intrinsic characteristic of
# firm-level financial data and reflect **structural economic differences rather
# than measurement errors.**
# 
# - Size-related variables such as `total_assets` and `total_debt` display pronounced
#   right tails, capturing the coexistence of **very large firms** alongside a
#   majority of small and medium enterprises.
# 
# - Leverage indicators (`debt_to_assets`, `leverage`) exhibit extreme upper values
#   that often correspond to situations of **negative or near-zero equity**,
#   signalling genuine financial distress rather than anomalies.
# 
# - Liquidity ratios (`current_ratio`, `quick_ratio`) show relatively compact
#   distributions, with occasional high outliers representing firms holding
#   unusually large liquidity buffers.
# 
# - Profitability measures highlight financial fragility:
#   - `profit_margin` presents a noticeable negative tail, indicating **loss-making firms;**
#   - `roi` shows moderate dispersion with both positive and negative extremes;
#   - `roe` exhibits extremely negative values, confirming how small equity bases
#   can **amplify losses.**
# 
# - Cost variables (`financial_expenses`, `production_costs`) remain strongly
#   right-skewed, reflecting firm size heterogeneity and differences in
#   financing burden.
# 
# Overall, the selected variables confirm that extreme observations carry economic
# meaning and are closely linked to financial distress dynamics. Rather than
# removing these values, preprocessing will focus on **mitigating scale distortions**
# while preserving the financial signals associated with bankruptcy risk.


# 


# ## **9. Data Preprocessing**


# To ensure a reliable and economically meaningful modelling pipeline, the preprocessing phase is divided into **two complementary parts**.
# 
# The **first part** focuses on preparing a **clean and consistent base dataset**. In this stage we handle **missing values** and apply **basic outlier control (winsorization)** using statistics computed **only on the development window (2018-2020)**. The **2021 validation year** is treated as future data: it receives those preprocessing rules, but it does not contribute to estimating them.
# 
# The resulting datasets represent clean **"pre-log / pre-feature-engineering" versions** of the data. They maintain the **original financial magnitudes** required for ratio construction and **time-based feature creation** in the next stage.
# 
# The **second part of preprocessing** is applied **after Feature Engineering**. Once the engineered features are created, we perform the model-oriented transformations required for machine learning, including **log transformations of selected magnitudes**, **categorical encoding fit only on 2018-2020**, and **feature scaling fit only on the development data**.
# 
# This separation ensures that:
# - **feature engineering** is computed on **economically meaningful raw magnitudes**
# - **all learned preprocessing rules** are estimated only on the **past development years**
# - the entire pipeline remains **time-aware and free from avoidable leakage**


# ###**Handling missing values**


# Before modelling, we ensure that the dataset is **complete and consistent**, since missing values can distort statistical patterns and degrade model performance.
# 
# Guided by the **Exploratory Data Analysis (EDA)**, we examine the extent and distribution of missingness and apply a **robust, train-driven imputation strategy** that preserves the economic meaning of the variables while preventing **data leakage**.
# 
# The preprocessing strategy follows three main principles:
# 
# - **Columns with excessive missing values** (e.g., above 40%) are identified using only the **2018-2020 development window** and then dropped consistently across development, validation, and test data.
# - **Numeric variables** are imputed using the **median**, a robust statistic that limits the influence of extreme values.
# - **Categorical variables** are imputed using the **mode** (most frequent value), ensuring consistency while preserving the most representative category.
# 
# Importantly, all imputation statistics are **computed only on 2018-2020** and then applied to:
# - the development rows used for model fitting
# - the **2021 validation year**
# - and the **2022-2023 official test period**
# 
# This preserves a realistic forecasting scenario in which the future can receive the learned rules, but cannot help define them.


df_train_fit_processed = train_fit_raw.copy()
df_val_2021_processed = val_2021_raw.copy()
df_test_processed = test_raw.copy()

# Targets exist only in labelled train data
target_cols = ["revenue_change", "financial_health_class", "bankruptcy_next_year"]

# Handle missing values
# 1) Drop columns with too many missing values (estimated on 2018-2020 only)
display(missing_df_train)

missing_pct_train_fit = df_train_fit_processed.isnull().mean() * 100
high_missing_cols = missing_pct_train_fit[missing_pct_train_fit > 40].index.tolist()
print(f"Dropping columns with > 40% missing (estimated on 2018-2020 only): {high_missing_cols}")

for frame in (df_train_fit_processed, df_val_2021_processed, df_test_processed):
    frame.drop(columns=high_missing_cols, inplace=True, errors="ignore")

# 2) Imputation: numeric and categorical
numeric_train = df_train_fit_processed.select_dtypes(include=[np.number]).columns.tolist()
numeric_train = [c for c in numeric_train if c not in target_cols]
categorical_train = df_train_fit_processed.select_dtypes(exclude=[np.number]).columns.tolist()

# Keep only columns available in development, validation, and test
numeric_cols = [
    c for c in numeric_train
    if c in df_val_2021_processed.columns and c in df_test_processed.columns
]
categorical_cols = [
    c for c in categorical_train
    if c in df_val_2021_processed.columns and c in df_test_processed.columns
]

# Numeric imputation with median (computed on 2018-2020, applied to all)
medians = df_train_fit_processed[numeric_cols].median()
for frame in (df_train_fit_processed, df_val_2021_processed, df_test_processed):
    frame[numeric_cols] = frame[numeric_cols].fillna(medians)

# Categorical imputation with mode (computed on 2018-2020, applied to all)
for col in categorical_cols:
    mode_series = df_train_fit_processed[col].mode(dropna=True)
    if len(mode_series) > 0:
        mode_value = mode_series.iloc[0]
        for frame in (df_train_fit_processed, df_val_2021_processed, df_test_processed):
            frame[col] = frame[col].fillna(mode_value)

print("Missing values handled using 2018-2020 preprocessing statistics only.")

# Save clean pre-log datasets for development, validation, and test
df_train_fit_clean_prelog = df_train_fit_processed.copy()
df_val_2021_clean_prelog = df_val_2021_processed.copy()
df_test_clean_prelog = df_test_processed.copy()

# Rebuild the combined labelled train table for downstream FE steps
df_train_clean_prelog = (
    pd.concat([df_train_fit_clean_prelog, df_val_2021_clean_prelog], ignore_index=True)
    .sort_values(["company_id", "fiscal_year"])
    .reset_index(drop=True)
)

print("Saved leakage-safe clean pre-log datasets for development, validation, and test.")
print(f"Development rows: {df_train_fit_clean_prelog.shape[0]}")
print(f"Validation rows:  {df_val_2021_clean_prelog.shape[0]}")
print(f"Test rows:        {df_test_clean_prelog.shape[0]}")

# 
# **Interpretation of the results**
# 
# The missing-value diagnostics indicate a **limited presence of incomplete observations** across the dataset.
# 
# - No variables exceed the **40% missing threshold**, therefore **no columns are removed**, preserving the full informational content of the dataset.
# - Missing values are concentrated in a small subset of variables:
#   - `revenue_change` shows a moderate proportion of missing observations, likely reflecting unavailable historical comparisons for some firms.
#   - `province`, `roe`, and `leverage` display only **marginal missingness**.
# 
# Numeric variables are imputed using the **median**, which preserves the distributional structure while reducing sensitivity to extreme values.  
# Categorical variables are imputed using the **mode**, ensuring consistency with the most common category observed in the training data.
# 
# Importantly, all imputation statistics are computed **only on the training set** and then applied to the test set, preventing **data leakage** and maintaining a realistic modelling framework.
# 
# Overall, the relatively low level of missingness confirms **good data quality**, and the chosen imputation strategy preserves statistical integrity while maintaining model robustness.
# 
# 
# 


# 
# **Saving a Clean Pre-Transformation Dataset**
# 
# After handling missing values, we store a **clean copy of the dataset before any further transformations**.
# 
# This intermediate version preserves the **original scale of financial variables**, which is essential for **accounting-based feature engineering** performed in the next stage, such as:
# 
# - Altman-like **Z-score indicators**
# - financial structure **ratios**
# - balance-sheet relationship features.
# 
# These indicators are economically meaningful **only when computed on variables expressed in their original accounting scale**.  
# Applying log transformations beforehand would distort these relationships.
# 
# Therefore, we save:
# 
# - **`df_train_clean_prelog`**
# - **`df_test_clean_prelog`**
# 
# which represent the **clean but non-transformed datasets** used as the starting point for the subsequent **Feature Engineering** phase.
# 


# ###**Winsorization and Outlier Handling**
# 
# 
# 
# 
# 
# 


# After handling missing values, we address the presence of **extreme observations** in financial ratios.
# 
# Financial ratios may exhibit unusually large or small values due to accounting effects such as **very small denominators**, temporary shocks, or reporting inconsistencies. If left untreated, these extreme observations can distort statistical distributions and negatively affect downstream modelling steps.
# 
# To mitigate this issue, we apply **winsorization** to the main ratio variables. Specifically, values are **capped at the 1st and 99th percentiles**, limiting the influence of extreme tails while preserving the overall structure of the data.
# 
# To prevent **data leakage**, percentile thresholds are computed **only on the 2018-2020 development dataset** and then applied consistently to the **2021 validation year** and the **2022-2023 test dataset**.
# 
# Importantly, this transformation is applied to the clean pre-log accounting tables so that the original financial scale is preserved for the subsequent feature-engineering steps.


# A. Winsorization on ratios (1st-99th percentile)
ratio_cols = [
    "current_ratio",
    "quick_ratio",
    "leverage",
    "debt_to_assets",
    "profit_margin",
    "roi",
    "roe",
]

# Keep only ratio columns available in development, validation, and test
ratio_cols = [
    c for c in ratio_cols
    if c in df_train_fit_clean_prelog.columns
    and c in df_val_2021_clean_prelog.columns
    and c in df_test_clean_prelog.columns
]

if len(ratio_cols) > 0:
    # Estimate clipping bounds on 2018-2020 only
    lower_bounds = df_train_fit_clean_prelog[ratio_cols].quantile(0.01)
    upper_bounds = df_train_fit_clean_prelog[ratio_cols].quantile(0.99)

    print("Winsorizing ratios (1st-99th percentile, estimated on 2018-2020 only):")
    for col in ratio_cols:
        orig_min = df_train_fit_clean_prelog[col].min()
        orig_max = df_train_fit_clean_prelog[col].max()

        # Apply the same bounds to development, validation, and test
        for frame in (df_train_fit_clean_prelog, df_val_2021_clean_prelog, df_test_clean_prelog):
            frame[col] = frame[col].clip(lower_bounds[col], upper_bounds[col])

        new_min = df_train_fit_clean_prelog[col].min()
        new_max = df_train_fit_clean_prelog[col].max()
        print(f"  - {col:15}: [{orig_min:10.4f}, {orig_max:10.4f}] -> [{new_min:10.4f}, {new_max:10.4f}]")
else:
    print("No ratio columns found across development, validation, and test. Skipping winsorization step.")

# Rebuild the combined labelled train table after development-based winsorization
df_train_clean_prelog = (
    pd.concat([df_train_fit_clean_prelog, df_val_2021_clean_prelog], ignore_index=True)
    .sort_values(["company_id", "fiscal_year"])
    .reset_index(drop=True)
)

print("Pre-log winsorization completed with development-only bounds.")

# **Interpretation of the results**
# 
# The output confirms that several financial ratios contain **extreme values** that can significantly distort the distribution of the data.
# 
# Winsorization works by **capping extreme observations** rather than removing them entirely. Values below the **1st percentile** are replaced by the 1st percentile value, and values above the **99th percentile** are replaced by the 99th percentile value.
# 
# This approach reduces the influence of rare extreme observations while preserving the **overall distribution and economic meaning** of the ratios.
# 
# The transformation is particularly relevant for variables such as:
# 
# - **leverage**, where extreme values can arise when equity becomes very small;
# - **roe**, which may produce very large negative values during periods of financial distress.
# 
# By limiting extreme tails, winsorization improves the **statistical robustness** of the dataset while maintaining the interpretability of the underlying financial indicators.
# 
# The cleaned datasets (`df_train_clean_prelog` and `df_test_clean_prelog`) will now be used as the **base input for the Feature Engineering phase**, where additional accounting-based and temporal features will be constructed.
# 


# ##**10. Feature Engineering**


# In this section we construct additional variables that capture
# economically meaningful patterns in firm financial data.
# 
# While preprocessing focused on data cleaning and stabilization,
# feature engineering aims to extract signals related to:
# 
# - **financial distress**
# - **temporal financial dynamics**
# - **sector-relative performance**
# - **structural firm profiles**
# 
# These derived indicators enrich the dataset with information that
# may help machine learning models identify patterns associated with
# **future bankruptcy risk** while maintaining economic interpretability.


# ###**Data Preparation for FE**


# 
# **Data Preparation for Feature Engineering**
# 
# Before constructing new features, we prepare the dataset to ensure that all variables are in a **consistent and economically meaningful format** for feature engineering.
# 
# The feature engineering phase starts from the **clean pre-log datasets** (`df_train_clean_prelog` and `df_test_clean_prelog`). These datasets already include:
# 
# - **missing-value imputation**
# - **outlier handling through winsorization**
# 
# but still preserve the **original accounting scale** of financial variables.
# 
# Maintaining the original scale is essential because many accounting-based indicators — such as **Altman-like scores, financial ratios, and balance-sheet relationships** — rely on raw financial magnitudes. Applying logarithmic transformations beforehand would distort these relationships.
# 
# At this stage we therefore:
# 
# - initialize the **Feature Engineering datasets** (`df_train_fe`, `df_test_fe`);
# - remove redundant variables not required for feature construction;
# - ensure key categorical variables are in a **usable format**;
# - preserve identifiers such as **`company_id`** and **`fiscal_year`**, which are necessary for company-level and time-based feature construction.
# 
# Finally, the datasets are **sorted by company and year** to prepare the structure required for **temporal feature generation** (e.g., growth rates, trends, and volatility measures).
# 


def combine_train_periods(train_fit_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    # Rebuild the full labelled train table after leakage-safe preprocessing / FE steps
    combined = pd.concat([train_fit_df, val_df], ignore_index=True)
    sort_cols = [c for c in ["company_id", "fiscal_year"] if c in combined.columns]
    if len(sort_cols) > 0:
        combined = combined.sort_values(sort_cols)
    return combined.reset_index(drop=True)


# Start FE from the clean pre-log tables
df_train_fit_fe = df_train_fit_clean_prelog.copy()
df_val_2021_fe = df_val_2021_clean_prelog.copy()
df_test_fe = df_test_clean_prelog.copy()

# Remove presentation-only label columns if still present
for frame in (df_train_fit_fe, df_val_2021_fe, df_test_fe):
    frame.drop(columns=["ateco_sector_label"], errors="ignore", inplace=True)

# Targets exist only in labelled train data
target_cols = [
    c for c in ["revenue_change", "financial_health_class", "bankruptcy_next_year"]
    if c in df_train_fit_fe.columns
]

# Standardize categorical columns before FE creation
for col in ["ateco_sector", "region", "legal_form"]:
    for frame in (df_train_fit_fe, df_val_2021_fe, df_test_fe):
        if col in frame.columns:
            frame[col] = frame[col].astype(str)

# Sort each table chronologically for time-based feature construction
for frame in (df_train_fit_fe, df_val_2021_fe, df_test_fe):
    sort_cols = [c for c in ["company_id", "fiscal_year"] if c in frame.columns]
    if len(sort_cols) > 0:
        frame.sort_values(sort_cols, inplace=True)
        frame.reset_index(drop=True, inplace=True)

# Rebuild the combined labelled train table for downstream FE steps
df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

# Keep track of columns shared by development, validation, and test
common_feature_cols = [
    c for c in df_train_fit_fe.columns
    if c in df_val_2021_fe.columns and c in df_test_fe.columns
]

print(f"Development FE shape: {df_train_fit_fe.shape}")
print(f"Validation FE shape:  {df_val_2021_fe.shape}")
print(f"Combined train FE:    {df_train_fe.shape}")
print(f"Test FE shape:        {df_test_fe.shape}")
print(f"Train-only target columns: {target_cols}")
print(f"Common columns available for FE: {len(common_feature_cols)}")
print("Feature Engineering base datasets prepared with an explicit temporal split.")

# **Interpretation of the results**
# 
# The preparation step confirms that the **Feature Engineering datasets are correctly initialized** from the cleaned preprocessing outputs.
# 
# At this point the notebook carries three aligned tables through the feature-engineering stage:
# 
# - **development data (2018-2020)** used to learn all train-driven rules
# - **validation data (2021)** treated as a future year
# - **official test data (2022-2023)** used only for final prediction
# 
# The difference in column counts between the combined training-side tables and the official test table still corresponds to the **target variables that exist only in the labelled data**:
# 
# - `revenue_change`
# - `financial_health_class`
# - `bankruptcy_next_year`
# 
# These variables are intentionally absent from the test set to avoid **information leakage**.
# 
# During this preparation stage, categorical variables such as **sector, region, and legal form** are converted into a consistent format so that they can be used in subsequent **group-based or sector-level feature calculations**.
# 
# Sorting the data by **`company_id`** and **`fiscal_year`** ensures that observations for each firm are chronologically ordered. This is a crucial prerequisite for constructing **temporal financial features**, including:
# 
# - year-over-year growth indicators
# - trend variables
# - historical volatility measures
# 
# The datasets are now ready for the **Feature Engineering stage**, where additional indicators capturing financial structure, temporal dynamics, and sector-relative performance will be created without breaking the temporal split.


# ###**Advanced Financial Ratios**


# **Altman-like Z-Score (Financial Distress Indicator)**
# 
# One of the most widely used indicators of corporate financial distress is the **Altman Z-score**, originally developed to assess bankruptcy risk using accounting ratios derived from firms' financial statements.
# 
# In this section we construct an **Altman-like Z-score indicator** adapted to the structure of our dataset. The objective is to capture a firm's **overall financial health** by combining multiple balance-sheet and performance ratios into a single synthetic indicator.
# 
# Because the dataset does not contain some variables required by the original Altman formulation, such as **retained earnings** and the **market value of equity**, we construct an adapted version of the indicator using available accounting variables. In particular:
# 
# - **shareholders' equity** is used as a proxy for retained earnings
# - an accounting-based leverage ratio (**equity / total liabilities**) replaces the original market-value-based leverage component
# 
# For this reason, the resulting indicator is referred to as an **Altman-like Z-score**, as it preserves the economic structure of the original model while adapting it to the variables available in the dataset.
# 
# The calculation relies on five key financial components:
# 
# - **Working Capital / Total Assets**
# - **Equity / Total Assets** (proxy for retained earnings)
# - **Operating Income / Total Assets** (EBIT-based profitability)
# - **Equity / Total Liabilities**
# - **Sales (Production Value) / Total Assets**
# 
# These ratios summarize important aspects of a firm's financial structure, including **liquidity, profitability, leverage, and asset productivity**.
# 
# The indicator is computed using the **clean pre-log accounting datasets** for development, validation, and test so that all ratios are calculated using the **original financial magnitudes**, preserving their economic interpretation. Any fallback median used to fill residual missing scores is estimated **only on 2018-2020**.
# 
# The resulting Altman-like score is then added as a new feature (`altman_z_like_score`) to the **Feature Engineering datasets**, providing the model with a compact measure of **financial distress risk**.


def calculate_altman_like_score(df):
    # Compute an Altman-like distress indicator from accounting-based ratios
    df = df.copy()

    total_assets = df["total_assets"].replace(0, np.nan)
    total_liabilities = df["total_debt"].replace(0, np.nan)

    # Approximate working capital using current assets and short-term debt when available
    if "short_term_debt" in df.columns:
        working_capital = df["current_assets"] - df["short_term_debt"]
    else:
        working_capital = df["current_assets"]

    x1 = working_capital / total_assets
    x2 = df["shareholders_equity"] / total_assets
    x3 = df["operating_income"] / total_assets
    x4 = df["shareholders_equity"] / total_liabilities
    x5 = df["production_value"] / total_assets

    z = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
    return z.replace([np.inf, -np.inf], np.nan)


# Compute the Altman-like score separately for development, validation, and test
z_train_fit = calculate_altman_like_score(df_train_fit_clean_prelog)
z_val_2021 = calculate_altman_like_score(df_val_2021_clean_prelog)
z_test = calculate_altman_like_score(df_test_clean_prelog)

# Fill missing score values using the DEVELOPMENT median only
z_median = z_train_fit.median()
z_train_fit = z_train_fit.fillna(z_median)
z_val_2021 = z_val_2021.fillna(z_median)
z_test = z_test.fillna(z_median)

# Add the engineered score to the FE tables
df_train_fit_fe["altman_z_like_score"] = z_train_fit.values
df_val_2021_fe["altman_z_like_score"] = z_val_2021.values
df_test_fe["altman_z_like_score"] = z_test.values
df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

# Report the score distribution on the development window
z_mean = z_train_fit.mean()
z_med = z_train_fit.median()

print("Altman-like Z-score calculated.")
print(f"Development mean score:   {z_mean:.2f}")
print(f"Development median score: {z_med:.2f}")
print(f"Development Safe Zone (>2.9): {(z_train_fit > 2.9).mean():.2%}")
print(f"Development Distress Zone (<1.23): {(z_train_fit < 1.23).mean():.2%}")

# 
# **Interpretation of the results**
# 
# The Altman-like Z-score provides a **synthetic measure of firm financial stability** by combining multiple accounting-based ratios into a single indicator.
# 
# Higher values of the score generally correspond to **stronger financial conditions**, while lower values indicate **greater financial distress risk**.
# 
# The summary statistics show:
# 
# - **Mean score:** approximately 2.89  
# - **Median score:** approximately 2.68  
# 
# These values suggest that, on average, firms in the dataset fall within a **relatively stable financial range**, although the distribution still contains a subset of companies experiencing significant financial stress.
# 
# Using commonly adopted Altman-style thresholds:
# 
# - Firms with **Z > 2.9** are considered to be in the **Safe Zone**, indicating strong financial conditions.
# - Firms with **Z < 1.23** fall into the **Distress Zone**, indicating elevated bankruptcy risk.
# 
# In the dataset:
# 
# - roughly **42% of firms** fall within the **Safe Zone**,  
# - while about **4% of firms** are classified within the **Distress Zone**.
# 
# These values are consistent with the expectation that **financial distress events are relatively rare**, which aligns with the typical **class imbalance** observed in bankruptcy prediction datasets.
# 
# By integrating this indicator into the feature set (`altman_z_like_score`), the model gains access to a **well-established financial distress signal** derived directly from firms’ accounting fundamentals.
# 


# ### **Temporal Financial Features (Growth, Trends, and Volatility)**
# 
# 


# Corporate financial distress rarely appears suddenly. In most cases, it emerges through **progressive changes in a firm's financial dynamics**, such as declining revenues, deteriorating profitability, increasing leverage, or unstable operating performance.
# 
# To capture these patterns, we construct a set of **temporal financial features** that describe how key firm variables evolve over time.
# 
# Using the company identifier (**`company_id`**) together with the chronological ordering provided by **`fiscal_year`**, we compute firm-level time-series indicators based on the historical evolution of important financial quantities.
# 
# Three types of temporal signals are extracted:
# 
# - **Year-over-year growth indicators**, measuring how core financial variables change relative to the previous year.
# - **Trend indicators**, capturing whether financial variables are improving or deteriorating over time.
# - **Historical volatility measures**, describing the stability or instability of key financial quantities across multiple years.
# 
# These features are computed using the **clean pre-log datasets** so that all growth rates and volatility measures are derived from the **original financial magnitudes**, preserving their economic interpretation.
# 
# To preserve a realistic forecasting structure, development, validation, and test observations are temporarily combined **only for backward-looking company histories**. After the temporal variables are computed in chronological order, the datasets are immediately separated again into **2018-2020**, **2021**, and **2022-2023** tables.
# 
# The resulting indicators enrich the dataset with information about **financial trajectories and instability patterns**, which are often strong early signals of **future bankruptcy risk**.


def calculate_temporal_features(dataset_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Build temporal features jointly, but keep development, validation, and test separable
    prepared_frames = []
    common_cols = sorted(set.intersection(*(set(df.columns) for df in dataset_map.values())))

    for dataset_name, df in dataset_map.items():
        frame = df.copy()
        frame["_dataset"] = dataset_name
        frame["_orig_idx"] = np.arange(len(frame))
        prepared_frames.append(frame[common_cols + ["_dataset", "_orig_idx"]])

    # Sort chronologically within each company so features remain backward-looking
    combined = pd.concat(prepared_frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["company_id", "fiscal_year"]).reset_index(drop=True)

    # Create growth and trend signals for key financial quantities
    growth_cols = ["production_value", "total_assets", "total_debt", "operating_income"]
    growth_cols = [c for c in growth_cols if c in combined.columns]

    for col in growth_cols:
        combined[f"{col}_growth"] = combined.groupby("company_id")[col].pct_change()
        combined[f"{col}_growth"] = (
            combined[f"{col}_growth"]
            .replace([np.inf, -np.inf], np.nan)
            .clip(-1, 5)
            .fillna(0)
        )
        combined[f"{col}_trend_up"] = (combined[f"{col}_growth"] > 0).astype(int)

    # Create expanding volatility measures for selected operating quantities
    vol_cols = ["production_value", "operating_income"]
    vol_cols = [c for c in vol_cols if c in combined.columns]

    for col in vol_cols:
        combined[f"{col}_volatility"] = (
            combined.groupby("company_id")[col]
            .expanding()
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    # Split the combined table back into the original datasets
    outputs = {}
    for dataset_name in dataset_map:
        outputs[dataset_name] = (
            combined[combined["_dataset"] == dataset_name]
            .sort_values("_orig_idx")
            .drop(columns=["_dataset", "_orig_idx"])
            .reset_index(drop=True)
        )

    return outputs


# Compute temporal features for development, validation, and test
temporal_sets = calculate_temporal_features(
    {
        "train_fit": df_train_fit_clean_prelog,
        "val_2021": df_val_2021_clean_prelog,
        "test": df_test_clean_prelog,
    }
)

# Keep only the newly created temporal feature columns
temporal_feature_cols = [
    c for c in temporal_sets["train_fit"].columns
    if c.endswith("_growth") or c.endswith("_trend_up") or c.endswith("_volatility")
]

# Add the temporal features to the FE tables
for col in temporal_feature_cols:
    df_train_fit_fe[col] = temporal_sets["train_fit"][col].values
    df_val_2021_fe[col] = temporal_sets["val_2021"][col].values
    df_test_fe[col] = temporal_sets["test"][col].values

df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

print("Temporal features extended with growth, volatility, and trend signals.")
print(f"Temporal features created: {len(temporal_feature_cols)}")

# 
# **Interpretation of the results**
# 
# The temporal feature construction successfully extends the dataset with additional indicators capturing the **dynamic evolution of firms’ financial performance**.
# 
# For each firm, the following information is extracted:
# 
# - **Growth indicators** measuring year-over-year changes in key financial quantities such as revenue (`production_value`), assets, debt, and operating income.
# - **Trend variables** identifying whether a firm's financial performance is improving or deteriorating relative to the previous year.
# - **Volatility measures** representing the historical variability of important financial variables, computed as the expanding standard deviation within each firm.
# 
# These variables capture aspects of firm behavior that static financial ratios cannot fully represent. For example:
# 
# - Persistent **declines in revenue or operating income** may signal emerging financial distress.
# - Rapid increases in **debt relative to assets** may indicate growing leverage pressure.
# - High **volatility in revenue or operating income** may reflect unstable business conditions or operational risk.
# 
# To ensure robustness, extreme growth values are capped and missing observations generated by the first time step are handled appropriately.
# 
# Overall, these temporal indicators provide the model with information about **financial trajectories**, allowing it to detect patterns of **gradual deterioration or instability** that often precede bankruptcy events.
# 


# ###**Sector-Relative Financial Features**


# 
# 
# Financial indicators often depend strongly on the **industry in which a firm operates**. For example, leverage levels, profitability margins, and asset utilization can vary substantially across sectors due to differences in capital intensity, business models, and competitive dynamics.
# 
# As a result, evaluating a firm’s financial ratios **in absolute terms alone may be misleading**. A leverage ratio that appears high in one sector may be perfectly normal in another.
# 
# To address this issue, we construct **sector-relative financial indicators** that compare each firm's ratios with the **typical values observed in its industry**.
# 
# Specifically, we compute the **median value of selected financial ratios within each ATECO sector**, using only observations from the **training period (2018–2020)**. Restricting the calculation to the training years prevents **information leakage into the validation period**.
# 
# For each firm, we then measure how its ratios differ from the corresponding **sector median**, generating features that capture **relative performance within the industry**.
# 
# The ratios used for this comparison include:
# 
# - **Return on Equity (ROE)**
# - **Return on Investment (ROI)**
# - **Leverage**
# - **Profit Margin**
# 
# These sector-relative indicators help the model identify firms that perform **significantly better or worse than their industry peers**, which can be an important signal of **financial strength or distress**.
# 


# Build sector-relative versions of key financial ratios
ratios = ["roe", "roi", "leverage", "profit_margin"]
existing_ratios = [c for c in ratios if c in df_train_fit_fe.columns]

# Estimate sector medians on DEVELOPMENT data only (2018-2020)
sector_medians = df_train_fit_fe.groupby("ateco_sector")[existing_ratios].median()
global_ratio_medians = df_train_fit_fe[existing_ratios].median()

def add_sector_relatives(df, medians, global_medians, ratio_cols):
    # Compare each firm’s ratio with the median level of its sector
    df = df.copy()
    for col in ratio_cols:
        sector_val = df["ateco_sector"].map(medians[col]).fillna(global_medians[col])
        df[f"{col}_vs_sector"] = df[col] - sector_val
    return df

# Apply the same sector benchmarks to development, validation, and test
df_train_fit_fe = add_sector_relatives(df_train_fit_fe, sector_medians, global_ratio_medians, existing_ratios)
df_val_2021_fe = add_sector_relatives(df_val_2021_fe, sector_medians, global_ratio_medians, existing_ratios)
df_test_fe = add_sector_relatives(df_test_fe, sector_medians, global_ratio_medians, existing_ratios)
df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

print(f"Sector-relative features calculated for: {existing_ratios}")
display(
    df_train_fit_fe[
        ["company_id", "fiscal_year", "ateco_sector"]
        + [c for c in df_train_fit_fe.columns if "_vs_sector" in c]
    ].head()
)

# 
# **Interpretation of the results**
# 
# The sector-relative feature construction successfully creates new indicators that measure **how each firm's financial ratios compare to the typical levels observed in its industry**.
# 
# For each selected ratio — **ROE, ROI, leverage, and profit margin** — the model now observes the **difference between a firm's value and the median value within its sector**.
# 
# This transformation provides several advantages:
# 
# - It adjusts financial indicators for **structural differences across industries**.
# - It highlights firms that **significantly outperform or underperform their sector peers**.
# - It allows the model to detect **relative financial weakness**, which may be more informative than absolute ratio levels.
# 
# The output above shows the first observations of these new features.  
# Each value represents the **difference between the firm's ratio and the sector median**.
# 
# For example:
# 
# - `roe_vs_sector = 0.1788` means the firm's **ROE is about 17.9 percentage points higher than the median ROE of its sector**.
# - `roe_vs_sector = -0.6894` indicates that the firm's ROE is **substantially below the sector median**, signaling potential underperformance.
# 
# Similarly:
# 
# - A firm with **positive profitability differences** may indicate stronger financial performance relative to competitors.
# - A firm with **higher-than-sector leverage** may face greater financial risk.
# - Firms consistently performing **below sector medians** may be more vulnerable to financial distress.
# 
# To ensure methodological correctness, sector medians are calculated **only using training data**, preventing information from the validation period from influencing the feature construction process.
# 
# Additionally, global medians are used as a fallback when a sector is **unseen or missing**, ensuring that the transformation remains robust for all observations.
# 
# Overall, these sector-relative indicators introduce a **contextual financial perspective**, enabling the model to evaluate firms not only based on their raw financial ratios but also relative to the **competitive environment in which they operate**.
# 


# ###**Correlation-Based Feature Diagnostics**


# Before performing feature selection, we examine the **correlation structure among numerical variables** to identify potential cases of **multicollinearity**.
# 
# Highly correlated variables often contain **redundant information**, meaning that several features may be capturing essentially the same underlying financial dimension.
# 
# To investigate this, we compute the **pairwise Pearson correlation matrix** using **only the 2018-2020 development data**, excluding:
# 
# - identifiers (e.g., `company_id`)
# - the target variable (`bankruptcy_next_year`)
# - leakage-prone variables
# - variables that are not intended for modeling
# 
# We then identify pairs of variables whose **absolute correlation exceeds 0.9**, which indicates very strong linear dependence.
# 
# The goal of this step is **diagnostic rather than prescriptive**: the analysis highlights candidate features that may later be removed or consolidated during the **feature selection stage**.
# 
# Importantly, the diagnostics are kept within the development window so that the **2021 validation year** does not influence any structural feature-pruning choice.


# Correlation diagnostics on DEVELOPMENT data only (2018-2020)
exclude_corr = ["company_id", "bankruptcy_next_year", "financial_health_class", "revenue_change"]
corr_source = df_train_fit_fe.copy()

numeric_df = corr_source.select_dtypes(include=[np.number]).drop(
    columns=[c for c in exclude_corr if c in corr_source.columns],
    errors="ignore"
)

# Identify very highly correlated features before manual pruning
corr_matrix = numeric_df.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.9)]

print("High-correlation features (>0.9) to consider dropping (estimated on 2018-2020 only):")
print(to_drop)

high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and abs(val) > 0.9:
            high_corr_pairs.append((row, col, val))

if len(high_corr_pairs) > 0:
    print("Highly correlated pairs:")
    for row, col, val in high_corr_pairs:
        print(f"{row}  <->  {col}: corr = {val:.3f}")

# Build a compact heatmap using the most redundancy-prone features
exclude_corr = ["company_id", "bankruptcy_next_year", "financial_health_class", "revenue_change"]
numeric_df = df_train_fit_fe.select_dtypes(include=[np.number]).drop(
    columns=[c for c in exclude_corr if c in df_train_fit_fe.columns],
    errors="ignore"
)

corr_matrix = numeric_df.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and abs(val) > 0.9:
            high_corr_pairs.append((row, col, val))

from collections import Counter

# Count how often each feature appears in very high-correlation pairs
feature_counts = Counter()
for row, col, _ in high_corr_pairs:
    feature_counts[row] += 1
    feature_counts[col] += 1

# Keep the most redundant features for a compact visualization
top_features = [feat for feat, _ in feature_counts.most_common(4)]

print("Selected features for compact heatmap (development years only):")
print(top_features)

corr_small = corr_matrix.loc[top_features, top_features]
mask = np.triu(np.ones_like(corr_small, dtype=bool))

plt.figure(figsize=(6, 5))
sns.heatmap(
    corr_small,
    mask=mask,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    cmap="Blues",
    vmin=0.9,
    vmax=1.0
)
plt.title("Correlation structure among the most redundant features")
plt.tight_layout()
plt.show()

# Full heatmap focused only on very high correlations
exclude_corr = ["company_id", "bankruptcy_next_year", "financial_health_class", "revenue_change"]
numeric_df = df_train_fit_fe.select_dtypes(include=[np.number]).drop(
    columns=[c for c in exclude_corr if c in df_train_fit_fe.columns],
    errors="ignore"
)

corr_matrix = numeric_df.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.9)]

print("High-correlation features (>0.9) to consider dropping (development years only):")
print(to_drop)

high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and abs(val) > 0.9:
            high_corr_pairs.append((row, col, val))

if len(high_corr_pairs) > 0:
    print()
    print("Highly correlated pairs:")
    for row, col, val in high_corr_pairs:
        print(f"{row}  <->  {col}: corr = {val:.3f}")

# Keep only correlations above the chosen threshold for visualization
corr_high_only = corr_matrix.where(corr_matrix.abs() > 0.9, 0)
annot_matrix = corr_matrix.where(corr_matrix.abs() > 0.9).round(2)

plt.figure(figsize=(16, 12))
ax = sns.heatmap(
    corr_high_only,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

# Grey out cells below the threshold to keep the focus on redundancy
for i in range(corr_high_only.shape[0]):
    for j in range(corr_high_only.shape[1]):
        if corr_high_only.iloc[i, j] == 0:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="lightgrey", ec="white", lw=0.5))

# Add text annotations only where the absolute correlation is high
for i in range(corr_high_only.shape[0]):
    for j in range(corr_high_only.shape[1]):
        val = annot_matrix.iloc[i, j]
        if pd.notna(val):
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}", ha="center", va="center", fontsize=8)

plt.title("Heatmap of correlations: only |corr| > 0.9 highlighted")
plt.tight_layout()
plt.show()

# Visualize the most highly correlated feature pairs on development data only
exclude_corr = ["company_id", "bankruptcy_next_year", "financial_health_class", "revenue_change"]
numeric_df = df_train_fit_fe.select_dtypes(include=[np.number]).drop(
    columns=[c for c in exclude_corr if c in df_train_fit_fe.columns],
    errors="ignore"
)

corr_matrix = numeric_df.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and abs(val) > 0.9:
            high_corr_pairs.append((row, col, val))

# Build a compact table of the strongest correlation pairs
corr_table = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"])
corr_table["Abs Correlation"] = corr_table["Correlation"].abs()
corr_table["Pair"] = corr_table["Feature 1"] + " ↔ " + corr_table["Feature 2"]
corr_table = corr_table.sort_values("Abs Correlation", ascending=False).head(12)

plt.figure(figsize=(10, 6))
sns.barplot(data=corr_table, y="Pair", x="Abs Correlation", color="#6D8F71")
plt.xlim(0.9, 1.0)
plt.xlabel("Absolute correlation")
plt.ylabel("")
plt.title("Top highly correlated feature pairs")
plt.tight_layout()
plt.show()

# 
# **Interpretation of the results**
# 
# The correlation analysis reveals several groups of variables with **very high pairwise correlations (above 0.9)**. This is expected in financial datasets because many accounting variables are mechanically linked through **balance-sheet identities and financial ratios**.
# 
# For example, strong correlations appear among variables such as:
# 
# - **total assets, shareholders’ equity, and total debt**, which are structurally related through accounting relationships;
# - **production value, production costs, and financial expenses**, which often scale together with firm size;
# - some **engineered features and their raw counterparts**, which naturally share overlapping information.
# 
# Similarly, certain derived indicators, such as sector-relative features or growth metrics, may exhibit high correlation with their base variables.
# 
# The purpose of this diagnostic output is to **highlight potential redundancy** in the feature space. In particular, variables listed in the “high-correlation features” output represent candidates that may later be simplified or removed during the **feature selection stage**.
# 
# However, these variables are **not automatically dropped at this stage**. Tree-based machine learning models typically handle correlated predictors reasonably well, and some correlated variables may still provide complementary information for prediction.
# 
# Therefore, the correlation diagnostics serve as a **guide for informed feature pruning**, ensuring that the final modeling dataset balances:
# 
# - **predictive power**,  
# - **economic interpretability**
# - **statistical efficiency**.


# ###**Redundancy-Based Feature Selection**


# After examining the correlation diagnostics, we perform a **manual redundancy-based feature selection** step to simplify the feature space and reduce unnecessary overlap between variables.
# 
# Financial datasets often contain groups of variables that are **structurally or mechanically related**, especially those derived from balance-sheet identities or closely related accounting measures. Including many highly correlated variables can increase model complexity without adding meaningful predictive information.
# 
# Based on the previous correlation analysis and financial interpretation of the variables, we remove selected features belonging to four main categories:
# 
# 1. **Highly redundant balance-sheet quantities**, such as multiple measures of firm size or liabilities that convey similar information.
# 2. **Overlapping operating variables**, where several accounting measures scale together with firm activity.
# 3. **Near-duplicate liquidity indicators**, where ratios capture essentially the same financial concept.
# 4. **Raw variables replaced by engineered features**, such as sector-relative or derived indicators that provide more informative signals.
# 
# Importantly, feature removal is decided using the **2018-2020 development data only** and then the same drop list is applied consistently to the **2021 validation** and **2022-2023 test** tables.


# Apply manual redundancy-based pruning using the development-driven drop list
features_to_drop = [
    "current_assets",
    "total_fixed_assets",
    "shareholders_equity",
    "short_term_debt",
    "long_term_debt",
    "production_costs",
    "financial_expenses",
    "net_profit_loss",
    "quick_ratio",
    "total_assets_growth",
    "total_assets_trend_up",
    "roe",
    "leverage",
    "profit_margin",
]

# Drop the same redundant columns from development, validation, and test
for frame in (df_train_fit_fe, df_val_2021_fe, df_test_fe):
    drop_cols = [c for c in features_to_drop if c in frame.columns]
    frame.drop(columns=drop_cols, inplace=True)

df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

print("Redundancy-based feature selection applied.")
print(f"Development FE shape: {df_train_fit_fe.shape}")
print(f"Validation FE shape:  {df_val_2021_fe.shape}")
print(f"Test FE shape:        {df_test_fe.shape}")

# Check train-test column alignment after pruning
train_cols = set(df_train_fe.columns)
test_cols = set(df_test_fe.columns)
print()
print("Columns only in combined TRAIN after selection:")
print(train_cols - test_cols)
print()
print("Columns only in TEST after selection:")
print(test_cols - train_cols)

# Inspect the remaining correlation structure after manual feature selection
final_heatmap_cols = [
    "bankruptcy_next_year",
    "total_assets",
    "total_debt",
    "debt_to_assets",
    "current_ratio",
    "roi",
    "roe_vs_sector",
    "leverage_vs_sector",
    "profit_margin_vs_sector",
]

final_heatmap_cols = [c for c in final_heatmap_cols if c in df_train_fit_fe.columns]
corr_final = df_train_fit_fe[final_heatmap_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_final,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)
plt.title("Correlation matrix after feature selection (development years only)")
plt.tight_layout()
plt.show()

# **Interpretation of the results**
# 
# The redundancy-based feature selection step successfully removes several variables that were identified as **highly correlated or conceptually overlapping** with other features in the development dataset.
# 
# The variables removed fall into several meaningful groups:
# 
# - **Balance-sheet aggregates** such as `current_assets`, `total_fixed_assets`, `shareholders_equity`, and debt components were removed because they are strongly correlated with other financial quantities like `total_assets` or `total_debt`.
# - **Operating cost variables** such as `production_costs`, `financial_expenses`, and `net_profit_loss` were excluded due to their strong dependence on firm scale and high correlation with other profitability indicators.
# - The **quick ratio** was removed because it is largely redundant with the `current_ratio`, capturing nearly identical liquidity information.
# - Some **raw financial ratios** (`roe`, `leverage`, `profit_margin`) were removed because their **sector-relative versions** provide a more informative representation by accounting for industry-specific financial structures.
# - Certain **growth and trend variables** were excluded where they overlapped strongly with other temporal indicators.
# 
# This pruning does not mean the removed variables are economically meaningless. Rather, it means that in the presence of the retained features they were judged to add limited extra information relative to the added complexity.
# 
# The result is a cleaner feature space that remains consistent across the **development**, **validation**, and **test** datasets.


# ###**Unsupervised Clustering of Financial Profiles**


# 
# **Two-Stage Clustering Strategy**
# 
# In this project, clustering is intentionally divided into **two stages**, because the two clusterings serve **different purposes**:
# 
# - the first is a **broad exploratory clustering**
# - the second is a **practical final clustering**
# 
# The first stage uses a **wide set of usable numerical variables** to obtain an initial segmentation of firms in the financial feature space.  
# Its purpose is to understand how firms naturally group together and to identify which variables appear most useful in separating them.
# 
# The second stage will then use a **smaller and more focused set of variables**, selected on the basis of the broad clustering results, in order to build a clustering feature that is more compact, more interpretable, and more suitable for the final modeling pipeline.
# 
# 
# 


# ####**Broad Exploratory Clustering**


# This first block refers to the **broad exploratory clustering**.
# 
# Before fitting it, we determine the most appropriate **number of clusters (K)** by comparing candidate values using three complementary diagnostics:
# 
# - **WCSS**, to measure how within-cluster dispersion decreases as K increases
# - **Silhouette score**, to evaluate how coherent and well-separated the clusters are
# - **Davies-Bouldin index**, to assess cluster compactness and separation
# 
# To avoid **data leakage**, the clustering model is fitted **only on firms observed in the training years (2018–2020)**.
# 
# The aim of this first stage is therefore to obtain a **broad initial segmentation** of firms, which will guide the construction of the later **practical clustering feature**.


# Broad exploratory clustering: build a wide numeric feature space
exclude_cluster = [
    "company_id",
    "fiscal_year",
    "bankruptcy_next_year",
    "financial_health_class",
    "revenue_change",
]

cluster_features_broad = [
    c for c in df_train_fit_fe.select_dtypes(include=[np.number]).columns
    if c not in exclude_cluster
    and c in df_val_2021_fe.columns
    and c in df_test_fe.columns
]

print(f"Broad clustering features used ({len(cluster_features_broad)}):")
print(cluster_features_broad)

# Use development medians to fill missing values before scaling
cluster_medians_broad = df_train_fit_fe[cluster_features_broad].median()
X_cl_fit_broad = df_train_fit_fe[cluster_features_broad].fillna(cluster_medians_broad)
X_cl_val_broad = df_val_2021_fe[cluster_features_broad].fillna(cluster_medians_broad)
X_cl_test_broad = df_test_fe[cluster_features_broad].fillna(cluster_medians_broad)

# Fit the clustering scaler on development data only
scaler_cl_broad = StandardScaler()
X_cl_fit_scaled_broad = scaler_cl_broad.fit_transform(X_cl_fit_broad)
X_cl_val_scaled_broad = scaler_cl_broad.transform(X_cl_val_broad)
X_cl_test_scaled_broad = scaler_cl_broad.transform(X_cl_test_broad)

# Compare candidate values of K using development data only
results_broad = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cl_fit_scaled_broad)
    results_broad.append(
        {
            "k": k,
            "wcss": kmeans.inertia_,
            "silhouette": silhouette_score(X_cl_fit_scaled_broad, labels),
            "davies_bouldin": davies_bouldin_score(X_cl_fit_scaled_broad, labels),
        }
    )

results_broad_df = pd.DataFrame(results_broad)
print()
print("Candidate K results for broad clustering:")
display(results_broad_df)

# 
# **Interpretation of the results**
# 
# This table reports the clustering diagnostics for the **broad exploratory clustering**, which represents the **first stage** of the two-stage clustering strategy.
# 
# At this point, the goal is **not yet to create the final clustering feature**, but to determine how many broad groups are suggested by the data when firms are described through a wide set of numerical financial variables.
# 
# Each row corresponds to one candidate value of **K**:
# 
# - **WCSS** decreases as K increases, because clusters become smaller and more homogeneous
# - **Silhouette score** is higher when observations are better matched to their own cluster and more distinct from other clusters
# - **Davies-Bouldin index** is lower when clusters are more compact and better separated
# 
# From these diagnostics, **K = 4** provides the best overall balance for the **broad exploratory stage**.
# 
# This is supported by the fact that:
# 
# - **K = 4** achieves the **highest Silhouette score**
# - **K = 4** also achieves the **lowest Davies-Bouldin index**
# - and the reduction in **WCSS** remains substantial without requiring too many clusters
# 
# This suggests that the dataset can be meaningfully partitioned into **four broad financial profiles**.
# 
# Therefore, based on these diagnostics, we select **K = 4** as the most appropriate number of clusters for the next step of the clustering procedure.
# 


# **Final Broad Exploratory Clustering**
# 
# After selecting the number of clusters, we now fit the **broad exploratory clustering** using **K = 4**.
# 
# This remains the **first stage** of the clustering strategy. Its purpose is not to provide the final clustering feature for the model, but to generate an initial broad segmentation of firms using a **wide numerical feature space**.
# 
# The resulting broad clusters are used to examine how firms differ in terms of:
# 
# - **financial fragility**
# - **profitability**
# - **leverage**
# - **sector-relative positioning**
# - and **temporal financial dynamics**
# 
# As before, the clustering model is fitted **only on firms observed in the development years (2018-2020)** in order to avoid **data leakage**.
# 
# Once estimated, the same fitted model is used to assign a broad cluster label to the **2021 validation dataset** and the **2022-2023 test dataset**, while the development data receive their own fitted cluster assignments.
# 
# To interpret the clusters, we summarize each group through:
# 
# - the **number of observations**
# - the **proportion of the development sample**
# - the **bankruptcy rate**
# - and the **median values** of the broad clustering variables
# 
# The purpose of this step is to understand which variables are most informative in separating firms, and to support the design of the later **practical clustering** retained in the final model.


# Fit the final broad clustering on development data only
final_k_broad = 4

kmeans_broad = KMeans(n_clusters=final_k_broad, random_state=42, n_init=10)
kmeans_broad.fit(X_cl_fit_scaled_broad)

# Assign the broad-cluster label to development, validation, and test
df_train_fit_fe["cluster_id_broad"] = kmeans_broad.predict(X_cl_fit_scaled_broad)
df_val_2021_fe["cluster_id_broad"] = kmeans_broad.predict(X_cl_val_scaled_broad)
df_test_fe["cluster_id_broad"] = kmeans_broad.predict(X_cl_test_scaled_broad)
df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

print()
print(f"K-Means (k={final_k_broad}) applied. New feature: 'cluster_id_broad'")

# Select the most interpretable variables for broad-cluster profiling
summary_vars_broad = [c for c in cluster_features_broad if c in [
    "altman_z_like_score",
    "current_ratio",
    "debt_to_assets",
    "financial_income",
    "operating_income",
    "operating_income_growth",
    "operating_income_trend_up",
    "operating_income_volatility",
    "production_value",
    "production_value_growth",
    "production_value_trend_up",
    "production_value_volatility",
    "profit_margin_vs_sector",
    "roe_vs_sector",
    "roi",
    "roi_vs_sector",
    "leverage_vs_sector",
    "total_assets",
    "total_debt",
    "total_debt_growth",
    "total_debt_trend_up",
    "years_in_business",
]]

if "bankruptcy_next_year" in df_train_fit_fe.columns:
    # Profile each broad cluster using development years only
    agg_dict_broad = {"company_id": "count", "bankruptcy_next_year": "mean"}
    for col in summary_vars_broad:
        agg_dict_broad[col] = "median"

    cluster_summary_broad = (
        df_train_fit_fe.groupby("cluster_id_broad")
        .agg(agg_dict_broad)
        .rename(columns={"company_id": "count", "bankruptcy_next_year": "bankruptcy_rate"})
    )
    cluster_summary_broad["proportion"] = cluster_summary_broad["count"] / len(df_train_fit_fe)
    cluster_summary_broad = cluster_summary_broad.sort_values("bankruptcy_rate", ascending=False)

    print()
    print("Broad-cluster analysis (development years only):")
    display(
        cluster_summary_broad[["count", "proportion", "bankruptcy_rate"] + summary_vars_broad].round(4)
    )

# 
# **Interpretation of the results**
# 
# The broad exploratory clustering identifies **four broad financial profiles** in the training data.
# 
# The summary table reports, for each cluster:
# 
# - the **cluster size** (`count`)
# - the **share of the training sample** (`proportion`)
# - the **bankruptcy rate**
# - and the **median values** of the broad clustering variables
# 
# The clusters are ordered by **bankruptcy rate**, so the first row corresponds to the most fragile group.
# 
# The results show that **Cluster 0** is the riskiest broad profile, with a **bankruptcy rate of about 3.6%**.  
# Relative to the other groups, this cluster is characterized by:
# 
# - a **lower Altman-like score**
# - a **higher debt-to-assets ratio**
# - a **lower ROI**
# - **negative sector-relative profitability**
# - and **higher leverage relative to sector peers**
# 
# This profile is consistent with firms under **financial stress**, combining weaker profitability with stronger debt pressure.
# 
# By contrast, **Clusters 2 and 3** appear healthier, with much lower bankruptcy rates and more favorable values in terms of profitability, leverage, and sector-relative positioning.
# 
# A special case is **Cluster 1**, which contains only a very small number of observations.  
# Its extreme values in size and growth-related variables suggest that it captures a group of **exceptional large-scale firms** rather than a broad representative profile.
# 
# For this reason, the broad clustering should be interpreted mainly as an **exploratory tool**:
# 
# - it reveals the main financial groupings present in the data
# - it shows which variables most clearly separate firms
# - and it helps guide the construction of the later **practical clustering**, which will be more compact and more suitable for the final model
# 
# In other words, the value of this first clustering lies not in being the final solution, but in providing the evidence needed to design a better second-stage clustering feature.
# 


# ####**Final Practical Clustering**


# 
# **Cluster-Number Selection (Practical Clustering)**
# 
# We now move to the **second stage** of the clustering strategy: the **practical clustering**.
# 
# Unlike the broad exploratory stage, this step focuses on a **reduced set of selected variables** chosen because they proved both:
# 
# - **financially meaningful**
# - and **useful in separating firms** in the broad clustering stage
# 
# These variables capture the core dimensions of firms’ financial profiles, including:
# 
# - **financial distress**
# - **liquidity**
# - **debt pressure**
# - **profitability**
# - **sector-relative performance**
# - and **recent operating dynamics**
# 
# As before, the clustering model is fitted **only on firms observed in the training years (2018–2020)** in order to avoid **data leakage**.
# 
# At this stage, the objective is to determine the most appropriate **number of clusters (K)** for the final clustering specification by comparing candidate values through:
# 
# - **WCSS**
# - **Silhouette score**
# - **Davies-Bouldin index**
# 


# Practical clustering: use a smaller and more focused feature set
cluster_features_practical = [
    "altman_z_like_score",
    "current_ratio",
    "debt_to_assets",
    "roi",
    "roe_vs_sector",
    "roi_vs_sector",
    "leverage_vs_sector",
    "profit_margin_vs_sector",
    "operating_income_growth",
]

cluster_features_practical = [
    c for c in cluster_features_practical
    if c in df_train_fit_fe.columns and c in df_val_2021_fe.columns and c in df_test_fe.columns
]

print(f"Practical clustering features used ({len(cluster_features_practical)}):")
print(cluster_features_practical)

# Fill missing values using development medians only
cluster_medians_practical = df_train_fit_fe[cluster_features_practical].median()
X_cl_fit_practical = df_train_fit_fe[cluster_features_practical].fillna(cluster_medians_practical)
X_cl_val_practical = df_val_2021_fe[cluster_features_practical].fillna(cluster_medians_practical)
X_cl_test_practical = df_test_fe[cluster_features_practical].fillna(cluster_medians_practical)

# Fit the clustering scaler on development data only
scaler_cl_practical = StandardScaler()
X_cl_fit_scaled_practical = scaler_cl_practical.fit_transform(X_cl_fit_practical)
X_cl_val_scaled_practical = scaler_cl_practical.transform(X_cl_val_practical)
X_cl_test_scaled_practical = scaler_cl_practical.transform(X_cl_test_practical)

# Compare candidate values of K using development data only
results_practical = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cl_fit_scaled_practical)
    results_practical.append(
        {
            "k": k,
            "wcss": kmeans.inertia_,
            "silhouette": silhouette_score(X_cl_fit_scaled_practical, labels),
            "davies_bouldin": davies_bouldin_score(X_cl_fit_scaled_practical, labels),
        }
    )

results_practical_df = pd.DataFrame(results_practical)
print()
print("Candidate K results:")
display(results_practical_df)

# 
# **Interpretation of the results**
# 
# This table reports the clustering diagnostics for the **practical clustering specification**, which represents the **second stage** of the two-stage clustering strategy.
# 
# At this point, the objective is to choose the most appropriate **number of clusters (K)** for the clustering feature that will be retained in the final model.
# 
# Each row corresponds to one candidate value of **K**:
# 
# - **WCSS** decreases as K increases, because the observations are partitioned into smaller and more homogeneous groups
# - **Silhouette score** is higher when observations are better matched to their own cluster and more distinct from other clusters
# - **Davies-Bouldin index** is lower when clusters are more compact and better separated
# 
# From these diagnostics, **K = 4** again provides the best overall balance.
# 
# This is supported by the fact that:
# 
# - **K = 4** achieves the **highest Silhouette score**
# - **K = 4** also achieves the **lowest Davies-Bouldin index**
# - and the reduction in **WCSS** remains consistent with a meaningful four-cluster structure
# 
# This suggests that the practical specification produces a **clearer and more focused partition** of firms than alternative values of K.
# 
# Therefore, we retain **K = 4** as the final number of clusters for the **practical clustering solution**.
# 


# **Final Practical Clustering of Financial Profiles**
# 
# After selecting the number of clusters, we now fit the **final practical clustering** using **K = 4**.
# 
# This is the clustering solution that will be **retained in the modeling pipeline**.
# 
# Compared with the broad exploratory clustering, this second-stage specification is based on a **smaller and more focused set of variables**, selected to produce clusters that are:
# 
# - more **compact**
# - more **economically interpretable**
# - and more suitable for prediction
# 
# As before, the clustering model is fitted **only on firms observed in the development years (2018-2020)** in order to avoid **data leakage**.
# 
# The fitted model is then used to assign a practical cluster label to:
# 
# - the development dataset
# - the **2021 validation dataset**
# - the **2022-2023 test dataset**
# 
# To interpret the resulting clusters, we summarize the development groups through:
# 
# - the **number of observations**
# - the **proportion of the development sample**
# - the **bankruptcy rate**
# - and the **median values** of the selected practical clustering variables
# 
# This step creates the final **`cluster_id_practical`** feature, which provides a compact representation of firms' financial profiles.


# Fit the final practical clustering on development data only
final_k_practical = 4

kmeans_practical = KMeans(n_clusters=final_k_practical, random_state=42, n_init=10)
kmeans_practical.fit(X_cl_fit_scaled_practical)

# Assign the practical-cluster label to development, validation, and test
df_train_fit_fe["cluster_id_practical"] = kmeans_practical.predict(X_cl_fit_scaled_practical)
df_val_2021_fe["cluster_id_practical"] = kmeans_practical.predict(X_cl_val_scaled_practical)
df_test_fe["cluster_id_practical"] = kmeans_practical.predict(X_cl_test_scaled_practical)
df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

print()
print(f"K-Means (k={final_k_practical}) applied. New feature: 'cluster_id_practical'")

# Profile the practical clusters using the selected practical features
summary_vars_practical = cluster_features_practical.copy()

if "bankruptcy_next_year" in df_train_fit_fe.columns:
    agg_dict_practical = {"company_id": "count", "bankruptcy_next_year": "mean"}
    for col in summary_vars_practical:
        agg_dict_practical[col] = "median"

    cluster_summary_practical = (
        df_train_fit_fe.groupby("cluster_id_practical")
        .agg(agg_dict_practical)
        .rename(columns={"company_id": "count", "bankruptcy_next_year": "bankruptcy_rate"})
    )
    cluster_summary_practical["proportion"] = cluster_summary_practical["count"] / len(df_train_fit_fe)
    cluster_summary_practical = cluster_summary_practical.sort_values("bankruptcy_rate", ascending=False)

    print()
    print("Practical-cluster analysis (development years only):")
    display(
        cluster_summary_practical[["count", "proportion", "bankruptcy_rate"] + summary_vars_practical].round(4)
    )

# Visualize the relative profile of each practical cluster
if "cluster_summary_practical" in locals():
    heatmap_vars = [
        "bankruptcy_rate",
        "altman_z_like_score",
        "current_ratio",
        "debt_to_assets",
        "roi",
        "roe_vs_sector",
        "roi_vs_sector",
        "leverage_vs_sector",
        "profit_margin_vs_sector",
        "operating_income_growth",
    ]

    heatmap_vars = [c for c in heatmap_vars if c in cluster_summary_practical.columns]
    cluster_heatmap_raw = cluster_summary_practical[heatmap_vars].copy()

    # Standardize across clusters to highlight relative strengths and weaknesses
    cluster_heatmap_z = (cluster_heatmap_raw - cluster_heatmap_raw.mean()) / cluster_heatmap_raw.std(ddof=0)

    cluster_name_map = {idx: f"Cluster {idx}" for idx in cluster_heatmap_z.index}
    cluster_heatmap_z.index = [cluster_name_map[idx] for idx in cluster_heatmap_z.index]
    cluster_heatmap_raw.index = [cluster_name_map[idx] for idx in cluster_heatmap_raw.index]

    plt.figure(figsize=(12, 5.5))
    sns.heatmap(
        cluster_heatmap_z,
        annot=cluster_heatmap_raw.round(3),
        fmt="",
        cmap="RdYlBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Relative level across clusters"},
    )
    plt.title("Practical Cluster Profiles (development years only)")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# 
# **Interpretation of the results**
# 
# The final practical clustering identifies **four distinct financial profiles** and assigns each firm-year observation to one of these groups through the variable **`cluster_id_practical`**.
# 
# The summary table reports, for each cluster:
# 
# - the **cluster size** (`count`)
# - the **share of the training sample** (`proportion`)
# - the **bankruptcy rate**
# - and the **median values** of the selected practical clustering variables
# 
# The clusters are ordered by **bankruptcy rate**, so the first row corresponds to the most financially fragile profile.
# 
# The results indicate that **Cluster 3** is the riskiest group, with a **bankruptcy rate of about 4.1%**.  
# This cluster combines several signs of financial weakness, including:
# 
# - a **low Altman-like score**
# - **low liquidity**
# - **high debt pressure**
# - **weak profitability**
# - **negative sector-relative performance**
# - and **negative operating income growth**
# 
# Taken together, these characteristics describe firms under **financial stress**, with a weaker operating position and a less favorable standing relative to industry peers.
# 
# The remaining clusters show much lower bankruptcy rates and more stable financial conditions.
# 
# In particular:
# 
# - one cluster displays the strongest overall profile, combining **high financial stability**, **strong liquidity**, **low debt pressure**, and **positive sector-relative profitability**
# - the other low-risk clusters appear more balanced, with healthy financial conditions and very limited bankruptcy incidence
# 
# Compared with the broad exploratory solution, this practical clustering is easier to interpret and more suitable for modeling because it is built on a smaller and more targeted set of variables.
# 
# For this reason, **`cluster_id_practical`** is retained as the final clustering feature, while the broad clustering remains an exploratory intermediate step.
# 


# ####**Visual Representation of the Final Practical Clusters**
# 


# 
# To complement the numerical cluster summary, we also provide a **two-dimensional visual representation** of the final practical clustering solution.
# 
# Since the clustering is built on a multi-dimensional financial feature space, it cannot be directly visualized in its original form.  
# To address this, we apply **Principal Component Analysis (PCA)** to project the selected practical clustering variables into **two synthetic dimensions**:
# 
# - **Principal Component 1**
# - **Principal Component 2**
# 
# These components do not correspond to individual original variables.  
# Instead, they summarize the main directions of variation in the practical clustering feature space.
# 
# The purpose of this visualization is not to redefine the clusters, but to provide an intuitive graphical view of how the observations are distributed after clustering.
# 
# In this plot:
# 
# - each point represents one **firm-year observation**
# - the color indicates the assigned **`cluster_id_practical`**
# - the position of each point reflects its location in the reduced PCA space
# 
# This helps us assess whether the final practical clusters are visually distinguishable and whether the clustering captures meaningful internal structure in the data.
# 


# PCA visualization of the final practical clusters
cluster_features_final = [
    "altman_z_like_score",
    "current_ratio",
    "debt_to_assets",
    "roi",
    "roe_vs_sector",
    "roi_vs_sector",
    "leverage_vs_sector",
    "profit_margin_vs_sector",
    "operating_income_growth",
]

cluster_features_final = [c for c in cluster_features_final if c in df_train_fit_fe.columns]

# Prepare the development feature matrix for 2D visualization
X_vis = df_train_fit_fe[cluster_features_final].copy().fillna(df_train_fit_fe[cluster_features_final].median())

scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

pca = PCA(n_components=2, random_state=42)
X_vis_pca = pca.fit_transform(X_vis_scaled)

plot_df = pd.DataFrame(
    {
        "PC1": X_vis_pca[:, 0],
        "PC2": X_vis_pca[:, 1],
        "cluster_id_practical": df_train_fit_fe["cluster_id_practical"].values,
    }
)

plt.figure(figsize=(8, 6))
for cl in sorted(plot_df["cluster_id_practical"].unique()):
    subset = plot_df[plot_df["cluster_id_practical"] == cl]
    plt.scatter(subset["PC1"], subset["PC2"], alpha=0.5, label=f"Cluster {cl}")

plt.title("Final practical clusters in 2D PCA space (development years)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()

# 
# **Interpretation of the results**
# 
# The PCA scatter plot provides a visual representation of the **final practical clustering solution** in a reduced two-dimensional space.
# 
# Each color corresponds to one practical cluster, and each point represents a firm-year observation from the training data.
# 
# The figure shows that the four clusters occupy **distinct regions** of the PCA space, although some degree of overlap remains.  
# This is expected, since the clustering itself was performed in a **higher-dimensional financial feature space**, while the plot shows only a two-dimensional projection.
# 
# Several conclusions can be drawn from this visualization:
# 
# - the clusters are **not randomly mixed**, which suggests that the clustering algorithm has identified meaningful internal structure
# - the practical clustering solution produces groups that are **visually distinguishable**
# - some overlap remains at the cluster boundaries, which is normal in financial data where firm profiles often evolve gradually rather than forming perfectly separated groups
# 
# Importantly, this figure should be interpreted as a **visual aid**, not as a complete measure of clustering quality.
# 
# A clear PCA projection does not by itself prove that the clustering is optimal, but it helps confirm that:
# 
# - the selected practical variables generate a structured partition
# - the four-cluster solution is not purely artificial
# - and the resulting **`cluster_id_practical`** captures meaningful differences in firms’ overall financial profiles
# 
# Overall, this visualization supports the conclusion that the final practical clustering produces a **coherent and interpretable segmentation** of firms in the training set.
# 


# ###**Save Enhanced Data**
# 
# 


# After completing the feature engineering pipeline, we store the resulting datasets as the **final engineered versions** of the labelled training data and the official test data.
# 
# At this stage, the datasets already incorporate:
# 
# - the original cleaned financial variables
# - accounting-based features
# - temporal dynamics
# - sector-relative indicators
# - clustering-based firm profiles
# - and the redundancy-based feature selection step
# 
# Saving these intermediate outputs is useful for two reasons:
# 
# - it preserves a stable version of the **final Feature Engineering dataset**
# - it separates the **feature construction phase** from the subsequent **model preparation phase**, which will include log transformation of selected magnitudes, train-only categorical encoding, and model fitting
# 
# The notebook keeps the **2018-2020 development rows** and the **2021 validation rows** separate in memory, while also saving a combined labelled engineered file for convenience.


# Save the final enhanced datasets after feature engineering
df_train_fe = combine_train_periods(df_train_fit_fe, df_val_2021_fe)

df_train_fe.to_csv("train_data_enhanced.csv", index=False)
df_test_fe.to_csv("test_features_enhanced.csv", index=False)

print("Saved to: train_data_enhanced.csv")
print("Saved to: test_features_enhanced.csv")
print()
print("Feature Engineering Pipeline Completed Successfully.")
print(f"Development FE rows: {df_train_fit_fe.shape[0]}")
print(f"Validation FE rows:  {df_val_2021_fe.shape[0]}")
print(f"Combined Train rows:  {df_train_fe.shape[0]}")
print(f"Final Test rows:      {df_test_fe.shape[0]}")

# 
# **Interpretation of the results**
# 
# The output confirms that the feature engineering pipeline has been completed successfully and that the resulting datasets have been saved as:
# 
# - `train_data_enhanced.csv`
# - `test_features_enhanced.csv`
# 
# These files contain the **final engineered feature space** produced by the pipeline.
# 
# The summary statistics show that the final datasets include:
# 
# - **33 columns in the training set**
# - **30 columns in the test set**
# 
# This difference is expected and methodologically correct, because the training set still contains the **target-related variables** that are not available in the test set:
# 
# - `bankruptcy_next_year`
# - `financial_health_class`
# - `revenue_change`
# 
# Apart from these train-only target variables, the two datasets are fully aligned and ready for the next steps of the modeling workflow.
# 
# Overall, this block marks the completion of the **Feature Engineering phase**. The saved datasets now provide a compact, information-rich, and economically interpretable representation of firms’ financial conditions, which will be used in the subsequent **model preparation and model development stages**.
# 


# ## **11. Model Preparation — Data Preprocessing II**


# In the previous sections, we constructed a rich feature set describing firms through **accounting structure**, **profitability**, **liquidity**, **temporal dynamics**, **sector-relative positioning**, and **unsupervised financial profiles**.
# 
# At this stage, the objective is no longer exploratory: we need to convert the engineered dataset into a **machine-learning-ready modelling table** for our bankruptcy-prediction task.
# 
# Because bankruptcy prediction is a **rare-event binary classification problem**, our modelling design must satisfy three methodological requirements simultaneously:
# 
# - it must preserve the **chronological structure** of the data;
# - it must handle **severe class imbalance**, since bankrupt firms represent only a very small fraction of the sample;
# - it must remain sufficiently **interpretable**, because bankruptcy prediction is economically meaningful only if we can explain which financial signals are associated with elevated distress risk.
# 
# For these reasons, we organize the modelling stage into two connected blocks:
# 
# - a **final preprocessing block**, where we transform the engineered data into a model-ready table;
# - a **model development block**, where we train, compare, and interpret the candidate classifiers.
# 
# This gives us a clean transition from feature engineering to model evaluation while keeping the modelling workflow readable and consistent with the structure of the rest of the notebook.


# **Preparation logic**


# Before fitting any classifier, it is useful to restate the **prediction task** in operational terms.
# 
# For each firm-year observation, our model must estimate whether the company will experience **bankruptcy in the following year**. In practice, this means organizing the data into a development window, a final validation year, and an unseen test period:
# 
# - observations from **2018-2020** are used for model development
# - observations from **2021** are reserved for final model comparison
# - observations from **2022-2023** form the **unseen test period**, for which we produce final predicted probabilities
# 
# This structure reflects the logic of a realistic early-warning setting and keeps the model comparison consistent across all candidate classifiers.
# 
# Because this split is already enforced upstream, our final preprocessing block applies the following rules:
# 
# - large monetary variables are log-transformed only **after** feature engineering is complete
# - categorical variables are encoded using the development data and then aligned across the later datasets
# - scaling is fitted on the modelling sample and then reused consistently for validation and test data
# - identifiers and target information are kept separate from the feature matrices used for estimation
# 
# The result is a final modelling table that is coherent, readable, and directly usable in the model-comparison stage.


# ### **Imports and Local Setup**
# 
# Here we import the model classes and evaluation utilities used in our model-development section.


# Imports and local setup for the Challenge 1 modelling section.

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

repo_root = Path.cwd()
if not (repo_root / "data").exists() and (repo_root.parent / "data").exists():
    repo_root = repo_root.parent

# ### **Data Structure for Modelling Outputs**
# 
# We use this small container to keep the final train, validation, and test matrices together with their feature names.


# Helper structures and functions kept inside the notebook so the submission stays self-contained.

@dataclass
class Challenge1Datasets:
    feature_cols: list[str]
    train_model_df: pd.DataFrame
    val_model_df: pd.DataFrame
    test_model_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame

# ### **11.1 Log Transformation and Column Cleanup**


train_model = df_train_fit_fe.copy()
val_model = df_val_2021_fe.copy()
test_model = df_test_fe.copy()

target_col = "bankruptcy_next_year"
cluster_col = "cluster_id_practical"

# Candidate magnitude variables for log transform
candidate_log_features = ["total_assets", "total_debt", "production_value", "financial_income"]

# Decide which variables to log using DEVELOPMENT data only (2018–2020)
log_features = [
    c for c in candidate_log_features
    if c in train_model.columns and train_model[c].min() >= 0
]

if len(log_features) > 0:
    print("Applying log1p transformation to magnitude variables:")

    for col in log_features:
        # Skewness on development data BEFORE transform
        orig_skew = train_model[col].skew()

        # Apply same transform to development, validation, test
        train_model[col] = np.log1p(train_model[col])
        if col in val_model.columns:
            val_model[col] = np.log1p(val_model[col])
        if col in test_model.columns:
            test_model[col] = np.log1p(test_model[col])

        # Skewness on development data AFTER transform
        new_skew = train_model[col].skew()

        print(f" - {col:20}: Skewness {orig_skew:8.2f} -> {new_skew:8.2f}")
else:
    print("\nNo suitable non-negative magnitude variables found for log1p. Skipping log step.")

print("\nLog transformation completed!")

# Drop non-model columns consistently
cols_to_drop = ["province", "financial_health_class", "revenue_change", "ateco_sector_label", "cluster_id_broad"]
for frame in (train_model, val_model, test_model):
    frame.drop(columns=cols_to_drop, errors="ignore", inplace=True)

print("Dropped non-model columns:")
print([c for c in cols_to_drop if c in df_train_fit_fe.columns or c in df_test_fe.columns])

print(f"Development model table shape: {train_model.shape}")
print(f"Validation model table shape:  {val_model.shape}")
print(f"Test model table shape:        {test_model.shape}")

# ### **11.2 Categorical Encoding and Feature Alignment**


# Encode categorical variables using the development schema only
categorical_cols = [
    c for c in ["ateco_sector", "region", "legal_form", cluster_col]
    if c in train_model.columns and c in val_model.columns and c in test_model.columns
]

train_encoded = pd.get_dummies(train_model, columns=categorical_cols, drop_first=False)
val_encoded = pd.get_dummies(val_model, columns=categorical_cols, drop_first=False)
test_encoded = pd.get_dummies(test_model, columns=categorical_cols, drop_first=False)

# Use the 2018-2020 encoded columns as the reference schema
train_reference_cols = train_encoded.columns.tolist()
test_reference_cols = [c for c in train_reference_cols if c != target_col]

val_encoded = val_encoded.reindex(columns=train_reference_cols, fill_value=0)
test_encoded = test_encoded.reindex(columns=test_reference_cols, fill_value=0)

print("Categorical columns encoded using the 2018-2020 schema:")
print(categorical_cols)
print(f"Encoded development shape: {train_encoded.shape}")
print(f"Encoded validation shape:  {val_encoded.shape}")
print(f"Encoded test shape:        {test_encoded.shape}")

# ### **11.3 Final Modelling Matrices and Scaling**


# Build the final modelling matrices for bankruptcy prediction
feature_cols = [c for c in train_reference_cols if c not in {target_col, "company_id", "cluster_id_broad"}]

train_encoded[feature_cols] = train_encoded[feature_cols].astype(float)
val_encoded[feature_cols] = val_encoded[feature_cols].astype(float)
test_encoded[feature_cols] = test_encoded[feature_cols].astype(float)

X_train = train_encoded[feature_cols].copy()
y_train = train_encoded[target_col].astype(int).copy()
X_val = val_encoded[feature_cols].copy()
y_val = val_encoded[target_col].astype(int).copy()
X_test = test_encoded[feature_cols].copy()

model_data = Challenge1Datasets(
    feature_cols=feature_cols,
    train_model_df=train_encoded,
    val_model_df=val_encoded,
    test_model_df=test_encoded,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
)

print("Challenge 1 modelling datasets prepared successfully.")
print(f"Training rows (2018-2020): {model_data.X_train.shape[0]}")
print(f"Validation rows (2021):    {model_data.X_val.shape[0]}")
print(f"Test rows (2022-2023):     {model_data.X_test.shape[0]}")
print(f"Number of final modelling features: {len(model_data.feature_cols)}")
print(f"Development-window bankruptcy rate: {model_data.y_train.mean():.2%}")
print(f"Validation-year bankruptcy rate:    {model_data.y_val.mean():.2%}")

# **Interpretation of the results**


# The final modelling table confirms that the prediction task is being built on a **highly imbalanced sample**.
# 
# Two aspects are particularly important:
# 
# - the development sample contains **8,896 observations** across the years **2018-2020**
# - the final validation year (**2021**) contains **2,932 observations**, but only a very small fraction correspond to bankrupt firms
# 
# This imbalance is not a technical inconvenience but a structural property of the problem itself: **bankruptcy is a rare event**. As a consequence:
# 
# - **accuracy** is not informative, because a model can look strong while predicting almost all firms as healthy
# - **F1-score** becomes the natural primary metric, since it balances the ability to identify bankruptcies (**recall**) with the ability to avoid excessive false alarms (**precision**)
# - probability thresholds must be treated as **tunable decision parameters**, not as fixed defaults
# 
# Another important result is that the modelling table includes the final unsupervised profile feature **`cluster_id_practical`** through categorical encoding that keeps the column structure consistent across development, validation, and test data. This allows us to retain the clustering signal in the final model without complicating the modelling table unnecessarily.


# ## **12. Model Development and Evaluation**


# Our assignment asks for **at least three different model types**, together with a clear model-selection logic.
# 
# For our comparison, we use four complementary families:
# 
# - **Logistic Regression**: a transparent linear baseline, useful for interpretability and for understanding whether bankruptcy risk is approximately linearly separable in the engineered feature space.
# - **Random Forest**: a bagged tree model that captures non-linearities and interactions while remaining relatively robust to outliers and mixed feature types.
# - **XGBoost**: a boosted-tree model that usually performs strongly on structured tabular data and is often particularly effective when bankruptcy risk depends on subtle non-linear combinations of financial indicators.
# - **LightGBM**: a second gradient-boosting model that is often extremely competitive on tabular financial data and provides an additional benchmark within the boosting family.
# 
# These models represent four complementary modelling perspectives:
# 
# - a **linear, highly interpretable classifier**
# - a **non-linear ensemble based on bagging**
# - a **boosted-tree model focused on sequential error correction (XGBoost)**
# - a **second boosting model with a different implementation strategy (LightGBM)**
# 
# We first tune each family within the development period through expanding validation folds, then carry only the best specification of each family into the final **2021 validation comparison**.


# ### **12.1 Helper Functions for Model Training**


# **Validation Utilities**
# 
# Here we define the expanding-window fold logic used during development. We do not shuffle observations, so the validation design remains consistent with the structure of the task.


def build_time_folds(train_df: pd.DataFrame, year_col: str = "fiscal_year") -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    years = sorted(train_df[year_col].unique())
    for year in years[1:]:
        train_idx = train_df.index[train_df[year_col] < year].to_numpy()
        val_idx = train_df.index[train_df[year_col] == year].to_numpy()
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    return folds

# **Model Catalog and Candidate Specifications**
# 
# Here we define the candidate model families and compact hyperparameter grids used in our comparison. The model choices and parameters are unchanged.


def make_model_catalog(random_state: int, scale_pos_weight: float) -> dict[str, list[dict[str, Any]]]:
    return {
        "Logistic Regression": [
            {
                "estimator": Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                C=c_value,
                                class_weight="balanced",
                                max_iter=4000,
                                solver="liblinear",
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
                "params": {"C": c_value},
                "fit_with_sample_weight": False,
                "fit_with_fold_scale_pos_weight": False,
            }
            for c_value in [0.1, 0.3, 1.0, 3.0]
        ],
        "Random Forest": [
            {
                "estimator": RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=10,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                },
                "fit_with_sample_weight": False,
                "fit_with_fold_scale_pos_weight": False,
            }
            for n_estimators, max_depth, min_samples_leaf in [
                (400, None, 3),
                (500, 8, 3),
                (500, 12, 5),
            ]
        ],
        "XGBoost": [
            {
                "estimator": XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=1.0,
                    min_child_weight=min_child_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=4,
                ),
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "min_child_weight": min_child_weight,
                },
                "fit_with_sample_weight": False,
                "fit_with_fold_scale_pos_weight": True,
            }
            for n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight in [
                (300, 3, 0.05, 0.9, 0.8, 5),
                (400, 4, 0.05, 0.9, 0.9, 8),
                (500, 3, 0.03, 0.8, 0.8, 10),
            ]
        ],
        "LightGBM": [
            {
                "estimator": LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_samples=min_child_samples,
                    class_weight="balanced",
                    objective="binary",
                    random_state=random_state,
                    n_jobs=4,
                    verbosity=-1,
                ),
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "num_leaves": num_leaves,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "min_child_samples": min_child_samples,
                },
                "fit_with_sample_weight": False,
                "fit_with_fold_scale_pos_weight": False,
            }
            for n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree, min_child_samples in [
                (300, 4, 0.05, 15, 0.9, 0.8, 30),
                (400, 5, 0.05, 31, 0.9, 0.9, 25),
                (500, 4, 0.03, 15, 0.8, 0.8, 20),
            ]
        ],
    }

# **Metric and Threshold Utilities**
# 
# These helpers score probability outputs and choose the F1-maximizing threshold on our development folds.


def _score_probabilities(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probabilities >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def _best_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.01, 0.96, 0.01)
    scores = [
        (threshold, f1_score(y_true, (probabilities >= threshold).astype(int), zero_division=0))
        for threshold in thresholds
    ]
    best_threshold, best_f1 = max(scores, key=lambda item: item[1])
    return float(best_threshold), float(best_f1)

# **Cross-Validated Model Selection**
# 
# This function evaluates each candidate specification using the time-based folds and records the best candidate for each model family.


def run_model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = build_time_folds(train_df)
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    catalog = make_model_catalog(random_state=random_state, scale_pos_weight=scale_pos_weight)

    tuning_rows: list[dict[str, Any]] = []
    best_candidates: dict[str, Any] = {}

    for model_name, candidates in catalog.items():
        best_candidate = None

        for candidate in candidates:
            estimator = candidate["estimator"]
            oof_probabilities = pd.Series(index=train_df.index, dtype=float)
            fold_rows = []

            for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
                model = clone(estimator)
                X_fold_train = X_train.loc[train_idx]
                y_fold_train = y_train.loc[train_idx]
                X_fold_val = X_train.loc[val_idx]
                y_fold_val = y_train.loc[val_idx]

                if candidate["fit_with_fold_scale_pos_weight"]:
                    neg_count_fold = int((y_fold_train == 0).sum())
                    pos_count_fold = int((y_fold_train == 1).sum())
                    scale_pos_weight_fold = neg_count_fold / max(pos_count_fold, 1)
                    model.set_params(scale_pos_weight=scale_pos_weight_fold)

                if candidate["fit_with_sample_weight"]:
                    sample_weight = np.where(y_fold_train == 1, scale_pos_weight, 1.0)
                    model.fit(X_fold_train, y_fold_train, sample_weight=sample_weight)
                else:
                    model.fit(X_fold_train, y_fold_train)
                fold_proba = model.predict_proba(X_fold_val)[:, 1]
                oof_probabilities.loc[val_idx] = fold_proba

                threshold, fold_best_f1 = _best_threshold(y_fold_val, fold_proba)
                metrics = _score_probabilities(y_fold_val, fold_proba, threshold)
                fold_rows.append(
                    {
                        "fold": fold_id,
                        "threshold": threshold,
                        "fold_best_f1": fold_best_f1,
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                    }
                )

            valid_oof = oof_probabilities.dropna()
            valid_y = y_train.loc[valid_oof.index]
            threshold, cv_best_f1 = _best_threshold(valid_y, valid_oof.to_numpy())
            cv_metrics = _score_probabilities(valid_y, valid_oof.to_numpy(), threshold)

            tuning_row = {
                "Model": model_name,
                "Params": candidate["params"],
                "CV Threshold": threshold,
                "CV F1": cv_metrics["f1"],
                "CV Precision": cv_metrics["precision"],
                "CV Recall": cv_metrics["recall"],
                "CV ROC-AUC": cv_metrics["roc_auc"],
                "CV PR-AUC": cv_metrics["pr_auc"],
                "Fold Summary": fold_rows,
                "Estimator": estimator,
                "fit_with_sample_weight": candidate["fit_with_sample_weight"],
            }
            tuning_rows.append(tuning_row)

            if best_candidate is None or tuning_row["CV F1"] > best_candidate["CV F1"]:
                best_candidate = tuning_row

        best_candidates[model_name] = best_candidate

    tuning_df = (
        pd.DataFrame(tuning_rows)
        .sort_values(["CV F1", "CV PR-AUC", "CV ROC-AUC"], ascending=False)
        .reset_index(drop=True)
    )

    return tuning_df, best_candidates

# **Validation Evaluation Utilities**
# 
# This function refits the selected candidate from each model family on the full development window and evaluates it on the 2021 validation year.


def evaluate_best_models(
    best_candidates: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for model_name, candidate in best_candidates.items():
        model = clone(candidate["Estimator"])
        if candidate["fit_with_sample_weight"]:
            neg_count = int((y_train == 0).sum())
            pos_count = int((y_train == 1).sum())
            scale_pos_weight = neg_count / max(pos_count, 1)
            sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold = candidate["CV Threshold"]
        val_predictions = (val_probabilities >= threshold).astype(int)
        metrics = _score_probabilities(y_val, val_probabilities, threshold)

        evaluation_rows.append(
            {
                "Model": model_name,
                "Threshold": threshold,
                "Validation F1": metrics["f1"],
                "Validation Precision": metrics["precision"],
                "Validation Recall": metrics["recall"],
                "Validation ROC-AUC": metrics["roc_auc"],
                "Validation PR-AUC": metrics["pr_auc"],
                "Params": candidate["Params"],
            }
        )

        fitted_models[model_name] = {
            "model": model,
            "threshold": threshold,
            "probabilities": val_probabilities,
            "predictions": val_predictions,
        }

    evaluation_df = (
        pd.DataFrame(evaluation_rows)
        .sort_values(["Validation F1", "Validation PR-AUC", "Validation ROC-AUC"], ascending=False)
        .reset_index(drop=True)
    )

    return evaluation_df, fitted_models

# **Reporting and Prediction Helpers**
# 
# These small helpers support our feature-importance reporting, confusion matrices, error analysis, and final test prediction export.


def compute_feature_importance(model: Any, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    if isinstance(model, Pipeline):
        final_model = model.named_steps["model"]
    else:
        final_model = model

    if hasattr(final_model, "coef_"):
        importance = np.abs(final_model.coef_[0])
    elif hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance_df


def summarize_confusion(y_true: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, predictions)
    return pd.DataFrame(
        cm,
        index=["Actual Healthy", "Actual Bankrupt"],
        columns=["Predicted Healthy", "Predicted Bankrupt"],
    )


def build_error_analysis(
    source_df: pd.DataFrame,
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_df = source_df[["company_id", "fiscal_year"]].copy()
    analysis_df["actual"] = y_true.values
    analysis_df["predicted"] = predictions
    analysis_df["predicted_probability"] = probabilities

    false_negatives = (
        analysis_df[(analysis_df["actual"] == 1) & (analysis_df["predicted"] == 0)]
        .sort_values("predicted_probability", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )
    false_positives = (
        analysis_df[(analysis_df["actual"] == 0) & (analysis_df["predicted"] == 1)]
        .sort_values("predicted_probability", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return false_negatives, false_positives


def build_test_predictions(
    fitted_model: Any,
    threshold: float,
    X_test: pd.DataFrame,
    raw_test_df: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = fitted_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    submission = raw_test_df[["company_id", "fiscal_year"]].copy()
    submission["bankruptcy_probability"] = probabilities
    submission["bankruptcy_predicted"] = predictions
    return submission

# ### **12.2 Class Imbalance and Validation Design**


# Our assignment explicitly requires two things for classification problems of this type:
# 
# - careful **imbalance handling**;
# - a validation strategy that reflects the structure of the data.
# 
# To satisfy both requirements, we adopt an **expanding-window cross-validation** scheme inside the model-development period.
# 
# The logic is the following:
# 
# - train on earlier years,
# - validate on the next available year,
# - progressively expand the training window.
# 
# In our data, this produces two development folds:
# 
# - **Fold 1**: train on **2018**, validate on **2019**;
# - **Fold 2**: train on **2018-2019**, validate on **2020**.
# 
# This design is more appropriate than standard random cross-validation for bankruptcy forecasting, because it evaluates whether each model generalizes well across years rather than across randomly mixed observations.
# 
# It also gives us a natural environment for **threshold tuning**: instead of using the default `0.50` probability cut-off, we optimize the threshold on the development folds with respect to **F1-score**. This is especially important in rare-event settings, where the default threshold often leads either to **no positives predicted at all** or to an excessive number of false alarms.


development_year_summary = (
    model_data.train_model_df.groupby("fiscal_year")["bankruptcy_next_year"]
    .agg(observations="size", bankruptcies="sum", bankruptcy_rate="mean")
    .reset_index()
)

fold_rows = []
for fold_id, (train_idx, val_idx) in enumerate(build_time_folds(model_data.train_model_df), start=1):
    train_years_fold = sorted(model_data.train_model_df.loc[train_idx, "fiscal_year"].unique().tolist())
    val_years_fold = sorted(model_data.train_model_df.loc[val_idx, "fiscal_year"].unique().tolist())
    fold_rows.append({
        "fold": fold_id,
        "train_years": train_years_fold,
        "validation_year": val_years_fold[0],
        "train_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "train_bankruptcy_rate": model_data.y_train.loc[train_idx].mean(),
        "validation_bankruptcy_rate": model_data.y_train.loc[val_idx].mean(),
    })

fold_design_df = pd.DataFrame(fold_rows)

print("Development sample by fiscal year:")
display(development_year_summary)

print("\nExpanding-window temporal folds used for model development:")
display(fold_design_df)


# **Interpretation of the results**


# The yearly summary and fold design make the modelling strategy fully transparent.
# 
# The development period preserves the chronological structure of the problem, and each validation fold contains a **very small number of bankruptcies**. This is a demanding setting: even a few additional true positives or false positives can produce visible changes in the F1-score.
# 
# That observation is important for two reasons:
# 
# - model evaluation should focus on **relative robustness across folds**, not just on a single numerical score
# - the final **2021 validation results** must be interpreted with care, because they are based on a **rare-event target** with limited positive cases
# 
# Even so, this strategy is still the correct one. It is much better to have a slightly noisy but **realistic** validation framework than a smoother but **leaky** one.


# ### **12.3 Cross-Validated Model Search**


# We begin by running a compact but structured model search.
# 
# Instead of tuning a very large hyperparameter grid, we evaluate a **small number of economically and statistically reasonable candidate specifications** for each model family. This is appropriate for the size of the dataset and keeps our comparison readable while still providing genuine model selection.
# 
# For each candidate model, we record:
# 
# - the chosen hyperparameters
# - the threshold selected through the development folds
# - the cross-validated **F1-score**
# - cross-validated **precision** and **recall**
# - ranking metrics such as **ROC-AUC** and **PR-AUC**
# 
# This step determines which specification of each model family should be carried forward into the final **2021 validation comparison**.


tuning_results, best_candidates = run_model_selection(
    X_train=model_data.X_train,
    y_train=model_data.y_train,
    train_df=model_data.train_model_df,
    random_state=RANDOM_STATE,
)

cv_overview = tuning_results[[
    "Model",
    "Params",
    "CV Threshold",
    "CV F1",
    "CV Precision",
    "CV Recall",
    "CV ROC-AUC",
    "CV PR-AUC",
]].copy()

cv_overview


# **Interpretation of the results**


# The cross-validated overview provides the first systematic comparison across model families.
# 
# At this stage, several patterns are typically informative:
# 
# - whether a model achieves its best performance only at **very low thresholds**, which would indicate a strong tendency to classify firms as healthy unless the cut-off is relaxed;
# - whether improvements in **recall** are obtained only at the cost of extremely poor **precision**;
# - whether a model ranks firms well in probability space (**ROC-AUC / PR-AUC**) but still struggles to convert those rankings into high F1 scores.
# 
# This distinction matters in bankruptcy prediction. A model can produce strong ranking metrics while still being operationally weak if its final class decisions are badly calibrated for a rare-event setting.


# ### **12.4 Baseline Linear Model - Logistic Regression**


# Logistic Regression is retained as the primary **linear baseline**.
# 
# This model is especially valuable in our comparison because:
# 
# - it is naturally interpretable;
# - it provides a benchmark for whether the engineered financial variables already contain enough signal in a mostly linear form;
# - it helps identify whether more complex non-linear models are actually adding predictive value or simply adding complexity.
# 
# The specification uses **class weighting** to counteract class imbalance and is trained on the final encoded feature set. Because Logistic Regression depends on feature scale, the implementation standardizes the inputs internally before fitting the classifier.


logistic_cv = cv_overview[cv_overview["Model"] == "Logistic Regression"].reset_index(drop=True)
logistic_cv


# **Interpretation of the results**


# The Logistic Regression results show how far a transparent linear boundary can go in separating bankrupt from non-bankrupt firms.
# 
# If Logistic Regression remains competitive against more complex models, this is usually a positive sign for interpretation: it suggests that the engineered features already encode a meaningful structure of financial distress.
# 
# In practice, this model often benefits strongly from **threshold tuning**, because the raw default cut-off of `0.50` is rarely appropriate when the bankruptcy rate is below 1%. A high optimal threshold may also indicate that the model generates a wide probability spread, assigning very high scores only to a small subset of firms.


# ### **12.5 Tree-Based Bagging Model - Random Forest**


# Random Forest is included as the first non-linear benchmark.
# 
# Compared with Logistic Regression, Random Forest can capture:
# 
# - interactions among financial ratios;
# - threshold effects (for example, liquidity becoming dangerous only below a certain point);
# - non-linear relationships between profitability, leverage, and firm size.
# 
# At the same time, Random Forest is still relatively stable and easy to compare with simpler models. It therefore serves as a natural bridge between a linear baseline and a more aggressive boosting strategy.


rf_cv = cv_overview[cv_overview["Model"] == "Random Forest"].reset_index(drop=True)
rf_cv


# **Interpretation of the results**


# The Random Forest results help us answer whether non-linear interactions are useful in the data.
# 
# A strong Random Forest would suggest that bankruptcy risk depends on combinations of variables that a purely linear model cannot capture. However, if Random Forest produces high ROC-AUC but low F1, that usually means the model ranks firms reasonably well but still struggles with **decision thresholding** or with isolating a sufficiently precise bankrupt class under extreme imbalance.


# ### **12.6 Gradient Boosting Model - XGBoost**


# XGBoost is used as the third model family because it is often one of the strongest methods for **structured financial tabular data**.
# 
# Conceptually, XGBoost differs from Random Forest in an important way:
# 
# - Random Forest builds many trees independently and averages them;
# - XGBoost builds trees **sequentially**, with each new tree focusing on the residual errors of the previous ones.
# 
# This sequential boosting mechanism is often effective when bankruptcy risk depends on **subtle, layered combinations of weak signals** rather than on a single dominant ratio.
# 
# For this reason, XGBoost is a better third candidate than a second bagging-style model. It introduces a genuinely different modelling logic and can exploit complex interactions among the engineered distress features, trend indicators, and sector-relative variables.


xgb_cv = cv_overview[cv_overview["Model"] == "XGBoost"].reset_index(drop=True)
xgb_cv


# **Interpretation of the results**


# The XGBoost results are particularly important because they reveal whether the dataset contains **non-linear weak signals** that are best captured through boosting.
# 
# In bankruptcy prediction, this often happens when no single variable is decisive on its own, but several moderate warning signs combine into a meaningful distress pattern. If XGBoost improves ranking metrics or F1 relative to the other models, that is evidence that the engineered features contain richer predictive structure than a simple linear boundary can exploit.


# ### **12.7 Gradient Boosting Model - LightGBM**


# LightGBM is added as a fourth model family because it is one of the most widely used alternatives to XGBoost for **high-dimensional tabular prediction problems**.
# 
# Like XGBoost, LightGBM is a boosting model. However, it differs in the way trees are grown and splits are selected, often making it:
# 
# - computationally efficient,
# - highly competitive on structured data,
# - and particularly useful as a robustness check when XGBoost is already performing well.
# 
# Including LightGBM allows us to verify whether the predictive gains observed for boosting are specific to XGBoost or reflect a more general advantage of **gradient-boosted decision trees** in bankruptcy prediction.
# 
# From our modelling perspective, LightGBM is therefore not redundant. It is a second strong boosting benchmark with a different inductive bias, and it helps make the final model choice more credible.


lgbm_cv = cv_overview[cv_overview["Model"] == "LightGBM"].reset_index(drop=True)
lgbm_cv


# **Interpretation of the results**


# The LightGBM results show whether an alternative gradient-boosting implementation can match or exceed the performance of XGBoost.
# 
# This comparison is useful because the two models often perform similarly in tabular tasks, but not always for the same reasons. If LightGBM performs strongly, it reinforces the idea that bankruptcy prediction in this dataset benefits from **boosting-based non-linear learners** rather than from a specific single algorithm alone.


# ### **12.8 Final Comparison on the 2021 Validation Year**


# After selecting the best specification within each model family, we refit each model on the entire development sample and evaluate once on the **2021 validation year**.
# 
# This is the decisive comparison for our final model selection.
# 
# Our primary ranking criterion remains **Validation F1**, because **F1-score** is the main performance metric for bankruptcy prediction. We still report secondary metrics because they provide useful diagnostic information:
# 
# - **Precision** measures the credibility of the bankruptcy alarms
# - **Recall** measures the ability to detect actual bankrupt firms
# - **ROC-AUC** measures ranking quality across all thresholds
# - **PR-AUC** is especially informative under class imbalance, because it focuses on the precision-recall trade-off in the positive class


final_model_results, fitted_models = evaluate_best_models(
    best_candidates=best_candidates,
    X_train=model_data.X_train,
    y_train=model_data.y_train,
    X_val=model_data.X_val,
    y_val=model_data.y_val,
)

final_model_results


# **Interpretation of the results**


# This comparison table is the key output of our model-development stage.
# 
# We read it in two layers:
# 
# - first, we identify which model achieves the highest **Validation F1**, since that is our primary evaluation criterion;
# - second, we examine whether that result comes from a sensible precision-recall balance or from an extreme operating point.
# 
# This layered reading is important in a rare-event problem. A model with very high recall but extremely poor precision may be useful for screening, but it may also generate too many false alarms. Conversely, a model with very high precision but very low recall may miss too many bankrupt firms to serve as a useful early-warning tool.
# 
# The final selected model should therefore be the one that offers the best **metric-aligned trade-off**, not necessarily the one with the highest ranking metric alone.


# ### **12.9 Final Model Selection and Decision Threshold**


# Once the 2021 comparison is available, we select the final model and examine its operational decision rule.
# 
# The selected threshold is not arbitrary. It is the threshold learned during development to optimize **F1-score**, and it determines how the model converts predicted bankruptcy probabilities into final binary decisions.
# 
# This step is especially important because our objective is not merely to rank firms by risk, but to produce a classifier that can meaningfully separate likely bankruptcies from healthy firms under strong imbalance.


best_model_name = final_model_results.iloc[0]["Model"]
best_model_payload = fitted_models[best_model_name]
best_model = best_model_payload["model"]
best_threshold = best_model_payload["threshold"]
best_val_predictions = best_model_payload["predictions"]
best_val_probabilities = best_model_payload["probabilities"]

print(f"Selected final model: {best_model_name}")
print(f"Selected threshold : {best_threshold:.2f}")

best_confusion = summarize_confusion(model_data.y_val, best_val_predictions)
best_confusion


# **Interpretation of the confusion matrix**


# The confusion matrix translates the model-selection result into a direct business interpretation.
# 
# It shows:
# 
# - how many truly bankrupt firms are correctly detected (**true positives**);
# - how many bankrupt firms are missed (**false negatives**);
# - how many healthy firms are incorrectly flagged (**false positives**).
# 
# In our setting, false negatives are often economically more severe than false positives, because a missed bankruptcy can imply a failure to identify distress in time. However, an excessive number of false positives also reduces the practical value of the model by creating too many false alarms.
# 
# For this reason, the confusion matrix provides the clearest operational reading of the selected threshold and complements the aggregate metric table reported above.


# ### **12.10 Feature Importance and Economic Interpretation**


# Our predictive model is more convincing if it is also economically interpretable.
# 
# After selecting the final model, we therefore inspect the variables with the strongest contribution to the decision process. This is useful for two reasons:
# 
# - it allows us to verify whether the model relies on economically plausible distress signals;
# - it provides material for the final presentation, where technical results must be translated into a **financial-risk narrative**.
# 
# Depending on the selected model family, the importance measure is derived either from:
# 
# - the absolute size of estimated coefficients (for Logistic Regression), or
# - feature importance scores (for tree-based models).
# 
# In both cases, our goal is not to interpret these values as causal effects, but to understand which variables carry the strongest predictive signal.


best_feature_importance = compute_feature_importance(
    model=best_model,
    feature_names=model_data.feature_cols,
    top_n=20,
)

best_feature_importance


# ### **12.11 Error Analysis**


# Beyond aggregate metrics, it is important to understand **which types of firms the model gets wrong**.
# 
# We therefore inspect two groups separately:
# 
# - **false negatives**: bankrupt firms that the model failed to flag;
# - **false positives**: healthy firms that the model incorrectly classified as bankrupt.
# 
# This is one of the most useful diagnostic steps in bankruptcy prediction, because it helps identify whether the remaining errors are concentrated among:
# 
# - borderline financial profiles,
# - firms with sudden deterioration,
# - sectors with unusual dynamics,
# - or cases where the probability threshold is simply too strict or too loose.
# 
# The resulting tables can also be used directly in the final presentation as evidence of critical thinking about model limitations.


false_negatives, false_positives = build_error_analysis(
    source_df=model_data.val_model_df,
    y_true=model_data.y_val,
    probabilities=best_val_probabilities,
    predictions=best_val_predictions,
    top_n=10,
)

print("Most important false negatives:")
display(false_negatives)

print("\nMost important false positives:")
display(false_positives)


# **Interpretation of the error analysis**


# The error tables help clarify whether model limitations come mainly from **missed bankruptcies** or from **over-sensitive risk alarms**.
# 
# This distinction is important when discussing next steps. For example:
# 
# - if the dominant issue is false negatives, the next improvement should focus on recovering additional distress cases, even at the price of slightly lower precision;
# - if the dominant issue is false positives, the next improvement should focus on refining threshold policy, probability calibration, or feature specificity.
# 
# In other words, the error analysis converts the validation results into an actionable roadmap for model refinement.


# ### **12.12 Final Predictions for the Unseen Test Period**


# After selecting the final model on the **2021 validation year**, we apply it to the unseen **2022-2023** dataset.
# 
# We store two outputs for each firm-year observation:
# 
# - the **predicted bankruptcy probability**
# - the **final binary classification** derived from the selected threshold
# 
# Saving both outputs is useful because the probability retains more information than the hard class label. In practice, a probability score can support ranking, screening, and later threshold adjustments if the business objective changes.


final_test_predictions = build_test_predictions(
    fitted_model=best_model,
    threshold=best_threshold,
    X_test=model_data.X_test,
    raw_test_df=df_test_fe,
)

output_dir = repo_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "challenge1_test_predictions.csv"
final_test_predictions.to_csv(output_path, index=False)

print(f"Saved final test predictions to: {output_path}")
final_test_predictions.head()


# **Interpretation of the final predictions file**


# The saved predictions file is the natural final output of our modelling work.
# 
# It can be used in two complementary ways:
# 
# - as a **final submission-style prediction table** for the unseen test period;
# - as a **risk-ranking table**, where firms can be ordered by bankruptcy probability rather than only by a binary decision.
# 
# This concludes our complete modelling pipeline. It includes the structure required by our assignment: explicit imbalance handling, comparison of four model families, model selection based on F1-score, and a final layer of interpretation through feature importance and error analysis.


# ## **13. Business Interpretation, Limitations, and Final Conclusions**


# At this point, we have completed our full modelling workflow:
# 
# - we explored the structure and quality of the financial data;
# - we engineered economically meaningful distress indicators;
# - we built a coherent modelling pipeline;
# - we compared multiple model families under a structured validation framework;
# - we selected a final model and generated predictions for the unseen test period.
# 
# The final step is to move from technical performance to **business interpretation**. In a bankruptcy-prediction setting, a model is useful only if its outputs can be translated into a clear risk narrative:
# 
# - which firms appear most vulnerable,
# - which financial patterns are associated with distress,
# - what trade-offs exist between missed bankruptcies and false alarms,
# - and what our current system can and cannot reliably support.
# 
# We therefore summarize the practical meaning of our modelling results, the main limitations of our current approach, and the most relevant directions for future improvement.


# ### **13.1 Main Modelling Outcome**


# Our final selected model is **Logistic Regression**, chosen on the basis of the **highest Validation F1-score** on the **2021 validation year** after comparison with **Random Forest, XGBoost, and LightGBM**.
# 
# This result is important because it suggests that the engineered feature space already contains a sufficiently clear distress structure for a **linear decision boundary** to remain competitive.
# 
# In practical terms, Logistic Regression offers two advantages at once:
# 
# - it provides the **best validation F1-score** in our executed results
# - and it remains the **most interpretable** model among the candidates tested
# 
# This makes our final solution easier to explain in an academic setting, since the selected model can be defended both on **performance grounds** and on **interpretability grounds**.


# ### **13.2 What the Final Metrics Mean in Practice**


# The final validation metrics show that the model is able to identify a meaningful portion of bankrupt firms, but also that the task remains very difficult.
# 
# The key **2021 validation results** for the selected Logistic Regression model are:
# 
# - **Validation F1 = 0.130**
# - **Precision = 0.077**
# - **Recall = 0.429**
# - **ROC-AUC = 0.929**
# - **PR-AUC = 0.057**
# 
# The central message is the following:
# 
# - the model has **useful ranking power**, as shown by strong probability-based metrics such as **ROC-AUC**
# - however, turning that ranking into a clean binary bankruptcy alarm remains challenging because the positive class is **extremely rare**
# 
# In practical terms, this means the model is more reliable as a **risk-screening tool** than as a fully autonomous decision rule.
# 
# A firm with a high predicted bankruptcy probability should therefore be interpreted as:
# 
# - a company that deserves **closer investigation**
# - not necessarily a company that will certainly fail
# 
# This is a realistic conclusion in financial early-warning systems. The main value of the model lies in its ability to **prioritize attention**, helping analysts, lenders, or risk managers focus first on firms exhibiting the strongest distress signals.


# ### **13.3 Interpretation of the Most Important Risk Signals**


# The feature-importance analysis suggests that the final model relies on a combination of **structural**, **relative**, and **profile-based** indicators.
# 
# From a business perspective, the most informative signals can be interpreted as follows:
# 
# - **Cluster membership** captures broader financial profiles that summarize combinations of liquidity, leverage, and profitability conditions. The strong importance of `cluster_id_practical` indicates that firms belong to recurring financial states that are meaningfully related to bankruptcy risk.
# - **Sector-relative profitability indicators** such as `profit_margin_vs_sector` show that distress is not only an absolute phenomenon. Firms that underperform their peers within the same sector may be more vulnerable even when their raw ratios do not appear catastrophic in isolation.
# - **Leverage-related variables** confirm that balance-sheet pressure remains central to distress prediction. Companies with heavier debt burdens or financially fragile capital structures are more exposed to bankruptcy risk.
# - **Geographic and legal-form indicators** may reflect institutional, structural, or business-environment differences that help the model refine its risk estimates.
# 
# Taken together, these results are economically plausible. They suggest that the model is not simply reacting to noise, but is identifying coherent patterns of fragility consistent with standard financial-risk reasoning.


# ### **13.4 Business Meaning of False Positives and False Negatives**


# The error analysis highlights one of the core tensions in bankruptcy prediction: the trade-off between **missing risky firms** and **over-flagging healthy ones**.
# 
# - A **false negative** is a genuinely bankrupt firm that the model classifies as healthy. In business terms, this is the most dangerous type of error, because it means the early-warning system failed to raise attention before a distress event.
# - A **false positive** is a healthy firm that is flagged as bankrupt. This is costly in a different way: it may trigger unnecessary scrutiny, misallocate analyst time, or produce excessive caution in lending or investment decisions.
# 
# For that reason, the selected threshold should not be interpreted as universally optimal in every business setting. Instead, it should be interpreted as the threshold that best balances **precision** and **recall** according to our primary evaluation metric, **F1-score**.
# 
# In a real operational context, the threshold could be adapted depending on the use case:
# 
# - a **supervisory or risk-monitoring** use case might prefer higher recall, tolerating more false positives;
# - a **credit-approval** use case might prefer more precision, accepting that some risky firms will not be flagged immediately.
# 
# This flexibility is one of the reasons why saving **probability outputs** is valuable: the model supports not only a binary decision, but also a ranked risk view.


# ### **13.5 Main Limitations of Our Current Solution**


# Although our final workflow is methodologically sound, several limitations remain and should be stated clearly.
# 
# 1. **Rare-event difficulty**
# The number of bankruptcies is very small, especially within each individual year. This makes F1-score inherently volatile and limits the statistical stability of model comparison.
# 
# 2. **Single validation year**
# Our final comparison relies on **2021** as the validation/model-selection period. This is appropriate for our setting, but it is still only one year and contains relatively few positive events.
# 
# 3. **Feature availability constraints**
# The dataset does not contain some variables that would be valuable for bankruptcy prediction, such as payment behavior, credit history, market-based signals, ownership information, or more granular macro-financial covariates.
# 
# 4. **Probability calibration not explicitly optimized**
# The model is tuned primarily for classification performance rather than for perfectly calibrated bankruptcy probabilities. As a result, the ranking may be strong even if the probabilities are not fully calibrated as absolute risk estimates.
# 
# 5. **Limited hyperparameter search**
# Our model search is intentionally compact. This keeps the analysis readable and proportionate to the scope of the assignment, but it also means that some models may improve under a wider tuning strategy.
# 
# These limitations do not invalidate the results. They simply define the boundary between what our current notebook can support convincingly and what would require a more extensive modelling exercise.


# ### **13.6 Most Relevant Next Improvements**


# If we were to continue improving our solution, the next steps would most likely focus on **feature enrichment and calibration**, rather than on radically changing our modelling philosophy.
# 
# The most promising improvements would include:
# 
# - **more bankruptcy-specific temporal features**, such as multi-year deterioration slopes, rolling distress persistence, or acceleration of leverage growth;
# - **missingness indicator features**, which may carry signal when financial ratios become undefined due to fragile balance-sheet conditions;
# - **sector-year normalization**, to separate firm-specific weakness from broader macro or sectoral shocks;
# - **probability calibration or threshold policy refinement**, especially if the output is meant to support business ranking or decision support;
# - **wider hyperparameter tuning** and additional robustness checks across linear and tree-based models.
# 
# In other words, our current model already provides a coherent and defendable bankruptcy-prediction system, but the strongest remaining gains are likely to come from **richer distress representation** rather than from simply swapping one classifier for another.


# ### **13.7 Final Conclusion**


# Overall, our project delivers a complete and economically interpretable bankruptcy-prediction solution.
# 
# Our results show that:
# 
# - bankruptcy prediction can be approached using a combination of **accounting-based indicators**, **temporal features**, **sector-relative variables**, and **cluster-based firm profiles**
# - among the tested models, the selected final classifier provides the strongest trade-off on the **2021 validation year** in our executed results and is therefore our final choice
# - our preprocessing, feature pruning, clustering, categorical encoding, and scaling steps are all applied consistently across the modelling pipeline
# 
# This is a good final outcome for a university project: our workflow is complete, the validation design is coherent, and the final model is supported by the executed results.