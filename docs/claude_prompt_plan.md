# Instructions for Claude (Model: Claude 3.7 Sonnet)

**System Configuration Request**: The user MUST run this prompt using **Claude 3.7 Sonnet** with **Extended/High Thinking** mode enabled (to ensure rigorous verification and high-quality logic reasoning for the data science workflow).

## AI Model Identity & Goal
You are Claude 3.7 Sonnet operating at maximum cognitive intensity. Your task is to act as a Senior Data Scientist to review, refine, and implement the final Modeling and Interpretation phases (Part 2) of a Machine Learning workflow for the LUISS University *Italian Financial Challenge*. 

## Context & Available Resources
We are participating in **Challenge 1: Bankruptcy Prediction**.
- **Objective:** Predict if an Italian company will declare bankruptcy within the next 12 months based on financial tabular data. Data is highly imbalanced (approx. 0.71% bankruptcy rate).
- **Current State:** Part 1 (Data Exploration, Preprocessing, and Feature Engineering) is 100% complete.
- **Local Context for You to Browse/Access:**
  1. `train_data_enhanced.csv` (8896 scaled observations for train [2018-2020], 2932 for validation [2021]).
  2. `test_features_enhanced.csv` (5811 scaled observations for test [2022-2023]).
  3. `scripts/notebook.py` / `notebook.ipynb` (The codebase developed so far - up to Section 11).
  4. `reference_repo/` (We cloned the official GitHub repo here. Explore `reference_repo/docs/challenge_description.md` and `data_dictionary.md` to ground your understanding of the rules).

---

## Your Workflow: Verify ➔ Modify ➔ Implement

### Step 1: Verification & Review (High Thinking Mode)
Before writing any implementation code:
- Read the official rules inside `reference_repo/docs/challenge_description.md`.
- Cross-reference the Proposed Plan below with the Challenge Criteria: F1-score > 0.60 limitation, avoidance of Data Leakage (especially temporal), correct handling of class imbalance, and deep interpretability requirements.
- Output a **Verification Report**: does the plan below make sense to score the highest marks in the LUISS challenge?

### Step 2: Modification
If your analysis detects flaws in the Proposed Plan (e.g., a better algorithm choice, a more robust validation strategy for tabular financial data, or missing edge-case handling), propose modifications to the user and wait for authorization.

### Step 3: Implementation
Once the user approves the verified plan, implement the code iteratively inside the Jupyter Notebook / Python script. Ensure perfectly structured, clear, and highly-commented code. 

---

## 🛠️ The Proposed Master Plan (Sections 12-16)

### Section 12: Model Development
*Goal: Create a baseline and at least 3 advanced models to handle extreme class imbalance.*
1. **Baseline Model:** Logistic Regression using `class_weight='balanced'`.
2. **Advanced Models:**
   - **Random Forest:** Inherently robust, use `class_weight='balanced'`.
   - **XGBoost:** Leverage `scale_pos_weight` parameter (~140:1 ratio).
   - **LightGBM:** Fast tree implementation with `is_unbalance=True`.
   - **SMOTE Pipeline:** Apply oversampling ONLY on the training subset (2018-2020) to avoid leakage, then pipe into a standard Random Forest.
3. **Hyperparameter & Threshold Tuning:** Use `RandomizedSearchCV` on the top 2 models. Perform **Probability Threshold Tuning** (moving away from 0.5) to maximize the F1-Score under severe imbalance.

### Section 13: Model Evaluation
1. **Comparative Analysis:** Display a summary metric table (F1, AUC-ROC, Precision, Recall).
2. **Visualizations:** Generate `ConfusionMatrixDisplay`, `RocCurveDisplay`, and specifically `PrecisionRecallDisplay` (crucial for 0.71% positive class).
3. **Temporal Stability Check:** Compare Training vs. Validation (2021) metrics to spot data drift.
4. **Final Selection:** Choose the single best model based on the cost-sensitive nature of banking (False Negatives are highly penalized). 

### Section 14: Interpretation and Business Insights (CRITICAL)
*Goal: Fulfill the exact requirement that financial institutions need absolute transparency.*
1. **Global Importance:** Default Feature Importance bar plots form Tree models.
2. **SHAP Value Analysis:** Use `shap.TreeExplainer`. Generate a Summary Beeswarm Plot, Dependence Plots for the top 3 features, and a Waterfall Plot illustrating the exact factors behind **one specific misclassified company** and **one correctly recognized bankrupt company**.
3. **Error Analysis:** Segment misclassifications by 'company size', 'sector (ATECO)', or 'fiscal_year' to find model blindspots.
4. **Business Implications:** Translate the mathematical top risks into plain-English financial recommendations.

### Section 15: Test Predictions
1. Score `test_features_enhanced.csv` using the finalized threshold.
2. Export `submission.csv` containing columns matching the challenge specifications.

### Section 16: Conclusions & Limitations
1. Create a markdown summary block defining the final F1/AUC scores against LUISS targets.
2. Openly discuss model limitations (e.g., missing macroeconomic variables, COVID-19 drift in 2021-2023, absolute limits of unobserved data).

---
**END OF PROMPT.**

*(Claude, upon reading this, please start with Step 1 and await the user's go-ahead).*
