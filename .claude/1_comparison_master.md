# Phase 1: Master Comparison & Briefing

**Instructions:** Provide this to Claude after he acknowledges Phase 0.

---

"Please perform a deep, side-by-side audit of the two attached files. I need a briefing that compares them across these 4 critical dimensions:

### **1. Modeling Rigor & Pipeline**
- Evaluate the model selection engine (manual vs. automated search).
- Analyze the anti-leakage discipline (how scaling and resampling like SMOTE are handled within CV folds).
- Compare the validation strategy (Temporal split vs. Cross-validation design).

### **2. Feature Engineering & Strategy**
- Compare the use of Altman-like scores, sector-relative ratios, and clustering.
- Evaluate the dimensionality reduction and feature selection logic in both.

### **3. Interpretation & Explainability**
- Evaluate the quality of SHAP usage (Summary plots vs. Waterfall plots for individual cases).
- Analyze how each notebook handles the 'Macroeconomic Narrative' (explaining results through the lens of the COVID shock).

### **4. Business Logic & Economic ROI**
- Search for a **Cost-Benefit Analysis**. Evaluate which notebook better quantifies the economic value (in Euros) for a financial institution.

**Output:** Provide a side-by-side summary table and a concise briefing of the relative strengths and weaknesses of each version. Do not suggest code changes yet."
