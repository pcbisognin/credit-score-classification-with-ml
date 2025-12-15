# credit-score-classification-with-ml
Credit score classification using supervised ML (Random Forest, XGBoost, LightGBM) with robust preprocessing, feature engineering, and metric-driven evaluation.

# Credit Score Classification with Machine Learning

End-to-end machine learning project to classify customer credit score into three categories: **Poor**, **Standard**, and **Good**.  
The workflow covers **EDA**, **data cleaning**, **feature engineering**, **model training**, **hyperparameter tuning**, and **evaluation with business-relevant metrics**.

## Objective
Build supervised classification models that support credit decisions by identifying higher-risk customers. In practice, correctly detecting the **Poor** class helps reduce default risk and improve portfolio quality.

## Dataset
- Source: Kaggle — Credit Score Classification (parisrohan)
- Training set: `train.csv` (labeled), **100,000 rows**
- Test set: `test.csv` (unlabeled)

Because the Kaggle test set is unlabeled, evaluation is performed using a **train/test split** from the labeled training data.

## Workflow

### 1) Data Cleaning & Feature Engineering
Key preprocessing steps:
- Removed identifier-like fields that do not add predictive signal:
  - `ID`, `Customer_ID`, `Name`, `SSN`
- Converted inconsistent numeric columns stored as strings (including trailing underscores) to proper numeric types.
- Handled invalid and unrealistic values:
  - `Age`: values `< 18` or `> 100` treated as invalid and imputed with the median.
  - Negative counts (e.g., loans, delayed payments) clipped to `0`.
- Addressed missing values with a mix of statistical and rule-based strategies:
  - `Payment_Behaviour`: replaced invalid tokens with the mode.
  - `Occupation`: imputed using income quantiles (mode occupation per income band).
  - `Credit_Mix`: imputed using debt tertiles (mode per debt band) due to strong correlation with `Outstanding_Debt`.
  - Remaining numeric features: median imputation where correlations were not informative.
- Transformed `Credit_History_Age` from text (e.g., “X Years and Y Months”) into total months.
- Dropped `Monthly_Balance` due to complete missingness in the processed dataset.

### 2) Exploratory Data Analysis (EDA)
- Distributions and summary statistics of numeric variables
- Categorical frequency analysis (occupation, loan type, payment behavior)
- Correlation analysis across numeric features to inform imputation and interpretation

### 3) Encoding & Scaling
- One-hot encoding applied to categorical variables (`Occupation`, `Type_of_Loan`, `Payment_Behaviour`)
- Robust scaling (RobustScaler) applied after the train/test split to reduce sensitivity to outliers

## Modeling
Three supervised models were trained and compared using pipelines and GridSearchCV:
- **Random Forest**
- **XGBoost**
- **LightGBM**

Hyperparameter tuning was performed with cross-validation (GridSearchCV). Due to compute/time constraints, the grids were intentionally limited and can be expanded in future iterations.

## Evaluation
Since misclassifying high-risk customers is costly, **recall for the Poor class** is treated as a key metric alongside overall performance.

### Results (held-out test split)
- **Random Forest**
  - Accuracy: ~0.78
  - Recall (Poor): **0.78**
- **XGBoost**
  - Accuracy: ~0.73
  - Recall (Poor): **0.68**
- **LightGBM**
  - Accuracy: ~0.75
  - Recall (Poor): **0.73**

Random Forest achieved the best performance under the tested grids, likely benefiting from fewer hyperparameters and a smaller search space.

## Practical Use (Business Value)
This type of model can be used to:
- Improve credit approval decisions by reducing exposure to high-risk borrowers
- Segment customers by risk profile for tailored products and credit limits
- Standardize and accelerate credit analysis workflows
- Support strategic reporting and risk management

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, XGBoost, LightGBM

└── README.md

