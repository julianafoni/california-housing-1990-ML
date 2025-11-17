# California Housing Price Prediction (1990 Baseline)
### Machine Learning Pipeline • Regression Modeling • Policy Insights
### By: Juliana Foni

---

## 🎯 1. Project Overview
This project builds a machine learning pipeline to predict housing prices in California using 1990 census data.
Beyond prediction, the project aims to understand socioeconomic and geographic factors shaping housing inequality.
Deliverables:
- Build an end-to-end machine learning pipeline  
- Compare multiple regression models  
- Identify the most influential predictors  
- Provide policy-oriented insights about inequality & affordability  
- Save a reusable baseline model (`.sav`) for future forecasting

---

## 🎯 2. Business Problem
California’s housing market is historically shaped by:
- Income inequality
- Coastal land scarcity
- Population density differences
- Housing structure disparities
Stakeholders (urban planners, policymakers, researchers) require a data-driven model to understand which factors most strongly influence housing values.
### Problem Statement
“Given district-level housing and demographic indicators, how can we predict median house value and identify the most influential socioeconomic and geographic factors?”

---

## 🎯 3. Project Objectives
1. Develop a regression model to predict 1990 median house values.
2. Identify key predictors (income, age of houses, geography, population density).
3. Build a reusable ML pipeline for future datasets (2000–2025).
4. Translate model findings into policy-oriented insights.
5. Export a production-ready ML model for simulation and forecasting.

---

## 🎯 4. Analytical Approach
This is a **supervised regression** project.
**Models Tested**
- Linear Regression (baseline)
- KNN Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- Random Forest Regressor
- LightGBM
- XGBoost (best-performing model)
  
**Pipeline Stages**
- Preprocessing
- Feature engineering
- Model training
- Cross-validation
- Evaluation
- Model exporting

---

## 🎯 5. Evaluation Metrics
- RMSE (Root Mean Square Error)  
- MAE (Mean Absolute Error)  
- R² Score  
- Cross-Validation Mean & Std  

XGBoost achieved:

- **RMSE ≈ 46,125**  
- **MAE ≈ 30,584**  
- **R² ≈ 0.84**

---

## 🎯 6. Preprocessing Steps
| Step                     | Description                                                    |
|-------------------------|----------------------------------------------------------------|
| Handle missing values   | Median imputation for numeric features                         |
| Encode categorical      | `ocean_proximity` → one-hot encoding                           |
| Feature scaling         | StandardScaler for numeric columns                             |
| Feature engineering     | Ratios such as `rooms_per_household`, `population_per_household` |
| Train-test split        | 80/20                                                          |

	•	src/data_preprocessing.py
	•	src/model_training.py
	•	src/model_evaluation.py

---

## 🎯 7. Selected Algorithm: XGBoost
XGBoost performs best due to:
- Capturing nonlinear relationships  
- Handling interaction effects  
- Robustness against multicollinearity  
- Strong generalization (low CV variance)
	* _learning_rate = 0.05
 	* n_estimators = 300
	* max_depth = 6
	* colsample_bytree = 0.8
	* subsample = 0.8
	* random_state = 42_

---

## 🎯 8. Key Findings

- **Median income** consistently appears as the strongest predictor of housing value.
- Geographic coordinates **(longitude, latitude)** capture coastal–inland inequality.
- Population density metrics correlate with lower prices.
- **Older housing stock** can indicate more valuable historic neighborhoods.
- The 1990 housing market was already **highly nonlinear**, making tree-based models ideal.

---

## 🎯 9. Policy Recommendations

- Introduce income-targeted housing subsidies.
- Reform coastal zoning to expand housing supply.
- Improve structural housing conditions in inland regions.
- Preserve historically valuable neighborhoods.
- Use XGBoost/Random Forest for future policy simulations.

---

## 🎯 10. Project Assets
| File | Description |
|------|-------------|
| [📘 Presentation PDF](assets/California Housing Price Prediction (1990 Baseline).pdf) | Final presentation slides for stakeholders |
| [🐍 CAPSS3 Notebook](notebook/CAPS3.ipynb) | End-to-end analysis & model training in Python |
| [💾 Final Model .sav](model/xgboost_final_model.sav) | Serialized XGBoost model ready for loading |
