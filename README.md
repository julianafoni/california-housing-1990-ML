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
	•	Income inequality
	•	Coastal land scarcity
	•	Population density differences
	•	Housing structure disparities
  


This is a **regression modeling** project using:

- Linear Regression  
- KNN Regressor  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- LightGBM  
- **XGBoost (Best Performing Model)**  

---

## 📊 4. Evaluation Metrics

- RMSE (Root Mean Square Error)  
- MAE (Mean Absolute Error)  
- R² Score  
- Cross-Validation Mean & Std  

XGBoost achieved:

- **RMSE ≈ 46,125**  
- **MAE ≈ 30,584**  
- **R² ≈ 0.84**

---

## 🛠 5. Preprocessing Steps

- Handling missing values  
- One-hot encoding categorical features  
- Creating derived ratios  
- Scaling numeric features  
- Train-test split  
- Saving cleaned dataset  

---

## 🌲 6. Selected Algorithm: XGBoost

XGBoost performs best due to:

- Capturing nonlinear relationships  
- Handling interaction effects  
- Robustness against multicollinearity  
- Strong generalization (low CV variance)  

Key hyperparameters:

```python
learning_rate=0.05
n_estimators=300
max_depth=6
colsample_bytree=0.8
subsample=0.8
random_state=42```

---

## 🧾 7. Key Findings

- Median income consistently appears as the strongest predictor of housing value.
- Geographic coordinates (longitude, latitude) capture coastal–inland inequality.
- Population density metrics correlate with lower prices.
- Older housing stock can indicate more valuable historic neighborhoods.
- The housing market already exhibited nonlinear structures as early as 1990.

---

## 🏛 8. Policy Recommendations

- Introduce income-targeted housing subsidies.
- Reform coastal zoning to expand housing supply.
- Improve structural housing conditions in inland regions.
- Preserve historically valuable neighborhoods.
- Use XGBoost/Random Forest for future policy simulations.

---

## 💾 9. Saving & Loading the Final Model

The trained model is exported as:
Load the model:

```python
import pickle
model = pickle.load(open("model/xgboost_final_model.sav", "rb"))

pip install -r requirements.txt
