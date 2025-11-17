
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

def train_model(
    df_clean: pd.DataFrame,
    target_col: str = "median_house_value",
    save_model_path: str = "xgboost_final_model.sav",
):
    """
    Train an XGBoost regression model on the cleaned dataset.
    Returns the trained model and test split.
    """

    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # 1. Split features & target
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scale numeric features
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # 4. Define XGBoost model
    model = XGBRegressor(
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    # 5. Train
    model.fit(X_train_scaled, y_train)

    # 6. Save model & scaler
    if save_model_path is not None:
        bundle = {
            "model": model,
            "scaler": scaler,
            "num_cols": num_cols.tolist(),
        }
        joblib.dump(bundle, save_model_path)

    return model, X_test_scaled, y_test
