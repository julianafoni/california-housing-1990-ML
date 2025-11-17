
import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess(
    raw_path: str = "data_california_house.csv",
    save_clean_path: str = "california_housing_clean.csv",
):
    """
    Load raw California housing data, clean it, engineer features,
    and (optionally) save a cleaned version.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV file.
    save_clean_path : str
        Where to save the cleaned CSV (set to None to skip saving).

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned and feature-engineered dataframe.
    """

    # 1. Load raw data
    df = pd.read_csv(raw_path)

    # 2. Feature engineering (if not already present)
    if "rooms_per_household" not in df.columns:
        df["rooms_per_household"] = df["total_rooms"] / df["households"]

    if "bedrooms_per_room" not in df.columns:
        df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

    if "population_per_household" not in df.columns:
        df["population_per_household"] = df["population"] / df["households"]

    # 3. Handle missing numeric values
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # 4. One-hot encode ocean_proximity if still object
    if "ocean_proximity" in df.columns and df["ocean_proximity"].dtype == "object":
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

    # 5. Save cleaned data (optional)
    if save_clean_path is not None:
        df.to_csv(save_clean_path, index=False)

    return df
