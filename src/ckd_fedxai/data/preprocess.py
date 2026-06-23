"""Data preprocessing for the CKD dataset (§3.3.3–3.3.5).

Takes the cleaned dataframe (from loader.clean) and produces a fully
numeric, imputed, scaled, model-ready dataset:
  - impute missing values (median for numeric, mode for categorical)
  - encode categorical features and the target to 0/1
  - scale numeric features (standard or minmax)

All randomness and choices come from config.yaml so the pipeline is
reproducible (§3.9.4).
"""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def impute(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Impute missing values: median for numeric, most-frequent for categorical."""
    df = df.copy()
    numeric = [c for c in config["data"]["numeric_features"] if c in df.columns]
    nominal = [c for c in config["data"]["nominal_features"] if c in df.columns]

    # Convert pandas nullable types (Int64/Float64/string with <NA>) to classic
    # NumPy types, so scikit-learn's imputer sees ordinary np.nan, not pd.NA.
    for col in numeric:
        df[col] = df[col].astype("float64")          # <NA> -> np.nan
    for col in nominal:
        df[col] = df[col].astype("object").where(df[col].notna(), other=float("nan"))

    # numeric: median (robust to outliers, §3.3.3)
    if numeric:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric] = num_imputer.fit_transform(df[numeric])

    # categorical: most frequent (mode)
    if nominal:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[nominal] = cat_imputer.fit_transform(df[nominal])

    return df

def encode(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode the binary categorical features and the target to 0/1 (§3.3.4)."""
    df = df.copy()
    nominal = [c for c in config["data"]["nominal_features"] if c in df.columns]
    target = config["data"]["target_column"]

    # Map each binary categorical to 0/1. The CKD nominal features are all
    # binary, so a deterministic map keeps encoding interpretable.
    binary_maps = {
        "rbc":   {"normal": 0, "abnormal": 1},
        "pc":    {"normal": 0, "abnormal": 1},
        "pcc":   {"notpresent": 0, "present": 1},
        "ba":    {"notpresent": 0, "present": 1},
        "htn":   {"no": 0, "yes": 1},
        "dm":    {"no": 0, "yes": 1},
        "cad":   {"no": 0, "yes": 1},
        "appet": {"good": 0, "poor": 1},
        "pe":    {"no": 0, "yes": 1},
        "ane":   {"no": 0, "yes": 1},
    }
    for col in nominal:
        if col in binary_maps:
            df[col] = df[col].map(binary_maps[col]).astype("int64")

    # target: ckd -> 1, notckd -> 0 (§3.3.4)
    pos = config["data"]["positive_class"]
    neg = config["data"]["negative_class"]
    df[target] = df[target].map({pos: 1, neg: 0}).astype("int64")

    return df


def scale(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, object]:
    """Scale numeric features (§3.3.5). Returns (scaled_df, fitted_scaler)."""
    df = df.copy()
    numeric = [c for c in config["data"]["numeric_features"] if c in df.columns]

    method = config["preprocess"]["scaling"]
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()

    if numeric:
        df[numeric] = scaler.fit_transform(df[numeric])

    return df, scaler


def preprocess(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, object]:
    """Full preprocessing pipeline: impute -> encode -> scale.

    Returns (processed_df, fitted_scaler). The scaler is returned so the
    same transformation can be reused later if needed.
    """
    df = impute(df, config)
    df = encode(df, config)
    df, scaler = scale(df, config)
    return df, scaler