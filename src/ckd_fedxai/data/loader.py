"""Data loading and inspection for the CKD dataset.

Stage 2 (diagnostic): load the raw CSV and report exactly what is in it —
columns, shape, dtypes, missing values, and the unique values of object
columns (where the stray-whitespace quirks hide). No cleaning yet; this
tells us what cleaning is actually needed (§3.2).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CKD CSV exactly as-is (no type coercion)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place kidney_disease.csv in data/raw/ (see README)."
        )
    # keep everything as read; don't let pandas guess-coerce yet
    return pd.read_csv(path)


def inspect(df: pd.DataFrame, max_unique: int = 15) -> None:
    """Print a full diagnostic of the dataframe's current state."""
    print("=" * 60)
    print(f"SHAPE: {df.shape[0]} rows x {df.shape[1]} columns")
    print("=" * 60)

    print("\nCOLUMNS (exact names, in order):")
    for i, col in enumerate(df.columns):
        print(f"  [{i:>2}] {col!r}")

    print("\nDTYPES:")
    print(df.dtypes.to_string())

    print("\nMISSING VALUES PER COLUMN:")
    missing = df.isna().sum()
    for col, n in missing.items():
        pct = 100 * n / len(df)
        print(f"  {col:<20} {n:>4}  ({pct:4.1f}%)")
    print(f"  {'TOTAL':<20} {missing.sum():>4}")

    print("\nUNIQUE VALUES OF OBJECT/TEXT COLUMNS")
    print("(this is where stray-whitespace and typo quirks show up):")
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) in ("str", "string"):
            uniques = df[col].dropna().unique()
            shown = uniques[:max_unique]
            # repr() so we can SEE trailing spaces / tab chars
            rendered = ", ".join(repr(u) for u in shown)
            more = "" if len(uniques) <= max_unique else f"  …(+{len(uniques)-max_unique} more)"
            print(f"  {col:<20} -> {rendered}{more}")

    print("\nFIRST 5 ROWS:")
    print(df.head().to_string())
    print("=" * 60)


def clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean the raw CKD dataframe (§3.3.1–3.3.2).

    - drop the id column
    - strip stray whitespace/tab characters from all text columns
    - coerce numeric-but-text columns (pcv, wc, rc) to real numbers
    - standardise the target to clean 'ckd' / 'notckd'
    Missing values are left as NaN here; imputation happens in Stage 3.
    """
    df = df.copy()

    # 1. drop id column (§3.3.2)
    id_col = config["data"]["id_column"]
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    # 2. strip whitespace/tabs from every text column (§3.3.1)
    #    fixes ' yes', '\tno', '\tyes', 'ckd\t', etc.
    text_cols = [c for c in df.columns
                 if df[c].dtype == object or str(df[c].dtype) in ("str", "string")]
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
        # normalise any remaining empty strings to NaN
        df[col] = df[col].replace({"": pd.NA})

    # 3. coerce numeric-but-text columns to real numbers (§3.3.1)
    #    these were forced to text by stray characters
    numeric_features = config["data"]["numeric_features"]
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. standardise the target (§3.3.1)
    target = config["data"]["target_column"]
    df[target] = df[target].astype("string").str.strip()

    return df