"""Stage 2–3: inspect and clean the raw CKD dataset.

Run from the project root:
    python scripts/01_prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# make src/ importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.data.loader import load_raw, inspect, clean
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.data.preprocess import preprocess


def main() -> None:
    config = load_config()
    raw_path = ROOT / config["paths"]["raw_data"]

    print(f"Loading raw data from: {raw_path}\n")
    df = load_raw(raw_path)
    inspect(df)

    # Reconciliation against config
    expected = (
        [config["data"]["id_column"]]
        + config["data"]["numeric_features"]
        + config["data"]["nominal_features"]
        + [config["data"]["target_column"]]
    )
    actual = list(df.columns)

    print("\nRECONCILIATION WITH config.yaml:")
    in_file_not_config = [c for c in actual if c not in expected]
    in_config_not_file = [c for c in expected if c not in actual]

    if not in_file_not_config and not in_config_not_file:
        print("  ✓ All column names match config exactly.")
    else:
        if in_file_not_config:
            print("  Columns in FILE but not in config:")
            for c in in_file_not_config:
                print(f"      {c!r}")
        if in_config_not_file:
            print("  Columns in CONFIG but not in file:")
            for c in in_config_not_file:
                print(f"      {c!r}")

    # --- Clean and re-inspect ---
    print("\n\n" + "#" * 60)
    print("# AFTER CLEANING")
    print("#" * 60)
    df_clean = clean(df, config)
    inspect(df_clean)
    # --- Preprocess: impute -> encode -> scale (§3.3.3–3.3.5) ---
    print("\n\n" + "#" * 60)
    print("# AFTER PREPROCESSING")
    print("#" * 60)
    df_processed, scaler = preprocess(df_clean, config)

    print(f"\nSHAPE: {df_processed.shape[0]} rows x {df_processed.shape[1]} columns")
    print(f"MISSING VALUES REMAINING: {df_processed.isna().sum().sum()}")
    print("\nDTYPES (all should be numeric now):")
    print(df_processed.dtypes.to_string())
    print("\nTARGET DISTRIBUTION (classification):")
    print(df_processed[config["data"]["target_column"]].value_counts().to_string())
    print("\nFIRST 5 ROWS:")
    print(df_processed.head().to_string())

    # --- Save the processed dataset ---
    out_path = ROOT / config["paths"]["processed_data"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(out_path, index=False)
    print(f"\n✓ Saved processed data to: {out_path}")


if __name__ == "__main__":
    main()