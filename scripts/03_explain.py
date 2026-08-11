"""Stage 5: SHAP (global) and LIME (local) explainability — §3.6.

Run from the project root:
    python scripts/03_explain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_fedxai.explain.explainer import lime_fidelity_and_stability, lime_local, shap_global
from ckd_fedxai.utils.config import load_config
from ckd_fedxai.utils.seed import set_seed


def main() -> None:
    config = load_config()
    set_seed(config["seed"])

    df = pd.read_csv(ROOT / config["paths"]["processed_data"])
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"],
        stratify=y, random_state=config["seed"],
    )

    figures_dir = ROOT / config["paths"]["figures_dir"]
    models_dir = ROOT / config["paths"]["models_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    for model_name in config["explainability"]["models_to_explain"]:
        print("\n" + "=" * 55)
        print(f"EXPLAINING: {model_name}")
        print("=" * 55)

        model_path = models_dir / f"central_{model_name}.joblib"
        if not model_path.exists():
            print(f"  ! Model not found ({model_path.name}). Run 02_train_central.py first.")
            continue
        model = joblib.load(model_path)

        if config["explainability"]["shap"]:
            ranking = shap_global(
                model, X_test, model_name, figures_dir,
                max_display=config["explainability"]["shap_max_display"],
            )
            print("\n  Top 10 features by mean |SHAP|:")
            print(ranking.head(10).to_string(index=False))

        if config["explainability"]["lime"]:
            lime_local(
                model, X_train, X_test, model_name, figures_dir,
                num_samples=config["explainability"]["num_lime_samples"],
            )

            print("\n  Computing LIME fidelity and stability (§3.8.2)...")
            fs = lime_fidelity_and_stability(
                model, X_train, X_test,
                num_samples=config["explainability"]["num_lime_samples"],
            )
            print(fs.to_string(index=False))
            print(f"  mean fidelity (R^2):        {fs['fidelity_r2'].mean():.4f}")
            print(f"  mean stability (Jaccard):   {fs['stability_jaccard'].mean():.4f}")

            metrics_dir = ROOT / config["paths"]["metrics_dir"]
            out = metrics_dir / f"lime_fidelity_stability_{model_name}.json"
            out.write_text(json.dumps({
                "per_patient": fs.to_dict(orient="records"),
                "fidelity_r2_mean": float(fs["fidelity_r2"].mean()),
                "fidelity_r2_std": float(fs["fidelity_r2"].std()),
                "stability_jaccard_mean": float(fs["stability_jaccard"].mean()),
                "stability_jaccard_std": float(fs["stability_jaccard"].std()),
            }, indent=2))
            print(f"  ✓ Saved: {out}")

    print(f"\n✓ All explainability figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()