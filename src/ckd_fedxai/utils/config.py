"""Configuration loading utility.

Loads the central YAML config so that no parameters are hard-coded in
scripts or modules (Chapter 3, §3.9.4 reproducibility).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_project_root(marker: str = "config/config.yaml") -> Path:
    """Walk upward from this file until the config marker is found."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / marker).exists():
            return parent
    # Fallback: three levels up from src/ckd_fedxai/utils/config.py
    return here.parents[3]


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the YAML configuration as a dictionary.

    Args:
        path: optional explicit path to a config file. If None, the
              default config/config.yaml at the project root is used.
    """
    if path is None:
        root = find_project_root()
        path = root / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from config['paths'] relative to the project root."""
    root = find_project_root()
    return root / config["paths"][key]