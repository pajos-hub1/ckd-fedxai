"""Reproducibility utilities.

Fixes all sources of randomness so experiments are repeatable
(Chapter 3, §3.9.4).
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and the hash seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)