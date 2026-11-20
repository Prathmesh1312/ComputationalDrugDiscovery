"""
seed.py — Deterministic Seeding and Compute Device Selection Utilities
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def get_device(prefer: str = "auto") -> torch.device:
    """Select compute device: MPS > CUDA > CPU (or explicit override)."""
    if prefer != "auto":
        return torch.device(prefer)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
