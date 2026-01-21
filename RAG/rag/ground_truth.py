import ast
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .utils import normalize_ecli


# Type alias used throughout evaluation code
GroundTruth = Dict[str, List[str]]


def norm_ecli(ecli: Optional[str]) -> str:
    """
    Small wrapper so other modules can import `norm_ecli` from here.
    """
    return normalize_ecli(ecli) if ecli is not None else ""


def _load_ground_truth_from_excel(path: Path) -> GroundTruth:
    """
    Internal helper: load ground-truth mapping from an Excel file.

    Returns:
        dict: advice_id (or zaaknummer) -> list of ECLI strings
    """
    if not path.exists():
        print(f"⚠ Advice file not found: {path}")
        return {}

    df = pd.read_excel(path)
    ground_truth: GroundTruth = {}

    # Get ID column
    id_col = None
    for col in ["Octopus zaaknummer", "zaaknummer", "id", "doc_id"]:
        if col in df.columns:
            id_col = col
            break

    if not id_col:
        print("⚠ Could not find ID column in advice file")
        return {}

    # Get ECLI column
    if "ECLI" not in df.columns:
        print("⚠ No ECLI column found in advice file")
        return {}

    for _, row in df.iterrows():
        advice_id = str(row[id_col])
        ecli_value = row["ECLI"]

        # Parse ECLI (might be string representation of list or actual list)
        ecli_list: List[str] = []
        if pd.notna(ecli_value):
            if isinstance(ecli_value, str):
                try:
                    # Try to parse as Python list
                    parsed = ast.literal_eval(ecli_value)
                    if isinstance(parsed, list):
                        ecli_list = [str(e) for e in parsed]
                    else:
                        ecli_list = [str(parsed)]
                except Exception:
                    # If parsing fails, treat as single ECLI
                    ecli_list = [ecli_value]
            elif isinstance(ecli_value, list):
                ecli_list = [str(e) for e in ecli_value]
            else:
                ecli_list = [str(ecli_value)]

        # Normalize ECLI numbers (remove duplicates, ensure they're strings)
        ecli_list = list(
            {
                norm_ecli(str(e).strip())
                for e in ecli_list
                if pd.notna(e) and str(e).strip()
            }
        )
        if ecli_list:
            ground_truth[advice_id] = ecli_list

    return ground_truth


def load_ground_truth_from_advice_excel(path: Path) -> GroundTruth:
    """
    Public API used by `scripts/evaluate.py`.

    Args:
        path: Path to the advice Excel file.
    """
    gt = _load_ground_truth_from_excel(path)
    if not hasattr(load_ground_truth_from_advice_excel, "_printed"):
        print(f"✓ Loaded ground truth for {len(gt)} advice letters from {path.name}")
        load_ground_truth_from_advice_excel._printed = True  # type: ignore[attr-defined]
    return gt


def load_ground_truth_ecli() -> GroundTruth:
    """
    Backwards-compatible helper used in some notebooks:
    load ground truth from the default advice Excel in the current working dir.
    """
    advice_file = Path.cwd() / "Dataset Advice letters on objections towing of bicycles.xlsx"
    return _load_ground_truth_from_excel(advice_file)


def split_train_test(
    ground_truth: GroundTruth,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split ground truth into train and test sets.
    
    Args:
        ground_truth: Full ground truth dictionary
        test_ratio: Ratio of test set (default: 0.2, i.e., 20%)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_ground_truth, test_ground_truth)
    """
    if not ground_truth:
        return {}, {}
    
    # Get all advice IDs
    advice_ids = list(ground_truth.keys())
    
    # Shuffle with fixed seed
    random.seed(random_seed)
    shuffled_ids = advice_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Split
    split_idx = int(len(shuffled_ids) * (1 - test_ratio))
    train_ids = set(shuffled_ids[:split_idx])
    test_ids = set(shuffled_ids[split_idx:])
    
    # Create split dictionaries
    train_gt = {aid: ecli_list for aid, ecli_list in ground_truth.items() if aid in train_ids}
    test_gt = {aid: ecli_list for aid, ecli_list in ground_truth.items() if aid in test_ids}
    
    return train_gt, test_gt


def get_train_ids(ground_truth: GroundTruth, test_ratio: float = 0.2, random_seed: int = 42) -> Set[str]:
    """
    Get set of training advice IDs for data leakage prevention.
    
    Args:
        ground_truth: Full ground truth dictionary
        test_ratio: Ratio of test set (default: 0.2)
        random_seed: Random seed for reproducibility
    
    Returns:
        Set of training advice IDs
    """
    train_gt, _ = split_train_test(ground_truth, test_ratio, random_seed)
    return set(train_gt.keys())
