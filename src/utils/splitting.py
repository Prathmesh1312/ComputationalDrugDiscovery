"""
splitting.py — Bemis-Murcko Scaffold Splitting for Zero-Leakage Molecular Evaluation
"""
import random
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_murcko_scaffold(smiles: str) -> str:
    """
    Compute canonical Bemis-Murcko scaffold SMILES (falls back to canonical SMILES if acyclic).
    Includes an exact ring-and-linker pruning fallback for RDKit builds where C++ MurckoDecompose
    raises `RuntimeError: bad any cast`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    if mol.GetRingInfo().NumRings() == 0:
        return canon

    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.error")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold:
            return scaffold
    except Exception:
        pass
    finally:
        from rdkit import RDLogger
        RDLogger.EnableLog("rdApp.error")

    # Exact Bemis-Murcko ring + linker core extraction via iterative degree-1 leaf pruning
    rw = Chem.RWMol(mol)
    while True:
        to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetDegree() <= 1]
        if not to_remove:
            break
        for idx in sorted(to_remove, reverse=True):
            rw.RemoveAtom(idx)

    if rw.GetNumAtoms() == 0:
        return canon
    m2 = rw.GetMol()
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.error")
        Chem.SanitizeMol(m2)
    except Exception:
        pass
    finally:
        from rdkit import RDLogger
        RDLogger.EnableLog("rdApp.error")
    scaf = Chem.MolToSmiles(m2, canonical=True, isomericSmiles=False)
    return scaf if scaf else canon


def _split_has_all_classes(
    df: pd.DataFrame, indices: List[int], task_cols: Sequence[str]
) -> bool:
    if not indices:
        return False
    sub = df.iloc[indices]
    for col in task_cols:
        if col not in sub.columns:
            continue
        vals = sub[col].dropna()
        if len(vals) == 0 or vals.nunique() < 2:
            return False
    return True


def bemis_murcko_scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    task_cols: Sequence[str] = ("bioavailability", "bbb", "herg"),
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Partition dataset into train/val/test by Bemis-Murcko scaffold so no molecular scaffold
    appears in more than one split (preventing structural memorization leakage).
    """
    df = df.copy().reset_index(drop=True)
    scaffolds = [get_murcko_scaffold(sm) for sm in df[smiles_col].tolist()]
    df["scaffold"] = scaffolds

    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, scaf in enumerate(scaffolds):
        scaffold_to_indices[scaf].append(idx)

    # Group by bucket size, shuffle deterministically within equal-size buckets, sort descending
    rng = random.Random(seed)
    size_buckets: Dict[int, List[List[int]]] = defaultdict(list)
    for idxs in scaffold_to_indices.values():
        size_buckets[len(idxs)].append(idxs)

    ordered_groups: List[List[int]] = []
    for sz in sorted(size_buckets.keys(), reverse=True):
        groups = size_buckets[sz]
        rng.shuffle(groups)
        ordered_groups.extend(groups)

    n_total = len(df)
    train_target = int(train_frac * n_total)
    val_target = int(val_frac * n_total)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    # Separate singletons vs multi-molecule scaffolds: place largest scaffold buckets into train first
    # while filling val/test up to target fractions with diverse medium/small scaffolds so all classes appear
    val_groups: List[List[int]] = []
    test_groups: List[List[int]] = []
    train_groups: List[List[int]] = []

    for group in ordered_groups:
        # Place into train until reaching half target, then interleave val/test/train proportionally
        if len(train_idx) + len(group) <= int(train_target * 0.65):
            train_idx.extend(group)
            train_groups.append(group)
        elif len(val_idx) + len(group) <= val_target:
            val_idx.extend(group)
            val_groups.append(group)
        elif len(test_idx) + len(group) <= (n_total - train_target - val_target):
            test_idx.extend(group)
            test_groups.append(group)
        else:
            train_idx.extend(group)
            train_groups.append(group)

    # Verify both binary classes present in val and test; swap whole scaffold groups with train if needed
    for split_name, target_groups, target_idx in [
        ("val", val_groups, val_idx),
        ("test", test_groups, test_idx),
    ]:
        if not _split_has_all_classes(df, target_idx, task_cols):
            # Swap small scaffold groups from train until all classes represented
            for t_i in range(len(train_groups) - 1, -1, -1):
                cand = train_groups[t_i]
                if len(cand) > 10:
                    continue
                if target_groups:
                    give_back = target_groups.pop(0)
                    target_groups.append(cand)
                    train_groups[t_i] = give_back
                    new_target = [idx for g in target_groups for idx in g]
                    if _split_has_all_classes(df, new_target, task_cols):
                        target_idx[:] = new_target
                        train_idx[:] = [idx for g in train_groups for idx in g]
                        break

    split_arr = np.empty(n_total, dtype=object)
    split_arr[train_idx] = "train"
    split_arr[val_idx] = "val"
    split_arr[test_idx] = "test"
    df["split"] = split_arr

    # Strict zero-leakage assertions
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    assert len(set(train_df["scaffold"]) & set(val_df["scaffold"])) == 0, "Scaffold leak train/val!"
    assert len(set(train_df["scaffold"]) & set(test_df["scaffold"])) == 0, "Scaffold leak train/test!"
    assert len(set(val_df["scaffold"]) & set(test_df["scaffold"])) == 0, "Scaffold leak val/test!"

    assert len(set(train_df[smiles_col]) & set(val_df[smiles_col])) == 0, "SMILES leak train/val!"
    assert len(set(train_df[smiles_col]) & set(test_df[smiles_col])) == 0, "SMILES leak train/test!"
    assert len(set(val_df[smiles_col]) & set(test_df[smiles_col])) == 0, "SMILES leak val/test!"

    for sp_name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        assert _split_has_all_classes(df, idxs, task_cols), f"Split {sp_name} missing binary class!"

    return df
