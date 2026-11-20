"""
chemistry.py — Chemoinformatics Standardization, Exact Properties, PAINS/Lipinski, and 3D Conformer Geometry
"""
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize

# Load RDKit contrib SA_Score module cleanly
_SA_DIR = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _SA_DIR not in sys.path:
    sys.path.append(_SA_DIR)
import sascorer  # type: ignore  # noqa: E402

_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

_PAINS_PARAMS = FilterCatalogParams()
_PAINS_PARAMS.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog(_PAINS_PARAMS)


def suppress_rdkit_warnings() -> None:
    """Disable noisy RDKit non-fatal warnings while preserving critical errors."""
    RDLogger.DisableLog("rdApp.warning")


def standardize_smiles(
    smiles: str, min_heavy: int = 2, max_heavy: int = 80
) -> Optional[str]:
    """
    Parse, desalt/strip counterions (largest fragment), validate organic carbon content
    and heavy atom count, and return canonical non-isomeric SMILES.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        mol = _LARGEST_FRAG.choose(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy < min_heavy or num_heavy > max_heavy:
        return None

    has_carbon = any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())
    if not has_carbon:
        return None

    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    if not canon or Chem.MolFromSmiles(canon) is None:
        return None
    return canon


def compute_exact_qed_sa(smiles: str) -> Tuple[float, float]:
    """Compute exact RDKit QED [0, 1] and Synthetic Accessibility Score [1, 10]."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.5, 5.0
    qed_val = float(QED.qed(mol))
    sa_val = float(np.clip(sascorer.calculateScore(mol), 1.0, 10.0))
    return qed_val, sa_val


def compute_lipinski_and_pains(smiles: str) -> Dict[str, Any]:
    """Compute Lipinski Rule-of-5 descriptors and RDKit PAINS structural alert check."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "mw": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "tpsa": 0.0,
            "rotatable_bonds": 0,
            "lipinski_violations": 4,
            "lipinski_pass": False,
            "pains_pass": False,
        }

    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    rot_bonds = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

    violations = (
        int(mw > 500.0)
        + int(logp > 5.0)
        + int(hbd > 5)
        + int(hba > 10)
    )
    pains_pass = not _PAINS_CATALOG.HasMatch(mol)

    return {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotatable_bonds": rot_bonds,
        "lipinski_violations": violations,
        "lipinski_pass": violations <= 1,
        "pains_pass": bool(pains_pass),
    }


def compute_morgan_fp_array(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute binary Morgan ECFP fingerprint NumPy array without deprecation warnings."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    gen = (
        _MORGAN_GEN
        if (radius == 2 and n_bits == 2048)
        else rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    )
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def compute_tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute Morgan ECFP4 Tanimoto similarity between two SMILES."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = _MORGAN_GEN.GetFingerprint(mol_a)
    fp_b = _MORGAN_GEN.GetFingerprint(mol_b)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def compute_3d_conformer_features(
    mol: Chem.Mol, random_seed: int = 42
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], np.ndarray]:
    """
    Generate a fast 3D conformer using RDKit ETKDGv3 and extract:
      1. atom_dist_to_centroid: [N] heavy-atom normalized distance to molecular centroid
      2. bond_lengths: dict[(min(i,j), max(i,j))] -> Euclidean bond length / 2.0
      3. geom_3d: [4] vector [NPR1, NPR2, Asphericity, RadiusOfGyration / 10.0]

    Gracefully returns safe zeros / empty dict on embedding failure or huge molecules.
    """
    num_heavy = mol.GetNumHeavyAtoms()
    zero_atom_dists = np.zeros(num_heavy, dtype=np.float32)
    zero_geom = np.zeros(4, dtype=np.float32)

    if num_heavy <= 1 or num_heavy > 80:
        return zero_atom_dists, {}, zero_geom

    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.maxIterations = 50
        params.useSmallRingTorsions = False
        res = AllChem.EmbedMolecule(mol_h, params)
        if res != 0:
            return zero_atom_dists, {}, zero_geom

        conf = mol_h.GetConformer()
        coords = np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(num_heavy)
            ],
            dtype=np.float32,
        )

        centroid = coords.mean(axis=0, keepdims=True)
        dists_to_cent = np.linalg.norm(coords - centroid, axis=1)

        rog = float(rdMolDescriptors.CalcRadiusOfGyration(mol_h))
        npr1 = float(rdMolDescriptors.CalcNPR1(mol_h))
        npr2 = float(rdMolDescriptors.CalcNPR2(mol_h))
        asph = float(rdMolDescriptors.CalcAsphericity(mol_h))

        norm_scale = max(rog, 1.0)
        atom_dist_norm = np.clip(dists_to_cent / norm_scale, 0.0, 5.0) / 5.0

        bond_lengths: Dict[Tuple[int, int], float] = {}
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < num_heavy and j < num_heavy:
                bl = float(np.linalg.norm(coords[i] - coords[j]))
                key = (min(i, j), max(i, j))
                bond_lengths[key] = float(np.clip(bl / 2.0, 0.0, 2.0))

        geom_3d = np.array(
            [npr1, npr2, asph, float(np.clip(rog / 10.0, 0.0, 2.0))],
            dtype=np.float32,
        )
        geom_3d = np.nan_to_num(geom_3d, nan=0.0, posinf=0.0, neginf=0.0)
        atom_dist_norm = np.nan_to_num(atom_dist_norm, nan=0.0, posinf=0.0, neginf=0.0)

        return atom_dist_norm.astype(np.float32), bond_lengths, geom_3d
    except Exception:
        return zero_atom_dists, {}, zero_geom
