"""
dataset.py — Chemoinformatics & 2D+3D Graph Data Loader for Biomolecular Graph-Transformers
"""
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..utils.chemistry import compute_3d_conformer_features, standardize_smiles

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
FORMAL_CHARGES = [-1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# 10 (element) + 6 (degree) + 5 (formal charge) + 4 (hybridization) + 1 (Hs) + 1 (aromatic) + 1 (ring) + 1 (3D dist)
NUM_ATOM_FEATURES = 29
# 5 (bond type) + 1 (conjugated) + 1 (in ring) + 1 (3D bond length)
NUM_BOND_FEATURES = 8
# 4 (Sign-canonicalized Laplacian PE) + 4 (Sign-invariant RWSE)
NUM_PE_FEATURES = 8
# 4 (NPR1, NPR2, Asphericity, RadiusOfGyration / 10.0)
NUM_GEOM_3D_FEATURES = 4

TASK_COLUMNS = ["bioavailability", "bbb", "herg", "qed", "sascore"]


def get_atom_features(atom: Chem.Atom, dist_to_centroid: float = 0.0) -> List[float]:
    """Extract 29-dimensional chemical + 3D spatial atom feature vector."""
    symbol = atom.GetSymbol()
    elem_oh = [1.0 if symbol == el else 0.0 for el in ATOM_TYPES]
    elem_oh.append(0.0 if 1.0 in elem_oh else 1.0)

    deg = atom.GetTotalDegree()
    deg_oh = [1.0 if deg == d else 0.0 for d in [0, 1, 2, 3, 4]]
    deg_oh.append(1.0 if deg >= 5 else 0.0)

    fc = atom.GetFormalCharge()
    fc_oh = [1.0 if fc == c else 0.0 for c in FORMAL_CHARGES]
    fc_oh.append(0.0 if 1.0 in fc_oh else 1.0)

    hyb = atom.GetHybridization()
    hyb_oh = [1.0 if hyb == h else 0.0 for h in HYBRIDIZATIONS]
    hyb_oh.append(0.0 if 1.0 in hyb_oh else 1.0)

    num_hs = [min(float(atom.GetTotalNumHs()), 4.0) / 4.0]
    arom = [1.0 if atom.GetIsAromatic() else 0.0]
    in_ring = [1.0 if atom.IsInRing() else 0.0]
    cent_dist = [float(dist_to_centroid)]

    return elem_oh + deg_oh + fc_oh + hyb_oh + num_hs + arom + in_ring + cent_dist


def get_bond_features(bond: Chem.Bond, bond_len_norm: float = 0.725) -> List[float]:
    """Extract 8-dimensional chemical + 3D bond feature vector."""
    btype = bond.GetBondType()
    bt_oh = [1.0 if btype == bt else 0.0 for bt in BOND_TYPES]
    bt_oh.append(0.0 if 1.0 in bt_oh else 1.0)

    conj = [1.0 if bond.GetIsConjugated() else 0.0]
    in_ring = [1.0 if bond.IsInRing() else 0.0]
    bl = [float(bond_len_norm)]
    return bt_oh + conj + in_ring + bl


def compute_laplacian_pe(edge_index: torch.Tensor, num_nodes: int, k: int = 4) -> torch.Tensor:
    """
    Compute sign-canonicalized Normalized Laplacian Positional Encodings (PE).
    Excludes all zero-eigenvalue trivial modes (safe even for disconnected graphs)
    and enforces deterministic eigenvector sign by making the max-magnitude element positive.
    """
    if num_nodes <= 1 or k <= 0:
        return torch.zeros((num_nodes, max(k, 0)), dtype=torch.float)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0

    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-5), -0.5)
    deg_inv_sqrt[deg == 0] = 0.0

    lap = torch.eye(num_nodes) - (deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0))

    try:
        e_vals, e_vecs = torch.linalg.eigh(lap)
        # Filter out trivial zero-eigenvalue modes (connected components)
        non_trivial_mask = e_vals > 1e-5
        non_trivial_vecs = e_vecs[:, non_trivial_mask]
        if non_trivial_vecs.size(1) == 0:
            return torch.zeros((num_nodes, k), dtype=torch.float)

        pe = non_trivial_vecs[:, :k].clone()
        # Sign canonicalization: flip each eigenvector so its max-abs entry is positive
        for col in range(pe.size(1)):
            vec = pe[:, col]
            max_idx = int(torch.argmax(torch.abs(vec)).item())
            if vec[max_idx] < 0:
                pe[:, col] = -vec

        if pe.size(1) < k:
            padding = torch.zeros((num_nodes, k - pe.size(1)), dtype=torch.float)
            pe = torch.cat([pe, padding], dim=1)
    except Exception:
        pe = torch.zeros((num_nodes, k), dtype=torch.float)

    return pe


def compute_rwse(edge_index: torch.Tensor, num_nodes: int, steps: int = 4) -> torch.Tensor:
    """
    Compute sign-invariant Random-Walk Structural Encodings (RWSE):
    return probabilities diag(P^1), diag(P^2), ..., diag(P^steps), where P = D^{-1}A.
    """
    if num_nodes <= 0 or steps <= 0:
        return torch.zeros((max(num_nodes, 0), max(steps, 0)), dtype=torch.float)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0

    deg = torch.sum(adj, dim=1, keepdim=True)
    p_mat = torch.where(deg > 0, adj / deg.clamp(min=1.0), torch.zeros_like(adj))

    rwse_cols = []
    curr = p_mat
    for _ in range(steps):
        rwse_cols.append(torch.diag(curr))
        curr = torch.mm(curr, p_mat)

    return torch.stack(rwse_cols, dim=1)


def smiles_to_graph_data(
    smiles: str,
    compute_pe: bool = True,
    pe_dim: int = NUM_PE_FEATURES,
    use_3d: bool = True,
    standardize: bool = True,
) -> Optional[Data]:
    """Converts a SMILES string into a PyG Data object with 2D+3D features and LapPE+RWSE."""
    canon_smiles = smiles
    if standardize:
        std = standardize_smiles(smiles, min_heavy=1, max_heavy=120)
        if std is None:
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if mol is None:
                return None
            canon_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        else:
            canon_smiles = std
            mol = Chem.MolFromSmiles(canon_smiles)
    else:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is not None:
            canon_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if use_3d:
        atom_dists, bond_lengths, geom_3d = compute_3d_conformer_features(mol)
    else:
        atom_dists = np.zeros(num_atoms, dtype=np.float32)
        bond_lengths = {}
        geom_3d = np.zeros(NUM_GEOM_3D_FEATURES, dtype=np.float32)

    atom_features = [
        get_atom_features(atom, dist_to_centroid=float(atom_dists[idx]))
        for idx, atom in enumerate(mol.GetAtoms())
    ]
    x = torch.tensor(atom_features, dtype=torch.float)

    src, dst, bond_attrs = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        key = (min(i, j), max(i, j))
        bl_norm = bond_lengths.get(key, 1.45 / 2.0)
        feats = get_bond_features(bond, bond_len_norm=bl_norm)

        # Consecutive bidirectional pair (i->j, j->i)
        src.extend([i, j])
        dst.extend([j, i])
        bond_attrs.extend([feats, feats])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(bond_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = x.size(0)
    data.smiles = canon_smiles
    data.geom_3d = torch.tensor(geom_3d, dtype=torch.float).unsqueeze(0)  # [1, 4]

    if compute_pe:
        if pe_dim == 4:
            data.pe = compute_laplacian_pe(edge_index, data.num_nodes, k=4)
        else:
            lap_k = pe_dim // 2
            rw_k = pe_dim - lap_k
            lap_pe = compute_laplacian_pe(edge_index, data.num_nodes, k=lap_k)
            rwse = compute_rwse(edge_index, data.num_nodes, steps=rw_k)
            data.pe = torch.cat([lap_pe, rwse], dim=1)

    return data


def _featurize_worker(args: Tuple[str, bool, int, bool, bool]) -> Optional[Data]:
    """Top-level picklable worker for multiprocessing graph featurization."""
    sm, compute_pe, pe_dim, use_3d, standardize = args
    return smiles_to_graph_data(
        sm,
        compute_pe=compute_pe,
        pe_dim=pe_dim,
        use_3d=use_3d,
        standardize=standardize,
    )


class MolecularDataset(Dataset):
    """PyTorch Dataset for 2D+3D Molecular Graphs with missing-label masks and disk caching."""

    def __init__(
        self,
        smiles_list: List[str],
        targets_df: Optional[pd.DataFrame] = None,
        labels: Optional[torch.Tensor] = None,
        compute_pe: bool = True,
        pe_dim: int = NUM_PE_FEATURES,
        use_3d: bool = True,
        standardize: bool = True,
        cache_path: Optional[str] = None,
        num_workers: int = 4,
    ):
        self.graphs: List[Data] = []

        if cache_path is not None and os.path.exists(cache_path):
            self.graphs = torch.load(cache_path, weights_only=False)
            return

        worker_args = [
            (sm, compute_pe, pe_dim, use_3d, standardize) for sm in smiles_list
        ]

        if len(worker_args) >= 100 and num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                raw_graphs = list(ex.map(_featurize_worker, worker_args, chunksize=32))
        else:
            raw_graphs = [_featurize_worker(a) for a in worker_args]

        for idx, g in enumerate(raw_graphs):
            if g is None:
                continue
            if targets_df is not None:
                row = targets_df.iloc[idx]
                vals = []
                masks = []
                for col in TASK_COLUMNS:
                    v = row[col] if col in targets_df.columns else np.nan
                    if pd.isna(v):
                        vals.append(0.0)
                        masks.append(False)
                    else:
                        vals.append(float(v))
                        masks.append(True)
                g.y = torch.tensor(vals, dtype=torch.float).unsqueeze(0)         # [1, 5]
                g.y_mask = torch.tensor(masks, dtype=torch.bool).unsqueeze(0)    # [1, 5]
            elif labels is not None:
                g.y = labels[idx].unsqueeze(0)
                g.y_mask = ~torch.isnan(g.y)
            self.graphs.append(g)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.graphs, cache_path)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]
