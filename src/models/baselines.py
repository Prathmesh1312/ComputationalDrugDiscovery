"""
baselines.py — Classical (Random Forest + ECFP4) and Standard GNN (GCN) Baselines for Ablation
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from ..data.dataset import NUM_ATOM_FEATURES, TASK_COLUMNS
from ..utils.chemistry import compute_morgan_fp_array


class StandardGCNBackbone(nn.Module):
    """
    Standard 3-Layer Message-Passing GCN Baseline (No Global Attention, No Edge Projection, No PE).
    Matches MolecularGraphTransformer's output interface (`hidden_dim`) for fair multi-task comparison.
    """

    def __init__(
        self,
        node_dim: int = NUM_ATOM_FEATURES,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        pe: Optional[torch.Tensor] = None,
        geom_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + self.drop(self.act(conv(h, edge_index))))

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        return self.readout_mlp(torch.cat([h_mean, h_max], dim=-1))


class MorganFingerprintRFBaseline:
    """
    Classical Chemoinformatics Baseline:
    Random Forest trained independently per task on 2048-bit Morgan ECFP4 Fingerprints.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models: Dict[str, object] = {}

    @staticmethod
    def featurize(smiles_list: List[str]) -> np.ndarray:
        return np.vstack([compute_morgan_fp_array(sm, n_bits=2048, radius=2) for sm in smiles_list])

    def fit(self, train_df: pd.DataFrame) -> "MorganFingerprintRFBaseline":
        x_all = self.featurize(train_df["smiles"].tolist())

        for task in ["bioavailability", "bbb", "herg"]:
            mask = ~train_df[task].isna().values
            y = train_df.loc[mask, task].values.astype(int)
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(x_all[mask], y)
            self.models[task] = clf

        for task in ["qed", "sascore"]:
            mask = ~train_df[task].isna().values
            y = train_df.loc[mask, task].values.astype(float)
            reg = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            reg.fit(x_all[mask], y)
            self.models[task] = reg

        return self

    def predict(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        x_test = self.featurize(test_df["smiles"].tolist())
        preds: Dict[str, np.ndarray] = {}
        for task in ["bioavailability", "bbb", "herg"]:
            clf = self.models[task]
            preds[task] = clf.predict_proba(x_test)[:, 1]
        for task in ["qed", "sascore"]:
            reg = self.models[task]
            preds[task] = reg.predict(x_test)
        return preds
