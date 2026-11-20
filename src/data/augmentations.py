"""
augmentations.py — Undirected-Safe Molecular Graph Augmentations for Contrastive Learning (GraphCL)
"""
import copy
import random
from typing import Dict, Tuple

import torch
from torch_geometric.data import Data


class MoleculeAugmenter:
    """
    Performs chemically consistent stochastic graph augmentations to create dual contrastive views (v_i, v_j).
    Supported augmentations:
      1. Node Feature Masking (Atom feature zeroing)
      2. Undirected-Symmetric Edge Dropping (Drops both i->j and j->i together)
      3. Subgraph Node Dropping (Removes atoms and incident edges cleanly)
      4. Random Laplacian PE Sign Flipping (Invariant to eigenvector sign ambiguity)
    """

    def __init__(
        self,
        node_mask_ratio: float = 0.15,
        edge_drop_ratio: float = 0.15,
        node_drop_ratio: float = 0.15,
    ):
        self.node_mask_ratio = node_mask_ratio
        self.edge_drop_ratio = edge_drop_ratio
        self.node_drop_ratio = node_drop_ratio

    def mask_nodes(self, data: Data) -> Data:
        """Stochastically mask/zero-out atom features."""
        d = copy.deepcopy(data)
        num_nodes = d.x.size(0)
        if num_nodes <= 1:
            return d

        mask_count = max(1, int(num_nodes * self.node_mask_ratio))
        masked_indices = random.sample(range(num_nodes), min(mask_count, num_nodes))
        d.x[masked_indices] = 0.0
        return d

    def drop_edges(self, data: Data) -> Data:
        """Stochastically drop undirected chemical bonds while strictly preserving bidirectionality."""
        d = copy.deepcopy(data)
        num_edges = d.edge_index.size(1)
        if num_edges <= 2:
            return d

        # Map each directed edge to its canonical undirected bond (min(u,v), max(u,v))
        src = d.edge_index[0].tolist()
        dst = d.edge_index[1].tolist()
        undir_keep: Dict[Tuple[int, int], bool] = {}
        keep_prob = 1.0 - self.edge_drop_ratio

        for u, v in zip(src, dst):
            key = (min(u, v), max(u, v))
            if key not in undir_keep:
                undir_keep[key] = random.random() < keep_prob

        # Guarantee at least one undirected bond survives
        if not any(undir_keep.values()):
            first_key = next(iter(undir_keep))
            undir_keep[first_key] = True

        mask = torch.tensor(
            [undir_keep[(min(u, v), max(u, v))] for u, v in zip(src, dst)],
            dtype=torch.bool,
            device=d.edge_index.device,
        )

        d.edge_index = d.edge_index[:, mask]
        if d.edge_attr is not None and d.edge_attr.size(0) == num_edges:
            d.edge_attr = d.edge_attr[mask]
        return d

    def drop_nodes(self, data: Data) -> Data:
        """Stochastically drop atoms and incident bonds, remapping node indices cleanly."""
        num_nodes = data.x.size(0)
        if num_nodes <= 5:
            return self.mask_nodes(data)

        d = copy.deepcopy(data)
        drop_count = max(1, int(num_nodes * self.node_drop_ratio))
        drop_count = min(drop_count, num_nodes - 3)
        drop_set = set(random.sample(range(num_nodes), drop_count))
        keep_nodes = [i for i in range(num_nodes) if i not in drop_set]

        if d.edge_index.numel() > 0:
            src = d.edge_index[0]
            dst = d.edge_index[1]
            keep_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=d.x.device)
            keep_tensor[keep_nodes] = True
            edge_mask = keep_tensor[src] & keep_tensor[dst]

            if edge_mask.sum() == 0:
                return self.mask_nodes(data)

            remap = torch.full((num_nodes,), -1, dtype=torch.long, device=d.x.device)
            remap[keep_nodes] = torch.arange(len(keep_nodes), dtype=torch.long, device=d.x.device)

            d.edge_index = remap[d.edge_index[:, edge_mask]]
            if d.edge_attr is not None and d.edge_attr.size(0) == src.size(0):
                d.edge_attr = d.edge_attr[edge_mask]

        d.x = d.x[keep_nodes]
        d.num_nodes = len(keep_nodes)
        if hasattr(d, "pe") and d.pe is not None and d.pe.size(0) == num_nodes:
            d.pe = d.pe[keep_nodes]

        return d

    @staticmethod
    def random_pe_sign_flip(data: Data) -> Data:
        """Randomly flip signs of Laplacian PE channels (first half) while preserving RWSE."""
        if not hasattr(data, "pe") or data.pe is None or data.pe.numel() == 0:
            return data
        pe_cols = data.pe.size(1)
        lap_cols = pe_cols if pe_cols <= 4 else pe_cols // 2
        signs = torch.where(
            torch.rand(lap_cols, device=data.pe.device) > 0.5,
            torch.ones(lap_cols, device=data.pe.device),
            -torch.ones(lap_cols, device=data.pe.device),
        )
        data.pe[:, :lap_cols] = data.pe[:, :lap_cols] * signs.unsqueeze(0)
        return data

    def _apply_mode(self, data: Data, mode: str) -> Data:
        if mode == "mask_nodes":
            out = self.mask_nodes(data)
        elif mode == "drop_edges":
            out = self.drop_edges(data)
        elif mode == "drop_nodes":
            out = self.drop_nodes(data)
        else:  # "mask_and_drop"
            out = self.drop_edges(self.mask_nodes(data))
        return self.random_pe_sign_flip(out)

    def augment_pair(self, data: Data) -> Tuple[Data, Data]:
        """Generate two distinct augmented views of the same molecular graph."""
        modes = ["mask_nodes", "drop_edges", "drop_nodes", "mask_and_drop"]
        m1, m2 = random.sample(modes, 2)
        return self._apply_mode(data, m1), self._apply_mode(data, m2)
