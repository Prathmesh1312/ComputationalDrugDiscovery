"""
graph_transformer.py — Hybrid Local + Global GraphGPS Transformer with 2D+3D Molecular Geometry
"""
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_max_pool, global_mean_pool
from torch_geometric.utils import to_dense_batch

from ..data.dataset import (
    NUM_ATOM_FEATURES,
    NUM_BOND_FEATURES,
    NUM_GEOM_3D_FEATURES,
    NUM_PE_FEATURES,
)


class GraphTransformerLayer(nn.Module):
    """
    Hybrid Local + Global GraphGPS Layer:
      1. Local Message Passing: Edge-aware multi-head attention (`TransformerConv`) over covalent bonds
      2. Global All-Atom Self-Attention: `nn.MultiheadAttention` across every atom within each molecule
      3. Feed-Forward Network (FFN) with GELU, Residual Connections & LayerNorm
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim

        # Shortcut projection if in_dim != out_dim
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # 1. Local edge-aware attention
        self.conv = TransformerConv(
            in_channels=in_dim,
            out_channels=out_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            beta=True,
        )
        self.norm_local = nn.LayerNorm(out_dim)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Global all-atom Transformer self-attention
        self.global_attn = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_global = nn.LayerNorm(out_dim)
        self.dropout_global = nn.Dropout(dropout)

        # 3. Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = self.shortcut(x)

        # Local covalent bond message passing
        h_loc = self.conv(x, edge_index, edge_attr)
        h_loc = self.norm_local(residual + self.dropout1(h_loc))

        # Global all-atom self-attention per molecule in batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h_dense, mask = to_dense_batch(residual, batch)
        h_glob_dense, _ = self.global_attn(
            h_dense, h_dense, h_dense, key_padding_mask=~mask, need_weights=False
        )
        h_glob = self.norm_global(residual + self.dropout_global(h_glob_dense[mask]))

        # Hybrid fusion + FFN
        h = 0.5 * (h_loc + h_glob)
        h_ffn = self.ffn(h)
        out = self.norm2(h + self.dropout2(h_ffn))
        return out


class MolecularGraphTransformer(nn.Module):
    """
    Biomolecular 2D+3D Hybrid GraphGPS Transformer Backbone.
    Maps variable-size molecular graphs + Laplacian/RWSE PE + 3D conformer shape descriptors
    to fixed-dimensional molecular fingerprints.
    """

    def __init__(
        self,
        node_dim: int = NUM_ATOM_FEATURES,
        edge_dim: int = NUM_BOND_FEATURES,
        pe_dim: int = NUM_PE_FEATURES,
        geom_3d_dim: int = NUM_GEOM_3D_FEATURES,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim
        self.geom_3d_dim = geom_3d_dim

        # 1. Input Projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim) if pe_dim > 0 else None
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.geom_proj = (
            nn.Sequential(nn.Linear(geom_3d_dim, hidden_dim // 4), nn.GELU())
            if geom_3d_dim > 0
            else None
        )

        # 2. Hybrid GraphGPS Layers Stack
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Readout Projection (Mean + Max + optional 3D global geometry)
        readout_in = hidden_dim * 2 + (hidden_dim // 4 if geom_3d_dim > 0 else 0)
        self.readout_mlp = nn.Sequential(
            nn.Linear(readout_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        geom_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.node_proj(x)

        if self.pe_proj is not None and pe is not None and pe.numel() > 0:
            if pe.size(1) < self.pe_dim:
                pad = torch.zeros((pe.size(0), self.pe_dim - pe.size(1)), device=pe.device, dtype=pe.dtype)
                pe = torch.cat([pe, pad], dim=1)
            elif pe.size(1) > self.pe_dim:
                pe = pe[:, : self.pe_dim]
            h = h + self.pe_proj(pe)

        if edge_attr is not None and edge_attr.numel() > 0:
            e = self.edge_proj(edge_attr)
        else:
            e = torch.empty((0, self.hidden_dim), device=x.device)

        for layer in self.layers:
            h = layer(h, edge_index, e, batch=batch)

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        parts = [h_mean, h_max]

        if self.geom_proj is not None:
            num_graphs = h_mean.size(0)
            if geom_3d is not None and geom_3d.numel() > 0:
                if geom_3d.dim() == 1:
                    geom_3d = geom_3d.unsqueeze(0)
                g_emb = self.geom_proj(geom_3d)
            else:
                g_emb = torch.zeros((num_graphs, self.hidden_dim // 4), device=x.device, dtype=h_mean.dtype)
            parts.append(g_emb)

        h_fused = torch.cat(parts, dim=-1)
        return self.readout_mlp(h_fused)
