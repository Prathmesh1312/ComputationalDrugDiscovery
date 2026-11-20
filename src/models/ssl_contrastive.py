"""
ssl_contrastive.py — Self-Supervised Contrastive Learning (GraphCL / InfoNCE / NT-Xent)
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_transformer import MolecularGraphTransformer


class InfoNCELoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) / InfoNCE Loss.
    Maximizes cosine similarity of positive pairs (augmented views of same molecule)
    while penalizing negative pairs (different molecules in the batch).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j: [B, projection_dim] (L2-normalized embeddings of view 1 and view 2)
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        sim_matrix = torch.mm(z, z.t()) / self.temperature

        labels_1 = torch.arange(batch_size, 2 * batch_size, device=z.device)
        labels_2 = torch.arange(0, batch_size, device=z.device)
        labels = torch.cat([labels_1, labels_2], dim=0)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        neg_inf = -1e4 if sim_matrix.dtype == torch.float16 else -1e9
        sim_matrix = sim_matrix.masked_fill(mask, neg_inf)

        return F.cross_entropy(sim_matrix, labels)


class MolecularContrastiveSSL(nn.Module):
    """
    Self-Supervised Pre-Training Model.
    Wraps the Graph-Transformer backbone with a BatchNorm non-linear Projection Head.
    """

    def __init__(
        self,
        backbone: MolecularGraphTransformer,
        proj_dim: int = 64,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(backbone.hidden_dim, backbone.hidden_dim),
            nn.BatchNorm1d(backbone.hidden_dim),
            nn.GELU(),
            nn.Linear(backbone.hidden_dim, proj_dim),
        )
        self.loss_fn = InfoNCELoss(temperature=temperature)

    def forward_repr(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        geom_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the base molecular fingerprint representation from the backbone."""
        return self.backbone(x, edge_index, edge_attr, batch, pe, geom_3d)

    def forward_proj(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        geom_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the L2-normalized projection vector for contrastive learning."""
        h = self.forward_repr(x, edge_index, edge_attr, batch, pe, geom_3d)
        z = self.projector(h)
        return F.normalize(z, dim=-1)

    def compute_ssl_loss(self, view1, view2) -> torch.Tensor:
        """Computes the InfoNCE contrastive loss between batch view 1 and batch view 2."""
        z1 = self.forward_proj(
            view1.x,
            view1.edge_index,
            view1.edge_attr,
            view1.batch,
            getattr(view1, "pe", None),
            getattr(view1, "geom_3d", None),
        )
        z2 = self.forward_proj(
            view2.x,
            view2.edge_index,
            view2.edge_attr,
            view2.batch,
            getattr(view2, "pe", None),
            getattr(view2, "geom_3d", None),
        )
        return self.loss_fn(z1, z2)
