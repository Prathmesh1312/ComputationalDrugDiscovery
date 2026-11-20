"""
multi_objective_heads.py — Multi-Objective ADMET & Safety Profiling Model + Masked Loss
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_task_head(hidden_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """Task-specific 2-layer MLP adapter head."""
    mid = max(hidden_dim // 2, 16)
    return nn.Sequential(
        nn.Linear(hidden_dim, mid),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mid, 1),
    )


class MultiObjectiveADMETModel(nn.Module):
    """
    Fine-Tuned Graph-Transformer (or GCN Baseline) for Multi-Task ADMET Profiling.
    Predicts:
      1. Human Oral Bioavailability P(F >= 20%) * 100% [0..100%]
      2. Blood-Brain Barrier (BBB) Permeability [0..1]
      3. hERG Cardiotoxicity Hazard [0..1]
      4. Drug-Likeness (QED Score) [0..1]
      5. Synthetic Accessibility (SAScore) [1..10]
    """

    def __init__(self, backbone: nn.Module, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone

        self.shared_mlp = nn.Sequential(
            nn.Linear(backbone.hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.head_bioavailability = _make_task_head(hidden_dim, dropout)
        self.head_bbb = _make_task_head(hidden_dim, dropout)
        self.head_herg = _make_task_head(hidden_dim, dropout)
        self.head_qed = _make_task_head(hidden_dim, dropout)
        self.head_sascore = _make_task_head(hidden_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
        pe: Optional[torch.Tensor] = None,
        geom_3d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mol_rep = self.backbone(x, edge_index, edge_attr, batch, pe, geom_3d)
        feat = self.shared_mlp(mol_rep)

        bio_logit = self.head_bioavailability(feat).squeeze(-1)
        bbb_logit = self.head_bbb(feat).squeeze(-1)
        herg_logit = self.head_herg(feat).squeeze(-1)
        qed_raw = self.head_qed(feat).squeeze(-1)
        sa_raw = self.head_sascore(feat).squeeze(-1)

        bioavailability = torch.sigmoid(bio_logit) * 100.0
        bbb_prob = torch.sigmoid(bbb_logit)
        herg_prob = torch.sigmoid(herg_logit)
        qed = torch.sigmoid(qed_raw)
        # Strictly bounded in [1.0, 10.0]
        sascore = 1.0 + 9.0 * torch.sigmoid(sa_raw)

        return {
            "bioavailability_logit": bio_logit,
            "bbb_logit": bbb_logit,
            "herg_logit": herg_logit,
            "bioavailability": bioavailability,
            "bbb": bbb_prob,
            "herg": herg_prob,
            "qed": qed,
            "sascore": sascore,
            "fingerprint": mol_rep,
        }


def compute_masked_multitask_loss(
    preds: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    masks: torch.Tensor,
    pos_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Numerically stable, balanced multi-task loss with NaN-safe label masking.
    `targets`: [B, 5] ordered as [bioavailability (0/1), bbb (0/1), herg (0/1), qed, sascore]
    `masks`: [B, 5] boolean mask (True = observed experimental label)
    """
    device = targets.device
    total_loss = preds["bioavailability_logit"].sum() * 0.0
    breakdown: Dict[str, float] = {}

    cls_specs = [
        ("bioavailability", 0, "bioavailability_logit", 1.0),
        ("bbb", 1, "bbb_logit", 1.0),
        ("herg", 2, "herg_logit", 1.0),
    ]
    for task_name, col_idx, logit_key, weight in cls_specs:
        m = masks[:, col_idx]
        if m.any():
            pw = pos_weights.get(task_name) if pos_weights is not None else None
            loss_t = F.binary_cross_entropy_with_logits(
                preds[logit_key][m], targets[:, col_idx][m], pos_weight=pw
            )
            total_loss = total_loss + weight * loss_t
            breakdown[task_name] = float(loss_t.detach().item())
        else:
            breakdown[task_name] = 0.0

    # QED regression [0, 1]
    m_qed = masks[:, 3]
    if m_qed.any():
        loss_qed = F.smooth_l1_loss(preds["qed"][m_qed], targets[:, 3][m_qed], beta=0.1)
        total_loss = total_loss + 2.0 * loss_qed
        breakdown["qed"] = float(loss_qed.detach().item())
    else:
        breakdown["qed"] = 0.0

    # SAScore normalized to [0, 1] via (score - 1.0) / 9.0 so loss scale matches QED & BCE
    m_sa = masks[:, 4]
    if m_sa.any():
        pred_sa_norm = (preds["sascore"][m_sa] - 1.0) / 9.0
        true_sa_norm = (targets[:, 4][m_sa] - 1.0) / 9.0
        loss_sa = F.smooth_l1_loss(pred_sa_norm, true_sa_norm, beta=0.1)
        total_loss = total_loss + 2.0 * loss_sa
        breakdown["sascore"] = float(loss_sa.detach().item())
    else:
        breakdown["sascore"] = 0.0

    return total_loss, breakdown
