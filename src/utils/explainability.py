"""
explainability.py — Atom-Level Gradient Saliency & Substructure Attribution (XAI)
"""
import io
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Batch


def compute_atom_saliency(
    model: torch.nn.Module,
    smiles: str,
    target_task: str = "herg",
    device: torch.device = torch.device("cpu"),
) -> Tuple[Chem.Mol, np.ndarray, Dict[str, float]]:
    """
    Compute atom-level gradient attribution (Input * Gradient L2 magnitude, normalized to [0, 1])
    for any predicted ADMET endpoint.
    """
    from ..data.dataset import smiles_to_graph_data

    g = smiles_to_graph_data(smiles, compute_pe=True, pe_dim=8, use_3d=True, standardize=True)
    mol = Chem.MolFromSmiles(g.smiles if g is not None and hasattr(g, "smiles") else smiles)
    if g is None or mol is None:
        raise ValueError(f"Invalid SMILES for explainability: {smiles}")

    was_training = model.training
    model.eval()

    batch = Batch.from_data_list([g]).to(device)
    x = batch.x.clone().detach().requires_grad_(True)
    pe = getattr(batch, "pe", None)
    geom_3d = getattr(batch, "geom_3d", None)

    model.zero_grad(set_to_none=True)
    preds = model(x, batch.edge_index, batch.edge_attr, batch.batch, pe, geom_3d)

    logit_key = f"{target_task}_logit"
    scalar_target = preds[logit_key].sum() if logit_key in preds else preds[target_task].sum()
    scalar_target.backward()

    grad = x.grad.detach()
    # Combine gradient sensitivity and input-gradient saliency per heavy atom
    raw_attr = torch.sqrt((x.detach() * grad).pow(2).sum(dim=-1) + 1e-12)
    raw_np = raw_attr.cpu().numpy()

    max_v = float(raw_np.max())
    min_v = float(raw_np.min())
    if max_v - min_v > 1e-8:
        norm_scores = (raw_np - min_v) / (max_v - min_v)
    else:
        norm_scores = np.zeros_like(raw_np)

    pred_summary = {
        "bioavailability": float(preds["bioavailability"].item()),
        "bbb": float(preds["bbb"].item()),
        "herg": float(preds["herg"].item()),
        "qed": float(preds["qed"].item()),
        "sascore": float(preds["sascore"].item()),
    }

    if was_training:
        model.train()

    return mol, np.clip(norm_scores, 0.0, 1.0), pred_summary


def _draw_mol_with_saliency(mol: Chem.Mol, scores: np.ndarray, width: int = 440, height: int = 330) -> np.ndarray:
    """Render an RDKit 2D molecule image with high-saliency atoms highlighted in coral/red."""
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.padding = 0.12

    highlight_atoms = []
    highlight_colors = {}
    highlight_radii = {}

    for idx in range(mol.GetNumAtoms()):
        w = float(scores[idx]) if idx < len(scores) else 0.0
        if w >= 0.35:
            highlight_atoms.append(idx)
            # Interpolate soft amber (w=0.35) -> deep crimson (w=1.0)
            intensity = (w - 0.35) / 0.65
            highlight_colors[idx] = (0.95, max(0.15, 0.72 - 0.57 * intensity), 0.18, 0.55 + 0.35 * intensity)
            highlight_radii[idx] = 0.28 + 0.18 * intensity

    Chem.rdDepictor.Compute2DCoords(mol)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return plt.imread(io.BytesIO(png_bytes))


def render_saliency_figure(
    model: torch.nn.Module,
    molecules: List[Tuple[str, str, str]],
    save_path: str,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Render multi-panel XAI atom attribution figure.
    `molecules`: list of (label, smiles, target_task) tuples.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(molecules)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), dpi=220)
    if n == 1:
        axes = [axes]

    for ax, (title, smi, task) in zip(axes, molecules):
        mol, scores, preds = compute_atom_saliency(model, smi, target_task=task, device=device)
        img = _draw_mol_with_saliency(mol, scores)
        ax.imshow(img)
        ax.axis("off")

        if task == "herg":
            stat_str = f"Pred hERG Hazard: {preds['herg']:.3f}"
        elif task == "bbb":
            stat_str = f"Pred BBB+ Prob: {preds['bbb']:.3f}"
        elif task == "bioavailability":
            stat_str = f"Pred Bioavail: {preds['bioavailability']:.1f}%"
        else:
            stat_str = f"Pred {task.upper()}: {preds[task]:.3f}"

        ax.set_title(f"{title}\n({task.upper()} Saliency | {stat_str})", fontsize=10.5, fontweight="bold", pad=8)

    fig.suptitle(
        "Model Explainability (XAI): Atom-Level Gradient Attribution Maps",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
