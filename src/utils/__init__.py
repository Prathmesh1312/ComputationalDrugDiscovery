"""
src/utils package — Reproducibility, Chemoinformatics, Splitting, Metrics, XAI, and Visualization
"""
from .chemistry import (
    compute_3d_conformer_features,
    compute_exact_qed_sa,
    compute_lipinski_and_pains,
    compute_morgan_fp_array,
    compute_tanimoto_similarity,
    standardize_smiles,
    suppress_rdkit_warnings,
)
from .explainability import compute_atom_saliency, render_saliency_figure
from .metrics import evaluate_multitask_predictions, format_metrics_markdown
from .seed import get_device, set_seed
from .splitting import bemis_murcko_scaffold_split, get_murcko_scaffold

__all__ = [
    "set_seed",
    "get_device",
    "suppress_rdkit_warnings",
    "standardize_smiles",
    "compute_exact_qed_sa",
    "compute_lipinski_and_pains",
    "compute_morgan_fp_array",
    "compute_tanimoto_similarity",
    "compute_3d_conformer_features",
    "get_murcko_scaffold",
    "bemis_murcko_scaffold_split",
    "evaluate_multitask_predictions",
    "format_metrics_markdown",
    "compute_atom_saliency",
    "render_saliency_figure",
]
