"""
generate_visual_assets.py — Generate Crisp Publication-Grade Architecture & Glossary Diagrams (300 DPI)
"""
import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NAVY = "#0F172A"
BLUE = "#1D4ED8"
BLUE_LT = "#EFF6FF"
EMERALD = "#059669"
EMERALD_LT = "#ECFDF5"
AMBER = "#D97706"
AMBER_LT = "#FFFBEB"
RED = "#DC2626"
RED_LT = "#FEF2F2"
SLATE = "#64748B"
SLATE_LT = "#F8FAFC"
PURPLE = "#6D28D9"
PURPLE_LT = "#F5F3FF"
BORDER = "#CBD5E1"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": BORDER,
        "text.color": NAVY,
    }
)


def _box(ax, x, y, w, h, title="", body="", fc="#FFFFFF", ec=NAVY, lw=1.4, title_color=NAVY, title_size=10, body_size=8.5, align="center"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)
    cx = x + w / 2 if align == "center" else x + 0.02
    ha = "center" if align == "center" else "left"
    if title and body:
        ax.text(cx, y + h * 0.72, title, ha=ha, va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        ax.text(cx, y + h * 0.34, body, ha=ha, va="center", fontsize=body_size, color="#334155", linespacing=1.35, zorder=3)
    elif title:
        ax.text(cx, y + h / 2, title, ha=ha, va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
    return patch


def _arrow(ax, p1, p2, color=NAVY, lw=1.6, rad=0.0, ls="-"):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>,head_length=6,head_width=3.8",
        color=color,
        linewidth=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
    )
    ax.add_patch(arr)


def build_arch_01_system_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(
        0.5,
        0.94,
        "End-to-End 4-Stage Computational Drug Discovery Pipeline (SSL-GraphGPS)",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=NAVY,
    )

    # Stage 0
    _box(ax, 0.02, 0.56, 0.21, 0.28, "Stage 0A: Unlabeled SSL", "ZINC250k (N = 5,000)\nZero val/test scaffold overlap\nLargestFragment Desalted", fc=BLUE_LT, ec=BLUE, title_color=BLUE)
    _box(ax, 0.02, 0.14, 0.21, 0.32, "Stage 0B: TDC Clinical ADMET", "5,007 Unique Compounds\n3,137 Bemis-Murcko Scaffolds\nBioavail (636) | BBB (1,921)\nhERG (2,899) | QED & SA (5,007)\nSplit: 4,005 / 500 / 502", fc=SLATE_LT, ec=SLATE, title_color=NAVY)

    # Stage 1
    _box(ax, 0.275, 0.56, 0.21, 0.28, "Stage 1: GraphCL / InfoNCE", "Undirected-Safe Edge Drop\nSubgraph Node Drop + Mask\nLapPE Sign Flip | tau = 0.1\nLoss: 0.8659 -> 0.0402", fc=PURPLE_LT, ec=PURPLE, title_color=PURPLE)

    # Stage 2
    _box(ax, 0.275, 0.14, 0.21, 0.32, "2D+3D Graph Featurizer", "Atoms: 29-dim + 3D Centroid\nBonds: 8-dim + 3D Length\nPE: 4 LapPE + 4 RWSE\nGlobal 3D: NPR1/2, Asph, RoG", fc=SLATE_LT, ec=BLUE, title_color=BLUE)
    _box(ax, 0.53, 0.26, 0.21, 0.52, "Stage 2: Hybrid GraphGPS\n& Masked ADMET Heads", "3x GraphGPS Layers (H=128)\nLocal TransformerConv +\nGlobal MultiheadAttention\n\n5 Bounded MLP Heads:\nBioavail [0-100%], BBB [0-1]\nhERG [0-1], QED, SA [1-10]\nNaN-Masked BCEWithLogits", fc=EMERALD_LT, ec=EMERALD, title_color=EMERALD, body_size=8.2)

    # Stage 3
    _box(ax, 0.785, 0.50, 0.195, 0.34, "Stage 3: Evolutionary\nPareto Lead Optimizer", "23 MedChem SMIRKS +\nBRICS Recombination\nPAINS & Lipinski Filters\nDerringer-Suich + NSGA-II", fc=AMBER_LT, ec=AMBER, title_color=AMBER, body_size=8.2)
    _box(ax, 0.785, 0.14, 0.195, 0.28, "Atom Saliency XAI\n& Benchmarks", "Input x Grad Attribution\nMacro ROC-AUC: 0.8299\nQED R2: 0.7165 | SA: 0.7584", fc=RED_LT, ec=RED, title_color=RED, body_size=8.3)

    # Arrows
    _arrow(ax, (0.235, 0.70), (0.270, 0.70), color=BLUE)
    _arrow(ax, (0.235, 0.30), (0.270, 0.30), color=NAVY)
    _arrow(ax, (0.490, 0.66), (0.525, 0.58), color=PURPLE)
    _arrow(ax, (0.490, 0.32), (0.525, 0.42), color=BLUE)
    _arrow(ax, (0.745, 0.60), (0.780, 0.66), color=EMERALD)
    _arrow(ax, (0.745, 0.40), (0.780, 0.28), color=EMERALD)
    _arrow(ax, (0.882, 0.495), (0.882, 0.425), color=AMBER)

    ax.text(0.508, 0.72, "Pre-trained\nWeights", fontsize=7.5, ha="center", va="bottom", color=PURPLE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFFFFF", ec="none", alpha=0.85))
    ax.text(0.5, 0.04, "Strict Zero-Leakage Protocol: Disjoint Murcko Scaffolds across Train / Val / Test and ZINC250k SSL Pre-Training", ha="center", fontsize=8.5, style="italic", color=SLATE)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "arch_01_system_pipeline.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_arch_02_graphgps_layer() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(0.5, 0.95, "Hybrid Local + Global 2D+3D GraphGPS Layer & Multi-Task Architecture", ha="center", fontsize=13.5, fontweight="bold", color=NAVY)

    # Left Input Embeddings
    _box(ax, 0.02, 0.69, 0.18, 0.17, "Atom Features", "x in R^{N x 29}\nnode_proj -> H=128", fc=BLUE_LT, ec=BLUE, title_color=BLUE)
    _box(ax, 0.02, 0.46, 0.18, 0.17, "LapPE + RWSE", "p in R^{N x 8}\npe_proj -> H=128", fc=PURPLE_LT, ec=PURPLE, title_color=PURPLE)
    _box(ax, 0.02, 0.23, 0.18, 0.17, "Bond Features", "e in R^{E x 8}\nedge_proj -> H=128", fc=SLATE_LT, ec=SLATE, title_color=NAVY)
    _box(ax, 0.02, 0.04, 0.18, 0.14, "Global 3D Shape", "g3d in R^{B x 4}\ngeom_proj -> 32-d", fc=AMBER_LT, ec=AMBER, title_color=AMBER)

    # Sum node + PE
    ax.add_patch(plt.Circle((0.245, 0.62), 0.024, fc="#FFFFFF", ec=NAVY, lw=1.5, zorder=3))
    ax.text(0.245, 0.62, "+", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    _arrow(ax, (0.205, 0.77), (0.233, 0.64), color=BLUE)
    _arrow(ax, (0.205, 0.54), (0.230, 0.60), color=PURPLE)

    # GraphGPS Layer Box container
    outer = FancyBboxPatch((0.29, 0.20), 0.43, 0.68, boxstyle="round,pad=0.015,rounding_size=0.025", fc="#F8FAFC", ec=BLUE, lw=1.6, ls="--", zorder=1)
    ax.add_patch(outer)
    ax.text(0.505, 0.85, "GraphTransformerLayer (x3 Layers, 4 Attention Heads)", ha="center", fontsize=10.2, fontweight="bold", color=BLUE)

    _box(ax, 0.315, 0.62, 0.185, 0.17, "Local Branch", "TransformerConv\nEdge-aware covalent\nattention + Residual\n+ LayerNorm", fc=BLUE_LT, ec=BLUE, title_color=BLUE, body_size=8.0)
    _box(ax, 0.315, 0.36, 0.185, 0.19, "Global Branch", "to_dense_batch\nnn.MultiheadAttention\nAll-atom attention +\nResidual + LayerNorm", fc=PURPLE_LT, ec=PURPLE, title_color=PURPLE, body_size=8.0)

    # Average + FFN
    _box(ax, 0.53, 0.48, 0.165, 0.21, "Fusion & FFN", "h = 0.5*(h_loc + h_glob)\nLinear(128->256)->GELU\nLinear(256->128)\nResidual + LayerNorm", fc=EMERALD_LT, ec=EMERALD, title_color=EMERALD, body_size=7.9)

    _arrow(ax, (0.269, 0.62), (0.310, 0.70), color=NAVY)
    _arrow(ax, (0.269, 0.60), (0.310, 0.47), color=NAVY)
    _arrow(ax, (0.205, 0.31), (0.310, 0.66), color=SLATE, rad=0.15)
    _arrow(ax, (0.505, 0.70), (0.525, 0.61), color=BLUE)
    _arrow(ax, (0.505, 0.46), (0.525, 0.56), color=PURPLE)

    # Readout & Heads
    _box(ax, 0.755, 0.47, 0.225, 0.25, "Global Readout (128-d)", "MeanPool(128) || MaxPool(128)\n|| 3D Shape Proj (32-d)\n-> MLP -> Fingerprint (128-d)", fc=SLATE_LT, ec=NAVY, title_color=NAVY, body_size=8.1)
    _arrow(ax, (0.700, 0.58), (0.750, 0.58), color=EMERALD)
    _arrow(ax, (0.205, 0.11), (0.750, 0.51), color=AMBER, rad=0.38)

    # Downstream Tasks
    _box(ax, 0.755, 0.13, 0.225, 0.26, "5 Multi-Task MLP Heads", "Bioavail [0-100%], BBB [0-1]\nhERG [0-1], QED [0-1]\nSAScore = 1 + 9*sig(.) [1-10]", fc=EMERALD_LT, ec=EMERALD, title_color=EMERALD, body_size=8.1)
    _arrow(ax, (0.867, 0.465), (0.867, 0.395), color=NAVY)

    ax.text(0.44, 0.29, "Eliminates 1-hop GCN myopia & oversmoothing", ha="center", fontsize=8.0, style="italic", color=SLATE)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "arch_02_graphgps_layer.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_glossary_01_smiles_to_graph() -> None:
    fig = plt.figure(figsize=(13.5, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.15, 1.15], wspace=0.18)

    ax0 = fig.add_subplot(gs[0])
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    Chem.rdDepictor.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(420, 340))
    ax0.imshow(img)
    ax0.axis("off")
    ax0.set_title("1. SMILES -> Desalted RDKit Mol\nCC(=O)Oc1ccccc1C(=O)O\n(13 Heavy Atoms, 13 Bonds)", fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title("2. Node & Bond Tensors (2D + 3D)", fontsize=10, fontweight="bold")
    _box(ax1, 0.03, 0.53, 0.94, 0.40, "Atom Feature Vector x_i (29-dim)", "Element one-hot (10) | Degree (6)\nFormal charge (5) | Hybridization (4)\nTotal H-count (1) | Aromatic (1) | In-Ring (1)\n+ ETKDGv3 3D Centroid Distance (1)", fc=BLUE_LT, ec=BLUE, title_color=BLUE, body_size=8.4)
    _box(ax1, 0.03, 0.06, 0.94, 0.40, "Bond Feature Vector e_ij (8-dim)", "Bond Type one-hot (5: 1/2/3/Ar/Other)\nConjugated (1) | In-Ring (1)\n+ ETKDGv3 3D Bond Length (1)\nStored symmetrically: (i->j) & (j->i)", fc=EMERALD_LT, ec=EMERALD, title_color=EMERALD, body_size=8.4)

    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title("3. Topology & 3D Shape Descriptors", fontsize=10, fontweight="bold")
    _box(ax2, 0.03, 0.53, 0.94, 0.40, "Positional + Structural Encodings (8-d)", "LapPE (4-d): Eigenmodes of L = I - D^-0.5 A D^-0.5\n  Sign-canonicalized (max-abs > 0)\nRWSE (4-d): Random-walk return probs\n  diag(P^1 .. P^4) -> Ring/Cycle Awareness", fc=PURPLE_LT, ec=PURPLE, title_color=PURPLE, body_size=8.2)
    _box(ax2, 0.03, 0.06, 0.94, 0.40, "Global 3D Conformer Vector (4-dim)", "Generated via RDKit ETKDGv3:\n[NPR1, NPR2] (rod/disc/sphere shape)\nAsphericity (molecular elongation)\nRadius of Gyration (spatial spread)", fc=AMBER_LT, ec=AMBER, title_color=AMBER, body_size=8.2)

    fig.suptitle("Glossary Visual 1: Transforming Raw SMILES into 2D+3D GraphGPS Inputs", fontsize=12.5, fontweight="bold", y=0.99)
    fig.savefig(os.path.join(FIG_DIR, "glossary_01_smiles_to_2d3d_graph.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_glossary_02_gcn_vs_graphgps_and_ssl() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Left: GCN vs GraphGPS
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Local GCN Myopia vs. Hybrid GraphGPS Global Attention", fontsize=11, fontweight="bold")

    # Draw hexagon molecule top (GCN) and bottom (GraphGPS)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    for row_idx, (cy, title, is_global) in enumerate([(0.72, "Standard GCN: Only 1-Hop Covalent Neighbors", False), (0.27, "Hybrid GraphGPS: Covalent + All-Atom Attention", True)]):
        ax.text(0.5, cy + 0.19, title, ha="center", fontsize=9.3, fontweight="bold", color=SLATE if not is_global else BLUE)
        pts = [(0.5 + 0.12 * np.cos(a), cy + 0.10 * np.sin(a)) for a in angles]
        pts.append((0.72, cy + 0.05))
        pts.append((0.28, cy - 0.05))
        # Covalent bonds
        cov_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (3, 7)]
        if is_global:
            for j in range(1, len(pts)):
                ax.plot([pts[0][0], pts[j][0]], [pts[0][1], pts[j][1]], color=PURPLE, lw=1.1, ls="--", alpha=0.65, zorder=1)
        for u, v in cov_edges:
            col = BLUE if (0 in (u, v) and not is_global) else "#475569"
            lw = 2.4 if (0 in (u, v) and not is_global) else 1.6
            ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color=col, lw=lw, zorder=2)
        for idx, (px, py) in enumerate(pts):
            fc = RED if idx == 0 else (BLUE_LT if is_global or idx in (1, 5, 6) else "#E2E8F0")
            ec = RED if idx == 0 else (BLUE if is_global else SLATE)
            ax.scatter([px], [py], s=140, color=fc, edgecolors=ec, linewidths=1.5, zorder=3)
            if idx == 0:
                ax.text(px, py, "q", ha="center", va="center", fontsize=8, fontweight="bold", color="#FFFFFF", zorder=4)

    # Right: Contrastive InfoNCE
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("GraphCL Self-Supervised Contrastive Learning (InfoNCE)", fontsize=11, fontweight="bold")

    _box(ax, 0.05, 0.72, 0.25, 0.18, "Unlabeled Mol G", "ZINC250k Graph", fc=SLATE_LT, ec=NAVY)
    _box(ax, 0.42, 0.80, 0.25, 0.14, "View 1 (v_i)", "Undirected Edge Drop", fc=BLUE_LT, ec=BLUE, title_color=BLUE, body_size=7.8)
    _box(ax, 0.42, 0.60, 0.25, 0.14, "View 2 (v_j)", "Subgraph Node Drop", fc=BLUE_LT, ec=BLUE, title_color=BLUE, body_size=7.8)
    _box(ax, 0.42, 0.26, 0.25, 0.15, "Other Mol (v_k)", "In-Batch Negative", fc=RED_LT, ec=RED, title_color=RED, body_size=7.8)
    _box(ax, 0.74, 0.58, 0.23, 0.30, "Positive Pull", "Maximize cos(z_i, z_j)\nSame chemical core", fc=EMERALD_LT, ec=EMERALD, title_color=EMERALD, body_size=8.0)
    _box(ax, 0.74, 0.20, 0.23, 0.25, "Negative Push", "Minimize cos(z_i, z_k)\nInfoNCE (tau=0.1)", fc=RED_LT, ec=RED, title_color=RED, body_size=8.0)

    _arrow(ax, (0.305, 0.83), (0.415, 0.86), color=BLUE)
    _arrow(ax, (0.305, 0.76), (0.415, 0.68), color=BLUE)
    _arrow(ax, (0.675, 0.86), (0.735, 0.76), color=EMERALD)
    _arrow(ax, (0.675, 0.67), (0.735, 0.69), color=EMERALD)
    _arrow(ax, (0.675, 0.34), (0.735, 0.33), color=RED)

    ax.text(0.5, 0.08, "No clinical labels needed — InfoNCE loss dropped 0.8659 -> 0.0402", ha="center", fontsize=8.8, style="italic", color=SLATE)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "glossary_02_gcn_vs_graphgps_and_ssl.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_glossary_03_scaffold_split_and_metrics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Left: Random Split vs Bemis-Murcko Scaffold Split
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Why Bemis-Murcko Scaffold Split Beats Random Split", fontsize=11, fontweight="bold")

    _box(ax, 0.04, 0.54, 0.92, 0.38, "Random Split (DATA LEAKAGE)", "Train:  [Scaffold A-1]  [Scaffold B-1]  [Scaffold C-1]\nTest:   [Scaffold A-2]  [Scaffold B-2]  [Scaffold C-2]\nProblem: Same core rings appear in both Train & Test -> Over-optimistic AUC", fc=RED_LT, ec=RED, title_color=RED, body_size=8.2)
    _box(ax, 0.04, 0.08, 0.92, 0.38, "Bemis-Murcko Scaffold Split (ZERO LEAKAGE - Used Here)", "Train (4,005 mols): 2,135 scaffolds (e.g., Scaffolds A, B, C...)\nVal   (500 mols):   500 UNSEEN scaffolds\nTest  (502 mols):   502 COMPLETELY UNSEEN scaffolds -> True Generalization", fc=EMERALD_LT, ec=EMERALD, title_color=EMERALD, body_size=8.2)

    # Right: Geometric vs Arithmetic Mean Desirability
    ax = axes[1]
    safety = np.linspace(0.01, 1.0, 200)
    bioavail = 1.0  # Best absorption
    arith = 0.5 * (bioavail + safety)
    geom = np.sqrt(bioavail * safety)

    ax.plot(safety, arith, "--", color=RED, lw=2.2, label="Arithmetic Mean (masks fatal toxicity at 0.50!)")
    ax.plot(safety, geom, "-", color=BLUE, lw=2.6, label="Derringer-Suich Geometric Mean (vetoes toxic leads)")
    ax.scatter([0.02], [0.51], color=RED, s=70, zorder=5)
    ax.scatter([0.02], [np.sqrt(0.02)], color=BLUE, s=70, zorder=5)
    ax.annotate("Toxic lead (Safety=0.02)\nArith=0.51 vs Geom=0.14", xy=(0.03, 0.16), xytext=(0.24, 0.20),
                arrowprops=dict(arrowstyle="->", color=NAVY), fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Cardiotoxicity Safety Score d_safety = (1 - hERG)")
    ax.set_ylabel("Composite Score (with Bioavailability = 1.0)")
    ax.set_title("Derringer-Suich Geometric vs. Arithmetic Mean")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "glossary_03_scaffold_split_and_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_glossary_04_masked_loss_and_pareto() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # Left: Sparse Masked Multi-Task Matrix
    ax = axes[0]
    matrix = np.array([
        [1, 0, 2, 1, 1],
        [2, 1, 0, 1, 1],
        [2, 2, 1, 1, 1],
        [1, 1, 2, 1, 1],
        [2, 0, 1, 1, 1],
    ])
    cmap_colors = {0: "#FEE2E2", 1: "#DBEAFE", 2: "#F1F5F9"}
    labels_txt = {0: "Obs: 0", 1: "Obs: 1 / Cont", 2: "NaN (Masked)"}
    cols = ["Bioavail", "BBB", "hERG", "QED", "SAScore"]
    rows = ["Mol 1", "Mol 2", "Mol 3", "Mol 4", "Mol 5"]

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    for r in range(5):
        for c in range(5):
            val = matrix[r, c]
            rect = mpatches.Rectangle((c - 0.46, 4 - r - 0.44), 0.92, 0.88, facecolor=cmap_colors[val], edgecolor=BORDER, lw=1.2)
            ax.add_patch(rect)
            txt = "NaN\n(Skip)" if val == 2 else ("Cont.\nLabel" if c >= 3 else f"Label={val}")
            tcol = SLATE if val == 2 else (BLUE if val == 1 else RED)
            ax.text(c, 4 - r, txt, ha="center", va="center", fontsize=8.0, fontweight="bold" if val != 2 else "normal", color=tcol)

    ax.set_xticks(range(5))
    ax.set_xticklabels(cols, fontweight="bold")
    ax.set_yticks(range(5))
    ax.set_yticklabels(reversed(rows), fontweight="bold")
    ax.set_title("NaN-Masked Multi-Task Learning Matrix", fontsize=11, fontweight="bold")

    # Right: Pareto Dominance Schematic
    ax = axes[1]
    pts_f1 = np.array([[0.04, 98], [0.08, 99], [0.16, 100]])
    pts_f2 = np.array([[0.12, 92], [0.22, 96], [0.32, 98]])
    pts_f3 = np.array([[0.28, 78], [0.40, 85], [0.55, 88]])

    ax.plot(pts_f1[:, 0], pts_f1[:, 1], "--", color=BLUE, lw=1.8, alpha=0.8)
    ax.scatter(pts_f1[:, 0], pts_f1[:, 1], s=130, color=BLUE, edgecolors="k", zorder=5, label="Pareto Front 1 (Non-Dominated)")
    ax.scatter(pts_f2[:, 0], pts_f2[:, 1], s=80, color=AMBER, edgecolors="k", zorder=4, label="Pareto Front 2")
    ax.scatter(pts_f3[:, 0], pts_f3[:, 1], s=60, color=SLATE, alpha=0.7, zorder=3, label="Dominated Candidates (Front 3)")

    ax.annotate("Ideal Region\n(High Bioavail, Low hERG)", xy=(0.04, 99.5), xytext=(0.24, 91),
                arrowprops=dict(arrowstyle="->", color=EMERALD, lw=1.6), fontsize=8.8, fontweight="bold", color=EMERALD)
    ax.set_xlabel("hERG Toxicity Risk (Minimize)")
    ax.set_ylabel("Oral Bioavailability % (Maximize)")
    ax.set_title("NSGA-II Non-Dominated Pareto Frontier", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 0.65)
    ax.set_ylim(72, 103)
    ax.legend(frameon=False, fontsize=8.3, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "glossary_04_masked_loss_and_pareto.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    build_arch_01_system_pipeline()
    build_arch_02_graphgps_layer()
    build_glossary_01_smiles_to_graph()
    build_glossary_02_gcn_vs_graphgps_and_ssl()
    build_glossary_03_scaffold_split_and_metrics()
    build_glossary_04_masked_loss_and_pareto()
    print("Generated 6 architecture & glossary figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
