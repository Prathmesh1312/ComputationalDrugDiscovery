"""
visualization.py — Publication-Grade Figures for EDA, SSL, Benchmarks, and Pareto Optimization
"""
import io
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

PALETTE = {
    "primary": "#1D4ED8",
    "danger": "#DC2626",
    "success": "#059669",
    "warning": "#D97706",
    "muted": "#64748B",
    "purple": "#7C3AED",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_dataset_eda(admet_df: pd.DataFrame, unlabeled_count: int, save_path: str) -> None:
    """Plot 01_dataset_eda.png: endpoint label counts, class balance, scaffold split, and QED/SA distributions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=220)

    # Panel A: Clinical endpoint positive vs negative counts
    ax = axes[0, 0]
    tasks = ["bioavailability", "bbb", "herg"]
    labels = ["Oral Bioavailability", "BBB Permeability", "hERG Cardiotoxicity"]
    pos_counts = [int((admet_df[t].dropna() >= 0.5).sum()) for t in tasks]
    neg_counts = [int((admet_df[t].dropna() < 0.5).sum()) for t in tasks]
    x = np.arange(len(tasks))
    ax.bar(x - 0.18, pos_counts, width=0.35, label="Positive (1)", color=PALETTE["primary"])
    ax.bar(x + 0.18, neg_counts, width=0.35, label="Negative (0)", color=PALETTE["danger"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Experimental Labels")
    ax.set_title(f"Clinical Endpoint Coverage (Total ADMET N={len(admet_df)}, SSL N={unlabeled_count})")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: Bemis-Murcko Scaffold Split Breakdown
    ax = axes[0, 1]
    split_order = ["train", "val", "test"]
    split_counts = [int((admet_df["split"] == s).sum()) for s in split_order]
    scaf_counts = [int(admet_df.loc[admet_df["split"] == s, "scaffold"].nunique()) for s in split_order]
    x2 = np.arange(len(split_order))
    bars = ax.bar(x2, split_counts, width=0.5, color=[PALETTE["primary"], PALETTE["warning"], PALETTE["success"]])
    for i, (cnt, sc) in enumerate(zip(split_counts, scaf_counts)):
        ax.text(i, cnt + max(split_counts) * 0.02, f"{cnt} mols\n({sc} scaf.)", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x2)
    ax.set_xticklabels(["Train (80%)", "Val (10%)", "Test (10%)"])
    ax.set_ylabel("Compounds")
    ax.set_ylim(0, max(split_counts) * 1.2)
    ax.set_title("Bemis-Murcko Disjoint Scaffold Split")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel C: QED Distribution
    ax = axes[1, 0]
    for sp, col in [("train", PALETTE["primary"]), ("val", PALETTE["warning"]), ("test", PALETTE["success"])]:
        vals = admet_df.loc[admet_df["split"] == sp, "qed"].dropna()
        ax.hist(vals, bins=25, alpha=0.45, density=True, label=sp.upper(), color=col)
    ax.set_xlabel("QED Drug-Likeness Score [0, 1]")
    ax.set_ylabel("Density")
    ax.set_title("QED Distribution Across Scaffold Splits")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel D: SAScore Distribution
    ax = axes[1, 1]
    for sp, col in [("train", PALETTE["primary"]), ("val", PALETTE["warning"]), ("test", PALETTE["success"])]:
        vals = admet_df.loc[admet_df["split"] == sp, "sascore"].dropna()
        ax.hist(vals, bins=25, alpha=0.45, density=True, label=sp.upper(), color=col)
    ax.set_xlabel("Synthetic Accessibility Score (SAScore) [1, 10]")
    ax.set_ylabel("Density")
    ax.set_title("SAScore Distribution Across Scaffold Splits")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ssl_pretraining(
    ssl_history: Dict[str, List[float]],
    embeddings_2d: np.ndarray,
    qed_vals: np.ndarray,
    herg_vals: np.ndarray,
    save_path: str,
) -> None:
    """Plot 02_ssl_pretraining.png: InfoNCE training curve + t-SNE embedding space colored by QED and hERG."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=220)

    # Panel 1: InfoNCE Loss
    ax = axes[0]
    epochs = ssl_history.get("epoch", list(range(1, len(ssl_history["loss"]) + 1)))
    ax.plot(epochs, ssl_history["loss"], marker="o", linewidth=2.2, markersize=4, color=PALETTE["primary"])
    ax.set_xlabel("Pre-Training Epoch")
    ax.set_ylabel("InfoNCE (NT-Xent) Contrastive Loss")
    ax.set_title("Self-Supervised GraphCL Convergence")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel 2: t-SNE colored by QED
    ax = axes[1]
    sc = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=qed_vals, cmap="viridis", s=16, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="RDKit QED")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title("Zero-Shot SSL Latent Space (by QED)")

    # Panel 3: t-SNE colored by hERG Hazard
    ax = axes[2]
    valid_herg = ~np.isnan(herg_vals)
    pts = embeddings_2d[valid_herg]
    lbls = herg_vals[valid_herg]
    neg = lbls < 0.5
    pos = lbls >= 0.5
    ax.scatter(pts[neg, 0], pts[neg, 1], c=PALETTE["success"], s=16, alpha=0.65, label="Safe (hERG=0)")
    ax.scatter(pts[pos, 0], pts[pos, 1], c=PALETTE["danger"], s=16, alpha=0.65, label="Toxic (hERG=1)")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title("Zero-Shot SSL Latent Space (by hERG)")
    ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(histories_dict: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """Plot 03_training_curves.png: Train loss and validation Macro ROC-AUC across ablation models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=220)
    colors = {
        "SSL-GraphGPS (Ours)": PALETTE["primary"],
        "GraphGPS-Scratch": PALETTE["warning"],
        "Standard-GCN": PALETTE["muted"],
    }

    for name, hist in histories_dict.items():
        col = colors.get(name, PALETTE["purple"])
        epochs = hist["epoch"]
        axes[0].plot(epochs, hist["train_loss"], linewidth=2.0, label=name, color=col)
        axes[1].plot(epochs, hist["val_macro_roc_auc"], linewidth=2.0, label=name, color=col)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked Multi-Task Loss")
    axes[0].set_title("Training Loss Convergence")
    axes[0].legend(frameon=False)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro ROC-AUC")
    axes[1].set_title("Held-Out Scaffold Validation ROC-AUC")
    axes[1].legend(frameon=False)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roc_comparison(roc_data_dict: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """Plot 04_roc_comparison.png: ROC curves for Bioavailability, BBB, and hERG on unseen scaffold test set."""
    from sklearn.metrics import auc, roc_curve

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=220)
    tasks = [("bioavailability", "Oral Bioavailability (F >= 20%)"), ("bbb", "BBB Permeability"), ("herg", "hERG Cardiotoxicity")]
    colors = {
        "SSL-GraphGPS (Ours)": PALETTE["primary"],
        "GraphGPS-Scratch": PALETTE["warning"],
        "Standard-GCN": PALETTE["muted"],
        "RF-ECFP4": PALETTE["success"],
    }

    for ax, (task_key, title) in zip(axes, tasks):
        for model_name, data in roc_data_dict.items():
            y_t = data[task_key]["y_true"]
            y_p = data[task_key]["y_prob"]
            fpr, tpr, _ = roc_curve(y_t, y_p)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                linewidth=2.0,
                color=colors.get(model_name, PALETTE["purple"]),
                label=f"{model_name} (AUC={roc_auc:.3f})",
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("Held-Out Unseen-Scaffold Test Set: Multi-Task ROC Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_regression_parity(
    y_true_qed: np.ndarray,
    y_pred_qed: np.ndarray,
    y_true_sa: np.ndarray,
    y_pred_sa: np.ndarray,
    save_path: str,
) -> None:
    """Plot 05_regression_parity.png: Predicted vs Actual parity plots for QED and SAScore."""
    from sklearn.metrics import mean_absolute_error, r2_score

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=220)

    for ax, yt, yp, label, bounds, col in [
        (axes[0], y_true_qed, y_pred_qed, "QED Drug-Likeness", (0.0, 1.0), PALETTE["primary"]),
        (axes[1], y_true_sa, y_pred_sa, "Synthetic Accessibility (SAScore)", (1.0, 8.0), PALETTE["success"]),
    ]:
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        ax.scatter(yt, yp, s=18, alpha=0.65, color=col, edgecolors="none")
        ax.plot(bounds, bounds, "k--", linewidth=1.3, label="Ideal y = x")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_xlabel(f"True {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label} Test Parity ($R^2={r2:.3f}$, MAE={mae:.3f})")
        ax.legend(frameon=False)
        ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_optimization(
    candidates: List[Dict[str, Any]],
    pareto_front1: List[Dict[str, Any]],
    seed_smiles: str,
    save_path: str,
) -> None:
    """Plot 06_pareto_optimization.png: Trade-off Pareto scatter + RDKit 2D grid of Seed & Top Pareto analogues."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=220, gridspec_kw={"width_ratios": [1.15, 1.0]})

    pareto_smiles = {p["smiles"] for p in pareto_front1}
    seed_canon = Chem.CanonSmiles(seed_smiles)

    ax = axes[0]
    other = [c for c in candidates if c["smiles"] not in pareto_smiles and c["smiles"] != seed_canon]
    p_only = [c for c in pareto_front1 if c["smiles"] != seed_canon]
    seed_item = [c for c in candidates if c["smiles"] == seed_canon]

    if other:
        ax.scatter(
            [c["herg"] for c in other],
            [c["bioavailability"] for c in other],
            c=PALETTE["muted"],
            s=45,
            alpha=0.6,
            label="Generated Analogues",
        )
    if p_only:
        p_sorted = sorted(p_only, key=lambda c: c["herg"])
        ax.scatter(
            [c["herg"] for c in p_sorted],
            [c["bioavailability"] for c in p_sorted],
            c=PALETTE["primary"],
            s=110,
            edgecolors="black",
            linewidth=0.8,
            zorder=4,
            label="Pareto-Optimal (Front 1)",
        )
        ax.plot(
            [c["herg"] for c in p_sorted],
            [c["bioavailability"] for c in p_sorted],
            linestyle="--",
            color=PALETTE["primary"],
            alpha=0.7,
            linewidth=1.5,
        )
    if seed_item:
        s0 = seed_item[0]
        ax.scatter(
            [s0["herg"]],
            [s0["bioavailability"]],
            c=PALETTE["danger"],
            marker="*",
            s=260,
            edgecolors="black",
            linewidth=0.8,
            zorder=5,
            label="Seed Lead",
        )

    ax.set_xlabel("Predicted hERG Cardiotoxicity Hazard (Lower is Safer)")
    ax.set_ylabel("Predicted Bioavailability Score % (Higher is Better)")
    ax.set_title("Multi-Objective Evolutionary Pareto Frontier")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Right panel: RDKit 2D structure grid of Seed + Top 3 Pareto-optimal analogues
    ax2 = axes[1]
    top_pareto = sorted(p_only, key=lambda c: c["desirability"], reverse=True)[:3]
    display_items = (seed_item[:1] + top_pareto) if seed_item else top_pareto[:4]

    mols = []
    legends = []
    for idx, item in enumerate(display_items):
        m = Chem.MolFromSmiles(item["smiles"])
        if m is not None:
            mols.append(m)
            tag = "SEED" if item["smiles"] == seed_canon else f"Pareto #{idx}"
            legends.append(
                f"{tag} | Desir={item['desirability']:.3f}\n"
                f"Bio={item['bioavailability']:.1f}% | hERG={item['herg']:.2f} | QED={item['qed']:.2f}"
            )

    if mols:
        pil_img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(320, 240), legends=legends)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        ax2.imshow(plt.imread(buf))
    ax2.axis("off")
    ax2.set_title("Seed Lead vs. Top Pareto-Optimal Analogues")

    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
