"""
4_evaluate_and_ablation.py — Held-Out Unseen-Scaffold Benchmark & Ablation Evaluation

Compares 4 models strictly on the unseen-scaffold test split:
  1. RF-ECFP4 (Classical Random Forest on 2048-bit Morgan Fingerprints)
  2. Standard-GCN (3-Layer Local Message Passing Baseline)
  3. GraphGPS-Scratch (2D+3D Hybrid Local+Global Transformer without SSL Pre-Training)
  4. SSL-GraphGPS (Ours: Self-Supervised GraphCL Pre-Trained + 2D+3D Hybrid GraphGPS)
"""
import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import MolecularDataset, TASK_COLUMNS
from src.models.baselines import MorganFingerprintRFBaseline, StandardGCNBackbone
from src.models.graph_transformer import MolecularGraphTransformer
from src.models.multi_objective_heads import MultiObjectiveADMETModel
from src.utils.chemistry import suppress_rdkit_warnings
from src.utils.metrics import evaluate_multitask_predictions, format_metrics_markdown
from src.utils.seed import get_device, set_seed
from src.utils.visualization import (
    plot_regression_parity,
    plot_roc_comparison,
    plot_training_curves,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@torch.no_grad()
def _predict_neural(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    preds = {t: [] for t in TASK_COLUMNS}
    trues = {t: [] for t in TASK_COLUMNS}
    masks = {t: [] for t in TASK_COLUMNS}

    for batch in loader:
        batch = batch.to(device)
        out = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            getattr(batch, "pe", None),
            getattr(batch, "geom_3d", None),
        )
        y = batch.y.cpu().numpy()
        m = batch.y_mask.cpu().numpy()
        for col_idx, task in enumerate(TASK_COLUMNS):
            vals = out[task].cpu().numpy()
            if task == "bioavailability":
                vals = vals / 100.0
            preds[task].append(vals)
            trues[task].append(y[:, col_idx])
            masks[task].append(m[:, col_idx])

    y_pred = {t: np.concatenate(preds[t]) for t in TASK_COLUMNS}
    y_true = {t: np.concatenate(trues[t]) for t in TASK_COLUMNS}
    mask = {t: np.concatenate(masks[t]) for t in TASK_COLUMNS}
    return {"y_true": y_true, "y_pred": y_pred, "mask": mask}


def main(data_path: str, device_str: str = "auto") -> None:
    set_seed(42)
    suppress_rdkit_warnings()
    device = get_device(device_str)

    print("=" * 75)
    print("Stage 4: Multi-Model Held-Out Scaffold Benchmark & Ablation Study")
    print("=" * 75)

    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    cache_path = os.path.join(BASE_DIR, "data", "processed", "graph_cache", "admet_test_3d.pt")
    test_ds = MolecularDataset(
        test_df["smiles"].tolist(),
        targets_df=test_df,
        use_3d=True,
        pe_dim=8,
        cache_path=cache_path if os.path.basename(data_path) == "admet_multitask_dataset.csv" else None,
        num_workers=4,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_metrics: Dict[str, Dict[str, Any]] = {}
    roc_data: Dict[str, Dict[str, Any]] = {}

    # 1. Classical Baseline: Random Forest on Morgan ECFP4
    print("Fitting & evaluating RF-ECFP4 baseline...")
    rf = MorganFingerprintRFBaseline(n_estimators=300, random_state=42).fit(train_df)
    rf_preds = rf.predict(test_df)
    rf_trues = {t: test_df[t].fillna(0.0).values for t in TASK_COLUMNS}
    rf_masks = {t: (~test_df[t].isna()).values for t in TASK_COLUMNS}
    all_metrics["RF-ECFP4"] = evaluate_multitask_predictions(rf_trues, rf_preds, rf_masks)
    roc_data["RF-ECFP4"] = {
        t: {"y_true": rf_trues[t][rf_masks[t]], "y_prob": rf_preds[t][rf_masks[t]]}
        for t in ["bioavailability", "bbb", "herg"]
    }

    # 2. Neural Models
    model_specs = [
        ("Standard-GCN", "gcn", os.path.join(BASE_DIR, "checkpoints", "admet_gcn_model.pth")),
        ("GraphGPS-Scratch", "scratch", os.path.join(BASE_DIR, "checkpoints", "admet_scratch_model.pth")),
        ("SSL-GraphGPS (Ours)", "ssl", os.path.join(BASE_DIR, "checkpoints", "admet_finetuned_model.pth")),
    ]

    ssl_outputs = None
    for display_name, kind, ckpt_path in model_specs:
        if not os.path.exists(ckpt_path):
            print(f"Skipping {display_name}: checkpoint not found at {ckpt_path}")
            continue
        if kind == "gcn":
            backbone = StandardGCNBackbone(hidden_dim=128, num_layers=3, dropout=0.1).to(device)
        else:
            backbone = MolecularGraphTransformer(hidden_dim=128, num_layers=3, heads=4, dropout=0.1).to(device)
        model = MultiObjectiveADMETModel(backbone, hidden_dim=128, dropout=0.1).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        res = _predict_neural(model, test_loader, device)
        all_metrics[display_name] = evaluate_multitask_predictions(res["y_true"], res["y_pred"], res["mask"])
        roc_data[display_name] = {
            t: {
                "y_true": res["y_true"][t][res["mask"][t]],
                "y_prob": res["y_pred"][t][res["mask"][t]],
            }
            for t in ["bioavailability", "bbb", "herg"]
        }
        if kind == "ssl":
            ssl_outputs = res

    res_dir = os.path.join(BASE_DIR, "results")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    summary_path = os.path.join(res_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    md_table = format_metrics_markdown(all_metrics)
    table_path = os.path.join(res_dir, "benchmark_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    print("\nHeld-Out Unseen-Scaffold Benchmark Table:")
    print(md_table)

    # Plots
    histories = {}
    for display_name, kind in [
        ("SSL-GraphGPS (Ours)", "ssl"),
        ("GraphGPS-Scratch", "scratch"),
        ("Standard-GCN", "gcn"),
    ]:
        h_path = os.path.join(res_dir, f"train_history_{kind}.json")
        if os.path.exists(h_path):
            with open(h_path, "r", encoding="utf-8") as f:
                histories[display_name] = json.load(f)

    if histories:
        plot_training_curves(histories, os.path.join(fig_dir, "03_training_curves.png"))
    if roc_data:
        plot_roc_comparison(roc_data, os.path.join(fig_dir, "04_roc_comparison.png"))
    if ssl_outputs is not None:
        m_qed = ssl_outputs["mask"]["qed"]
        m_sa = ssl_outputs["mask"]["sascore"]
        plot_regression_parity(
            ssl_outputs["y_true"]["qed"][m_qed],
            ssl_outputs["y_pred"]["qed"][m_qed],
            ssl_outputs["y_true"]["sascore"][m_sa],
            ssl_outputs["y_pred"]["sascore"][m_sa],
            os.path.join(fig_dir, "05_regression_parity.png"),
        )

    print(f"\nSaved metrics JSON -> {summary_path}")
    print(f"Saved Markdown table -> {table_path}")
    print(f"Saved evaluation figures -> {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Multi-Model Held-Out Evaluation & Ablation")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(BASE_DIR, "data", "processed", "admet_multitask_dataset.csv"),
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    main(args.data, device_str=args.device)
