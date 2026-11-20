"""
2_finetune_admet.py — Multi-Objective ADMET Fine-Tuning with Scaffold Splitting & Early Stopping
"""
import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import MolecularDataset, TASK_COLUMNS
from src.models.baselines import StandardGCNBackbone
from src.models.graph_transformer import MolecularGraphTransformer
from src.models.multi_objective_heads import MultiObjectiveADMETModel, compute_masked_multitask_loss
from src.utils.chemistry import suppress_rdkit_warnings
from src.utils.metrics import evaluate_multitask_predictions
from src.utils.seed import get_device, set_seed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _compute_train_pos_weights(train_df: pd.DataFrame, device: torch.device) -> Dict[str, torch.Tensor]:
    weights: Dict[str, torch.Tensor] = {}
    for task in ["bioavailability", "bbb", "herg"]:
        vals = train_df[task].dropna().values
        n_pos = float((vals >= 0.5).sum())
        n_neg = float((vals < 0.5).sum())
        pw = float(np.clip(n_neg / max(n_pos, 1.0), 0.5, 4.0))
        weights[task] = torch.tensor(pw, dtype=torch.float, device=device)
    return weights


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    all_preds = {t: [] for t in TASK_COLUMNS}
    all_trues = {t: [] for t in TASK_COLUMNS}
    all_masks = {t: [] for t in TASK_COLUMNS}

    for batch in loader:
        batch = batch.to(device)
        preds = model(
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
            p_vals = preds[task].cpu().numpy()
            if task == "bioavailability":
                p_vals = p_vals / 100.0
            all_preds[task].append(p_vals)
            all_trues[task].append(y[:, col_idx])
            all_masks[task].append(m[:, col_idx])

    y_pred_dict = {t: np.concatenate(all_preds[t]) for t in TASK_COLUMNS}
    y_true_dict = {t: np.concatenate(all_trues[t]) for t in TASK_COLUMNS}
    mask_dict = {t: np.concatenate(all_masks[t]) for t in TASK_COLUMNS}
    metrics = evaluate_multitask_predictions(y_true_dict, y_pred_dict, mask_dict)
    return metrics, y_true_dict, y_pred_dict, mask_dict


def finetune_admet(
    data_path: str,
    init_mode: str = "ssl",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 5e-4,
    patience: int = 10,
    seed: int = 42,
    device_str: str = "auto",
) -> None:
    set_seed(seed)
    suppress_rdkit_warnings()
    device = get_device(device_str)

    mode_titles = {
        "ssl": "SSL-GraphGPS (Ours)",
        "scratch": "GraphGPS-Scratch",
        "gcn": "Standard-GCN",
    }
    ckpt_names = {
        "ssl": "admet_finetuned_model.pth",
        "scratch": "admet_scratch_model.pth",
        "gcn": "admet_gcn_model.pth",
    }

    print("=" * 70)
    print(f"Stage 2: Multi-Objective ADMET Fine-Tuning [{mode_titles[init_mode]}]")
    print("=" * 70)
    print(f"Hardware Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size}")

    df = pd.read_csv(data_path)
    cache_dir = os.path.join(BASE_DIR, "data", "processed", "graph_cache")
    use_cache = os.path.basename(data_path) == "admet_multitask_dataset.csv"

    datasets = {}
    splits_df = {}
    t0 = time.time()
    for sp in ["train", "val", "test"]:
        sp_df = df[df["split"] == sp].reset_index(drop=True)
        splits_df[sp] = sp_df
        c_path = os.path.join(cache_dir, f"admet_{sp}_3d.pt") if use_cache else None
        datasets[sp] = MolecularDataset(
            sp_df["smiles"].tolist(),
            targets_df=sp_df,
            use_3d=True,
            pe_dim=8,
            cache_path=c_path,
            num_workers=4,
        )
    print(
        f"Loaded splits in {time.time() - t0:.1f}s -> "
        f"train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}"
    )

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    pos_weights = _compute_train_pos_weights(splits_df["train"], device)
    print(
        "Train Pos Weights -> "
        + ", ".join(f"{k}: {float(v.item()):.2f}" for k, v in pos_weights.items())
    )

    # Model Initialization
    if init_mode == "gcn":
        backbone = StandardGCNBackbone(hidden_dim=128, num_layers=3, dropout=0.1).to(device)
    else:
        backbone = MolecularGraphTransformer(hidden_dim=128, num_layers=3, heads=4, dropout=0.1).to(device)
        if init_mode == "ssl":
            ssl_ckpt = os.path.join(BASE_DIR, "checkpoints", "ssl_pretrained_backbone.pth")
            if os.path.exists(ssl_ckpt):
                backbone.load_state_dict(torch.load(ssl_ckpt, map_location=device, weights_only=True))
                print("Loaded SSL Pre-Trained Weights into GraphGPS Backbone!")
            else:
                print("Warning: SSL checkpoint not found; falling back to random init.")

    model = MultiObjectiveADMETModel(backbone, hidden_dim=128, dropout=0.1).to(device)

    # Discriminative LR for SSL-initialized backbone
    if init_mode == "ssl":
        head_params = (
            list(model.shared_mlp.parameters())
            + list(model.head_bioavailability.parameters())
            + list(model.head_bbb.parameters())
            + list(model.head_herg.parameters())
            + list(model.head_qed.parameters())
            + list(model.head_sascore.parameters())
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": lr * 0.5},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "model": mode_titles[init_mode],
        "init_mode": init_mode,
        "epoch": [],
        "train_loss": [],
        "val_macro_roc_auc": [],
        "val_mean_reg_r2": [],
        "val_composite_score": [],
    }

    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                getattr(batch, "pe", None),
                getattr(batch, "geom_3d", None),
            )
            loss, _ = compute_masked_multitask_loss(preds, batch.y, batch.y_mask, pos_weights=pos_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)

        val_metrics, _, _, _ = evaluate_loader(model, val_loader, device)
        val_macro = val_metrics["aggregate"]["macro_roc_auc"]
        val_r2 = val_metrics["aggregate"]["mean_reg_r2"]
        val_comp = val_metrics["aggregate"]["composite_val_score"]

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_macro_roc_auc"].append(val_macro)
        history["val_mean_reg_r2"].append(val_r2)
        history["val_composite_score"].append(val_comp)

        if val_comp > best_score + 1e-4:
            best_score = val_comp
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | "
                f"Val Macro-AUC: {val_macro:.4f} | Val Reg-R2: {val_r2:.4f} | "
                f"Best Epoch: {best_epoch:02d}"
            )

        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch}).")
            break

    # Save best model checkpoint
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    res_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, ckpt_names[init_mode])
    torch.save(best_state, ckpt_path)

    # Final Held-Out Scaffold Test Evaluation
    model.load_state_dict(best_state)
    test_metrics, _, _, _ = evaluate_loader(model, test_loader, device)
    history["best_epoch"] = best_epoch
    history["best_val_composite_score"] = best_score
    history["test_metrics"] = test_metrics

    hist_path = os.path.join(res_dir, f"train_history_{init_mode}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved best model to: {ckpt_path}")
    print(
        f"Held-Out Test Results [{mode_titles[init_mode]}] -> "
        f"Macro ROC-AUC: {test_metrics['aggregate']['macro_roc_auc']:.4f} | "
        f"QED R2: {test_metrics['qed']['r2']:.4f} | SAScore R2: {test_metrics['sascore']['r2']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Multi-Objective ADMET Fine-Tuning")
    parser.add_argument("--data", type=str, default=os.path.join(BASE_DIR, "data", "processed", "admet_multitask_dataset.csv"))
    parser.add_argument("--init", type=str, choices=["ssl", "scratch", "gcn"], default="ssl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    finetune_admet(
        data_path=args.data,
        init_mode=args.init,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
    )
