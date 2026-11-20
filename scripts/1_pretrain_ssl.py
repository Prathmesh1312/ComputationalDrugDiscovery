"""
1_pretrain_ssl.py — Self-Supervised Pre-Training (GraphCL / InfoNCE) on Unlabeled Drug-Like Molecules
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.augmentations import MoleculeAugmenter
from src.data.dataset import MolecularDataset
from src.models.graph_transformer import MolecularGraphTransformer
from src.models.ssl_contrastive import MolecularContrastiveSSL
from src.utils.chemistry import suppress_rdkit_warnings
from src.utils.seed import get_device, set_seed
from src.utils.visualization import plot_ssl_pretraining

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def train_ssl(
    data_path: str,
    admet_path: str,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device_str: str = "auto",
) -> None:
    set_seed(seed)
    suppress_rdkit_warnings()
    device = get_device(device_str)

    print("=" * 70)
    print("Stage 1: Self-Supervised Pre-Training (GraphCL / InfoNCE)")
    print("=" * 70)
    print(f"Hardware Device: {device} | Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size}")

    df = pd.read_csv(data_path)
    smiles_list = df["smiles"].dropna().tolist()
    cache_path = os.path.join(BASE_DIR, "data", "processed", "graph_cache", "ssl_5000_3d.pt")
    if os.path.basename(data_path) != "unlabeled_molecules.csv":
        cache_path = None

    t0 = time.time()
    dataset = MolecularDataset(smiles_list, use_3d=True, pe_dim=8, cache_path=cache_path, num_workers=4)
    print(f"Loaded {len(dataset)} unlabeled 2D+3D molecular graphs in {time.time() - t0:.1f}s.")

    backbone = MolecularGraphTransformer(hidden_dim=128, num_layers=3, heads=4, dropout=0.1).to(device)
    ssl_model = MolecularContrastiveSSL(backbone, proj_dim=64, temperature=0.1).to(device)

    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    augmenter = MoleculeAugmenter(node_mask_ratio=0.15, edge_drop_ratio=0.15, node_drop_ratio=0.15)

    idx_loader = DataLoader(
        list(range(len(dataset))),
        batch_size=min(batch_size, max(2, len(dataset))),
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
    )

    history = {"epoch": [], "loss": [], "lr": []}
    best_loss = float("inf")
    best_state = copy.deepcopy(backbone.state_dict())

    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss = 0.0
        num_batches = 0

        for idx_batch in idx_loader:
            if len(idx_batch) < 2:
                continue
            views1, views2 = [], []
            for idx in idx_batch.tolist():
                v1, v2 = augmenter.augment_pair(dataset[idx])
                views1.append(v1)
                views2.append(v2)

            b1 = Batch.from_data_list(views1).to(device)
            b2 = Batch.from_data_list(views2).to(device)

            optimizer.zero_grad()
            loss = ssl_model.compute_ssl_loss(b1, b2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        curr_lr = float(optimizer.param_groups[0]["lr"])
        history["epoch"].append(epoch)
        history["loss"].append(avg_loss)
        history["lr"].append(curr_lr)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(backbone.state_dict())

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | InfoNCE Loss: {avg_loss:.4f} | LR: {curr_lr:.2e}")

    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    res_dir = os.path.join(BASE_DIR, "results")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    save_path = os.path.join(ckpt_dir, "ssl_pretrained_backbone.pth")
    torch.save(best_state, save_path)
    print(f"\nSaved best SSL GraphGPS backbone (loss={best_loss:.4f}) to: {save_path}")

    hist_path = os.path.join(res_dir, "ssl_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Zero-shot t-SNE visualization on ADMET val+test compounds if dataset exists
    if os.path.exists(admet_path):
        admet_df = pd.read_csv(admet_path)
        holdout_df = admet_df[admet_df["split"].isin(["val", "test"])].reset_index(drop=True)
        if len(holdout_df) >= 30:
            backbone.load_state_dict(best_state)
            backbone.eval()
            eval_ds = MolecularDataset(
                holdout_df["smiles"].tolist(),
                targets_df=holdout_df,
                use_3d=True,
                pe_dim=8,
                cache_path=None,
                num_workers=4,
            )
            eval_loader = PyGDataLoader(eval_ds, batch_size=64, shuffle=False)
            embs = []
            with torch.no_grad():
                for batch in eval_loader:
                    batch = batch.to(device)
                    rep = backbone(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                        getattr(batch, "pe", None),
                        getattr(batch, "geom_3d", None),
                    )
                    embs.append(rep.cpu().numpy())
            emb_mat = np.vstack(embs)
            perp = min(30, max(5, len(emb_mat) // 4))
            emb_2d = TSNE(n_components=2, perplexity=perp, random_state=seed, init="pca", learning_rate="auto").fit_transform(emb_mat)
            fig_path = os.path.join(fig_dir, "02_ssl_pretraining.png")
            plot_ssl_pretraining(
                history,
                emb_2d,
                holdout_df["qed"].values[: len(emb_mat)],
                holdout_df["herg"].values[: len(emb_mat)],
                fig_path,
            )
            print(f"Saved SSL pre-training & t-SNE figure to: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Self-Supervised GraphGPS Pre-Training")
    parser.add_argument("--data", type=str, default=os.path.join(BASE_DIR, "data", "raw", "unlabeled_molecules.csv"))
    parser.add_argument("--admet-data", type=str, default=os.path.join(BASE_DIR, "data", "processed", "admet_multitask_dataset.csv"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_ssl(
        data_path=args.data,
        admet_path=args.admet_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_str=args.device,
    )
