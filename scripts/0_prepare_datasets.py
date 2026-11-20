"""
0_prepare_datasets.py — Real-World Drug Discovery Dataset Curation & Bemis-Murcko Scaffold Splitting

Sources:
  - Therapeutics Data Commons (TDC) via Harvard Dataverse:
      * Bioavailability_Ma (n ~ 640)
      * BBB_Martins (n ~ 2,039)
      * hERG Wang (n ~ 655)
      * hERG Karim (stratified n = 2,500 subsample from 13,445)
  - ZINC250k Drug-Like Library:
      * 5,000 unlabeled molecules strictly filtered to have ZERO scaffold/SMILES overlap
        with ADMET validation and test splits for leakage-free Self-Supervised Pre-Training.
"""
import io
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.utils.chemistry import compute_exact_qed_sa, standardize_smiles, suppress_rdkit_warnings
from src.utils.seed import set_seed
from src.utils.splitting import bemis_murcko_scaffold_split, get_murcko_scaffold
from src.utils.visualization import plot_dataset_eda

TDC_URLS = {
    "bioavailability": "https://dataverse.harvard.edu/api/access/datafile/4259567",
    "bbb": "https://dataverse.harvard.edu/api/access/datafile/4259566",
    "herg_wang": "https://dataverse.harvard.edu/api/access/datafile/4259588",
    "herg_karim": "https://dataverse.harvard.edu/api/access/datafile/6822246",
}
ZINC_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)


def download_text(url: str, retries: int = 3, timeout: int = 30) -> str:
    """Download URL text with retry and exponential backoff."""
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_err = exc
            wait_s = 2 ** attempt
            print(f"   Retry {attempt}/{retries} failed ({exc}); waiting {wait_s}s...")
            time.sleep(wait_s)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def _aggregate_endpoint_rows(
    df: pd.DataFrame, smiles_col: str, y_col: str, out_dict: Dict[str, List[float]]
) -> int:
    valid = 0
    for sm, y_val in zip(df[smiles_col].tolist(), df[y_col].tolist()):
        if pd.isna(y_val):
            continue
        std = standardize_smiles(str(sm), min_heavy=2, max_heavy=80)
        if std is None:
            continue
        out_dict[std].append(float(int(float(y_val) >= 0.5)))
        valid += 1
    return valid


def _majority_vote(vals: List[float]) -> float:
    if not vals:
        return np.nan
    avg = float(np.mean(vals))
    if avg > 0.5:
        return 1.0
    if avg < 0.5:
        return 0.0
    return np.nan


def main() -> None:
    set_seed(42)
    suppress_rdkit_warnings()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    proc_dir = os.path.join(base_dir, "data", "processed")
    fig_dir = os.path.join(base_dir, "results", "figures")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("Stage 0: Real-World ADMET & Unlabeled SSL Dataset Curation")
    print("=" * 70)

    endpoint_obs: Dict[str, Dict[str, List[float]]] = {
        "bioavailability": defaultdict(list),
        "bbb": defaultdict(list),
        "herg": defaultdict(list),
    }

    # 1. TDC Bioavailability_Ma
    print("Downloading TDC Bioavailability_Ma...")
    bio_df = pd.read_csv(io.StringIO(download_text(TDC_URLS["bioavailability"])), sep="\t")
    n_bio = _aggregate_endpoint_rows(bio_df, "Drug", "Y", endpoint_obs["bioavailability"])
    print(f"   Bioavailability standardized rows: {n_bio}")

    # 2. TDC BBB_Martins
    print("Downloading TDC BBB_Martins...")
    bbb_df = pd.read_csv(io.StringIO(download_text(TDC_URLS["bbb"])), sep="\t")
    n_bbb = _aggregate_endpoint_rows(bbb_df, "Drug", "Y", endpoint_obs["bbb"])
    print(f"   BBB standardized rows: {n_bbb}")

    # 3. TDC hERG Wang + Stratified hERG Karim (2,500)
    print("Downloading TDC hERG (Wang) & hERG_Karim...")
    herg_w_df = pd.read_csv(io.StringIO(download_text(TDC_URLS["herg_wang"])), sep="\t")
    n_hw = _aggregate_endpoint_rows(herg_w_df, "Drug", "Y", endpoint_obs["herg"])

    herg_k_df = pd.read_csv(io.StringIO(download_text(TDC_URLS["herg_karim"])), sep="\t")
    # Stratified balanced subsample of 2,500 compounds
    pos_k = herg_k_df[herg_k_df["Y"] == 1].sample(n=1250, random_state=42)
    neg_k = herg_k_df[herg_k_df["Y"] == 0].sample(n=1250, random_state=42)
    herg_k_sub = pd.concat([pos_k, neg_k], ignore_index=True).sample(frac=1.0, random_state=42)
    n_hk = _aggregate_endpoint_rows(herg_k_sub, "Drug", "Y", endpoint_obs["herg"])
    print(f"   hERG standardized rows: Wang={n_hw}, Karim(sub)={n_hk}")

    # Merge unique canonical SMILES across endpoints
    all_smiles = sorted(
        set(endpoint_obs["bioavailability"].keys())
        | set(endpoint_obs["bbb"].keys())
        | set(endpoint_obs["herg"].keys())
    )

    records = []
    for smi in all_smiles:
        b_val = _majority_vote(endpoint_obs["bioavailability"].get(smi, []))
        bbb_val = _majority_vote(endpoint_obs["bbb"].get(smi, []))
        h_val = _majority_vote(endpoint_obs["herg"].get(smi, []))
        if np.isnan(b_val) and np.isnan(bbb_val) and np.isnan(h_val):
            continue
        qed_val, sa_val = compute_exact_qed_sa(smi)
        records.append(
            {
                "smiles": smi,
                "bioavailability": b_val,
                "bbb": bbb_val,
                "herg": h_val,
                "qed": round(qed_val, 5),
                "sascore": round(sa_val, 5),
            }
        )

    admet_df = pd.DataFrame(records)
    print(f"\nMerged unique standardized ADMET compounds: {len(admet_df)}")

    # 4. Bemis-Murcko Disjoint Scaffold Split (80/10/10)
    admet_df = bemis_murcko_scaffold_split(
        admet_df,
        smiles_col="smiles",
        task_cols=("bioavailability", "bbb", "herg"),
        train_frac=0.8,
        val_frac=0.1,
        seed=42,
    )

    admet_out = os.path.join(proc_dir, "admet_multitask_dataset.csv")
    admet_df.to_csv(admet_out, index=False)
    print(f"Saved multi-task ADMET dataset to: {admet_out}")

    # 5. ZINC250k Unlabeled SSL Library (5,000 molecules with zero val/test leakage)
    print("\nDownloading ZINC250k library for SSL pre-training...")
    zinc_raw = pd.read_csv(io.StringIO(download_text(ZINC_URL)))
    zinc_pool = zinc_raw["smiles"].dropna().sample(n=18000, random_state=42).tolist()

    holdout_df = admet_df[admet_df["split"].isin(["val", "test"])]
    forbidden_smiles = set(holdout_df["smiles"].tolist())
    forbidden_scaffolds = set(holdout_df["scaffold"].tolist())

    ssl_smiles: List[str] = []
    seen_ssl = set()
    for raw_sm in zinc_pool:
        std = standardize_smiles(str(raw_sm).strip(), min_heavy=5, max_heavy=70)
        if std is None or std in seen_ssl or std in forbidden_smiles:
            continue
        scaf = get_murcko_scaffold(std)
        if scaf in forbidden_scaffolds:
            continue
        seen_ssl.add(std)
        ssl_smiles.append(std)
        if len(ssl_smiles) >= 5000:
            break

    unlabeled_df = pd.DataFrame({"smiles": ssl_smiles})
    unlabeled_out = os.path.join(raw_dir, "unlabeled_molecules.csv")
    unlabeled_df.to_csv(unlabeled_out, index=False)
    print(f"Saved {len(unlabeled_df)} leakage-free SSL molecules to: {unlabeled_out}")

    # 6. Save exact dataset statistics & EDA plot
    stats = {
        "unlabeled_ssl_count": int(len(unlabeled_df)),
        "admet_total_count": int(len(admet_df)),
        "unique_scaffolds": int(admet_df["scaffold"].nunique()),
        "splits": {},
        "tasks": {},
    }
    for sp in ["train", "val", "test"]:
        sub = admet_df[admet_df["split"] == sp]
        stats["splits"][sp] = {
            "count": int(len(sub)),
            "unique_scaffolds": int(sub["scaffold"].nunique()),
            "bioavailability_count": int(sub["bioavailability"].notna().sum()),
            "bioavailability_pos_rate": float(sub["bioavailability"].dropna().mean()),
            "bbb_count": int(sub["bbb"].notna().sum()),
            "bbb_pos_rate": float(sub["bbb"].dropna().mean()),
            "herg_count": int(sub["herg"].notna().sum()),
            "herg_pos_rate": float(sub["herg"].dropna().mean()),
        }
    for t in ["bioavailability", "bbb", "herg"]:
        vals = admet_df[t].dropna()
        stats["tasks"][t] = {
            "count": int(len(vals)),
            "pos_count": int((vals >= 0.5).sum()),
            "neg_count": int((vals < 0.5).sum()),
            "pos_rate": float(vals.mean()),
        }
    for t in ["qed", "sascore"]:
        vals = admet_df[t].dropna()
        stats["tasks"][t] = {
            "count": int(len(vals)),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    stats_path = os.path.join(proc_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    eda_path = os.path.join(fig_dir, "01_dataset_eda.png")
    plot_dataset_eda(admet_df, len(unlabeled_df), eda_path)

    print("\nDataset Summary:")
    print(json.dumps(stats, indent=2))
    print(f"Saved EDA figure to: {eda_path}")


if __name__ == "__main__":
    main()
