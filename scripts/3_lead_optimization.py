"""
3_lead_optimization.py — Multi-Generation Evolutionary Lead Optimization, Pareto Ranking & XAI
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from torch_geometric.data import Batch

from src.data.dataset import smiles_to_graph_data
from src.models.graph_transformer import MolecularGraphTransformer
from src.models.multi_objective_heads import MultiObjectiveADMETModel
from src.optimization.lead_mutator import MolecularLeadMutator
from src.optimization.pareto_scorer import MultiObjectiveScorer, ParetoFrontier
from src.utils.chemistry import (
    compute_exact_qed_sa,
    compute_lipinski_and_pains,
    compute_tanimoto_similarity,
    standardize_smiles,
    suppress_rdkit_warnings,
)
from src.utils.explainability import render_saliency_figure
from src.utils.seed import get_device, set_seed
from src.utils.visualization import plot_pareto_optimization

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_lead_optimization(
    seed_smiles: str,
    generations: int = 2,
    max_analogues: int = 35,
    mode: str = "peripheral",
    top_k: int = 5,
    save_plots: bool = False,
    device_str: str = "auto",
) -> List[Dict[str, Any]]:
    set_seed(42)
    suppress_rdkit_warnings()
    device = get_device(device_str)

    seed_std = standardize_smiles(seed_smiles, min_heavy=2, max_heavy=80) or seed_smiles

    print("=" * 80)
    print("Stage 3: Multi-Generation Evolutionary Lead Optimization & Pareto Scoring")
    print("=" * 80)
    print(f"Seed Lead SMILES: {seed_std} | Mode: {mode} | Generations: {generations}")

    backbone = MolecularGraphTransformer(hidden_dim=128, num_layers=3, heads=4, dropout=0.1).to(device)
    model = MultiObjectiveADMETModel(backbone, hidden_dim=128, dropout=0.1).to(device)

    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "admet_finetuned_model.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("Loaded SSL-GraphGPS Fine-Tuned Weights!")
    else:
        print("Warning: Fine-tuned checkpoint not found; running with initialized weights.")

    model.eval()
    scorer = MultiObjectiveScorer(target_mode=mode)
    mutator = MolecularLeadMutator()

    @torch.no_grad()
    def batch_score_fn(smiles_batch: List[str]) -> List[Dict[str, Any]]:
        graphs = []
        valid_smiles = []
        for smi in smiles_batch:
            g = smiles_to_graph_data(smi, compute_pe=True, pe_dim=8, use_3d=True, standardize=True)
            if g is not None:
                graphs.append(g)
                valid_smiles.append(g.smiles)
        if not graphs:
            return []

        batch = Batch.from_data_list(graphs).to(device)
        preds = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            getattr(batch, "pe", None),
            getattr(batch, "geom_3d", None),
        )

        out_list: List[Dict[str, Any]] = []
        for idx, smi in enumerate(valid_smiles):
            qed_rdkit, sa_rdkit = compute_exact_qed_sa(smi)
            mc = compute_lipinski_and_pains(smi)
            sim = compute_tanimoto_similarity(seed_std, smi)
            profile = {
                "smiles": smi,
                "bioavailability": float(preds["bioavailability"][idx].item()),
                "bbb": float(preds["bbb"][idx].item()),
                "herg": float(preds["herg"][idx].item()),
                "qed": float(preds["qed"][idx].item()),
                "sascore": float(preds["sascore"][idx].item()),
                "qed_rdkit": round(qed_rdkit, 4),
                "sascore_rdkit": round(sa_rdkit, 4),
                "tanimoto_sim": round(sim, 4),
                "lipinski_violations": mc["lipinski_violations"],
                "pains_pass": mc["pains_pass"],
            }
            profile["desirability"] = scorer.compute_composite_desirability(profile)
            out_list.append(profile)
        return out_list

    all_candidates = mutator.evolve_leads(
        seed_std,
        score_fn=batch_score_fn,
        generations=generations,
        pop_size=max_analogues,
        top_parents=6,
        min_similarity=0.35,
    )

    ParetoFrontier.assign_pareto_ranks(all_candidates, target_mode=mode)
    pareto_front1 = ParetoFrontier.filter_pareto_candidates(all_candidates, target_mode=mode)
    sorted_candidates = sorted(
        all_candidates,
        key=lambda x: (-int(x.get("pareto_rank", 99)), x["desirability"]),
        reverse=True,
    )

    print(f"Total Valid MedChem-Filtered Candidates Explored: {len(all_candidates)}")
    print(f"Non-Dominated Pareto Front-1 Candidates: {len(pareto_front1)}\n")

    print("Top Optimized Molecular Leads (Pareto Front-1 First, Ranked by Desirability):")
    print("-" * 92)
    print(
        f"{'Rank':<5} {'Front':<6} {'Bioavail':<10} {'hERG':<8} {'BBB':<7} "
        f"{'QED(P/RD)':<12} {'SA(P/RD)':<11} {'Sim':<6} {'Desir':<7}"
    )
    print("-" * 92)

    for rank, c in enumerate(sorted_candidates[:top_k], 1):
        front_tag = f"F{c['pareto_rank']}" + ("*" if c["pareto_rank"] == 1 else "")
        print(
            f"#{rank:<4} {front_tag:<6} "
            f"{c['bioavailability']:>6.1f}%   "
            f"{c['herg']:>5.3f}   "
            f"{c['bbb']:>5.3f}  "
            f"{c['qed']:>4.2f}/{c['qed_rdkit']:<4.2f}   "
            f"{c['sascore']:>4.2f}/{c['sascore_rdkit']:<4.2f}  "
            f"{c['tanimoto_sim']:>4.2f}  "
            f"{c['desirability']:>5.3f}"
        )
        print(f"      SMILES: {c['smiles']}\n")

    if save_plots:
        res_dir = os.path.join(BASE_DIR, "results")
        fig_dir = os.path.join(res_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        out_json = os.path.join(res_dir, "lead_opt_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed_smiles": seed_std,
                    "mode": mode,
                    "total_candidates": len(all_candidates),
                    "pareto_front1_count": len(pareto_front1),
                    "top_candidates": sorted_candidates[:15],
                },
                f,
                indent=2,
            )

        pareto_fig = os.path.join(fig_dir, "06_pareto_optimization.png")
        plot_pareto_optimization(all_candidates, pareto_front1, seed_std, pareto_fig)

        # Render XAI Atom Saliency on Held-Out Test hERG Blocker + Aspirin + Caffeine
        xai_mols = []
        admet_csv = os.path.join(BASE_DIR, "data", "processed", "admet_multitask_dataset.csv")
        blocker_smi = "Oc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1"  # Astemizole metabolite
        if os.path.exists(admet_csv):
            df_admet = pd.read_csv(admet_csv)
            test_herg_pos = df_admet[(df_admet["split"] == "test") & (df_admet["herg"] == 1.0)]
            if len(test_herg_pos) > 0:
                # Pick first mid-sized test hERG blocker
                for sm in test_herg_pos["smiles"].tolist():
                    if 18 <= len(sm) <= 55:
                        blocker_smi = sm
                        break

        xai_mols.append(("Test hERG Blocker", blocker_smi, "herg"))
        xai_mols.append(("Aspirin (Seed)", seed_std, "herg"))
        xai_mols.append(("Caffeine (CNS+)", "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "bbb"))

        xai_fig = os.path.join(fig_dir, "07_atom_saliency_xai.png")
        render_saliency_figure(model, xai_mols, xai_fig, device=device)
        print(f"Saved lead optimization JSON -> {out_json}")
        print(f"Saved Pareto figure -> {pareto_fig}")
        print(f"Saved XAI saliency figure -> {xai_fig}")

    return sorted_candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Multi-Objective Evolutionary Lead Optimization")
    parser.add_argument("--smiles", type=str, default="CC(=O)Oc1ccccc1C(=O)O", help="Seed lead molecule SMILES")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--max-analogues", type=int, default=35)
    parser.add_argument("--mode", type=str, choices=["peripheral", "cns", "neutral"], default="peripheral")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_lead_optimization(
        seed_smiles=args.smiles,
        generations=args.generations,
        max_analogues=args.max_analogues,
        mode=args.mode,
        top_k=args.top_k,
        save_plots=args.save_plots,
        device_str=args.device,
    )
