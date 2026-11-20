"""
app.py — FastAPI Backend for the ADMET GraphGPS Scientific Workbench Demo UI
"""
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
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
from src.utils.explainability import compute_atom_saliency, render_mol_png_base64
from src.utils.seed import get_device, set_seed
from src.utils.splitting import get_murcko_scaffold

WEB_DIR = os.path.join(BASE_DIR, "web")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints", "admet_finetuned_model.pth")


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=300)


class SaliencyRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=300)
    task: Literal["herg", "bbb", "bioavailability", "qed", "sascore"] = "herg"


class OptimizeRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=300)
    generations: int = Field(default=2, ge=1, le=2)
    max_analogues: int = Field(default=30, ge=10, le=40)
    mode: Literal["peripheral", "cns", "neutral"] = "peripheral"
    min_similarity: float = Field(default=0.35, ge=0.2, le=0.8)


def _load_test_herg_blocker() -> str:
    fallback = "Oc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1"
    csv_path = os.path.join(BASE_DIR, "data", "processed", "admet_multitask_dataset.csv")
    if not os.path.exists(csv_path):
        return fallback
    df = pd.read_csv(csv_path)
    sub = df[(df["split"] == "test") & (df["herg"] == 1.0)]
    for sm in sub["smiles"].tolist():
        if 18 <= len(sm) <= 55:
            return sm
    return fallback


def _load_runtime_state(target_app: FastAPI) -> None:
    if getattr(target_app.state, "initialized", False):
        return

    set_seed(42)
    suppress_rdkit_warnings()
    device = get_device()

    backbone = MolecularGraphTransformer(hidden_dim=128, num_layers=3, heads=4, dropout=0.1).to(device)
    model = MultiObjectiveADMETModel(backbone, hidden_dim=128, dropout=0.1).to(device)
    if not os.path.exists(CKPT_PATH):
        raise RuntimeError(f"Trained ADMET checkpoint missing at {CKPT_PATH}")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    model.eval()

    with open(os.path.join(BASE_DIR, "results", "metrics_summary.json"), "r", encoding="utf-8") as f:
        metrics_summary = json.load(f)
    with open(os.path.join(BASE_DIR, "data", "processed", "dataset_stats.json"), "r", encoding="utf-8") as f:
        dataset_stats = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "ssl_history.json"), "r", encoding="utf-8") as f:
        ssl_history = json.load(f)

    target_app.state.device = device
    target_app.state.model = model
    target_app.state.mutator = MolecularLeadMutator()
    target_app.state.param_count = sum(p.numel() for p in model.parameters())
    target_app.state.metrics_summary = metrics_summary
    target_app.state.dataset_stats = dataset_stats
    target_app.state.ssl_history = ssl_history
    target_app.state.test_herg_blocker = _load_test_herg_blocker()
    target_app.state.initialized = True


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    _load_runtime_state(fastapi_app)
    yield


app = FastAPI(
    title="ADMET GraphGPS Workbench",
    version="1.0.0",
    lifespan=lifespan,
)

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
if os.path.isdir(FIG_DIR):
    app.mount("/figures", StaticFiles(directory=FIG_DIR), name="figures")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/health")
def health():
    _load_runtime_state(app)
    return {
        "status": "ok",
        "device": str(app.state.device),
        "checkpoint": "admet_finetuned_model.pth",
        "param_count": app.state.param_count,
        "macro_roc_auc": round(
            float(app.state.metrics_summary["SSL-GraphGPS (Ours)"]["aggregate"]["macro_roc_auc"]), 4
        ),
    }


@app.get("/api/presets")
def get_presets():
    _load_runtime_state(app)
    return {
        "presets": [
            {
                "name": "Aspirin",
                "smiles": "CC(=O)Oc1ccccc1C(=O)O",
                "role": "NSAID · Peripheral",
            },
            {
                "name": "Ibuprofen",
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                "role": "NSAID · Peripheral",
            },
            {
                "name": "Caffeine",
                "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                "role": "Stimulant · CNS+",
            },
            {
                "name": "Test hERG Blocker",
                "smiles": app.state.test_herg_blocker,
                "role": "Unseen Test · Cardiotoxic",
            },
        ]
    }


@app.post("/api/predict")
def predict_admet(req: PredictRequest):
    _load_runtime_state(app)
    canon = standardize_smiles(req.smiles, min_heavy=2, max_heavy=80)
    if canon is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid or unsupported SMILES string. Enter a valid organic molecule (e.g., CC(=O)Oc1ccccc1C(=O)O).",
        )

    t0 = time.perf_counter()
    graph = smiles_to_graph_data(canon, compute_pe=True, pe_dim=8, use_3d=True, standardize=False)
    if graph is None:
        raise HTTPException(status_code=422, detail="Unable to construct molecular graph from SMILES.")

    batch = Batch.from_data_list([graph]).to(app.state.device)
    with torch.no_grad():
        preds = app.state.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            getattr(batch, "pe", None),
            getattr(batch, "geom_3d", None),
        )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    mol = Chem.MolFromSmiles(canon)
    qed_rd, sa_rd = compute_exact_qed_sa(canon)
    mc = compute_lipinski_and_pains(canon)
    g3d = graph.geom_3d.squeeze(0).tolist()
    scaffold = get_murcko_scaffold(canon)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    img_uri = render_mol_png_base64(mol, scores=None, width=520, height=340)

    return {
        "predictions": {
            "bioavailability": round(float(preds["bioavailability"].item()), 2),
            "bbb": round(float(preds["bbb"].item()), 4),
            "herg": round(float(preds["herg"].item()), 4),
            "qed": round(float(preds["qed"].item()), 4),
            "sascore": round(float(preds["sascore"].item()), 3),
        },
        "rdkit_audit": {
            "qed": round(qed_rd, 4),
            "sascore": round(sa_rd, 3),
        },
        "medchem": {
            "mw": round(mc["mw"], 2),
            "logp": round(mc["logp"], 2),
            "hbd": mc["hbd"],
            "hba": mc["hba"],
            "tpsa": round(mc["tpsa"], 2),
            "rotatable_bonds": mc["rotatable_bonds"],
            "lipinski_violations": mc["lipinski_violations"],
            "lipinski_pass": mc["lipinski_pass"],
            "pains_pass": mc["pains_pass"],
        },
        "geometry_3d": {
            "npr1": round(float(g3d[0]), 3),
            "npr2": round(float(g3d[1]), 3),
            "asphericity": round(float(g3d[2]), 3),
            "radius_of_gyration": round(float(g3d[3]) * 10.0, 2),
        },
        "graph_info": {
            "canonical_smiles": canon,
            "formula": formula,
            "scaffold": scaffold,
            "num_atoms": int(mol.GetNumAtoms()),
            "num_bonds": int(mol.GetNumBonds()),
        },
        "latency_ms": latency_ms,
        "image_data_uri": img_uri,
    }


@app.post("/api/saliency")
def explain_saliency(req: SaliencyRequest):
    _load_runtime_state(app)
    canon = standardize_smiles(req.smiles, min_heavy=2, max_heavy=80)
    if canon is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid or unsupported SMILES string. Enter a valid organic molecule.",
        )

    mol, scores, pred_summary = compute_atom_saliency(
        app.state.model,
        canon,
        target_task=req.task,
        device=app.state.device,
    )
    img_uri = render_mol_png_base64(mol, scores=scores, width=560, height=380)

    atom_rows = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_rows.append(
            {
                "atom_idx": idx,
                "symbol": atom.GetSymbol(),
                "degree": int(atom.GetTotalDegree()),
                "formal_charge": int(atom.GetFormalCharge()),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "in_ring": bool(atom.IsInRing()),
                "attribution": round(float(scores[idx]), 4),
            }
        )
    atom_rows.sort(key=lambda r: r["attribution"], reverse=True)

    return {
        "canonical_smiles": canon,
        "task": req.task,
        "predictions": {k: round(float(v), 4) for k, v in pred_summary.items()},
        "image_data_uri": img_uri,
        "top_atoms": atom_rows[:8],
    }


@app.post("/api/optimize")
def optimize_lead(req: OptimizeRequest):
    _load_runtime_state(app)
    seed_std = standardize_smiles(req.smiles, min_heavy=2, max_heavy=80)
    if seed_std is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid or unsupported SMILES string for lead optimization.",
        )

    t0 = time.perf_counter()
    scorer = MultiObjectiveScorer(target_mode=req.mode)
    model = app.state.model
    device = app.state.device

    @torch.no_grad()
    def batch_score(smiles_batch: List[str]) -> List[Dict[str, Any]]:
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
        out: List[Dict[str, Any]] = []
        for idx, smi in enumerate(valid_smiles):
            qed_rd, sa_rd = compute_exact_qed_sa(smi)
            mc = compute_lipinski_and_pains(smi)
            sim = compute_tanimoto_similarity(seed_std, smi)
            prof = {
                "smiles": smi,
                "is_seed": smi == seed_std,
                "bioavailability": round(float(preds["bioavailability"][idx].item()), 2),
                "bbb": round(float(preds["bbb"][idx].item()), 4),
                "herg": round(float(preds["herg"][idx].item()), 4),
                "qed": round(float(preds["qed"][idx].item()), 4),
                "sascore": round(float(preds["sascore"][idx].item()), 3),
                "qed_rdkit": round(qed_rd, 4),
                "sascore_rdkit": round(sa_rd, 3),
                "tanimoto_sim": round(sim, 3),
                "lipinski_violations": mc["lipinski_violations"],
                "pains_pass": mc["pains_pass"],
            }
            prof["desirability"] = round(scorer.compute_composite_desirability(prof), 4)
            out.append(prof)
        return out

    all_candidates = app.state.mutator.evolve_leads(
        seed_std,
        score_fn=batch_score,
        generations=req.generations,
        pop_size=req.max_analogues,
        top_parents=6,
        min_similarity=req.min_similarity,
    )
    ParetoFrontier.assign_pareto_ranks(all_candidates, target_mode=req.mode)
    pareto_f1_count = sum(1 for c in all_candidates if c.get("pareto_rank") == 1)

    sorted_candidates = sorted(
        all_candidates,
        key=lambda c: (int(c.get("pareto_rank", 99)), -float(c["desirability"])),
    )

    seed_profile = next((c for c in sorted_candidates if c["is_seed"]), sorted_candidates[0])

    # Render PNG cards for Seed + Top 5 non-seed Front-1 / highest-ranked candidates
    top_cards = []
    card_items = [seed_profile] + [c for c in sorted_candidates if not c["is_seed"]][:5]
    for c in card_items:
        mol = Chem.MolFromSmiles(c["smiles"])
        card_copy = dict(c)
        card_copy["image_data_uri"] = (
            render_mol_png_base64(mol, scores=None, width=360, height=240) if mol is not None else ""
        )
        top_cards.append(card_copy)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "seed_smiles": seed_std,
        "mode": req.mode,
        "total_candidates": len(sorted_candidates),
        "pareto_front1_count": pareto_f1_count,
        "latency_ms": latency_ms,
        "seed_profile": seed_profile,
        "candidates": sorted_candidates,
        "top_cards": top_cards,
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    _load_runtime_state(app)
    return {
        "metrics_summary": app.state.metrics_summary,
        "dataset_stats": app.state.dataset_stats,
        "ssl_history": app.state.ssl_history,
        "figures": [
            {
                "id": "roc",
                "title": "Held-Out Unseen-Scaffold ROC Curves",
                "caption": "ROC comparison across Bioavailability (n=44), BBB (n=153), and hERG (n=330).",
                "url": "/figures/04_roc_comparison.png",
            },
            {
                "id": "ssl",
                "title": "Stage 1 GraphCL Convergence & Zero-Shot t-SNE",
                "caption": "InfoNCE loss drop (0.8659 to 0.0402) and unsupervised latent clustering.",
                "url": "/figures/02_ssl_pretraining.png",
            },
            {
                "id": "curves",
                "title": "Multi-Task Training & Validation Curves",
                "caption": "Training loss and unseen-scaffold validation Macro ROC-AUC across neural models.",
                "url": "/figures/03_training_curves.png",
            },
            {
                "id": "parity",
                "title": "QED & SAScore Test Regression Parity",
                "caption": "SSL-GraphGPS achieves R^2 = 0.7165 on QED and R^2 = 0.7584 on SAScore.",
                "url": "/figures/05_regression_parity.png",
            },
            {
                "id": "pipeline",
                "title": "4-Stage System Architecture",
                "caption": "End-to-end curation, GraphCL pre-training, 2D+3D GraphGPS, and Pareto XAI.",
                "url": "/figures/arch_01_system_pipeline.png",
            },
            {
                "id": "eda",
                "title": "Curated TDC + ZINC250k Dataset EDA",
                "caption": "Clinical endpoint coverage and disjoint Bemis-Murcko scaffold split proportions.",
                "url": "/figures/01_dataset_eda.png",
            },
        ],
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
