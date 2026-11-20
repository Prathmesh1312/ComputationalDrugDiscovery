# Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling and Molecular Lead Scoring

An end-to-end Deep Graph Learning & Computational Drug Discovery (CADD) engineering semester project combining **Self-Supervised Contrastive Pre-Training (GraphCL / InfoNCE)**, a **Hybrid Local+Global 2D+3D GraphGPS Transformer**, **Masked Multi-Task ADMET Profiling**, **Evolutionary SMIRKS + BRICS Pareto Lead Optimization**, and **Atom-Level Explainability (XAI)**.

---

## Key Highlights & Verified Held-Out Results

1. **Real-World Curated Corpora (`10,007` Standardized Compounds):**
   - **5,000** unlabeled drug-like ZINC250k molecules for SSL pre-training (zero SMILES or scaffold overlap with ADMET validation/test splits).
   - **5,007** clinical multi-task ADMET compounds across **3,137** unique Bemis-Murcko scaffolds curated from Therapeutics Data Commons (`Bioavailability_Ma`, `BBB_Martins`, `hERG` Wang + `hERG_Karim`, plus exact RDKit `QED` and `SAScore`).
   - **Strict Zero-Leakage Scaffold Split:** Train (`4,005`), Val (`500`), Test (`502`).
2. **2D+3D Hybrid Local+Global GraphGPS Architecture:**
   - 29-dim atom features + 8-dim bond features + **ETKDGv3 3D conformer geometry** (`NPR1`, `NPR2`, `Asphericity`, `RadiusOfGyration`) + sign-canonicalized **Laplacian PE** & **RWSE**.
   - Parallel local covalent `TransformerConv` and dense all-atom `MultiheadAttention`.
3. **Held-Out Unseen-Scaffold Benchmark (`n = 502`):**

| Model | Bioavail. ROC-AUC | BBB ROC-AUC | hERG ROC-AUC | Macro ROC-AUC | QED $R^2$ (MAE) | SAScore $R^2$ (MAE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF-ECFP4** | 0.7658 | 0.8758 | 0.7794 | **0.8070** | 0.5612 (0.101) | 0.7341 (0.317) |
| **Standard-GCN** | 0.7466 | 0.8945 | 0.7786 | **0.8066** | 0.3699 (0.127) | 0.6044 (0.421) |
| **GraphGPS-Scratch** | 0.7052 | 0.8883 | 0.7931 | **0.7956** | 0.6935 (0.087) | 0.6844 (0.380) |
| **SSL-GraphGPS (Ours)** | **0.7769** | **0.9001** | **0.8128** | **0.8299** | **0.7165 (0.082)** | **0.7584 (0.324)** |

---

## Project Structure

```
course_project/
├── README.md                          # Quickstart, benchmark summary & usage
├── PROJECT_REPORT.md                  # Full Engineering Semester Project Report
├── AUDIT_REPORT.md                    # 13-point code, math & ML audit report
├── PROJECT_GUIDE.md                   # Theoretical handbook
├── PRESENTATION_DECK.md               # Slide deck & faculty viva Q&A
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/unlabeled_molecules.csv    # 5,000 leakage-free ZINC250k SMILES
│   └── processed/                     # admet_multitask_dataset.csv, dataset_stats.json & graph cache
├── checkpoints/                       # Pre-trained SSL & fine-tuned ADMET/ablation weights
├── results/                           # metrics_summary.json, benchmark_table.md, figures/ (01-07)
├── src/
│   ├── data/                          # 2D+3D graph featurization & undirected-safe augmentations
│   ├── models/                        # GraphGPS, InfoNCE SSL, masked heads, GCN & RF-ECFP4 baselines
│   ├── optimization/                  # 23 SMIRKS + BRICS evolutionary mutator & NSGA-II Pareto engine
│   └── utils/                         # Seed, chemistry/3D, scaffold split, metrics, XAI, visualization
├── scripts/
│   ├── 0_prepare_datasets.py          # Download, standardize, merge & scaffold-split TDC + ZINC250k
│   ├── 1_pretrain_ssl.py              # Stage 1: Contrastive SSL pre-training + t-SNE
│   ├── 2_finetune_admet.py            # Stage 2: Masked multi-task fine-tuning (ssl/scratch/gcn)
│   ├── 3_lead_optimization.py         # Stage 3: Evolutionary Pareto optimization & XAI saliency
│   ├── 4_evaluate_and_ablation.py     # Stage 4: Held-out benchmark & publication plots
│   └── run_all.py                     # End-to-end reproducible pipeline runner
├── notebooks/
│   └── 01_explore_and_demo.ipynb      # Interactive 6-part demonstration & XAI notebook
└── tests/
    └── test_course_project.py         # 14-test unit & integration suite
```

---

## Execution Pipeline

Run all commands from the `course_project/` root directory using the project virtual environment (`./env/bin/python` or your active Python 3.12 environment):

```bash
# 1. Run all 14 unit & integration tests
PYTHONPATH=. ./env/bin/python -m unittest discover -s tests -v

# 2. Run the full end-to-end pipeline (Stages 0 -> 1 -> 2 -> 4 -> 3)
./env/bin/python scripts/run_all.py

# Or run individual stages:
./env/bin/python scripts/0_prepare_datasets.py
./env/bin/python scripts/1_pretrain_ssl.py --epochs 25 --batch-size 64
./env/bin/python scripts/2_finetune_admet.py --init ssl --epochs 40 --batch-size 64
./env/bin/python scripts/4_evaluate_and_ablation.py
./env/bin/python scripts/3_lead_optimization.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --generations 2 --mode peripheral --save-plots
```
