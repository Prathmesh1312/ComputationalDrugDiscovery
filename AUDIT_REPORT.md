# Codebase Audit Report: Logical, Mathematical & Engineering Fixes

This report documents all **13 logical, chemical, architectural, and machine-learning methodology issues** discovered during a full audit of the original `course_project` codebase, along with the exact fixes implemented and verified.

---

## Summary Table

| # | Severity | Component | Issue Found in Original Codebase | Fix Implemented |
| :-: | :--- | :--- | :--- | :--- |
| 1 | Critical | `data/` | Tiny toy datasets (20 unlabeled SMILES, 10 labeled rows) | Curated **5,000** ZINC250k unlabeled molecules + **5,007** real TDC clinical compounds |
| 2 | Critical | `scripts/2_finetune_admet.py` | 100% data leakage (trained & reported loss on all 10 rows; no split) | Implemented **Bemis-Murcko Scaffold Split** (80/10/10) with zero-leakage assertions |
| 3 | High | `scripts/1_pretrain_ssl.py` | Unshuffled mini-batches (`range(0, N, B)`) gave static negative pairs every epoch | Switched to `DataLoader(..., shuffle=True, drop_last=True)` + `CosineAnnealingLR` |
| 4 | High | Evaluation & Ablations | Zero held-out metrics, no baselines, no SSL-vs-Scratch ablation | Added `metrics.py`, `StandardGCN`, `RF-ECFP4`, `GraphGPS-Scratch`, and `4_evaluate_and_ablation.py` |
| 5 | High | Training Hygiene | No RNG seeds, LR scheduler, grad clipping, or val checkpointing | Added `seed.py`, `CosineAnnealingLR`, `clip_grad_norm_(1.0)`, early stopping & best val checkpointing |
| 6 | Critical | `scripts/2_finetune_admet.py` | Post-sigmoid `F.binary_cross_entropy`, no `pos_weight`, ~100x loss scale imbalance, no `NaN` mask | Replaced with `BCEWithLogitsLoss(pos_weight=...)`, normalized Smooth-L1 regression, and `NaN` mask |
| 7 | High | `src/data/augmentations.py` | `drop_edges` broke undirected symmetry (~25% one-way edges); `drop_nodes` advertised but unimplemented | Fixed canonical `(min(u,v), max(u,v))` undirected bond masking + implemented `drop_nodes` + PE sign flip |
| 8 | High | `src/data/dataset.py` | Laplacian PE sign ambiguity; `.[Cl]` salts created disconnected graphs; missing charge/hybridization/ring/3D features; dead `MolecularDataset` | Added `LargestFragmentChooser` desalting, 29-dim atom + 8-dim bond + 3D ETKDGv3 geometry, sign-canonicalized LapPE + RWSE, and cached `MolecularDataset` |
| 9 | High | `src/models/graph_transformer.py` | Claimed global all-atom attention, but `TransformerConv` only attended 1-hop neighbors | Upgraded to **Hybrid Local+Global GraphGPS** (`TransformerConv` + global `nn.MultiheadAttention`) + 3D shape fusion |
| 10 | Medium | `src/models/multi_objective_heads.py` | `F.softplus(...) + 1.0` could exceed `10.0` for SAScore; single linear heads; unsafe `torch.load` | Fixed to `1.0 + 9.0 * sigmoid(...)` (`[1, 10]` strict), 2-layer MLP task heads, raw logits, and `weights_only=True` |
| 11 | Medium | `src/models/ssl_contrastive.py` | Hardcoded `temperature=0.1` ignored; no `BatchNorm1d` in projector; fixed `-1e9` mask | Exposed `temperature` parameter, added `BatchNorm1d`, dtype-safe masking, and forwarded `geom_3d` |
| 12 | Medium | `src/optimization/` | Unmapped-product SMIRKS emitted 8 RDKit stderr warnings; broken inner `break`; narrow aromatic-only SMIRKS; BBB ignored in scorer | Cleaned SMIRKS (0 warnings), expanded to 23 medchem SMIRKS + BRICS recombination + PAINS/Lipinski + evolutionary loop + CNS/Peripheral BBB modes |
| 13 | Medium | Deliverables | Empty `src/utils/`; broken `course_project/...` README paths; stale toy Slide 7; no XAI; no report | Populated `src/utils/` (7 modules), added Atom Saliency XAI, rebuilt notebook, updated docs, and wrote `PROJECT_REPORT.md` |

---

## Detailed Findings & Resolutions

### 1. Toy Dataset Bottleneck (`data/`)
- **Original State:** `unlabeled_sample_smiles.csv` contained 20 rows and `admet_multitask_sample.csv` contained 10 rows (Aspirin, Ibuprofen, Paracetamol, etc.).
- **Resolution:** Built `scripts/0_prepare_datasets.py` curating real clinical data from **Therapeutics Data Commons (TDC)** (`Bioavailability_Ma`, `BBB_Martins`, `hERG` Wang, stratified `hERG_Karim`) and **ZINC250k**. Final standardized corpora: **5,000** unlabeled SSL molecules and **5,007** multi-task ADMET molecules across **3,137** unique Murcko scaffolds.

### 2. Zero-Split Data Leakage (`scripts/2_finetune_admet.py`)
- **Original State:** Loaded all 10 rows into a single `Batch.from_data_list(graphs)` and trained/evaluated on the exact same rows without validation or test splits.
- **Resolution:** Implemented `bemis_murcko_scaffold_split` in `src/utils/splitting.py` (80% train / 10% val / 10% test = 4,005 / 500 / 502 compounds) with explicit assertions confirming zero shared scaffolds and zero shared canonical SMILES across splits, plus zero overlap between ZINC250k SSL molecules and ADMET val/test scaffolds.

### 3. Unshuffled SSL Mini-Batches (`scripts/1_pretrain_ssl.py`)
- **Original State:** `for i in range(0, len(graphs), batch_size)` iterated deterministically in fixed order, producing identical negative pairs across every epoch.
- **Resolution:** Replaced with `DataLoader(..., shuffle=True, drop_last=True)` + `CosineAnnealingLR` + gradient norm clipping (`1.0`).

### 4. Missing Quantitative Evaluation & Ablations
- **Original State:** No held-out evaluation script, no ROC-AUC/PR-AUC/F1/R²/MAE metrics, and no baselines to prove whether SSL or Graph-Transformers helped.
- **Resolution:** Implemented `src/utils/metrics.py`, `src/models/baselines.py` (`MorganFingerprintRFBaseline`, `StandardGCNBackbone`), and `scripts/4_evaluate_and_ablation.py` comparing all 4 models identically on the unseen-scaffold test set.

### 5. Training Hygiene & Reproducibility
- **Original State:** Neither `1_pretrain_ssl.py` nor `2_finetune_admet.py` set deterministic RNG seeds, applied learning-rate decay or gradient clipping, or used mini-batch `DataLoader` + early-stopping validation checkpoints during fine-tuning.
- **Resolution:** Added `src/utils/seed.py` (`set_seed(42)` across Python/NumPy/PyTorch/MPS), `CosineAnnealingLR`, `clip_grad_norm_(..., 1.0)`, mini-batch PyG `DataLoader` (`batch_size=64`), discriminative backbone LR (`0.5 * lr` for SSL initialization), and best-validation early-stopping checkpoint selection.

### 6. Loss Stability, Class Imbalance & Missing-Label Masking (`multi_objective_heads.py`)
- **Original State:** Called `F.binary_cross_entropy` on post-sigmoid outputs (numerically unstable), omitted `pos_weight` despite ~77% positive imbalance in Bioavailability and BBB, multiplied raw Bioavailability MSE `[0..100]^2` by `0.01` (still dominating ~100x over BCE and QED), and crashed on `NaN` targets.
- **Resolution:** Built `compute_masked_multitask_loss` using `F.binary_cross_entropy_with_logits` with training-split `pos_weight` clamping (`[0.5, 4.0]`), `NaN` boolean masks, and Smooth-L1 loss on `QED` and `[0, 1]`-normalized `SAScore` (`(sa - 1) / 9`).

### 7. Broken Undirected Graph Augmentations (`src/data/augmentations.py`)
- **Original State:** `drop_edges` sampled `torch.rand(num_edges) < keep_prob` independently per directed edge `(i,j)` and `(j,i)`, leaving ~25% of dropped bonds unidirectional despite its docstring claiming *"preserving bidirectionality"*. Its docstring also advertised Node Dropping, which did not exist.
- **Resolution:** Rewrote `drop_edges` to sample one keep flag per canonical undirected edge `(min(u,v), max(u,v))`, implemented true `drop_nodes` with edge filtering and tensor index remapping, and added `random_pe_sign_flip`.

### 8. Graph Featurization & Positional Encoding Fixes (`src/data/dataset.py`)
- **Original State:** Laplacian PE sliced `[:, 1:k+1]` without filtering zero eigenvalues (fragile on multi-component salts like `.[Cl]`), suffered arbitrary eigenvector sign ambiguity, omitted formal charge / hybridization / ring flags / 3D conformer geometry, and left `MolecularDataset` unused.
- **Resolution:** Added RDKit `LargestFragmentChooser` desalting, 29-dim atom + 8-dim bond features with **ETKDGv3 3D conformer geometry**, sign-canonicalized LapPE (`e_vals > 1e-5` filter + max-abs positive sign rule) concatenated with sign-invariant **RWSE**, and multiprocess-cached `MolecularDataset`.

### 9. True Global Self-Attention Architecture (`src/models/graph_transformer.py`)
- **Original State:** Documentation described global all-atom self-attention, but `TransformerConv` only computes attention over 1-hop `edge_index` neighbors.
- **Resolution:** Upgraded `GraphTransformerLayer` to a **Hybrid Local + Global GraphGPS** block combining local edge-aware `TransformerConv` with dense batched all-atom `nn.MultiheadAttention` (`to_dense_batch`), fused with 3D global shape descriptors (`NPR1, NPR2, Asphericity, RadiusOfGyration`).

### 10. Physical Bound Enforcement (`src/models/multi_objective_heads.py`)
- **Original State:** Predicted SAScore via `F.softplus(raw) + 1.0`, which is unbounded above `[1, 10]` and could output invalid scores `> 10`.
- **Resolution:** Constrained `sascore = 1.0 + 9.0 * torch.sigmoid(sa_raw)` strictly within `[1.0, 10.0]`, added 2-layer task-specific adapter MLPs, and switched checkpoint loading to `weights_only=True`.

### 11. Contrastive Loss & Projector Refinements (`src/models/ssl_contrastive.py`)
- **Original State:** `MolecularContrastiveSSL` hardcoded `temperature=0.1` instead of taking a parameter, lacked `BatchNorm1d` in the projector, and masked with fixed `-1e9`.
- **Resolution:** Parameterized `temperature`, added `BatchNorm1d` to prevent representation collapse, used dtype-safe masking (`-1e4` for FP16 / `-1e9` for FP32), and forwarded `geom_3d`.

### 12. Optimization, Warning Hygiene & BBB Integration (`src/optimization/`)
- **Original State:** Reaction SMIRKS mapped leaving hydrogens (`[c:1][H:2]>>[c:1]F`), emitting 8 RDKit warnings per run; `break` exited only the inner loop; mutations covered only 8 rules (7 aromatic); `pareto_scorer.py` ignored predicted BBB; `3_lead_optimization.py` ran inference molecule-by-molecule; and `ParetoFrontier.is_pareto_optimal` used an order-dependent loop.
- **Resolution:** Unmapped leaving atoms (0 RDKit warnings), expanded to **23 SMIRKS + BRICS fragment recombination**, added PAINS & Lipinski filtering + Tanimoto similarity constraints + multi-generation `evolve_leads`, added configurable `peripheral`/`cns` BBB objectives + NSGA-II non-dominated peeling + order-independent dominance check, and batched inference in `3_lead_optimization.py`.

### 13. Engineering Deliverables & Explainability
- **Original State:** `src/utils/` was empty, `README.md` had broken `course_project/...` relative commands, `PRESENTATION_DECK.md` contained a stale toy-data disclaimer, `notebooks/01_explore_and_demo.ipynb` showed old toy results and an empty trailing cell, and no XAI or formal report existed.
- **Resolution:** Populated 7 modules in `src/utils/`, implemented atom-level gradient attribution XAI (`explainability.py`), generated 7 publication figures in `results/figures/`, rebuilt and executed `notebooks/01_explore_and_demo.ipynb`, created `PROJECT_REPORT.md`, and updated `README.md`, `PROJECT_GUIDE.md`, and `PRESENTATION_DECK.md` with verified benchmark numbers.
