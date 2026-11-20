# Faculty Presentation Deck & Word-for-Word Speaker Script

**Project Title:** Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling and Molecular Lead Scoring  
**Duration:** 8–10 Minutes (8 Slides)  
**Tone:** Confident, Technically Rigorous, and Empirically Backed  

---

## Slide-by-Slide Overview

| Slide # | Slide Title | Key Theme |
| :---: | :--- | :--- |
| **Slide 1** | Title & Team Introduction | Project identity, problem domain, team details |
| **Slide 2** | Problem Statement & Clinical Motivation | Why 90% of drugs fail; the ADMET bottleneck |
| **Slide 3** | Why Traditional ML & GNNs Fall Short | Data scarcity, GCN myopia, random-split leakage & single-target flaws |
| **Slide 4** | Proposed Architecture: 2D+3D Hybrid GraphGPS | SSL Pre-training + GraphGPS + Masked Multi-Task Heads + Pareto + XAI |
| **Slide 5** | Core Innovation 1: GraphCL SSL & 2D+3D Encodings | Undirected-safe augmentations, LapPE+RWSE, ETKDGv3 3D geometry |
| **Slide 6** | Core Innovation 2: Evolutionary Pareto & XAI | 23 SMIRKS + BRICS, Derringer-Suich, NSGA-II, Atom Saliency XAI |
| **Slide 7** | Quantitative Held-Out Results & Case Study | Unseen-scaffold benchmark (`0.8299` Macro AUC) & Aspirin case study |
| **Slide 8** | Conclusion, Limitations Solved & Future Scope | Summary of deliverables & viva Q&A |

---

### Slide 1: Title & Introduction

#### Slide Content:
- **Title:** Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling and Molecular Lead Scoring
- **Subtitle:** An End-to-End Deep Graph Learning Framework for In-Silico Safety Prediction & Lead Optimization
- **Presented By:** [Your Names / Roll Numbers]
- **Course & Department:** Engineering Semester Machine Learning Course Project | Department of CSE / AI & ML

#### Visual / Diagram Recommendation:
- Clean molecular graph transitioning into a 3D conformer and GraphGPS neural network block.

#### Word-for-Word Speaker Script:
> *"Good morning respected faculty and evaluators. Today we present our engineering semester machine learning project: **Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling and Molecular Lead Scoring**.*  
>  
> *In this project, we address one of the hardest challenges in computational drug discovery: predicting human safety and pharmacokinetic properties directly from molecular structure using **10,007 curated compounds** and optimizing drug leads across conflicting clinical objectives."*

---

### Slide 2: Problem Statement & Clinical Motivation

#### Slide Content:
- **The $2.6 Billion Bottleneck:** Bringing a drug to market takes 10–15 years and >$2.6B, with **>90% clinical trial attrition**.
- **The Primary Culprit:** Poor **ADMET** (Absorption, Distribution, Metabolism, Excretion, Toxicity):
  - *hERG Cardiotoxicity:* Fatal drug-induced cardiac arrhythmia.
  - *Oral Bioavailability ($F \ge 20\%$):* Insufficient systemic drug absorption.
  - *Blood-Brain Barrier (BBB):* Necessary for CNS therapeutics, hazardous side effect for peripheral drugs.
- **Project Goal:** Build an interpretable, *in-silico* multi-task AI pipeline evaluated strictly on **unseen chemical scaffolds**.

#### Visual / Diagram Recommendation:
- Embed [results/figures/01_dataset_eda.png](results/figures/01_dataset_eda.png) showing our 5,007 real TDC clinical compounds + 5,000 ZINC250k SSL compounds across 3,137 Bemis-Murcko scaffolds.

#### Word-for-Word Speaker Script:
> *"More than 90% of candidate drugs fail in clinical trials—not because they miss their target protein, but because of poor ADMET properties such as heart toxicity via hERG channel blockage or poor oral absorption. To build a realistic predictor, we curated **5,007 clinical molecules from Therapeutics Data Commons** alongside **5,000 unlabeled drug-like molecules from ZINC250k**, partitioned via **Bemis-Murcko scaffold splitting** so test molecules belong to completely unseen chemical scaffolds."*

---

### Slide 3: Why Existing ML & GNN Approaches Fall Short

#### Slide Content:
- **1. Sparse & Incomplete Clinical Labels:** Experimental assays contain many `NaN` entries (e.g., 636 Bioavailability vs. 2,899 hERG labels) and strong class imbalance (~77% positive in Bioavailability/BBB).
- **2. Local Myopia & 2D Blindness of GCNs:** Standard GCNs only pass messages over 1-hop covalent bonds, ignore 3D conformer shape, and suffer oversmoothing.
- **3. Random-Split Leakage:** Random splits leak near-identical chemical scaffolds between training and testing.
- **4. Single-Objective Blindness:** Optimizing absorption alone often degrades cardiac safety or synthetic accessibility.

#### Word-for-Word Speaker Script:
> *"Existing models face four major bottlenecks: clinical assay tables are small and sparse with missing values; standard GCNs only see immediate 1-hop neighbors and ignore 3D geometry; random data splits leak identical ring cores into the test set; and optimizing a single metric hides toxic trade-offs."*

---

### Slide 4: Proposed Architecture — 4-Stage Pipeline

#### Slide Content:
1. **Stage 0 — Real-World Curation & Zero-Leakage Splitting:** 5,000 ZINC250k (SSL) + 5,007 TDC compounds split 80/10/10 by Bemis-Murcko scaffold (`4,005 / 500 / 502`).
2. **Stage 1 — Self-Supervised Contrastive Pre-Training (GraphCL / InfoNCE):** 25 epochs on 5,000 unlabeled graphs (`InfoNCE` loss `0.8659 -> 0.0402`).
3. **Stage 2 — 2D+3D Hybrid GraphGPS + Masked Multi-Task Heads:** Local `TransformerConv` + Global `MultiheadAttention` + 5 strictly bounded endpoints with `BCEWithLogitsLoss(pos_weight)` and normalized Smooth-L1 loss.
4. **Stage 3 — Evolutionary Pareto Lead Optimization & XAI:** 23 SMIRKS + BRICS recombination + PAINS/Lipinski + NSGA-II + Atom Saliency.

#### Visual / Diagram Recommendation:
- Embed [results/figures/02_ssl_pretraining.png](results/figures/02_ssl_pretraining.png) showing InfoNCE convergence and zero-shot t-SNE latent space organization.

#### Word-for-Word Speaker Script:
> *"Our pipeline overcomes each bottleneck systematically. First, we pre-train on 5,000 unlabeled ZINC250k molecules via InfoNCE contrastive learning, driving loss from 0.8659 down to 0.0402 and organizing chemical space prior to seeing clinical labels. Next, our Hybrid GraphGPS backbone fuses local bond attention, global all-atom attention, and 3D conformer shape to feed five bounded, NaN-masked ADMET prediction heads."*

---

### Slide 5: Core Innovation 1 — 2D+3D GraphGPS & Undirected-Safe SSL

#### Slide Content:
- **Rich 2D+3D Representation:**
  - **29-dim Atom Features:** Element, degree, formal charge, hybridization, H-count, aromaticity, ring flag, **3D centroid distance**.
  - **8-dim Bond Features:** Bond type, conjugation, ring flag, **3D ETKDGv3 bond length**.
  - **8-dim Structural Encodings:** Sign-canonicalized **Laplacian PE** ($k=4$) + sign-invariant **RWSE** ($k=4$).
  - **4-dim Global 3D Shape:** `NPR1`, `NPR2`, `Asphericity`, `RadiusOfGyration`.
- **Undirected-Symmetric Contrastive Augmentations:** Drops both $(u, v)$ and $(v, u)$ together, performs subgraph node dropping with index remapping, atom feature masking, and random LapPE sign flips.

#### Word-for-Word Speaker Script:
> *"In a Graph-Transformer, molecules have no natural 1D sequence order and eigenvectors have arbitrary signs. We solve this using sign-canonicalized Laplacian eigenvectors paired with sign-invariant Random-Walk Structural Encodings, enriched with fast ETKDGv3 3D conformer geometry. Furthermore, our contrastive augmentations strictly preserve undirected covalent bond symmetry."*

---

### Slide 6: Core Innovation 2 — Evolutionary Pareto Lead Scoring & XAI

#### Slide Content:
- **Chemical Exploration:** 23 warning-free medicinal chemistry SMIRKS rules + **BRICS fragment recombination** + PAINS & Lipinski Rule-of-5 filters across multiple evolutionary generations.
- **Derringer-Suich Geometric Desirability:**
  $$D(x) = \left( d_{\text{bio}}^{w_1} \cdot d_{\text{safety\_herg}}^{w_2} \cdot d_{\text{QED}}^{w_3} \cdot d_{\text{SA}}^{w_4} \cdot d_{\text{BBB}}^{w_5} \right)^{1/\sum w_i}$$
  Supports configurable **`peripheral`** (penalizes CNS exposure) and **`cns`** modes.
- **NSGA-II Non-Dominated Sorting:** Extracts Pareto Front-1 leads where no objective can be improved without harming another.
- **Atom-Level XAI Saliency:** Input × Gradient attribution highlights structural toxicophores.

#### Word-for-Word Speaker Script:
> *"Instead of using an arithmetic average that can mask fatal heart toxicity, we combine the Derringer-Suich geometric mean with NSGA-II Pareto non-dominated sorting across 23 medicinal chemistry reactions and BRICS fragment recombination. We also compute atom-level gradient saliency maps so chemists can inspect exactly which atoms drive toxicity or permeability."*

---

### Slide 7: Quantitative Held-Out Results & Case Study

#### Slide Content:
- **Held-Out Unseen-Scaffold Test Benchmark ($n = 502$, Zero Scaffold Overlap):**

| Model | Bioavail. ROC-AUC | BBB ROC-AUC | hERG ROC-AUC | Macro ROC-AUC | QED $R^2$ (MAE) | SAScore $R^2$ (MAE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF-ECFP4** | 0.7658 | 0.8758 | 0.7794 | **0.8070** | 0.5612 (0.101) | 0.7341 (0.317) |
| **Standard-GCN** | 0.7466 | 0.8945 | 0.7786 | **0.8066** | 0.3699 (0.127) | 0.6044 (0.421) |
| **GraphGPS-Scratch** | 0.7052 | 0.8883 | 0.7931 | **0.7956** | 0.6935 (0.087) | 0.6844 (0.380) |
| **SSL-GraphGPS (Ours)** | **0.7769** | **0.9001** | **0.8128** | **0.8299** | **0.7165 (0.082)** | **0.7584 (0.324)** |

- **Key Finding:** SSL Pre-Training improves Macro ROC-AUC by **+3.43 pp** over `GraphGPS-Scratch` (and **+7.17 pp** on sparse Oral Bioavailability), beating every baseline on every endpoint.
- **Aspirin Evolutionary Case Study (`peripheral` mode, 70 candidates, 45 Pareto Front-1):**
  - Top candidate `CCc1cc(OC(C)=O)c(C(=O)O)cc1CC` achieves Desirability **0.919** (vs. **0.898** for Aspirin), Bioavailability **100.0%**, hERG **0.005**, BBB **0.081** (reduced CNS exposure), and improved RDKit QED **0.644** (vs. **0.550**).

#### Visual / Diagram Recommendation:
- Embed [results/figures/04_roc_comparison.png](results/figures/04_roc_comparison.png), [results/figures/06_pareto_optimization.png](results/figures/06_pareto_optimization.png), and [results/figures/07_atom_saliency_xai.png](results/figures/07_atom_saliency_xai.png).

#### Word-for-Word Speaker Script:
> *"On our strictly unseen-scaffold held-out test set of 502 compounds, SSL-GraphGPS achieves **0.8299 Macro ROC-AUC**, **0.9001** on BBB, **0.8128** on hERG cardiotoxicity, and **0.7769** on Oral Bioavailability—outperforming Random Forest ECFP4, Standard GCN, and GraphGPS trained from scratch across every single task. Self-supervised pre-training alone adds **+3.43 percentage points** in Macro ROC-AUC and **+7.17 points** on data-scarce Bioavailability.*  
>  
> *On the Aspirin optimization case study, two evolutionary generations produced 70 PAINS-free analogues with 45 Pareto Front-1 candidates, boosting desirability from 0.898 to 0.919 while keeping hERG hazard near zero. Finally, our XAI saliency map on an unseen test-set hERG blocker accurately pinpoints the basic tertiary amine nitrogen—matching known medicinal chemistry rules."*

---

### Slide 8: Conclusion, Limitations Solved & Future Scope

#### Slide Content:
- **Key Engineering & Research Contributions:**
  1. Scaled from toy data to **10,007 curated compounds** with zero-leakage Bemis-Murcko splits and 14 passing unit tests.
  2. Built a true **2D+3D Hybrid GraphGPS** backbone combining local bond attention, global attention, and ETKDGv3 conformers.
  3. Demonstrated consistent quantitative superiority over classical ML, GCN, and non-pretrained baselines.
- **Remaining Future Work:**
  - Full explicit-solvent Molecular Dynamics (MD) trajectories for flexible macrocycles.
  - SE(3)-equivariant 3D diffusion models for unconstrained *de novo* scaffold generation.

#### Word-for-Word Speaker Script:
> *"In conclusion, we built, audited, and rigorously benchmarked an end-to-end Self-Supervised 2D+3D GraphGPS pipeline for ADMET profiling and lead optimization. In the future, we plan to extend this foundation to explicit-solvent molecular dynamics and SE(3)-equivariant chemical diffusion. Thank you—we are now ready for your questions."*

---

## Anticipated Faculty Viva Questions & Answers

#### Q1: Why did you evaluate using Bemis-Murcko scaffold splitting instead of a standard random 80/20 split?
> **Answer:** *"In drug discovery, molecule datasets contain many close structural analogues sharing the same ring core. A random split places nearly identical compounds in both training and test sets, letting models memorize scaffolds rather than generalize. Bemis-Murcko scaffold splitting guarantees that every validation and test compound has a completely unseen ring-and-linker backbone, measuring true out-of-distribution generalization."*

#### Q2: How does your model train on a multi-task dataset where many experimental endpoints are `NaN`?
> **Answer:** *"Clinically, almost no compound is tested across every assay simultaneously—for example, we have 636 Bioavailability labels vs. 2,899 hERG labels across 5,007 compounds. Our `MolecularDataset` builds a per-sample boolean mask `y_mask`, and `compute_masked_multitask_loss` computes `BCEWithLogitsLoss` (with training-split `pos_weight`) and Smooth-L1 regression strictly on observed entries, allowing all 5,007 compounds to share structural representations without label imputation."*

#### Q3: What makes GraphGPS better than a standard GCN or standalone `TransformerConv`?
> **Answer:** *"A standard GCN or `TransformerConv` only propagates information between directly bonded 1-hop neighbors, so distant functional groups need many hops to interact and suffer from oversmoothing. Our Hybrid GraphGPS layer runs edge-aware local `TransformerConv` in parallel with dense global `MultiheadAttention` across every atom in the molecule, enriched with sign-canonicalized Laplacian PE, RWSE, and 3D ETKDGv3 shape descriptors—yielding a +2.33 pp Macro ROC-AUC gain over Standard-GCN and +0.347 QED R² improvement."*

#### Q4: Why use the Derringer-Suich geometric mean instead of an arithmetic weighted average?
> **Answer:** *"An arithmetic average can mask fatal flaws: a drug with 100% bioavailability and 0.0 safety (100% cardiotoxicity) averages to 0.50. The geometric mean multiplies desirabilities together, forcing the composite score toward zero whenever any single critical safety or synthesizability constraint fails."*

#### Q5: How do we trust the model's hERG or BBB predictions chemically?
> **Answer:** *"Using Input × Gradient atom-level attribution (`compute_atom_saliency`), we visualize exactly which atoms contribute most to a prediction. On an unseen test-set hERG blocker (`Pred hERG = 0.996`), the model highlights the basic tertiary amine nitrogen and adjacent aromatic linker—aligning directly with established medicinal chemistry hERG channel pharmacophores."*
