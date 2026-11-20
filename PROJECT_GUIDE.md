# Master Project Guide & Theoretical Handbook

**Project Title:** Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling and Molecular Lead Scoring  
**Domain:** Artificial Intelligence in Healthcare & Computer-Aided Drug Discovery (CADD)  
**Target Audience:** Team Members, Reviewers, and Faculty Evaluators  

---

## 1. Executive Summary & Core Objective

Developing a new medicine takes **10–15 years** and costs **over $2.6 Billion**, with more than **90% of drug candidates failing** during clinical trials. Over half of clinical failures stem from poor **ADMET** (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties rather than lack of target binding.

This engineering semester project delivers an end-to-end Graph Machine Learning system built on **10,007 standardized compounds** (**5,000** ZINC250k unlabeled molecules + **5,007** real clinical ADMET compounds from Therapeutics Data Commons across **3,137** Bemis-Murcko scaffolds):
1. **Self-Supervised Contrastive Pre-Training (GraphCL/InfoNCE):** Learns chemical valence, aromaticity, and scaffold geometry from 5,000 unlabeled ZINC250k molecules before clinical supervision, with zero scaffold overlap with validation/test splits.
2. **2D+3D Hybrid Local+Global GraphGPS Backbone:** Fuses covalent-bond `TransformerConv` with dense all-atom `MultiheadAttention`, **sign-canonicalized Laplacian PE + RWSE**, and **ETKDGv3 3D conformer geometry** (`NPR1`, `NPR2`, `Asphericity`, `RadiusOfGyration`).
3. **NaN-Masked Multi-Objective ADMET Profiling:** Simultaneously predicts 5 bounded endpoints—Oral Bioavailability ($[0, 100]\%$), BBB Permeability ($[0, 1]$), hERG Cardiotoxicity ($[0, 1]$), QED ($[0, 1]$), and SAScore ($[1, 10]$)—using `BCEWithLogitsLoss` with training-split `pos_weight` and normalized Smooth-L1 regression.
4. **Evolutionary Lead Optimization & Atom-Level XAI:** Explores chemically valid analogues via 23 medicinal chemistry SMIRKS rules, **BRICS fragment recombination**, PAINS/Lipinski filtering, **NSGA-II Pareto sorting**, and **Input × Gradient atom attribution maps**.

---

## 2. Why Traditional Approaches Fail vs. Our Hybrid Architecture

| Challenge | Shortcoming of Early Approaches | Our Solution |
| :--- | :--- | :--- |
| **Clinical Data Scarcity** | Labeled clinical datasets have missing (`NaN`) endpoints and limited sizes; supervised deep models overfit rapidly | **5,000-molecule GraphCL InfoNCE Pre-Training** with undirected-symmetric bond drops, subgraph node drops, atom masking, and PE sign flips |
| **GCN Local Myopia & Oversmoothing** | Standard GCNs only aggregate 1-hop neighbors and ignore bond types and 3D shape | **Hybrid Local+Global GraphGPS** combining local edge-aware `TransformerConv` + global `MultiheadAttention` + **ETKDGv3 3D conformers** + **LapPE & RWSE** |
| **Random-Split Data Leakage** | Random splits place near-identical isomers in both train and test, inflating accuracy | **Bemis-Murcko Disjoint Scaffold Splitting** (4,005 train / 500 val / 502 test) with zero shared scaffolds or SMILES |
| **Single-Metric Optimization** | Arithmetic averages or single-property models hide lethal flaws (e.g., high bioavailability + lethal hERG toxicity) | **Derringer-Suich Geometric Desirability** (with `peripheral`/`cns` BBB control) + **NSGA-II Non-Dominated Pareto Frontier** |

---

## 3. End-to-End Technical Workflow

```
[Input: Standardized Desalted SMILES]
       │
       ▼
[2D+3D RDKit Chemoinformatics Parser]
       │  Extracts: 29-dim Atom Features + 8-dim Bond Features + 8-dim (LapPE + RWSE) + 4-dim 3D Geometry
       ▼
[Hybrid Local+Global GraphGPS Backbone (SSL Pre-Trained on 5,000 ZINC250k Molecules)]
       │  3 Layers: Local TransformerConv + Global All-Atom MultiheadAttention + GELU FFN + 3D Shape Fusion
       ▼
[128-dim Molecular Representation Vector]
       │
       ├──► Head 1: Oral Bioavailability P(F >= 20%)  [0 - 100%]   (Sigmoid * 100)
       ├──► Head 2: BBB Permeability                   [0 - 1 Prob] (Sigmoid)
       ├──► Head 3: hERG Cardiotoxicity Hazard         [0 - 1 Prob] (Sigmoid)
       ├──► Head 4: QED Drug-Likeness                  [0 - 1]      (Sigmoid)
       └──► Head 5: Synthetic Accessibility (SAScore)  [1 - 10]     (1 + 9 * Sigmoid)
       │
       ▼
[Multi-Generation Evolutionary Pareto Optimizer & XAI]
       │  1. Mutates leads across 23 SMIRKS + BRICS recombination + PAINS/Lipinski filters
       │  2. Ranks via Derringer-Suich Desirability & NSGA-II Pareto Fronts
       │  3. Highlights key toxicophores via Atom-Level Gradient Saliency Maps
       ▼
[Output: Verified Non-Dominated Drug Candidates & Interpretable Plots]
```

---

## 4. How to Run the Project (Step-by-Step)

Run all commands from the `course_project/` root directory:

1. **Verify System Integrity (14 Unit & Integration Tests):**
   ```bash
   PYTHONPATH=. ./env/bin/python -m unittest discover -s tests -v
   ```
2. **Run Entire Reproducible Pipeline (Stages 0 → 1 → 2 → 4 → 3):**
   ```bash
   ./env/bin/python scripts/run_all.py
   ```
3. **Or Execute Stages Individually:**
   ```bash
   ./env/bin/python scripts/0_prepare_datasets.py
   ./env/bin/python scripts/1_pretrain_ssl.py --epochs 25 --batch-size 64
   ./env/bin/python scripts/2_finetune_admet.py --init ssl --epochs 40 --batch-size 64
   ./env/bin/python scripts/4_evaluate_and_ablation.py
   ./env/bin/python scripts/3_lead_optimization.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --generations 2 --mode peripheral --save-plots
   ```
4. **Interactive Exploration:**
   Open [notebooks/01_explore_and_demo.ipynb](notebooks/01_explore_and_demo.ipynb) for live interactive execution.

---

## 5. Limitations Resolved & Remaining Future Work

### Original Limitations Now Resolved
- **Real Clinical Scale & Zero-Leakage Splits:** Upgraded from 10 toy rows to **5,007** real TDC compounds + **5,000** ZINC250k SSL molecules evaluated strictly on unseen Bemis-Murcko scaffolds (`0.8299 Macro ROC-AUC` vs `0.7956` without SSL).
- **3D Conformational Awareness:** Incorporated fast **ETKDGv3 3D geometry** (`NPR1`, `NPR2`, `Asphericity`, `RadiusOfGyration`, atom centroid distances, and 3D bond lengths).
- **Expanded Chemical Space & BBB Integration:** Expanded from 8 aromatic rules to **23 warning-free SMIRKS + BRICS fragment recombination**, PAINS/Lipinski filters, evolutionary selection, and `peripheral`/`cns` BBB trade-offs.
- **Model Explainability:** Added Input × Gradient atom attribution maps verifying medicinal chemistry pharmacophores (e.g., tertiary amine hERG toxicophores).

### Remaining Future Work
- Explicit-solvent Molecular Dynamics (MD) ensemble scoring for highly flexible macrocycles and PROTACs.
- SE(3)-equivariant 3D generative diffusion models for unconstrained *de novo* scaffold generation.
- Expanding multi-task heads to CYP450 metabolic inhibition and Ames mutagenicity.
