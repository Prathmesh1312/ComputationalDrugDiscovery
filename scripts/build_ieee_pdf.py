"""
build_ieee_pdf.py — Compile Authentic 6–7 Page Two-Column IEEE Conference PDF via Headless Chrome
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_DIR = os.path.join(BASE_DIR, "paper")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _fig_uri(filename: str) -> str:
    return "file://" + os.path.join(FIG_DIR, filename)


def build_html() -> str:
    with open(os.path.join(BASE_DIR, "data", "processed", "dataset_stats.json"), "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "metrics_summary.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "ssl_history.json"), "r", encoding="utf-8") as f:
        ssl_hist = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "lead_opt_results.json"), "r", encoding="utf-8") as f:
        lead_opt = json.load(f)

    ours = metrics["SSL-GraphGPS (Ours)"]
    scratch = metrics["GraphGPS-Scratch"]
    gcn = metrics["Standard-GCN"]
    rf = metrics["RF-ECFP4"]

    def _row(name: str, m: dict, bold: bool = False) -> str:
        cls = ' class="ours-row"' if bold else ""
        b_open = "<b>" if bold else ""
        b_close = "</b>" if bold else ""
        return (
            f"<tr{cls}><td>{b_open}{name}{b_close}</td>"
            f"<td>{b_open}{m['bioavailability']['roc_auc']:.4f}{b_close}</td>"
            f"<td>{b_open}{m['bbb']['roc_auc']:.4f}{b_close}</td>"
            f"<td>{b_open}{m['herg']['roc_auc']:.4f}{b_close}</td>"
            f"<td>{b_open}{m['aggregate']['macro_roc_auc']:.4f}{b_close}</td>"
            f"<td>{b_open}{m['qed']['r2']:.4f} ({m['qed']['mae']:.3f}){b_close}</td>"
            f"<td>{b_open}{m['sascore']['r2']:.4f} ({m['sascore']['mae']:.3f}){b_close}</td>"
            f"<td>{b_open}{m['aggregate']['composite_val_score']:.4f}{b_close}</td></tr>"
        )

    def _cls_detail_row(name: str, m: dict, bold: bool = False) -> str:
        cls = ' class="ours-row"' if bold else ""
        b_open = "<b>" if bold else ""
        b_close = "</b>" if bold else ""
        return (
            f"<tr{cls}><td>{b_open}{name}{b_close}</td>"
            f"<td>{b_open}{m['bioavailability']['pr_auc']:.3f} / {m['bioavailability']['f1']:.3f}{b_close}</td>"
            f"<td>{b_open}{m['bbb']['pr_auc']:.3f} / {m['bbb']['f1']:.3f}{b_close}</td>"
            f"<td>{b_open}{m['herg']['pr_auc']:.3f} / {m['herg']['f1']:.3f}{b_close}</td>"
            f"<td>{b_open}{m['qed']['pearson_r']:.3f} / {m['sascore']['pearson_r']:.3f}{b_close}</td></tr>"
        )

    top_cands = lead_opt["top_candidates"][:5]
    cand_names = ["#1 Di-Et", "#2 4-Et", "#3 4-Me", "#4 Me,Cl", "#5 4-Cl"]
    lead_rows = ""
    for cname, c in zip(cand_names, top_cands):
        lead_rows += (
            f"<tr><td><b>{cname}</b></td>"
            f"<td>{c['bioavailability']:.1f}</td>"
            f"<td>{c['herg']:.3f}</td>"
            f"<td>{c['bbb']:.3f}</td>"
            f"<td>{c['qed']:.2f} / {c['qed_rdkit']:.2f}</td>"
            f"<td>{c['sascore']:.2f} / {c['sascore_rdkit']:.2f}</td>"
            f"<td><b>{c['desirability']:.3f}</b></td></tr>"
        )

    references = [
        'J. A. DiMasi, H. G. Grabowski, and R. W. Hansen, &ldquo;Innovation in the pharmaceutical industry: New estimates of R&amp;D costs,&rdquo; <i>J. Health Econ.</i>, vol. 47, pp. 20&ndash;33, 2016.',
        'M. J. Waring <i>et al.</i>, &ldquo;An analysis of the attrition of drug candidates from four major pharmaceutical companies,&rdquo; <i>Nat. Rev. Drug Discov.</i>, vol. 14, no. 7, pp. 475&ndash;486, 2015.',
        'K. Huang <i>et al.</i>, &ldquo;Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development,&rdquo; in <i>Proc. NeurIPS Datasets &amp; Benchmarks</i>, 2021.',
        'C.-Y. Ma, S.-Y. Yang, H. Zhang, M.-L. Xiang, Q. Huang, and Y.-Q. Wei, &ldquo;Prediction models of human plasma protein binding rate and oral bioavailability derived by using GA&ndash;CG&ndash;SVM method,&rdquo; <i>J. Pharm. Biomed. Anal.</i>, vol. 47, no. 4&ndash;5, pp. 677&ndash;682, 2008.',
        'I. F. Martins, A. L. Teixeira, L. Pinheiro, and A. O. Falc&atilde;o, &ldquo;A Bayesian approach to in silico blood-brain barrier penetration modeling,&rdquo; <i>J. Chem. Inf. Model.</i>, vol. 52, no. 6, pp. 1686&ndash;1697, 2012.',
        'S. Wang, H. Sun, H. Liu, D. Li, Y. Li, and T. Hou, &ldquo;ADMET evaluation in drug discovery. 16. Predicting hERG blockers by combining multiple pharmacophores and machine learning approaches,&rdquo; <i>Mol. Pharm.</i>, vol. 13, no. 8, pp. 2855&ndash;2866, 2016.',
        'A. Karim, M. Lee, T. Balle, and A. Sattar, &ldquo;CardioTox net: a robust predictor for hERG channel blockade based on deep learning meta-feature ensembles,&rdquo; <i>J. Cheminform.</i>, vol. 13, no. 1, Art. no. 60, 2021.',
        'R. G&oacute;mez-Bombarelli <i>et al.</i>, &ldquo;Automatic chemical design using a data-driven continuous representation of molecules,&rdquo; <i>ACS Cent. Sci.</i>, vol. 4, no. 2, pp. 268&ndash;276, 2018.',
        'T. Sterling and J. J. Irwin, &ldquo;ZINC 15 &mdash; Ligand discovery for everyone,&rdquo; <i>J. Chem. Inf. Model.</i>, vol. 55, no. 11, pp. 2324&ndash;2337, 2015.',
        'G. W. Bemis and M. A. Murcko, &ldquo;The properties of known drugs. 1. Molecular frameworks,&rdquo; <i>J. Med. Chem.</i>, vol. 39, no. 15, pp. 2887&ndash;2893, 1996.',
        'G. R. Bickerton, G. V. Paolini, J. Besnard, S. Muresan, and A. L. Hopkins, &ldquo;Quantifying the chemical beauty of drugs,&rdquo; <i>Nat. Chem.</i>, vol. 4, no. 2, pp. 90&ndash;98, 2012.',
        'P. Ertl and A. Schuffenhauer, &ldquo;Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions,&rdquo; <i>J. Cheminform.</i>, vol. 1, no. 1, Art. no. 8, 2009.',
        'S. Riniker and G. A. Landrum, &ldquo;Better informed distance geometry: Using what we know to improve conformation generation,&rdquo; <i>J. Chem. Inf. Model.</i>, vol. 55, no. 12, pp. 2562&ndash;2574, 2015.',
        'D. Rogers and M. Hahn, &ldquo;Extended-connectivity fingerprints,&rdquo; <i>J. Chem. Inf. Model.</i>, vol. 50, no. 5, pp. 742&ndash;754, 2010.',
        'L. Breiman, &ldquo;Random forests,&rdquo; <i>Mach. Learn.</i>, vol. 45, no. 1, pp. 5&ndash;32, 2001.',
        'T. N. Kipf and M. Welling, &ldquo;Semi-supervised classification with graph convolutional networks,&rdquo; in <i>Proc. ICLR</i>, 2017.',
        'Y. Shi, Z. Huang, S. Feng, H. Zhong, W. Wang, and Y. Sun, &ldquo;Masked label prediction: Unified message passing model for semi-supervised classification,&rdquo; in <i>Proc. IJCAI</i>, 2021, pp. 1548&ndash;1554.',
        'L. Ramp&aacute;&scaron;ek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini, &ldquo;Recipe for a general, powerful, scalable graph transformer,&rdquo; in <i>Proc. NeurIPS</i>, 2022.',
        'V. P. Dwivedi, A. T. Luu, T. Laurent, Y. Bengio, and X. Bresson, &ldquo;Graph neural networks with learnable structural and positional representations,&rdquo; in <i>Proc. ICLR</i>, 2022.',
        'Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, &ldquo;Graph contrastive learning with augmentations,&rdquo; in <i>Proc. NeurIPS</i>, 2020.',
        'T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, &ldquo;A simple framework for contrastive learning of visual representations,&rdquo; in <i>Proc. ICML</i>, 2020, pp. 1597&ndash;1607.',
        'Y. Wang, J. Wang, Z. Cao, and A. Barati Farimani, &ldquo;Molecular contrastive learning of representations via graph neural networks,&rdquo; <i>Nat. Mach. Intell.</i>, vol. 4, no. 3, 2022.',
        'J. Degen, C. Wegscheid-Gerlach, A. Zaliani, and M. Rarey, &ldquo;On the art of compiling and using &lsquo;drug-like&rsquo; chemical fragment spaces,&rdquo; <i>ChemMedChem</i>, vol. 3, no. 10, pp. 1503&ndash;1507, 2008.',
        'J. B. Baell and G. A. Holloway, &ldquo;New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries and for their exclusion in bioassays,&rdquo; <i>J. Med. Chem.</i>, vol. 53, no. 7, pp. 2719&ndash;2740, 2010.',
        'C. A. Lipinski, F. Lombardo, B. W. Dominy, and P. J. Feeney, &ldquo;Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings,&rdquo; <i>Adv. Drug Deliv. Rev.</i>, vol. 46, no. 1&ndash;3, pp. 3&ndash;26, 2001.',
        'G. Derringer and R. Suich, &ldquo;Simultaneous optimization of several response variables,&rdquo; <i>J. Qual. Technol.</i>, vol. 12, no. 4, pp. 214&ndash;219, 1980.',
        'K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, &ldquo;A fast and elitist multiobjective genetic algorithm: NSGA-II,&rdquo; <i>IEEE Trans. Evol. Comput.</i>, vol. 6, no. 2, pp. 182&ndash;197, 2002.',
        'A. Shrikumar, P. Greenside, and A. Kundaje, &ldquo;Learning important features through propagating activation differences,&rdquo; in <i>Proc. ICML</i>, PMLR 70, 2017, pp. 3145&ndash;3153.',
    ]
    ref_html = "\n".join(
        f'<div class="ref-item"><span class="ref-num">[{i}]</span><span>{txt}</span></div>'
        for i, txt in enumerate(references, 1)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IEEE Paper — Self-Supervised 2D+3D Hybrid GraphGPS for ADMET Profiling</title>
<style>
  @page {{
    size: letter;
    margin: 0.72in 0.64in 0.72in 0.64in;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 9.8pt;
    line-height: 1.31;
    color: #111111;
    margin: 0;
    padding: 0;
  }}
  .header-block {{
    text-align: center;
    margin-bottom: 12pt;
  }}
  .paper-title {{
    font-size: 19pt;
    font-weight: bold;
    line-height: 1.15;
    margin: 0 0 8pt 0;
    color: #000000;
  }}
  .author-name {{
    font-size: 11pt;
    font-weight: normal;
    margin: 0 0 2pt 0;
  }}
  .author-affil {{
    font-size: 9.2pt;
    font-style: italic;
    color: #333333;
    margin: 0;
  }}
  .two-col {{
    column-count: 2;
    column-gap: 0.24in;
    text-align: justify;
    hyphens: auto;
  }}
  .abstract-box {{
    margin-bottom: 9pt;
    font-size: 9.1pt;
    line-height: 1.28;
  }}
  .abstract-box b {{
    font-style: italic;
  }}
  .keywords {{
    margin-bottom: 10pt;
    font-size: 8.9pt;
  }}
  .keywords b {{
    font-style: italic;
  }}
  h2 {{
    font-size: 10.2pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin: 11pt 0 5pt 0;
    break-after: avoid;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 9.6pt;
    font-weight: bold;
    font-style: italic;
    text-align: left;
    margin: 8pt 0 3.5pt 0;
    break-after: avoid;
    page-break-after: avoid;
  }}
  p {{
    margin: 0 0 5.5pt 0;
    text-indent: 1.15em;
  }}
  p.noindent {{
    text-indent: 0;
  }}
  ul, ol {{
    margin: 3pt 0 6pt 0;
    padding-left: 14pt;
  }}
  li {{
    margin-bottom: 3pt;
  }}
  .eq-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 5pt 0 6pt 0;
    font-size: 9.2pt;
    break-inside: avoid;
  }}
  .eq-math {{
    flex: 1;
    text-align: center;
    font-style: italic;
  }}
  .eq-num {{
    width: 24pt;
    text-align: right;
  }}
  .span-all {{
    column-span: all;
    margin: 7pt 0 9pt 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  figure {{
    margin: 7pt 0 8pt 0;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  figure img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }}
  figcaption {{
    font-size: 8.2pt;
    line-height: 1.22;
    text-align: justify;
    margin-top: 3.5pt;
    color: #1f2937;
  }}
  .table-wrap {{
    margin: 7pt 0 8pt 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  .table-caption {{
    font-size: 8.4pt;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.2px;
    margin-bottom: 3.5pt;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.0pt;
    line-height: 1.22;
    border-top: 1.4pt solid #000000;
    border-bottom: 1.4pt solid #000000;
  }}
  th {{
    border-bottom: 0.8pt solid #000000;
    padding: 3pt 3.5pt;
    text-align: center;
    font-weight: bold;
    background: #ffffff;
  }}
  th:first-child, td:first-child {{
    text-align: left;
  }}
  td {{
    padding: 2.5pt 3.5pt;
    text-align: center;
  }}
  tr.mid-rule td {{
    border-top: 0.6pt solid #444444;
  }}
  tr.ours-row {{
    background-color: #f3f7ff;
  }}
  code {{
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
  }}
  .ref-list {{
    font-size: 7.9pt;
    line-height: 1.24;
    text-align: left;
  }}
  .ref-item {{
    display: flex;
    margin-bottom: 2.4pt;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  .ref-num {{
    width: 19pt;
    flex-shrink: 0;
  }}
</style>
</head>
<body>

<div class="header-block">
  <div class="paper-title">Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling, Explainable Safety Scoring, and Evolutionary Lead Optimization</div>
  <div class="author-name">Prathmesh Toke</div>
  <div class="author-affil">Department of Computer Science and Engineering &mdash; Computational Drug Discovery &amp; Graph Representation Learning<br>Email: tokeprathmesh54321@gmail.com</div>
</div>

<div class="two-col">

<div class="abstract-box">
<b>Abstract&mdash;</b>Pre-clinical drug discovery exhibits over a 90% attrition rate, predominantly driven by poor absorption, distribution, metabolism, excretion, and toxicity (ADMET) profiles such as fatal hERG cardiotoxicity and low oral bioavailability. Training supervised graph neural networks (GNNs) on clinical assays is severely constrained by sparse, imbalanced, and incomplete endpoint tables, while standard message-passing GNNs suffer from 1-hop local myopia, oversmoothing, and insensitivity to three-dimensional (3D) conformation. Furthermore, naive random dataset splits leak near-identical chemical scaffolds across training and test sets, yielding inflated generalization estimates. To address these bottlenecks, we introduce an end-to-end Self-Supervised 2D+3D Hybrid GraphGPS framework evaluated on 10,007 standardized compounds: {stats['unlabeled_ssl_count']:,} unlabeled ZINC250k molecules for leakage-free contrastive pre-training and {stats['admet_total_count']:,} unique clinical compounds across {stats['unique_scaffolds']:,} Bemis-Murcko scaffolds curated from Therapeutics Data Commons (TDC). Each molecular graph combines 29-dimensional chemical and 3D conformer atom features, 8-dimensional bond descriptors, sign-canonicalized Laplacian positional encodings (LapPE), sign-invariant random-walk structural encodings (RWSE), and ETKDGv3 global 3D shape vectors. A parallel local covalent <code>TransformerConv</code> and global all-atom <code>MultiheadAttention</code> backbone is pre-trained via undirected-symmetric GraphCL InfoNCE loss (converging from {ssl_hist['loss'][0]:.4f} to {min(ssl_hist['loss']):.4f}) and fine-tuned via NaN-masked, class-weighted multi-task heads. On a strictly unseen-scaffold held-out test set (<i>n</i> = {stats['splits']['test']['count']}), our SSL-GraphGPS achieves <b>{ours['aggregate']['macro_roc_auc']:.4f} Macro ROC-AUC</b> across Oral Bioavailability ({ours['bioavailability']['roc_auc']:.4f}), BBB Permeability ({ours['bbb']['roc_auc']:.4f}), and hERG Cardiotoxicity ({ours['herg']['roc_auc']:.4f}), plus <i>R</i><sup>2</sup> = {ours['qed']['r2']:.4f} on QED and <i>R</i><sup>2</sup> = {ours['sascore']['r2']:.4f} on SAScore&mdash;outperforming Random Forest ECFP4 ({rf['aggregate']['macro_roc_auc']:.4f}), Standard-GCN ({gcn['aggregate']['macro_roc_auc']:.4f}), and GraphGPS-Scratch ({scratch['aggregate']['macro_roc_auc']:.4f}) on every endpoint. Finally, pairing 23 medicinal chemistry SMIRKS rules, BRICS fragment recombination, PAINS/Lipinski filters, NSGA-II Pareto sorting, and Input &times; Gradient atom saliency yields interpretable, synthesizable lead candidates.
</div>

<div class="keywords">
<b>Index Terms&mdash;</b>ADMET prediction, GraphGPS, Graph Transformer, self-supervised learning, InfoNCE, Bemis-Murcko scaffold split, Pareto optimization, explainable AI, hERG cardiotoxicity.
</div>

<h2>I. Introduction</h2>
<p class="noindent">Bringing a single new drug to market requires 10&ndash;15 years and over $2.6 billion in capitalized R&amp;D expenditure [1], yet more than 90% of candidates fail during human clinical trials [2]. Retrospective pharmaceutical analyses demonstrate that poor pharmacokinetics and safety liabilities&mdash;collectively termed ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity)&mdash;account for a primary fraction of late-stage attrition [2]. Specifically, low oral bioavailability (<i>F</i> &lt; 20%) prevents therapeutic systemic exposure [4], unintended Blood-Brain Barrier (BBB) penetration triggers central nervous system (CNS) side effects for peripheral drugs [5], and off-target inhibition of the cardiac hERG potassium channel induces fatal ventricular arrhythmia [6], [7].</p>

<p>Despite the urgency of <i>in-silico</i> ADMET screening, machine learning frameworks face four coupled challenges: (1) <i>clinical data scarcity and sparse missing labels</i> across merged public assay tables [3]; (2) <i>architectural myopia and 2D blindness</i> of local Message-Passing Graph Convolutional Networks (GCNs) [16], [18]; (3) <i>structural data leakage</i> from naive random splits that distribute identical ring cores across train and test sets [10]; and (4) <i>single-objective optimization flaws</i> that sacrifice cardiac safety or synthetic ease for absorption [26], [27].</p>

<p>To resolve all four gaps rigorously, we make four contributions:</p>
<ul>
  <li><b>Zero-Leakage Multi-Task Benchmark:</b> Standardized {stats['admet_total_count']:,} TDC clinical compounds across {stats['unique_scaffolds']:,} disjoint Bemis-Murcko scaffolds [3]&ndash;[7], [10] plus {stats['unlabeled_ssl_count']:,} unlabeled ZINC250k molecules [8], [9] verified to have zero scaffold/SMILES overlap with validation and test sets.</li>
  <li><b>2D+3D Hybrid Local+Global GraphGPS:</b> Extending GraphGPS [18] and edge-aware <code>TransformerConv</code> [17] with sign-canonicalized LapPE + RWSE [19] and fast ETKDGv3 3D conformer geometry [13].</li>
  <li><b>Undirected-Safe GraphCL &amp; Masked Multi-Task Learning:</b> Enforces undirected bond symmetry during GraphCL InfoNCE pre-training [20]&ndash;[22] and trains 5 strictly bounded heads under <code>NaN</code>-masked class-weighted losses.</li>
  <li><b>Evolutionary Pareto Optimization &amp; XAI:</b> Combines 23 SMIRKS rules, BRICS recombination [23], PAINS/Lipinski filters [24], [25], Derringer-Suich desirability [26], NSGA-II sorting [27], and Input &times; Gradient atom attribution [28].</li>
</ul>

<h2>II. Related Work</h2>
<p class="noindent">Classical Quantitative Structure-Activity Relationship (QSAR) pipelines encode molecules using Extended-Connectivity Fingerprints (ECFP4) [14] paired with Random Forests [15]. While strong at local substructure memorization, circular bit-vectors suffer hash collisions, lose 3D shape, and cannot transfer representations across partially labeled clinical endpoints.</p>

<p>Graph Convolutional Networks (GCNs) [16] learn directly on molecular connectivity, but 1-hop message passing misses long-range functional interactions and suffers oversmoothing. Shi <i>et al.</i> [17] integrated bond features into graph attention (<code>TransformerConv</code>), while Ramp&aacute;&scaron;ek <i>et al.</i> [18] formulated GraphGPS by coupling local message passing with global Transformer self-attention and structural/positional encodings (LSPE/RWSE) [19]. We adapt GraphGPS to pharmacology by resolving Laplacian eigenvector sign ambiguity and embedding ETKDGv3 3D conformer features [13].</p>

<p>To alleviate clinical label scarcity, GraphCL [20] and MolCLR [22] applied SimCLR-style contrastive InfoNCE loss [21] over perturbed molecular graphs. Critically, naive edge-dropping on bidirectional PyG tensors corrupts undirected symmetry (~25% one-way covalent edges); we enforce canonical undirected edge masking and validate pre-training strictly on unseen Bemis-Murcko scaffolds [10]. Downstream lead optimization integrates BRICS fragment reassembly [23], Lipinski [25] and PAINS [24] drug-likeness gates, Derringer-Suich geometric desirability [26], NSGA-II fronts [27], and gradient saliency XAI [28].</p>

<h2>III. Dataset Curation &amp; Zero-Leakage Protocol</h2>
<p class="noindent">All molecules are standardized through RDKit <code>LargestFragmentChooser</code> desalting (removing counterions like <code>.[Cl]</code>), valence sanitization, heavy-atom filtering ([2, 80] heavy atoms containing carbon), and canonical non-isomeric SMILES deduplication.</p>

<p>Clinical labels are aggregated from TDC [3]: <code>Bioavailability_Ma</code> [4] (<i>F</i> &ge; 20%), <code>BBB_Martins</code> [5], <code>hERG</code> [6] (<i>n</i> = 655), and a stratified balanced subsample (<i>n</i> = 2,500) from <code>hERG_Karim</code> [7]. Conflicts across duplicate canonical SMILES are resolved via majority vote, while exact RDKit QED [11] and SAScore [12] targets are computed across all {stats['admet_total_count']:,} compounds. For SSL pre-training, {stats['unlabeled_ssl_count']:,} molecules are sampled from ZINC250k [8], [9] after excluding any compound sharing a canonical SMILES or Bemis-Murcko scaffold [10] with ADMET validation or test sets. Table I and Fig. 1 detail the 80/10/10 disjoint scaffold splits (zero shared scaffolds or SMILES verified by assertion).</p>

<div class="table-wrap">
  <div class="table-caption">TABLE I. Curated Multi-Task ADMET Dataset &amp; Disjoint Scaffold Splits</div>
  <table>
    <thead>
      <tr><th>Split / Set</th><th>Mols (Scaf.)</th><th>Bioavail. (+%)</th><th>BBB (+%)</th><th>hERG (+%)</th></tr>
    </thead>
    <tbody>
      <tr><td>SSL Unlabeled</td><td>5,000 (ZINC250k)</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td></tr>
      <tr><td>Train (80%)</td><td>4,005 (2,135)</td><td>536 (78.0%)</td><td>1,611 (75.9%)</td><td>2,252 (54.2%)</td></tr>
      <tr><td>Val (10%)</td><td>500 (500)</td><td>56 (67.9%)</td><td>157 (80.3%)</td><td>317 (55.5%)</td></tr>
      <tr><td>Test (10%)</td><td>502 (502)</td><td>44 (75.0%)</td><td>153 (82.4%)</td><td>330 (48.5%)</td></tr>
      <tr class="mid-rule"><td><b>Total ADMET</b></td><td><b>5,007 (3,137)</b></td><td><b>636 (76.9%)</b></td><td><b>1,921 (76.7%)</b></td><td><b>2,899 (53.7%)</b></td></tr>
    </tbody>
  </table>
</div>

<figure>
  <img src="{_fig_uri('01_dataset_eda.png')}" alt="Dataset EDA">
  <figcaption><b>Fig. 1.</b> Exploratory summary of TDC clinical endpoint class distributions, disjoint Bemis-Murcko scaffold counts across splits, and QED/SAScore distributions.</figcaption>
</figure>

<h2>IV. Proposed Methodology</h2>
<p class="noindent">Fig. 2 presents our end-to-end 4-stage computational drug discovery architecture integrating curation, GraphCL pre-training, 2D+3D Hybrid GraphGPS, Pareto optimization, and XAI.</p>

<figure class="span-all">
  <img src="{_fig_uri('arch_01_system_pipeline.png')}" style="width: 96%;" alt="System Pipeline">
  <figcaption><b>Fig. 2.</b> End-to-end 4-stage pipeline: (0) zero-leakage TDC + ZINC250k curation and Bemis-Murcko scaffold splitting, (1) undirected-safe GraphCL InfoNCE pre-training, (2) 2D+3D Hybrid GraphGPS with 5 <code>NaN</code>-masked bounded ADMET heads, and (3) 23-SMIRKS + BRICS evolutionary Pareto optimizer and atom saliency XAI.</figcaption>
</figure>

<h3>A. 2D+3D Molecular Featurization &amp; Structural Encodings</h3>
<p class="noindent">Each molecule becomes a bidirectional graph <i>G</i> = (<i>V</i>, <i>E</i>). Heavy atoms carry 29-dimensional vectors <i>x<sub>i</sub></i> encoding element type (10), degree (6), formal charge (5, critical for cationic hERG blockers and BBB ionization), hybridization (4), normalized H-count (1), aromaticity (1), ring flag (1), and normalized 3D distance to centroid (1). Bonds carry 8-dimensional vectors <i>e<sub>ij</sub></i> encoding bond order (5), conjugation (1), ring flag (1), and 3D bond length (1). Global 3D shape descriptors <i>g</i><sub>3D</sub> = [NPR1, NPR2, Asphericity, RoG/10] &isin; &Ropf;<sup>4</sup> are extracted via RDKit ETKDGv3 [13] with deterministic zero fallbacks.</p>

<p>Following [19], we compute eigenvectors of the normalized graph Laplacian <i>L</i> = <i>I</i> &minus; <i>D</i><sup>&minus;1/2</sup><i>AD</i><sup>&minus;1/2</sup> excluding zero eigenvalues (&lambda; &gt; 10<sup>&minus;5</sup>) and canonicalize eigenvector signs so the max-magnitude element of each mode is positive. Concatenating 4 LapPE modes with 4 sign-invariant Random-Walk Structural Encoding (RWSE) diagonals diag(<i>P</i><sup>1</sup>)&hellip;diag(<i>P</i><sup>4</sup>) yields <i>p<sub>i</sub></i> &isin; &Ropf;<sup>8</sup>.</p>

<h3>B. Hybrid Local+Global 2D+3D GraphGPS Backbone</h3>
<p class="noindent">As shown in Fig. 3, inputs are linearly projected to <i>H</i> = 128 (<i>h</i><sup>(0)</sup> = <i>W<sub>x</sub>x</i> + <i>W<sub>p</sub>p</i>). Across <i>L</i> = 3 layers (4 heads), each layer pairs covalent-edge <code>TransformerConv</code> [17] with dense all-atom <code>MultiheadAttention</code> [18]:</p>

<div class="eq-row"><div class="eq-math">h<sub>loc</sub><sup>(l)</sup> = LN( h<sup>(l&minus;1)</sup> + Drop(TransformerConv(h<sup>(l&minus;1)</sup>, E, W<sub>e</sub>e)) )</div><div class="eq-num">(1)</div></div>
<div class="eq-row"><div class="eq-math">h<sub>glob</sub><sup>(l)</sup> = LN( h<sup>(l&minus;1)</sup> + Drop(MHA<sub>dense</sub>(h<sup>(l&minus;1)</sup>, mask)) )</div><div class="eq-num">(2)</div></div>
<div class="eq-row"><div class="eq-math">h<sup>(l)</sup> = LN( h&#772;<sup>(l)</sup> + FFN(h&#772;<sup>(l)</sup>) ), &nbsp; h&#772;<sup>(l)</sup> = 0.5(h<sub>loc</sub><sup>(l)</sup> + h<sub>glob</sub><sup>(l)</sup>)</div><div class="eq-num">(3)</div></div>

<p>Readout concatenates global mean pooling, global max pooling, and projected 3D shape descriptors into a 128-dim molecular embedding <i>h</i><sub>mol</sub>.</p>

<figure>
  <img src="{_fig_uri('arch_02_graphgps_layer.png')}" alt="GraphGPS Layer">
  <figcaption><b>Fig. 3.</b> Hybrid Local+Global 2D+3D GraphGPS layer fusing covalent bond attention, all-atom self-attention, and global ETKDGv3 3D shape vectors.</figcaption>
</figure>

<h3>C. Undirected-Safe GraphCL Contrastive Pre-Training</h3>
<p class="noindent">Following GraphCL [20] and MolCLR [22], stochastic view pairs (<i>v<sub>i</sub></i>, <i>v<sub>j</sub></i>) are drawn from atom masking (15%), undirected-symmetric edge dropping (15%, one keep mask per canonical bond (min(<i>u</i>,<i>v</i>), max(<i>u</i>,<i>v</i>))), subgraph node dropping (15% with edge index remapping), and random LapPE sign flipping. A <code>BatchNorm1d</code> 2-layer MLP projects <i>h</i><sub>mol</sub> to L2-normalized vectors <i>z<sub>i</sub></i> &isin; &Ropf;<sup>64</sup> trained via InfoNCE (&tau; = 0.1) [21]:</p>

<div class="eq-row"><div class="eq-math">L<sub>InfoNCE</sub> = &minus;log [ exp(sim(z<sub>i</sub>, z<sub>j</sub>)/&tau;) / &Sigma;<sub>k&ne;i</sub> exp(sim(z<sub>i</sub>, z<sub>k</sub>)/&tau;) ]</div><div class="eq-num">(4)</div></div>

<h3>D. NaN-Masked Multi-Task Heads &amp; Bounded Losses</h3>
<p class="noindent">Five 2-layer MLP adapter heads predict raw logits (<i>l</i><sub>bio</sub>, <i>l</i><sub>bbb</sub>, <i>l</i><sub>herg</sub>) mapped via sigmoid to [0, 100]% oral bioavailability probability, [0, 1] BBB, and [0, 1] hERG hazard, plus QED &isin; [0, 1] and strictly bounded SAScore = 1 + 9&sigma;(<i>s</i><sub>raw</sub>) &isin; [1, 10]. Using per-sample mask <i>M<sub>i,t</sub></i> &isin; &#123;0, 1&#125; to ignore missing <code>NaN</code> endpoints and training-split positive weights pw<sub><i>t</i></sub> = clip(<i>N</i><sub>neg</sub>/<i>N</i><sub>pos</sub>, 0.5, 4.0), the balanced loss is:</p>

<div class="eq-row"><div class="eq-math">L<sub>MT</sub> = &Sigma;<sub>t&isin;&#123;bio,bbb,herg&#125;</sub> L<sub>BCE</sub><sup>pw</sup>(l<sub>t</sub>, y<sub>t</sub>; M<sub>t</sub>) + 2 L<sub>SL1</sub>(q&#770;, q) + 2 L<sub>SL1</sub>(s&#771;&#770;, s&#771;)</div><div class="eq-num">(5)</div></div>
<p class="noindent">where s&#771; = (s &minus; 1)/9 scales SAScore to [0, 1] to match QED and BCE gradient magnitudes.</p>

<h3>E. Evolutionary Pareto Lead Design &amp; Atom Saliency XAI</h3>
<p class="noindent">From a seed molecule, 23 warning-free medicinal chemistry SMIRKS reactions plus RDKit BRICS fragment recombination [23] evolve analogues filtered by PAINS alerts [24], Lipinski Ro5 (&le; 1 violation) [25], and ECFP4 Tanimoto similarity &ge; 0.35 [14]. Candidates are scored via Derringer-Suich geometric desirability [26] in <code>peripheral</code> mode (<i>d</i><sub>BBB</sub> = 1 &minus; BBB):</p>

<div class="eq-row"><div class="eq-math">D(x) = ( d<sub>bio</sub><sup>1.0</sup> &middot; d<sub>1&minus;hERG</sub><sup>1.5</sup> &middot; d<sub>QED</sub><sup>1.0</sup> &middot; d<sub>SA</sub><sup>0.8</sup> &middot; d<sub>BBB</sub><sup>0.4</sup> )<sup>1/4.7</sup></div><div class="eq-num">(6)</div></div>
<p class="noindent">and partitioned into Pareto fronts via NSGA-II peeling [27]. Finally, Input &times; Gradient saliency [28] quantifies per-atom importance:</p>

<div class="eq-row"><div class="eq-math">A<sub>i</sub> = &radic;( &Sigma;<sub>d=1..29</sub> (x<sub>i,d</sub> &middot; &part;l<sub>task</sub>/&part;x<sub>i,d</sub>)<sup>2</sup> )</div><div class="eq-num">(7)</div></div>

<h2>V. Experimental Setup &amp; Ablation Results</h2>
<p class="noindent">All experiments use seed <code>42</code>, MPS acceleration, <code>AdamW</code> (wd = 10<sup>&minus;4</sup>), cosine LR annealing (&eta;<sub>min</sub> = 10<sup>&minus;5</sup>), gradient clipping (1.0), and batch size 64. Stage 1 SSL trains for 25 epochs (lr = 10<sup>&minus;3</sup>). Stage 2 fine-tunes for up to 40 epochs (lr = 5&times;10<sup>&minus;4</sup>, 0.5&times; backbone LR for SSL init, <code>pos_weight</code> = 0.50, 0.50, 0.84) with validation early stopping (patience = 10) maximizing 0.7&middot;Macro-ROC-AUC + 0.3&middot;Mean-<i>R</i><sup>2</sup>. On the unseen-scaffold test set (<i>n</i> = 502), we compare <b>RF-ECFP4</b> (300 balanced trees [14], [15]), <b>Standard-GCN</b> (3-layer <code>GCNConv</code> [16]), <b>GraphGPS-Scratch</b>, and <b>SSL-GraphGPS (Ours)</b>.</p>

<p>During Stage 1 pre-training on 5,000 ZINC250k graphs, InfoNCE loss converged smoothly from <b>{ssl_hist['loss'][0]:.4f}</b> (epoch 1) to <b>{ssl_hist['loss'][4]:.4f}</b> (epoch 5), <b>{ssl_hist['loss'][9]:.4f}</b> (epoch 10), and <b>{ssl_hist['loss'][-1]:.4f}</b> at epoch 25 (best loss <b>{min(ssl_hist['loss']):.4f}</b>). Fig. 4 shows zero-shot t-SNE clustering along QED and hERG prior to fine-tuning.</p>

<figure class="span-all">
  <img src="{_fig_uri('02_ssl_pretraining.png')}" style="width: 95%;" alt="SSL Pretraining">
  <figcaption><b>Fig. 4.</b> Stage 1 InfoNCE loss trajectory (left) and zero-shot t-SNE latent representations on unseen val/test scaffolds colored by QED (center) and hERG hazard (right).</figcaption>
</figure>

<div class="table-wrap span-all">
  <div class="table-caption">TABLE II. Held-Out Unseen-Scaffold Test Benchmark &amp; Ablation Comparison (<i>n</i> = 502)</div>
  <table>
    <thead>
      <tr>
        <th>Model</th><th>Bioavail. AUC</th><th>BBB AUC</th><th>hERG AUC</th><th>Macro ROC-AUC</th><th>QED <i>R</i><sup>2</sup> (MAE)</th><th>SAScore <i>R</i><sup>2</sup> (MAE)</th><th>Comp. Score</th>
      </tr>
    </thead>
    <tbody>
      {_row("RF-ECFP4 [14], [15]", rf)}
      {_row("Standard-GCN [16]", gcn)}
      {_row("GraphGPS-Scratch (No SSL)", scratch)}
      {_row("SSL-GraphGPS (Ours)", ours, bold=True)}
      <tr class="mid-rule">
        <td><i>SSL Gain vs. Scratch</i></td>
        <td><i>+{ours['bioavailability']['roc_auc'] - scratch['bioavailability']['roc_auc']:.4f}</i></td>
        <td><i>+{ours['bbb']['roc_auc'] - scratch['bbb']['roc_auc']:.4f}</i></td>
        <td><i>+{ours['herg']['roc_auc'] - scratch['herg']['roc_auc']:.4f}</i></td>
        <td><b><i>+{ours['aggregate']['macro_roc_auc'] - scratch['aggregate']['macro_roc_auc']:.4f}</i></b></td>
        <td><i>+{ours['qed']['r2'] - scratch['qed']['r2']:.4f}</i></td>
        <td><i>+{ours['sascore']['r2'] - scratch['sascore']['r2']:.4f}</i></td>
        <td><i>+{ours['aggregate']['composite_val_score'] - scratch['aggregate']['composite_val_score']:.4f}</i></td>
      </tr>
    </tbody>
  </table>
</div>

<p>Tables II&ndash;III and Figs. 5&ndash;7 demonstrate that <b>SSL-GraphGPS</b> achieves the strongest generalization across <b>every single endpoint</b>: <b>{ours['aggregate']['macro_roc_auc']:.4f} Macro ROC-AUC</b> (Bioavailability {ours['bioavailability']['roc_auc']:.4f} / PR-AUC {ours['bioavailability']['pr_auc']:.4f}; BBB {ours['bbb']['roc_auc']:.4f} / PR-AUC {ours['bbb']['pr_auc']:.4f}; hERG {ours['herg']['roc_auc']:.4f} / PR-AUC {ours['herg']['pr_auc']:.4f}), QED <i>R</i><sup>2</sup> = {ours['qed']['r2']:.4f} (MAE {ours['qed']['mae']:.3f}, <i>r</i> = {ours['qed']['pearson_r']:.4f}), and SAScore <i>R</i><sup>2</sup> = {ours['sascore']['r2']:.4f} (MAE {ours['sascore']['mae']:.3f}, <i>r</i> = {ours['sascore']['pearson_r']:.4f}). Contrastive pre-training adds <b>+3.43 percentage points (pp) Macro ROC-AUC</b> over <code>GraphGPS-Scratch</code>&mdash;with the steepest +7.17 pp boost on data-scarce Bioavailability&mdash;while hybrid 2D+3D GraphGPS outperforms <code>Standard-GCN</code> by +2.33 pp Macro ROC-AUC and +0.347 QED <i>R</i><sup>2</sup>.</p>

<div class="table-wrap">
  <div class="table-caption">TABLE III. Secondary Classification (PR-AUC / F1) &amp; Regression (Pearson <i>r</i>)</div>
  <table>
    <thead>
      <tr><th>Model</th><th>Bioavail. PR/F1</th><th>BBB PR/F1</th><th>hERG PR/F1</th><th>QED / SA <i>r</i></th></tr>
    </thead>
    <tbody>
      {_cls_detail_row("RF-ECFP4", rf)}
      {_cls_detail_row("Standard-GCN", gcn)}
      {_cls_detail_row("GraphGPS-Scratch", scratch)}
      {_cls_detail_row("SSL-GraphGPS", ours, bold=True)}
    </tbody>
  </table>
</div>

<figure>
  <img src="{_fig_uri('03_training_curves.png')}" alt="Training Curves">
  <figcaption><b>Fig. 5.</b> Training loss and unseen-scaffold validation Macro ROC-AUC across SSL-GraphGPS, GraphGPS-Scratch, and Standard-GCN.</figcaption>
</figure>

<figure class="span-all">
  <img src="{_fig_uri('04_roc_comparison.png')}" style="width: 94%;" alt="ROC Comparison">
  <figcaption><b>Fig. 6.</b> Unseen-scaffold test ROC curves across Oral Bioavailability (<i>n</i> = 44), BBB Permeability (<i>n</i> = 153), and hERG Cardiotoxicity (<i>n</i> = 330).</figcaption>
</figure>

<figure>
  <img src="{_fig_uri('05_regression_parity.png')}" alt="Regression Parity">
  <figcaption><b>Fig. 7.</b> Unseen-scaffold test parity plots for SSL-GraphGPS on QED (<i>R</i><sup>2</sup> = 0.7165) and SAScore (<i>R</i><sup>2</sup> = 0.7584).</figcaption>
</figure>

<h2>VI. Lead Optimization Case Study &amp; XAI</h2>
<p class="noindent">Two evolutionary generations around <b>Aspirin</b> (<code>CC(=O)Oc1ccccc1C(=O)O</code>) in <code>peripheral</code> mode evaluated {lead_opt['total_candidates']} PAINS-free, Lipinski-compliant analogues (<i>T</i> &ge; 0.35) and discovered {lead_opt['pareto_front1_count']} Pareto Front-1 candidates (Table IV and Fig. 8). Top analogues improve desirability from 0.898 (Aspirin seed) to <b>{top_cands[0]['desirability']:.3f}</b> while lowering predicted BBB penetration (0.216 &rarr; 0.032&ndash;0.081), keeping hERG &le; 0.013, and raising exact RDKit QED from 0.550 to 0.644.</p>

<div class="table-wrap">
  <div class="table-caption">TABLE IV. Aspirin Evolutionary Pareto Front-1 Candidates</div>
  <table>
    <thead>
      <tr><th>Candidate</th><th>Bio%</th><th>hERG</th><th>BBB</th><th>QED(P/RD)</th><th>SA(P/RD)</th><th>Desir.</th></tr>
    </thead>
    <tbody>
      <tr><td>Aspirin (Seed)</td><td>99.9</td><td>0.004</td><td>0.216</td><td>0.74 / 0.55</td><td>2.06 / 1.58</td><td>0.898</td></tr>
      {lead_rows}
    </tbody>
  </table>
</div>

<figure class="span-all">
  <img src="{_fig_uri('06_pareto_optimization.png')}" style="width: 94%;" alt="Pareto Optimization">
  <figcaption><b>Fig. 8.</b> Bioavailability vs. hERG hazard Pareto Front-1 (left) and 2D structures of Aspirin seed vs. top Pareto-optimal analogues (right).</figcaption>
</figure>

<p>Fig. 9 visualizes atom-level Input &times; Gradient saliency [28]. On an unseen test hERG blocker (pred = 0.996), saliency pinpoints the basic tertiary amine nitrogen and adjacent linker carbons&mdash;matching medicinal chemistry hERG pharmacophore rules [6]&mdash;while Aspirin scores safe (0.004) and Caffeine predicts high BBB penetration (0.941).</p>

<figure class="span-all">
  <img src="{_fig_uri('07_atom_saliency_xai.png')}" style="width: 94%;" alt="Atom Saliency XAI">
  <figcaption><b>Fig. 9.</b> Input &times; Gradient atom saliency highlighting the tertiary amine hERG toxicophore on an unseen test blocker (0.996), safe Aspirin (0.004), and CNS+ Caffeine (0.941).</figcaption>
</figure>

<h2>VII. Discussion, Honest Limitations &amp; Future Work</h2>
<p class="noindent">To uphold transparent reporting, five empirical boundaries apply: (1) metrics reflect deterministic seed <code>42</code> rather than multi-seed intervals; (2) the unseen-scaffold Bioavailability test fold holds <i>n</i> = 44 labeled compounds (out of 636 total in TDC), causing higher variance than hERG (<i>n</i> = 330); (3) the bioavailability head models binary <i>P</i>(<i>F</i> &ge; 20%) scaled to [0, 100]% rather than continuous absorption percentage; (4) on Aspirin, neural QED (0.739) overestimates exact RDKit QED (0.550) despite test <i>R</i><sup>2</sup> = 0.7165, motivating exact RDKit cross-auditing during Stage 3; and (5) featurization relies on a single static ETKDGv3 conformer rather than solvent molecular dynamics ensembles. Future extensions include Boltzmann multi-conformer SE(3) encoders, CYP450/Ames endpoints, and 3D generative diffusion.</p>

<h2>VIII. Conclusion</h2>
<p class="noindent">We presented an audited, zero-leakage Self-Supervised 2D+3D Hybrid GraphGPS framework for multi-objective ADMET profiling and evolutionary lead optimization. Across {stats['admet_total_count']:,} TDC clinical compounds tested on unseen Bemis-Murcko scaffolds, GraphCL pre-training over {stats['unlabeled_ssl_count']:,} ZINC250k molecules boosted Macro ROC-AUC by +3.43 pp to <b>{ours['aggregate']['macro_roc_auc']:.4f}</b>, outperforming ECFP4 Random Forests, Standard-GCN, and non-pretrained GraphGPS across every endpoint while providing atom-level XAI and PAINS-free Pareto-optimal leads.</p>

<h2>References</h2>
<div class="ref-list">
{ref_html}
</div>

</div>
</body>
</html>
"""
    html_path = os.path.join(PAPER_DIR, "ieee_paper.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def compile_pdf(html_path: str, pdf_path: str) -> int:
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        "file://" + html_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    gs_cmd = [
        "gs",
        "-q",
        "-dNODISPLAY",
        "-dNOSAFER",
        "-c",
        f"({pdf_path}) (r) file runpdfbegin pdfpagecount = quit",
    ]
    res = subprocess.run(gs_cmd, check=True, capture_output=True, text=True)
    return int(res.stdout.strip())


def render_pages_to_png(pdf_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        if fn.endswith(".png"):
            os.remove(os.path.join(out_dir, fn))
    subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=png16m",
            "-r150",
            f"-sOutputFile={os.path.join(out_dir, 'page_%02d.png')}",
            pdf_path,
        ],
        check=True,
    )


if __name__ == "__main__":
    os.makedirs(PAPER_DIR, exist_ok=True)
    html_file = build_html()
    pdf_file = os.path.join(PAPER_DIR, "IEEE_Research_Paper.pdf")
    pages = compile_pdf(html_file, pdf_file)
    scratch_pages = "/Users/prathmeshtoke/.gemini/antigravity/brain/d1daf4d3-6c11-4304-87f6-6facba93a79f/scratch/paper_pages"
    render_pages_to_png(pdf_file, scratch_pages)
    print(f"Compiled IEEE PDF -> {pdf_file} ({pages} pages)")
    print(f"Rendered page previews -> {scratch_pages}")
