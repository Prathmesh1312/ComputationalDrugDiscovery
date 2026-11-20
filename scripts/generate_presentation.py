"""
generate_presentation.py — Generate Professional 12-Slide 16:9 End-Sem Review PowerPoint Deck
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
OUT_DIR = os.path.join(BASE_DIR, "deliverables")
os.makedirs(OUT_DIR, exist_ok=True)

NAVY = RGBColor(0x0F, 0x17, 0x2A)
BLUE = RGBColor(0x1D, 0x4E, 0xD8)
EMERALD = RGBColor(0x05, 0x96, 0x69)
AMBER = RGBColor(0xD9, 0x77, 0x06)
RED = RGBColor(0xDC, 0x26, 0x26)
SLATE = RGBColor(0x47, 0x55, 0x69)
MUTED = RGBColor(0x94, 0xA3, 0xB8)
LIGHT_BG = RGBColor(0xF8, 0xFA, 0xFC)
BLUE_BG = RGBColor(0xEF, 0xF6, 0xFF)
EMERALD_BG = RGBColor(0xEC, 0xFD, 0xF5)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT_NAME = "Helvetica Neue"


def _add_image_fit(slide, img_path: str, left: float, top: float, max_w: float, max_h: float, center: bool = True):
    with Image.open(img_path) as im:
        px_w, px_h = im.size
    ratio = min(max_w / px_w, max_h / px_h)
    w = px_w * ratio
    h = px_h * ratio
    x = left + (max_w - w) / 2.0 if center else left
    y = top + (max_h - h) / 2.0 if center else top
    return slide.shapes.add_picture(img_path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def _add_header_footer(slide, section: str, title: str, page_num: int, total_pages: int = 12) -> None:
    # Section eyebrow
    tb_sec = slide.shapes.add_textbox(Inches(0.55), Inches(0.22), Inches(10.0), Inches(0.28))
    tf_sec = tb_sec.text_frame
    tf_sec.word_wrap = True
    tf_sec.margin_left = tf_sec.margin_top = tf_sec.margin_right = tf_sec.margin_bottom = 0
    p_sec = tf_sec.paragraphs[0]
    p_sec.text = section.upper()
    p_sec.font.name = FONT_NAME
    p_sec.font.size = Pt(10)
    p_sec.font.bold = True
    p_sec.font.color.rgb = BLUE

    # Slide title
    tb_t = slide.shapes.add_textbox(Inches(0.55), Inches(0.45), Inches(12.2), Inches(0.45))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title
    p_t.font.name = FONT_NAME
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = NAVY

    # Divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(0.95), Inches(12.23), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    # Footer line & text
    f_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(7.05), Inches(12.23), Inches(0.01))
    f_line.fill.solid()
    f_line.fill.fore_color.rgb = BORDER
    f_line.line.fill.background()

    tb_f = slide.shapes.add_textbox(Inches(0.55), Inches(7.10), Inches(9.5), Inches(0.25))
    tf_f = tb_f.text_frame
    tf_f.margin_left = tf_f.margin_top = tf_f.margin_right = tf_f.margin_bottom = 0
    pf = tf_f.paragraphs[0]
    pf.text = "Self-Supervised 2D+3D Hybrid GraphGPS for ADMET Profiling & Lead Optimization | End-Sem Review"
    pf.font.name = FONT_NAME
    pf.font.size = Pt(9.5)
    pf.font.color.rgb = MUTED

    tb_p = slide.shapes.add_textbox(Inches(11.5), Inches(7.10), Inches(1.28), Inches(0.25))
    tf_p = tb_p.text_frame
    tf_p.margin_left = tf_p.margin_top = tf_p.margin_right = tf_p.margin_bottom = 0
    pp = tf_p.paragraphs[0]
    pp.alignment = PP_ALIGN.RIGHT
    pp.text = f"{page_num} / {total_pages}"
    pp.font.name = FONT_NAME
    pp.font.size = Pt(9.5)
    pp.font.bold = True
    pp.font.color.rgb = SLATE


def _add_card(slide, left: float, top: float, w: float, h: float, title: str, bullets: list,
              bg_color=LIGHT_BG, border_color=BORDER, title_color=NAVY, title_size: int = 13, body_size: int = 11.5):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(w), Inches(h))
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1.0)

    tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.14), Inches(w - 0.36), Inches(h - 0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.name = FONT_NAME
    p0.font.size = Pt(title_size)
    p0.font.bold = True
    p0.font.color.rgb = title_color
    p0.space_after = Pt(6)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"•  {b}"
        p.font.name = FONT_NAME
        p.font.size = Pt(body_size)
        p.font.color.rgb = SLATE
        p.space_after = Pt(4)
    return card


def _set_notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def build_deck() -> str:
    with open(os.path.join(BASE_DIR, "data", "processed", "dataset_stats.json"), "r", encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "metrics_summary.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "ssl_history.json"), "r", encoding="utf-8") as f:
        ssl_hist = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "lead_opt_results.json"), "r", encoding="utf-8") as f:
        lead_opt = json.load(f)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ------------------------------------------------------------------
    # SLIDE 1: Title Cover
    # ------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank)
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    accent_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.45), Inches(0.10), Inches(3.25))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = BLUE
    accent_bar.line.fill.background()

    tb_tag = s1.shapes.add_textbox(Inches(1.05), Inches(1.45), Inches(11.0), Inches(0.35))
    p = tb_tag.text_frame.paragraphs[0]
    p.text = "ENGINEERING SEMESTER MACHINE LEARNING COURSE PROJECT  |  END-SEM REVIEW"
    p.font.name = FONT_NAME
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x60, 0xA5, 0xFA)

    tb_title = s1.shapes.add_textbox(Inches(1.05), Inches(1.90), Inches(11.4), Inches(1.75))
    tf = tb_title.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Self-Supervised 2D+3D Hybrid GraphGPS for\nMulti-Objective ADMET Profiling & Lead Scoring"
    p.font.name = FONT_NAME
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = WHITE

    tb_sub = s1.shapes.add_textbox(Inches(1.05), Inches(3.85), Inches(11.2), Inches(0.9))
    tf_s = tb_sub.text_frame
    tf_s.word_wrap = True
    p = tf_s.paragraphs[0]
    p.text = "Pre-trained with GraphCL/InfoNCE on ZINC250k, evaluated on disjoint Bemis-Murcko scaffolds of TDC clinical assays, and coupled with evolutionary NSGA-II Pareto optimization & atom-level XAI."
    p.font.name = FONT_NAME
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)

    # 3 metric pills at bottom
    pills = [
        ("10,007 Compounds", "5,000 SSL + 5,007 TDC ADMET"),
        (f"{metrics['SSL-GraphGPS (Ours)']['aggregate']['macro_roc_auc']:.4f} Macro ROC-AUC", "Held-Out Unseen Scaffolds (+3.43 pp vs Scratch)"),
        ("14 / 14 Unit Tests Passed", "Zero RDKit Warnings | Zero Scaffold Leakage"),
    ]
    for i, (m_val, m_sub) in enumerate(pills):
        px = 1.05 + i * 3.85
        pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(px), Inches(5.15), Inches(3.60), Inches(1.10))
        pill.adjustments[0] = 0.08
        pill.fill.solid()
        pill.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        pill.line.color.rgb = RGBColor(0x33, 0x41, 0x55)
        tb = s1.shapes.add_textbox(Inches(px + 0.18), Inches(5.28), Inches(3.25), Inches(0.85))
        tff = tb.text_frame
        tff.word_wrap = True
        p1 = tff.paragraphs[0]
        p1.text = m_val
        p1.font.name = FONT_NAME
        p1.font.size = Pt(16)
        p1.font.bold = True
        p1.font.color.rgb = RGBColor(0x34, 0xD3, 0x99) if i == 1 else WHITE
        p2 = tff.add_paragraph()
        p2.text = m_sub
        p2.font.name = FONT_NAME
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    tb_auth = s1.shapes.add_textbox(Inches(1.05), Inches(6.68), Inches(11.2), Inches(0.35))
    pa = tb_auth.text_frame.paragraphs[0]
    pa.text = "Presented by: Prathmesh Toke  |  Computational Drug Discovery & Graph Representation Learning"
    pa.font.name = FONT_NAME
    pa.font.size = Pt(12)
    pa.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    _set_notes(s1, "Welcome panel. Introduce the project title, scale (10,007 compounds), unseen-scaffold Macro ROC-AUC (0.8299), and end-to-end engineering scope.")

    # ------------------------------------------------------------------
    # SLIDE 2: Clinical Motivation & Core Objectives
    # ------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank)
    _add_header_footer(s2, "01 / Motivation & Objectives", "Clinical Drug Attrition & Project Objectives", 2)

    _add_card(
        s2, 0.55, 1.15, 5.9, 2.70,
        "The Pre-Clinical ADMET Bottleneck",
        [
            "Drug development requires 10–15 years and >$2.6 Billion per approved molecule.",
            ">90% of clinical candidates fail; over half fail due to poor ADMET pharmacokinetics & toxicity.",
            "hERG potassium channel blockade causes fatal QT arrhythmia.",
            "Low oral bioavailability (F < 20%) and unintended Blood-Brain Barrier (BBB) penetration drive late-stage attrition.",
        ],
        bg_color=RGBColor(0xFE, 0xF2, 0xF2), border_color=RGBColor(0xFE, 0xCA, 0xCA), title_color=RED,
    )
    _add_card(
        s2, 0.55, 4.05, 5.9, 2.75,
        "Five Simultaneous ADMET Endpoints",
        [
            "Oral Bioavailability P(F >= 20%) [0–100%] — TDC Bioavailability_Ma (n=636).",
            "BBB Permeability [0–1] — TDC BBB_Martins (n=1,921).",
            "hERG Cardiotoxicity [0–1] — TDC hERG Wang + Karim (n=2,899).",
            "Drug-Likeness (QED [0–1]) & Synthetic Accessibility (SAScore [1–10]) across all 5,007 molecules.",
        ],
        bg_color=LIGHT_BG, border_color=BORDER, title_color=NAVY,
    )
    _add_card(
        s2, 6.70, 1.15, 6.08, 5.65,
        "Four Core Engineering & Research Objectives",
        [
            "1. Contrastive Pre-Training (GraphCL / InfoNCE): Exploit 5,000 unlabeled ZINC250k compounds to learn chemical structure before seeing clinical labels.",
            "2. Hybrid Local+Global 2D+3D GraphGPS: Overcome 1-hop GCN myopia using edge-aware TransformerConv + all-atom MultiheadAttention + ETKDGv3 3D conformers.",
            "3. Zero-Leakage Multi-Task Evaluation: Partition compounds strictly by disjoint Bemis-Murcko scaffolds with NaN-masked, class-weighted losses.",
            "4. Interpretable Evolutionary Lead Design: Combine 23 SMIRKS + BRICS recombination, PAINS/Lipinski rules, NSGA-II Pareto sorting, and Atom Saliency XAI.",
        ],
        bg_color=BLUE_BG, border_color=RGBColor(0xBF, 0xDB, 0xFE), title_color=BLUE, title_size=14, body_size=12,
    )
    _set_notes(s2, "Explain why ADMET failures cost billions, introduce the 5 target endpoints, and state the 4 concrete engineering objectives.")

    # ------------------------------------------------------------------
    # SLIDE 3: Limitations of Prior Work & Codebase Engineering Audit
    # ------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank)
    _add_header_footer(s3, "02 / Gap Analysis & Audit", "Literature Gaps & Systematic Codebase Audit Fixes", 3)

    _add_card(
        s3, 0.55, 1.15, 5.95, 5.65,
        "Shortcomings of Conventional Approaches",
        [
            "Classical Morgan ECFP4 + Random Forest: Strong substructure memorization, but discards 3D shape, global topology, and cross-task correlations.",
            "Standard 2D GCNs: Suffer 1-hop local myopia, oversmoothing beyond 3 layers, and ignore covalent bond types and 3D conformers.",
            "Random 80/20 Data Splits: Leak near-identical ring scaffolds across train and test, over-estimating real-world accuracy.",
            "Single-Metric Optimization: Arithmetic averages disguise fatal cardiotoxicity behind high absorption scores.",
        ],
        bg_color=LIGHT_BG, border_color=BORDER, title_color=NAVY, body_size=12,
    )
    _add_card(
        s3, 6.75, 1.15, 6.03, 5.65,
        "13 Critical Defects Found & Fixed in Project Audit",
        [
            "Toy Data -> Curated 10,007 real standardized compounds (ZINC250k + TDC).",
            "100% Train/Test Leakage -> Disjoint Bemis-Murcko scaffold split (zero overlap).",
            "Unshuffled SSL Batches -> DataLoader(shuffle=True, drop_last=True) + Cosine LR.",
            "Unstable Post-Sigmoid BCE -> Masked BCEWithLogitsLoss + pos_weight clamping.",
            "Asymmetric Edge Drop -> Canonical (min(u,v), max(u,v)) undirected masking.",
            "Unbounded Softplus SAScore -> Strictly bounded 1 + 9*sigmoid(z) in [1, 10].",
            "Local-Only TransformerConv -> True Hybrid Local+Global GraphGPS + 3D shape.",
            "Noisy 8-Rule Mutator -> 23 warning-free SMIRKS + BRICS + PAINS/Lipinski + XAI.",
        ],
        bg_color=EMERALD_BG, border_color=RGBColor(0xA7, 0xF3, 0xD0), title_color=EMERALD, body_size=11.5,
    )
    _set_notes(s3, "Highlight both scientific motivation against ECFP4/GCN/random splits and the rigorous 13-point engineering audit performed on the codebase.")

    # ------------------------------------------------------------------
    # SLIDE 4: Curated Datasets & Bemis-Murcko Scaffold Split
    # ------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank)
    _add_header_footer(s4, "03 / Data Curation", "Real-World TDC + ZINC250k Corpora & Disjoint Scaffold Split", 4)

    _add_image_fit(s4, os.path.join(FIG_DIR, "01_dataset_eda.png"), 0.55, 1.10, 7.85, 5.80)
    _add_card(
        s4, 8.60, 1.15, 4.18, 2.70,
        "Corpus & Standardization",
        [
            f"SSL Corpus: {stats['unlabeled_ssl_count']:,} ZINC250k molecules (0 shared scaffolds/SMILES with val/test).",
            f"ADMET Corpus: {stats['admet_total_count']:,} unique compounds across {stats['unique_scaffolds']:,} Murcko scaffolds.",
            "Desalted via LargestFragmentChooser & deduplicated via assay majority vote.",
        ],
        bg_color=BLUE_BG, border_color=RGBColor(0xBF, 0xDB, 0xFE), title_color=BLUE, body_size=11,
    )
    _add_card(
        s4, 8.60, 4.05, 4.18, 2.75,
        "Disjoint Scaffold Split (80/10/10)",
        [
            f"Train: {stats['splits']['train']['count']:,} mols ({stats['splits']['train']['unique_scaffolds']:,} scaffolds).",
            f"Val: {stats['splits']['val']['count']} mols ({stats['splits']['val']['unique_scaffolds']} unseen scaffolds).",
            f"Test: {stats['splits']['test']} mols ({stats['splits']['test']['unique_scaffolds']} unseen scaffolds).",
            "Automated assertions verify zero scaffold & SMILES leakage across splits.",
        ],
        bg_color=LIGHT_BG, border_color=BORDER, title_color=NAVY, body_size=11,
    )
    _set_notes(s4, "Show dataset distribution plots, explain missing NaN assay sparsity across Bioavailability/BBB/hERG, and emphasize strict zero-leakage scaffold splitting.")

    # ------------------------------------------------------------------
    # SLIDE 5: End-to-End 4-Stage Pipeline
    # ------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank)
    _add_header_footer(s5, "04 / System Architecture", "End-to-End 4-Stage Computational Drug Discovery Pipeline", 5)

    _add_image_fit(s5, os.path.join(FIG_DIR, "arch_01_system_pipeline.png"), 0.55, 1.05, 12.23, 4.55)

    stages_summary = [
        ("Stage 0: Curation", "TDC + ZINC250k desalting & Bemis-Murcko split (4,005 / 500 / 502)", BLUE),
        ("Stage 1: GraphCL SSL", "5,000 unlabeled graphs trained via InfoNCE (loss 0.8659 -> 0.0402)", RGBColor(0x6D, 0x28, 0xD9)),
        ("Stage 2: GraphGPS", "2D+3D Hybrid Local+Global attention + 5 NaN-masked ADMET heads", EMERALD),
        ("Stage 3: Pareto & XAI", "23 SMIRKS + BRICS evolution, NSGA-II ranking & Atom Saliency", AMBER),
    ]
    for idx, (st_t, st_b, st_c) in enumerate(stages_summary):
        lx = 0.55 + idx * 3.10
        _add_card(s5, lx, 5.72, 2.93, 1.20, st_t, [st_b], bg_color=LIGHT_BG, border_color=st_c, title_color=st_c, title_size=11.5, body_size=9.8)
    _set_notes(s5, "Walk cleanly left-to-right through the 4 stages of the architecture diagram.")

    # ------------------------------------------------------------------
    # SLIDE 6: 2D+3D Hybrid GraphGPS Architecture
    # ------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank)
    _add_header_footer(s6, "05 / Neural Backbone", "2D+3D Hybrid Local+Global GraphGPS Architecture", 6)

    _add_card(
        s6, 0.55, 1.15, 4.35, 5.65,
        "Rich 2D+3D Molecular Representation",
        [
            "29-dim Atom Features: Element (10), Degree (6), Formal Charge (5), Hybridization (4), Hs (1), Aromatic (1), In-Ring (1), 3D Centroid Dist (1).",
            "8-dim Bond Features: Bond Type (5), Conjugated (1), In-Ring (1), 3D ETKDGv3 Bond Length (1).",
            "8-dim Structural PE: 4 Sign-Canonicalized Laplacian PE + 4 Sign-Invariant RWSE.",
            "4-dim Global 3D Shape: NPR1, NPR2, Asphericity, Radius of Gyration via RDKit ETKDGv3.",
            "Parallel Fusion: Local TransformerConv + Global MultiheadAttention (H=128, 3 layers, 4 heads).",
        ],
        bg_color=LIGHT_BG, border_color=BORDER, title_color=NAVY, title_size=13, body_size=10.8,
    )
    _add_image_fit(s6, os.path.join(FIG_DIR, "arch_02_graphgps_layer.png"), 5.05, 1.10, 7.73, 5.75)
    _set_notes(s6, "Explain how combining covalent local TransformerConv with dense global MultiheadAttention + 3D ETKDGv3 geometry solves GCN myopia and oversmoothing.")

    # ------------------------------------------------------------------
    # SLIDE 7: Stage 1 Self-Supervised Contrastive Pre-Training
    # ------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank)
    _add_header_footer(s7, "06 / Stage 1: Self-Supervised Learning", "GraphCL Contrastive Pre-Training (5,000 ZINC250k Molecules)", 7)

    _add_image_fit(s7, os.path.join(FIG_DIR, "02_ssl_pretraining.png"), 0.55, 1.10, 12.23, 3.85)
    _add_card(
        s7, 0.55, 5.10, 5.95, 1.80,
        "Undirected-Safe Graph Augmentations",
        [
            "Undirected Edge Dropping: drops (u,v) & (v,u) symmetrically (15%).",
            "Subgraph Node Dropping (15%), Atom Masking (15%) & LapPE Sign Flipping.",
        ],
        bg_color=BLUE_BG, border_color=RGBColor(0xBF, 0xDB, 0xFE), title_color=BLUE, title_size=12.5, body_size=10.8,
    )
    _add_card(
        s7, 6.75, 5.10, 6.03, 1.80,
        "Verified Convergence & Zero-Shot Organization",
        [
            f"InfoNCE Loss (tau=0.1, BatchNorm projector): {ssl_hist['loss'][0]:.4f} -> {min(ssl_hist['loss']):.4f} over 25 epochs.",
            "t-SNE on unseen val/test scaffolds shows smooth QED & hERG structure pre-finetuning.",
        ],
        bg_color=EMERALD_BG, border_color=RGBColor(0xA7, 0xF3, 0xD0), title_color=EMERALD, title_size=12.5, body_size=10.8,
    )
    _set_notes(s7, "Highlight the steep InfoNCE loss convergence (0.8659 -> 0.0402) and the zero-shot latent clustering by QED and hERG before fine-tuning.")

    # ------------------------------------------------------------------
    # SLIDE 8: Stage 2 Masked Multi-Task ADMET Fine-Tuning
    # ------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank)
    _add_header_footer(s8, "07 / Stage 2: Fine-Tuning", "NaN-Masked Multi-Task Learning & Class Imbalance Handling", 8)

    _add_image_fit(s8, os.path.join(FIG_DIR, "03_training_curves.png"), 0.55, 1.15, 7.65, 5.65)
    _add_card(
        s8, 8.40, 1.20, 4.38, 2.70,
        "Multi-Task Loss Engineering",
        [
            "NaN Boolean Masking: trains across partial assay tables without imputing labels.",
            "BCEWithLogitsLoss + training-split pos_weight (0.50–0.84) handles class imbalance.",
            "Smooth-L1 loss on QED and [0,1]-normalized SAScore keeps gradients balanced.",
        ],
        bg_color=LIGHT_BG, border_color=BORDER, title_color=NAVY, body_size=11,
    )
    _add_card(
        s8, 8.40, 4.10, 4.38, 2.70,
        "Optimization & Regularization",
        [
            "Discriminative LR: 0.5x LR on SSL backbone, 1.0x LR on 2-layer MLP task heads.",
            "AdamW (wd=1e-4), CosineAnnealingLR, gradient clipping (max_norm=1.0).",
            "Early stopping selects best unseen-scaffold validation composite score (epoch 33).",
        ],
        bg_color=BLUE_BG, border_color=RGBColor(0xBF, 0xDB, 0xFE), title_color=BLUE, body_size=11,
    )
    _set_notes(s8, "Discuss why NaN masking and normalized Smooth-L1 prevent gradient explosions and allow SSL-GraphGPS to converge cleanly beyond Standard-GCN.")

    # ------------------------------------------------------------------
    # SLIDE 9: Held-Out Unseen-Scaffold Benchmark & Ablation
    # ------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank)
    _add_header_footer(s9, "08 / Empirical Benchmark", "Held-Out Unseen-Scaffold Test Results & Ablation (n = 502)", 9)

    # Native PPT Table
    headers = ["Model", "Bioavail AUC", "BBB AUC", "hERG AUC", "Macro AUC", "QED R² (MAE)", "SAScore R² (MAE)"]
    model_order = ["RF-ECFP4", "Standard-GCN", "GraphGPS-Scratch", "SSL-GraphGPS (Ours)"]
    tbl_shape = s9.shapes.add_table(5, 7, Inches(0.55), Inches(1.12), Inches(12.23), Inches(2.15))
    tbl = tbl_shape.table
    col_widths = [2.43, 1.45, 1.35, 1.35, 1.45, 2.10, 2.10]
    for idx, cw in enumerate(col_widths):
        tbl.columns[idx].width = Inches(cw)

    for c_idx, h_text in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
        p.font.name = FONT_NAME
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE

    for r_idx, m_name in enumerate(model_order, start=1):
        m = metrics[m_name]
        is_ours = m_name == "SSL-GraphGPS (Ours)"
        row_vals = [
            m_name,
            f"{m['bioavailability']['roc_auc']:.4f}",
            f"{m['bbb']['roc_auc']:.4f}",
            f"{m['herg']['roc_auc']:.4f}",
            f"{m['aggregate']['macro_roc_auc']:.4f}",
            f"{m['qed']['r2']:.4f} ({m['qed']['mae']:.3f})",
            f"{m['sascore']['r2']:.4f} ({m['sascore']['mae']:.3f})",
        ]
        for c_idx, val in enumerate(row_vals):
            cell = tbl.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = EMERALD_BG if is_ours else (WHITE if r_idx % 2 == 1 else LIGHT_BG)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
            p.font.name = FONT_NAME
            p.font.size = Pt(10.8)
            p.font.bold = is_ours or (c_idx == 4)
            p.font.color.rgb = EMERALD if (is_ours and c_idx in (0, 4)) else NAVY

    _add_image_fit(s9, os.path.join(FIG_DIR, "04_roc_comparison.png"), 0.55, 3.40, 12.23, 3.55)
    _set_notes(s9, "Emphasize that SSL-GraphGPS wins on ALL 5 endpoints simultaneously on strictly unseen scaffolds, gaining +3.43 pp Macro ROC-AUC over GraphGPS-Scratch and +7.17 pp on data-scarce Bioavailability.")

    # ------------------------------------------------------------------
    # SLIDE 10: Regression Parity & Aspirin Pareto Lead Optimization
    # ------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank)
    _add_header_footer(s10, "09 / Regression & Lead Design", "QED/SAScore Parity & Evolutionary Pareto Lead Optimization", 10)

    _add_image_fit(s10, os.path.join(FIG_DIR, "05_regression_parity.png"), 0.55, 1.12, 5.45, 3.35)
    _add_image_fit(s10, os.path.join(FIG_DIR, "06_pareto_optimization.png"), 6.15, 1.12, 6.63, 3.85)

    top1 = lead_opt["top_candidates"][0]
    _add_card(
        s10, 0.55, 4.65, 5.45, 2.25,
        "Physicochemical Generalization",
        [
            f"QED Test R² = {metrics['SSL-GraphGPS (Ours)']['qed']['r2']:.4f} (MAE = {metrics['SSL-GraphGPS (Ours)']['qed']['mae']:.3f}, r = {metrics['SSL-GraphGPS (Ours)']['qed']['pearson_r']:.3f}).",
            f"SAScore Test R² = {metrics['SSL-GraphGPS (Ours)']['sascore']['r2']:.4f} (MAE = {metrics['SSL-GraphGPS (Ours)']['sascore']['mae']:.3f}, r = {metrics['SSL-GraphGPS (Ours)']['sascore']['pearson_r']:.3f}).",
        ],
        bg_color=LIGHT_BG, border_color=BORDER, title_color=NAVY, title_size=12, body_size=10.5,
    )
    _add_card(
        s10, 6.15, 5.10, 6.63, 1.80,
        f"Aspirin Evolutionary Optimization ({lead_opt['total_candidates']} Analogues, {lead_opt['pareto_front1_count']} Front-1)",
        [
            f"Top Lead ({top1['smiles']}): Desirability {top1['desirability']:.3f} vs Seed 0.898.",
            f"Bioavail {top1['bioavailability']:.1f}% | hERG {top1['herg']:.3f} | BBB {top1['bbb']:.3f} | RDKit QED {top1['qed_rdkit']:.3f} | SA {top1['sascore_rdkit']:.2f}.",
        ],
        bg_color=EMERALD_BG, border_color=RGBColor(0xA7, 0xF3, 0xD0), title_color=EMERALD, title_size=11.5, body_size=10.3,
    )
    _set_notes(s10, "Walk through QED/SAScore parity plots and show how 2 evolutionary generations of SMIRKS + BRICS yielded 45 Pareto Front-1 Aspirin analogues.")

    # ------------------------------------------------------------------
    # SLIDE 11: Explainable AI (Atom-Level Saliency)
    # ------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank)
    _add_header_footer(s11, "10 / Explainable AI (XAI)", "Atom-Level Gradient Attribution Validates Chemical Reasoning", 11)

    _add_image_fit(s11, os.path.join(FIG_DIR, "07_atom_saliency_xai.png"), 0.55, 1.12, 12.23, 4.15)
    _add_card(
        s11, 0.55, 5.40, 4.00, 1.50,
        "Unseen Test hERG Blocker",
        ["Pred hERG = 0.996; Input x Grad highlights the basic tertiary amine nitrogen & linker, matching known hERG pharmacophores."],
        bg_color=RGBColor(0xFE, 0xF2, 0xF2), border_color=RGBColor(0xFE, 0xCA, 0xCA), title_color=RED, title_size=12, body_size=10.2,
    )
    _add_card(
        s11, 4.70, 5.40, 3.95, 1.50,
        "Aspirin (Non-Blocker Seed)",
        ["Pred hERG = 0.004; low uniform risk without basic amine toxicophore, confirming safe cardiac profile."],
        bg_color=EMERALD_BG, border_color=RGBColor(0xA7, 0xF3, 0xD0), title_color=EMERALD, title_size=12, body_size=10.2,
    )
    _add_card(
        s11, 8.80, 5.40, 3.98, 1.50,
        "Caffeine (CNS+ Control)",
        ["Pred BBB+ = 0.941; attribution focuses on the lipophilic xanthine ring N-methyl & carbonyl groups."],
        bg_color=BLUE_BG, border_color=RGBColor(0xBF, 0xDB, 0xFE), title_color=BLUE, title_size=12, body_size=10.2,
    )
    _set_notes(s11, "Point out that on an unseen test-set hERG blocker, gradient saliency independently discovers the basic tertiary amine nitrogen toxicophore.")

    # ------------------------------------------------------------------
    # SLIDE 12: Conclusion, Honest Limitations & Future Scope
    # ------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank)
    _add_header_footer(s12, "11 / Conclusion & Future Scope", "Key Contributions, Honest Limitations & Future Directions", 12)

    _add_card(
        s12, 0.55, 1.15, 3.95, 5.65,
        "Key Contributions",
        [
            "Scaled from 10 toy samples to 10,007 standardized compounds with disjoint Bemis-Murcko splits.",
            "Built 2D+3D Hybrid Local+Global GraphGPS pre-trained via undirected-safe GraphCL.",
            "Best performance on all 5 endpoints (0.8299 Macro ROC-AUC; 0.7165 QED R²; 0.7584 SA R²).",
            "Delivered evolutionary Pareto optimizer, XAI saliency, 14/14 unit tests, IEEE paper & report.",
        ],
        bg_color=BLUE_BG, border_color=RGBColor(0xBF, 0xDB, 0xFE), title_color=BLUE, body_size=11.3,
    )
    _add_card(
        s12, 4.68, 1.15, 4.00, 5.65,
        "Honest Limitations",
        [
            "Single-seed benchmark (seed=42); multi-seed error bars remain future work.",
            "Oral Bioavailability test split is small (n=44) due to public TDC dataset size (636 total).",
            "Bioavailability is binary (F >= 20%) mapped to [0–100]% probability rather than continuous %F.",
            "Aspirin's predicted QED (0.739) overestimates exact RDKit QED (0.550), justifying exact RDKit audits in Stage 3.",
            "Single static ETKDGv3 conformer per molecule.",
        ],
        bg_color=RGBColor(0xFF, 0xFB, 0xEB), border_color=RGBColor(0xFD, 0xE6, 0x8A), title_color=AMBER, body_size=11,
    )
    _add_card(
        s12, 8.85, 1.15, 3.93, 5.65,
        "Future Work",
        [
            "Multi-conformer Boltzmann ensembles & explicit-solvent Molecular Dynamics scoring.",
            "SE(3)-equivariant 3D generative diffusion for de novo scaffold hopping.",
            "Expand multi-task head bank to CYP450 inhibition, Ames mutagenicity & DILI.",
            "Interactive local Web UI (FastAPI / Streamlit) for real-time medicinal chemistry screening.",
            "Thank You — Open for Faculty Q&A.",
        ],
        bg_color=EMERALD_BG, border_color=RGBColor(0xA7, 0xF3, 0xD0), title_color=EMERALD, body_size=11.3,
    )
    _set_notes(s12, "Present balanced scientific honesty about dataset sizes, single-seed evaluation, and exact RDKit QED/SA cross-checks, then open for viva Q&A.")

    out_path = os.path.join(OUT_DIR, "EndSem_Review_Presentation.pptx")
    prs.save(out_path)
    return out_path


def validate_presentation(pptx_path: str) -> None:
    prs = Presentation(pptx_path)
    assert len(prs.slides) == 12, f"Expected 12 slides, got {len(prs.slides)}"
    max_w = prs.slide_width
    max_h = prs.slide_height
    pic_count = 0
    for s_idx, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            assert sh.left >= 0 and sh.top >= 0, f"Negative coord on slide {s_idx}"
            assert sh.left + sh.width <= max_w + 5000, f"Width overflow on slide {s_idx}: {sh.left + sh.width} > {max_w}"
            assert sh.top + sh.height <= max_h + 5000, f"Height overflow on slide {s_idx}: {sh.top + sh.height} > {max_h}"
            if sh.shape_type == 13:  # PICTURE
                pic_count += 1
    assert pic_count >= 9, f"Expected >= 9 embedded images, got {pic_count}"
    print(f"Validated PPTX cleanly: {len(prs.slides)} slides, {pic_count} embedded figures, 0 out-of-bounds shapes -> {pptx_path}")


if __name__ == "__main__":
    path = build_deck()
    validate_presentation(path)
