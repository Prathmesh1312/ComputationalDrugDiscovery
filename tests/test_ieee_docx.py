"""
test_ieee_docx.py — Comprehensive OOXML, IEEE Layout, OMML, and Metric Verification Suite
for paper/IEEE_Research_Paper.docx.
"""
import json
import os
import unittest
import zipfile

import docx
from docx.oxml.ns import qn
from lxml import etree
from PIL import Image

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCX_PATH = os.path.join(BASE_DIR, "paper", "IEEE_Research_Paper.docx")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _dxa(inches: float) -> int:
    return int(round(inches * 1440.0))


class TestIEEEDocxPaper(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = docx.Document(DOCX_PATH)
        with zipfile.ZipFile(DOCX_PATH, "r") as zf:
            cls.bad_zip_member = zf.testzip()
            cls.names = set(zf.namelist())
            cls.xml_parts = {
                name: zf.read(name)
                for name in cls.names
                if name.endswith(".xml") or name.endswith(".rels")
            }
        cls.doc_xml = etree.fromstring(cls.xml_parts["word/document.xml"])
        cls.rels_xml = etree.fromstring(cls.xml_parts["word/_rels/document.xml.rels"])
        cls.settings_xml = etree.fromstring(cls.xml_parts["word/settings.xml"])

    def test_01_valid_ooxml_zip_and_image_relationships(self):
        """Verify valid ZIP archive, well-formed XML parts, and resolved media relationships."""
        self.assertIsNone(self.bad_zip_member)
        for name, raw in self.xml_parts.items():
            etree.fromstring(raw)

        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in self.rels_xml.findall("{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        }
        blips = self.doc_xml.xpath(".//*[local-name()='blip']")
        self.assertEqual(len(blips), 9, "Expected all 9 embedded figure blips")
        for blip in blips:
            r_id = blip.attrib.get(f"{{{R_NS}}}embed")
            self.assertIn(r_id, rel_targets)
            media_path = "word/" + rel_targets[r_id].lstrip("/")
            self.assertIn(media_path, self.names)

    def test_02_ieee_letter_geometry_sections_and_hyphenation(self):
        """Verify US Letter size, exact IEEE margins across all sections, 1-col/2-col counts, and auto-hyphenation."""
        sections = self.doc.sections
        # 12 content sections + 1 trailing continuous balancing section = 13
        self.assertEqual(len(sections), 13)

        expected_cols = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        for idx, (sec, exp_col) in enumerate(zip(sections, expected_cols)):
            self.assertEqual(sec.page_width.twips, _dxa(8.5), f"Section {idx} width")
            self.assertEqual(sec.page_height.twips, _dxa(11.0), f"Section {idx} height")
            self.assertEqual(sec.top_margin.twips, _dxa(0.75), f"Section {idx} top margin")
            self.assertEqual(sec.bottom_margin.twips, _dxa(1.00), f"Section {idx} bottom margin")
            self.assertEqual(sec.left_margin.twips, _dxa(0.625), f"Section {idx} left margin")
            self.assertEqual(sec.right_margin.twips, _dxa(0.625), f"Section {idx} right margin")

            cols_el = sec._sectPr.find(qn("w:cols"))
            self.assertIsNotNone(cols_el, f"Section {idx} missing <w:cols>")
            self.assertEqual(int(cols_el.get(qn("w:num"), "1")), exp_col, f"Section {idx} col count")
            if exp_col == 2:
                self.assertEqual(int(cols_el.get(qn("w:space"))), 360)

            self.assertEqual("".join(p.text for p in sec.header.paragraphs).strip(), "")
            self.assertEqual("".join(p.text for p in sec.footer.paragraphs).strip(), "")

        self.assertIsNotNone(self.settings_xml.find(qn("w:autoHyphenation")))
        hz = self.settings_xml.find(qn("w:hyphenationZone"))
        self.assertIsNotNone(hz)
        self.assertEqual(hz.get(qn("w:val")), "360")

    def test_03_native_omml_equations_and_schema_order(self):
        """Verify 7 native OMML equations in 3.50 in borderless tables with <w:rPr> strictly before <m:t>."""
        omaths = self.doc_xml.xpath(".//*[local-name()='oMath']")
        self.assertEqual(len(omaths), 7)

        for eq_idx, om in enumerate(omaths, 1):
            runs = om.xpath(f".//*[local-name()='r' and namespace-uri()='{M_NS}']")
            self.assertGreater(len(runs), 0, f"Equation ({eq_idx}) has no <m:r> runs")
            for mr in runs:
                tags = [etree.QName(child).localname for child in mr]
                if "t" in tags and "rPr" in tags:
                    # Both <m:rPr> and <w:rPr> use localname 'rPr' and must precede <m:t>
                    last_rpr_idx = max(i for i, t in enumerate(tags) if t == "rPr")
                    first_t_idx = tags.index("t")
                    self.assertLess(last_rpr_idx, first_t_idx, f"Eq ({eq_idx}) has rPr after m:t: {tags}")

        # First 7 tables are Eq (1)-(7) tables with exact total width 3.50 in (5040 dxa)
        eq_tables = []
        for tbl in self.doc.tables:
            if tbl.columns and len(tbl.rows) == 1 and len(tbl.columns) == 3:
                txt = tbl.rows[0].cells[2].text.strip()
                if txt.startswith("(") and txt.endswith(")"):
                    eq_tables.append((tbl, txt))
        self.assertEqual([lbl for _, lbl in eq_tables], [f"({i})" for i in range(1, 8)])
        for tbl, lbl in eq_tables:
            grid_sum = sum(int(gc.get(qn("w:w"))) for gc in tbl._tbl.tblGrid.gridCol_lst)
            self.assertAlmostEqual(grid_sum, _dxa(3.50), delta=2, msg=f"Equation table {lbl} width")

    def test_04_fixed_width_ieee_data_tables(self):
        """Verify TABLE I-IV fixed layout, exact widths (3.50 or 7.25 in), tblHeader, cantSplit, zero vertical rules."""
        data_tables = [t for t in self.doc.tables if len(t.rows) >= 4]
        self.assertEqual(len(data_tables), 4, "Expected TABLE I, II, III, IV")

        expected_widths_in = [3.50, 7.25, 3.50, 3.50]
        for idx, (tbl, exp_in) in enumerate(zip(data_tables, expected_widths_in), 1):
            tbl_pr = tbl._tbl.tblPr
            layout = tbl_pr.find(qn("w:tblLayout"))
            self.assertIsNotNone(layout)
            self.assertEqual(layout.get(qn("w:type")), "fixed")

            grid_sum = sum(int(gc.get(qn("w:w"))) for gc in tbl._tbl.tblGrid.gridCol_lst)
            self.assertAlmostEqual(grid_sum, _dxa(exp_in), delta=5, msg=f"TABLE {idx} gridCol sum")

            header_tr_pr = tbl.rows[0]._tr.trPr
            self.assertIsNotNone(header_tr_pr.find(qn("w:tblHeader")))
            for row in tbl.rows:
                self.assertIsNotNone(row._tr.trPr.find(qn("w:cantSplit")))

            borders = tbl_pr.find(qn("w:tblBorders"))
            self.assertEqual(borders.find(qn("w:insideV")).get(qn("w:val")), "none")
            self.assertEqual(borders.find(qn("w:left")).get(qn("w:val")), "none")
            self.assertEqual(borders.find(qn("w:right")).get(qn("w:val")), "none")

    def test_05_embedded_figures_aspect_ratio_and_caption_lock(self):
        """Verify all 9 figures preserve natural aspect ratios within container bounds and lock onto captions."""
        expected_figs = [
            ("01_dataset_eda.png", 3.42, 3.46),
            ("arch_01_system_pipeline.png", 7.05, 7.15),
            ("arch_02_graphgps_layer.png", 3.42, 3.46),
            ("02_ssl_pretraining.png", 7.05, 7.15),
            ("03_training_curves.png", 3.42, 3.46),
            ("04_roc_comparison.png", 7.05, 7.15),
            ("05_regression_parity.png", 3.42, 3.46),
            ("06_pareto_optimization.png", 7.05, 7.15),
            ("07_atom_saliency_xai.png", 7.05, 7.15),
        ]
        shapes = list(self.doc.inline_shapes)
        self.assertEqual(len(shapes), 9)

        img_paras = [p for p in self.doc.paragraphs if p.style.name == "IEEE Figure Image"]
        self.assertEqual(len(img_paras), 9)
        for p in img_paras:
            effective_kwn = (
                p.paragraph_format.keep_with_next
                if p.paragraph_format.keep_with_next is not None
                else p.style.paragraph_format.keep_with_next
            )
            self.assertTrue(effective_kwn)

        for shape, (fn, exp_w_in, max_w_in) in zip(shapes, expected_figs):
            w_in = shape.width / 914400.0
            h_in = shape.height / 914400.0
            self.assertLessEqual(w_in, max_w_in + 1e-4, f"{fn} exceeds container width")
            self.assertAlmostEqual(w_in, exp_w_in, delta=0.02, msg=f"{fn} width mismatch")
            with Image.open(os.path.join(FIG_DIR, fn)) as im:
                px_w, px_h = im.size
            nat_aspect = px_w / px_h
            doc_aspect = w_in / h_in
            self.assertAlmostEqual(doc_aspect, nat_aspect, delta=nat_aspect * 0.005, msg=f"{fn} aspect distorted")

    def test_06_sections_citations_and_live_ground_truth_metrics(self):
        """Verify all Roman sections, 28 IEEE references, and exact metrics matching JSON artifacts."""
        all_text_parts = [p.text for p in self.doc.paragraphs]
        for tbl in self.doc.tables:
            for row in tbl.rows:
                for cell in row.cells:
                    all_text_parts.append(cell.text)
        full_text = "\n".join(all_text_parts)

        for token in [
            "Abstract\u2014",
            "Index Terms\u2014",
            "I.  Introduction",
            "II.  Related Work",
            "III.  Dataset Curation and Zero-Leakage Protocol",
            "IV.  Proposed Methodology",
            "V.  Experimental Setup and Ablation Results",
            "VI.  Lead Optimization Case Study and Explainability",
            "VII.  Discussion, Honest Limitations, and Future Work",
            "VIII.  Conclusion",
            "References",
        ]:
            self.assertIn(token, full_text)

        for i in range(1, 29):
            self.assertIn(f"[{i}]", full_text, f"Missing reference [{i}]")
        for i in range(1, 10):
            self.assertIn(f"Fig. {i}.", full_text, f"Missing Fig. {i}.")

        with open(os.path.join(BASE_DIR, "results", "metrics_summary.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
        ours = metrics["SSL-GraphGPS (Ours)"]
        scratch = metrics["GraphGPS-Scratch"]
        gain_macro = ours["aggregate"]["macro_roc_auc"] - scratch["aggregate"]["macro_roc_auc"]

        for num_str in [
            "5,007",
            "3,137",
            "5,000",
            f"{ours['aggregate']['macro_roc_auc']:.4f}",
            f"{ours['bioavailability']['roc_auc']:.4f}",
            f"{ours['bbb']['roc_auc']:.4f}",
            f"{ours['herg']['roc_auc']:.4f}",
            f"{ours['qed']['r2']:.4f}",
            f"{ours['sascore']['r2']:.4f}",
            f"+{gain_macro:.4f}",
        ]:
            self.assertIn(num_str, full_text, f"Missing verified ground-truth value: {num_str}")


if __name__ == "__main__":
    unittest.main()
