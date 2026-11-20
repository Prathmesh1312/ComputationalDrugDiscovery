"""
build_ieee_docx.py — Build Authentic Editable IEEE Two-Column Conference Word Document (.docx)
Conforms to official IEEE US Letter conference formatting, uses named Word styles,
native editable OMML equations via Pandoc, fixed-layout booktabs tables, and live metrics.
"""
import copy
import io
import json
import os
import subprocess
import sys
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import docx
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_DIR = os.path.join(BASE_DIR, "paper")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
OUT_DOCX = os.path.join(PAPER_DIR, "IEEE_Research_Paper.docx")
PANDOC_BIN = "/opt/homebrew/bin/pandoc"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# Official IEEE US Letter dimensions & margins (1440 dxa = 1 inch)
PAGE_W_IN = 8.5
PAGE_H_IN = 11.0
MARGIN_TOP_IN = 0.75
MARGIN_BOTTOM_IN = 1.00
MARGIN_LR_IN = 0.625
HEADER_DIST_IN = 0.35
FOOTER_DIST_IN = 0.50
GUTTER_DXA = 360  # 0.25 in between the two 3.50 in columns
FULL_WIDTH_IN = 7.25
COL_WIDTH_IN = 3.50
FONT_FAMILY = "Times New Roman"
CODE_FAMILY = "Courier New"
HIGHLIGHT_FILL = "F2F6FC"

Segment = Union[str, Tuple[str, ...]]


def _dxa(inches: float) -> int:
    return int(round(inches * 1440.0))


def _set_rfonts(rpr: etree._Element, font_name: str = FONT_FAMILY) -> None:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)
    rfonts.set(qn("w:cs"), font_name)
    rfonts.set(qn("w:eastAsia"), font_name)


def _set_style_font(style, name: str, size_pt: float, bold: bool = False, italic: bool = False, small_caps: bool = False) -> None:
    style.font.name = name
    style.font.size = Pt(size_pt)
    style.font.bold = bold
    style.font.italic = italic
    style.font.small_caps = small_caps
    style.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    rpr = style.element.get_or_add_rPr()
    _set_rfonts(rpr, name)


def _set_outline_level(style, level: int) -> None:
    ppr = style.element.get_or_add_pPr()
    existing = ppr.find(qn("w:outlineLvl"))
    if existing is not None:
        ppr.remove(existing)
    el = OxmlElement("w:outlineLvl")
    el.set(qn("w:val"), str(level))
    ppr.append(el)


def _register_styles(doc: docx.Document) -> None:
    styles = doc.styles

    # Normal baseline
    normal = styles["Normal"]
    _set_style_font(normal, FONT_FAMILY, 10.0)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing = 1.05
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(3.0)
    normal.paragraph_format.widow_control = True

    def make_style(name: str, base_name: str = "Normal") -> Any:
        if name in styles:
            return styles[name]
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = styles[base_name]
        st.quick_style = True
        return st

    # Title / Authors (1-col top banner)
    s_title = make_style("IEEE Paper Title")
    _set_style_font(s_title, FONT_FAMILY, 24.0, bold=False)
    s_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_title.paragraph_format.line_spacing = 1.08
    s_title.paragraph_format.space_before = Pt(0)
    s_title.paragraph_format.space_after = Pt(10.0)
    s_title.paragraph_format.keep_with_next = True

    s_author = make_style("IEEE Author Name")
    _set_style_font(s_author, FONT_FAMILY, 11.0, bold=False)
    s_author.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_author.paragraph_format.line_spacing = 1.05
    s_author.paragraph_format.space_before = Pt(0)
    s_author.paragraph_format.space_after = Pt(2.0)
    s_author.paragraph_format.keep_with_next = True

    s_affil = make_style("IEEE Author Affiliation")
    _set_style_font(s_affil, FONT_FAMILY, 10.0, bold=False, italic=True)
    s_affil.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_affil.paragraph_format.line_spacing = 1.05
    s_affil.paragraph_format.space_before = Pt(0)
    s_affil.paragraph_format.space_after = Pt(1.0)
    s_affil.paragraph_format.keep_with_next = True

    s_email = make_style("IEEE Author Email")
    _set_style_font(s_email, FONT_FAMILY, 9.0, bold=False)
    s_email.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_email.paragraph_format.line_spacing = 1.05
    s_email.paragraph_format.space_before = Pt(1.0)
    s_email.paragraph_format.space_after = Pt(12.0)

    # Abstract & Index Terms
    s_abs = make_style("IEEE Abstract")
    _set_style_font(s_abs, FONT_FAMILY, 9.0, bold=True)
    s_abs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    s_abs.paragraph_format.first_line_indent = Inches(0)
    s_abs.paragraph_format.line_spacing = 1.05
    s_abs.paragraph_format.space_before = Pt(0)
    s_abs.paragraph_format.space_after = Pt(4.0)
    s_abs.paragraph_format.widow_control = True

    s_kw = make_style("IEEE Index Terms")
    _set_style_font(s_kw, FONT_FAMILY, 9.0, bold=True)
    s_kw.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    s_kw.paragraph_format.first_line_indent = Inches(0)
    s_kw.paragraph_format.line_spacing = 1.05
    s_kw.paragraph_format.space_before = Pt(0)
    s_kw.paragraph_format.space_after = Pt(8.0)
    s_kw.paragraph_format.widow_control = True

    # Headings
    s_h1 = make_style("IEEE Heading 1")
    _set_style_font(s_h1, FONT_FAMILY, 10.0, bold=False, small_caps=True)
    s_h1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_h1.paragraph_format.first_line_indent = Inches(0)
    s_h1.paragraph_format.line_spacing = 1.05
    s_h1.paragraph_format.space_before = Pt(10.0)
    s_h1.paragraph_format.space_after = Pt(4.0)
    s_h1.paragraph_format.keep_with_next = True
    _set_outline_level(s_h1, 0)

    s_h2 = make_style("IEEE Heading 2")
    _set_style_font(s_h2, FONT_FAMILY, 10.0, bold=False, italic=True)
    s_h2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    s_h2.paragraph_format.first_line_indent = Inches(0)
    s_h2.paragraph_format.line_spacing = 1.05
    s_h2.paragraph_format.space_before = Pt(6.0)
    s_h2.paragraph_format.space_after = Pt(2.5)
    s_h2.paragraph_format.keep_with_next = True
    _set_outline_level(s_h2, 1)

    # Body paragraphs
    s_body = make_style("IEEE Body")
    _set_style_font(s_body, FONT_FAMILY, 10.0)
    s_body.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    s_body.paragraph_format.first_line_indent = Inches(0.15)
    s_body.paragraph_format.line_spacing = 1.05
    s_body.paragraph_format.space_before = Pt(0)
    s_body.paragraph_format.space_after = Pt(3.0)
    s_body.paragraph_format.widow_control = True

    s_body_first = make_style("IEEE Body First", base_name="IEEE Body")
    s_body_first.paragraph_format.first_line_indent = Inches(0)

    # Lists (0.18 in marker width avoids 10pt "1)" overflowing to 0.50 in default tab)
    s_list = make_style("IEEE List Item")
    _set_style_font(s_list, FONT_FAMILY, 10.0)
    s_list.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    s_list.paragraph_format.left_indent = Inches(0.28)
    s_list.paragraph_format.first_line_indent = Inches(-0.18)
    s_list.paragraph_format.line_spacing = 1.05
    s_list.paragraph_format.space_before = Pt(0)
    s_list.paragraph_format.space_after = Pt(2.5)
    s_list.paragraph_format.widow_control = True
    s_list.paragraph_format.tab_stops.add_tab_stop(Inches(0.28), WD_TAB_ALIGNMENT.LEFT)

    # Tables & Figures
    s_tlbl = make_style("IEEE Table Label")
    _set_style_font(s_tlbl, FONT_FAMILY, 8.0)
    s_tlbl.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_tlbl.paragraph_format.first_line_indent = Inches(0)
    s_tlbl.paragraph_format.line_spacing = 1.05
    s_tlbl.paragraph_format.space_before = Pt(6.0)
    s_tlbl.paragraph_format.space_after = Pt(1.0)
    s_tlbl.paragraph_format.keep_with_next = True

    s_ttitle = make_style("IEEE Table Title")
    _set_style_font(s_ttitle, FONT_FAMILY, 8.0, small_caps=True)
    s_ttitle.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_ttitle.paragraph_format.first_line_indent = Inches(0)
    s_ttitle.paragraph_format.line_spacing = 1.05
    s_ttitle.paragraph_format.space_before = Pt(0)
    s_ttitle.paragraph_format.space_after = Pt(3.0)
    s_ttitle.paragraph_format.keep_with_next = True

    s_fimg = make_style("IEEE Figure Image")
    _set_style_font(s_fimg, FONT_FAMILY, 8.0)
    s_fimg.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_fimg.paragraph_format.first_line_indent = Inches(0)
    s_fimg.paragraph_format.line_spacing = 1.0
    s_fimg.paragraph_format.space_before = Pt(6.0)
    s_fimg.paragraph_format.space_after = Pt(2.5)
    s_fimg.paragraph_format.keep_with_next = True

    s_fcap = make_style("IEEE Figure Caption")
    _set_style_font(s_fcap, FONT_FAMILY, 8.0)
    s_fcap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    s_fcap.paragraph_format.first_line_indent = Inches(0)
    s_fcap.paragraph_format.line_spacing = 1.06
    s_fcap.paragraph_format.space_before = Pt(0)
    s_fcap.paragraph_format.space_after = Pt(6.0)
    s_fcap.paragraph_format.widow_control = True

    # References
    s_ref = make_style("IEEE Reference")
    _set_style_font(s_ref, FONT_FAMILY, 8.0)
    s_ref.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    s_ref.paragraph_format.left_indent = Inches(0.26)
    s_ref.paragraph_format.first_line_indent = Inches(-0.26)
    s_ref.paragraph_format.line_spacing = 1.05
    s_ref.paragraph_format.space_before = Pt(0)
    s_ref.paragraph_format.space_after = Pt(2.0)
    s_ref.paragraph_format.keep_together = True
    s_ref.paragraph_format.widow_control = True
    s_ref.paragraph_format.tab_stops.add_tab_stop(Inches(0.26), WD_TAB_ALIGNMENT.LEFT)


def _configure_section(section, cols: int = 2) -> None:
    """Apply IEEE US Letter page size, margins, blank header/footer, and columns."""
    section.page_width = Inches(PAGE_W_IN)
    section.page_height = Inches(PAGE_H_IN)
    section.top_margin = Inches(MARGIN_TOP_IN)
    section.bottom_margin = Inches(MARGIN_BOTTOM_IN)
    section.left_margin = Inches(MARGIN_LR_IN)
    section.right_margin = Inches(MARGIN_LR_IN)
    section.header_distance = Inches(HEADER_DIST_IN)
    section.footer_distance = Inches(FOOTER_DIST_IN)

    # Clear any inherited header/footer text
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False

    sect_pr = section._sectPr
    existing_cols = sect_pr.find(qn("w:cols"))
    if existing_cols is not None:
        sect_pr.remove(existing_cols)

    cols_el = OxmlElement("w:cols")
    if cols == 2:
        cols_el.set(qn("w:num"), "2")
        cols_el.set(qn("w:space"), str(GUTTER_DXA))
        cols_el.set(qn("w:equalWidth"), "1")
    else:
        cols_el.set(qn("w:num"), "1")
        cols_el.set(qn("w:space"), "720")

    # Place <w:cols> in valid CT_SectPr order (right before docGrid if present)
    doc_grid = sect_pr.find(qn("w:docGrid"))
    if doc_grid is not None:
        sect_pr.insert(sect_pr.index(doc_grid), cols_el)
    else:
        sect_pr.append(cols_el)


def _add_continuous_section(doc: docx.Document, cols: int) -> Any:
    """Add a continuous section break and format its anchor paragraph tightly (no blank gap)."""
    new_sec = doc.add_section(WD_SECTION.CONTINUOUS)
    _configure_section(new_sec, cols=cols)
    # Compact the hidden paragraph holding the previous section's <w:sectPr>
    sect_paras = doc._body._body.xpath("./w:p[w:pPr/w:sectPr]")
    if sect_paras:
        p_el = sect_paras[-1]
        ppr = p_el.find(qn("w:pPr"))
        for tag in ("w:spacing", "w:rPr"):
            old = ppr.find(qn(tag))
            if old is not None:
                ppr.remove(old)
        spacing = OxmlElement("w:spacing")
        spacing.set(qn("w:before"), "0")
        spacing.set(qn("w:after"), "0")
        spacing.set(qn("w:line"), "20")
        spacing.set(qn("w:lineRule"), "exact")
        ppr.insert(0, spacing)
        rpr = OxmlElement("w:rPr")
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "2")
        rpr.append(sz)
        ppr.insert(1, rpr)
    return new_sec


def _enable_auto_hyphenation(doc: docx.Document) -> None:
    """Enable automatic Word hyphenation in word/settings.xml using strict CT_Settings order."""
    settings_el = doc.settings.element
    for tag in ("w:autoHyphenation", "w:hyphenationZone"):
        old = settings_el.find(qn(tag))
        if old is not None:
            settings_el.remove(old)

    auto_hyph = OxmlElement("w:autoHyphenation")
    hyph_zone = OxmlElement("w:hyphenationZone")
    hyph_zone.set(qn("w:val"), "360")

    anchor = settings_el.find(qn("w:defaultTabStop"))
    if anchor is not None:
        idx = settings_el.index(anchor) + 1
        settings_el.insert(idx, auto_hyph)
        settings_el.insert(idx + 1, hyph_zone)
    else:
        settings_el.insert(0, auto_hyph)
        settings_el.insert(1, hyph_zone)


def _emit_rich(
    p,
    segments: Sequence[Segment],
    code_pt: float = 8.5,
    base_font: str = FONT_FAMILY,
    inherit_size_pt: Optional[float] = None,
) -> None:
    """Append mixed formatted runs cleanly to paragraph `p`."""
    for seg in segments:
        if isinstance(seg, str):
            r = p.add_run(seg)
            _set_rfonts(r._r.get_or_add_rPr(), base_font)
            if inherit_size_pt is not None:
                r.font.size = Pt(inherit_size_pt)
            continue

        tag = seg[0]
        if tag == "b":
            r = p.add_run(seg[1])
            r.bold = True
            _set_rfonts(r._r.get_or_add_rPr(), base_font)
        elif tag == "i":
            r = p.add_run(seg[1])
            r.italic = True
            _set_rfonts(r._r.get_or_add_rPr(), base_font)
        elif tag == "bi":
            r = p.add_run(seg[1])
            r.bold = True
            r.italic = True
            _set_rfonts(r._r.get_or_add_rPr(), base_font)
        elif tag == "code":
            r = p.add_run(seg[1])
            r.font.name = CODE_FAMILY
            r.font.size = Pt(code_pt)
            _set_rfonts(r._r.get_or_add_rPr(), CODE_FAMILY)
        elif tag == "sup":
            r = p.add_run(seg[1])
            r.font.superscript = True
            _set_rfonts(r._r.get_or_add_rPr(), base_font)
        elif tag == "sub":
            r = p.add_run(seg[1])
            r.font.subscript = True
            _set_rfonts(r._r.get_or_add_rPr(), base_font)
        elif tag == "isub":
            # (var, sub) with optional 4th element italic_sub bool
            var_txt, sub_txt = seg[1], seg[2]
            sub_italic = seg[3] if len(seg) > 3 else True
            r1 = p.add_run(var_txt)
            r1.italic = True
            _set_rfonts(r1._r.get_or_add_rPr(), base_font)
            r2 = p.add_run(sub_txt)
            r2.italic = sub_italic
            r2.font.subscript = True
            _set_rfonts(r2._r.get_or_add_rPr(), base_font)
        elif tag == "isup":
            var_txt, sup_txt = seg[1], seg[2]
            r1 = p.add_run(var_txt)
            r1.italic = True
            _set_rfonts(r1._r.get_or_add_rPr(), base_font)
            r2 = p.add_run(sup_txt)
            r2.font.superscript = True
            _set_rfonts(r2._r.get_or_add_rPr(), base_font)
        elif tag == "isubsup":
            var_txt, sub_txt, sup_txt = seg[1], seg[2], seg[3]
            r1 = p.add_run(var_txt)
            r1.italic = True
            _set_rfonts(r1._r.get_or_add_rPr(), base_font)
            r2 = p.add_run(sub_txt)
            r2.italic = True
            r2.font.subscript = True
            _set_rfonts(r2._r.get_or_add_rPr(), base_font)
            r3 = p.add_run(sup_txt)
            r3.font.superscript = True
            _set_rfonts(r3._r.get_or_add_rPr(), base_font)
        else:
            raise ValueError(f"Unknown rich-text segment tag: {tag}")


def _add_p(doc: docx.Document, style_name: str, segments: Sequence[Segment], code_pt: float = 8.5) -> Any:
    p = doc.add_paragraph(style=style_name)
    _emit_rich(p, segments, code_pt=code_pt)
    return p


def _latex_to_omml(latex_str: str, font_pt: float = 8.5) -> etree._Element:
    """Compile LaTeX display math through Pandoc to native editable Word OMML (<m:oMath>)."""
    proc = subprocess.run(
        [PANDOC_BIN, "-f", "latex", "-t", "docx", "-o", "-"],
        input=f"$${latex_str}$$\n".encode("utf-8"),
        capture_output=True,
        check=True,
    )
    with zipfile.ZipFile(io.BytesIO(proc.stdout)) as zf:
        root = etree.fromstring(zf.read("word/document.xml"))
    omaths = root.xpath(".//*[local-name()='oMath']")
    if not omaths:
        raise RuntimeError(f"Pandoc produced no OMML for equation: {latex_str}")

    omath = copy.deepcopy(omaths[0])
    sz_half_pt = str(int(round(font_pt * 2)))

    def _make_wrpr() -> etree._Element:
        wrpr = OxmlElement("w:rPr")
        rfonts = OxmlElement("w:rFonts")
        rfonts.set(qn("w:ascii"), "Cambria Math")
        rfonts.set(qn("w:hAnsi"), "Cambria Math")
        rfonts.set(qn("w:cs"), "Cambria Math")
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), sz_half_pt)
        sz_cs = OxmlElement("w:szCs")
        sz_cs.set(qn("w:val"), sz_half_pt)
        wrpr.append(rfonts)
        wrpr.append(sz)
        wrpr.append(sz_cs)
        return wrpr

    # CRITICAL OOXML ORDER inside <m:r>: <m:rPr> (optional), <w:rPr> (optional), <m:t>
    for mr in omath.xpath(".//*[local-name()='r' and namespace-uri()='http://schemas.openxmlformats.org/officeDocument/2006/math']"):
        existing_wrpr = mr.find(qn("w:rPr"))
        if existing_wrpr is not None:
            mr.remove(existing_wrpr)
        mrpr = mr.find(f"{{{M_NS}}}rPr")
        insert_idx = (mr.index(mrpr) + 1) if mrpr is not None else 0
        mr.insert(insert_idx, _make_wrpr())

    for ctrl in omath.xpath(".//*[local-name()='ctrlPr' and namespace-uri()='http://schemas.openxmlformats.org/officeDocument/2006/math']"):
        existing_wrpr = ctrl.find(qn("w:rPr"))
        if existing_wrpr is not None:
            ctrl.remove(existing_wrpr)
        ctrl.insert(0, _make_wrpr())

    return omath


def _set_cell_margins(tc_pr: etree._Element, top: int = 36, bottom: int = 36, left: int = 50, right: int = 50) -> None:
    old = tc_pr.find(qn("w:tcMar"))
    if old is not None:
        tc_pr.remove(old)
    tcmar = OxmlElement("w:tcMar")
    for edge, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        tcmar.append(el)
    tc_pr.append(tcmar)


def _set_cell_width_and_shading(cell, width_dxa: int, fill_hex: Optional[str] = None, margins: Tuple[int, int, int, int] = (36, 36, 50, 50)) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    old_w = tc_pr.find(qn("w:tcW"))
    if old_w is not None:
        tc_pr.remove(old_w)
    tcw = OxmlElement("w:tcW")
    tcw.set(qn("w:w"), str(width_dxa))
    tcw.set(qn("w:type"), "dxa")
    tc_pr.insert(0, tcw)

    old_shd = tc_pr.find(qn("w:shd"))
    if old_shd is not None:
        tc_pr.remove(old_shd)
    if fill_hex is not None:
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill_hex}"/>')
        tc_pr.append(shd)

    _set_cell_margins(tc_pr, *margins)


def _configure_fixed_table(table, col_widths_in: Sequence[float]) -> List[int]:
    col_widths_dxa = [_dxa(w) for w in col_widths_in]
    total_dxa = sum(col_widths_dxa)

    tbl = table._tbl
    tbl_pr = tbl.tblPr
    for tag in ("w:tblW", "w:tblBorders", "w:tblLayout", "w:tblLook"):
        old = tbl_pr.find(qn(tag))
        if old is not None:
            tbl_pr.remove(old)

    tbl_w = OxmlElement("w:tblW")
    tbl_w.set(qn("w:w"), str(total_dxa))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_pr.append(tbl_w)

    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)

    old_grid = tbl.find(qn("w:tblGrid"))
    if old_grid is not None:
        tbl.remove(old_grid)
    grid = OxmlElement("w:tblGrid")
    for w_dxa in col_widths_dxa:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w_dxa))
        grid.append(gc)
    tbl.insert(tbl.index(tbl_pr) + 1, grid)

    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    return col_widths_dxa


def _add_equation(doc: docx.Document, latex_str: str, eq_num: int, font_pt: float = 8.5) -> None:
    """Insert a numbered IEEE display equation with native Word OMML in a borderless 1x3 fixed table."""
    omath = _latex_to_omml(latex_str, font_pt=font_pt)

    tbl = doc.add_table(rows=1, cols=3)
    col_dxa = _configure_fixed_table(tbl, (0.04, 3.14, 0.32))

    # Zero table borders
    tbl_pr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
    )
    tbl_pr.insert(tbl_pr.index(tbl_pr.find(qn("w:tblLayout"))), borders)

    row = tbl.rows[0]
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))

    for c_idx, cell in enumerate(row.cells):
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        _set_cell_width_and_shading(cell, col_dxa[c_idx], margins=(20, 20, 0, 0))
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2.0)
        p.paragraph_format.space_after = Pt(2.0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.first_line_indent = Inches(0)

    # Col 0 spacer (keep minimal 1 pt font)
    p0 = row.cells[0].paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r0 = p0.add_run("")
    r0.font.size = Pt(font_pt)

    # Col 1 centered native OMML math
    p1 = row.cells[1].paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1._p.append(omath)

    # Col 2 right-aligned equation number (n)
    p2 = row.cells[2].paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rn = p2.add_run(f"({eq_num})")
    rn.font.name = FONT_FAMILY
    rn.font.size = Pt(9.5)
    _set_rfonts(rn._r.get_or_add_rPr(), FONT_FAMILY)


def _set_cell_horizontal_border(cell, top_sz: Optional[int] = None, bottom_sz: Optional[int] = None) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    old = tc_pr.find(qn("w:tcBorders"))
    if old is not None:
        tc_pr.remove(old)
    top_xml = f'<w:top w:val="single" w:sz="{top_sz}" w:space="0" w:color="000000"/>' if top_sz else '<w:top w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
    bot_xml = f'<w:bottom w:val="single" w:sz="{bottom_sz}" w:space="0" w:color="000000"/>' if bottom_sz else '<w:bottom w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f"{top_xml}"
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        f"{bot_xml}"
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        "</w:tcBorders>"
    )
    tcw = tc_pr.find(qn("w:tcW"))
    tc_pr.insert(tc_pr.index(tcw) + 1 if tcw is not None else 0, borders)


def _add_ieee_table(
    doc: docx.Document,
    roman_label: str,
    title_text: str,
    headers: Sequence[Sequence[Segment]],
    rows: Sequence[Sequence[Sequence[Segment]]],
    col_widths_in: Sequence[float],
    font_pt: float = 8.0,
    lr_pad_dxa: int = 44,
    midrule_rows: Sequence[int] = (),
    highlight_rows: Sequence[int] = (),
    bold_rows: Sequence[int] = (),
    italic_rows: Sequence[int] = (),
) -> None:
    """Create a strict fixed-width IEEE booktabs-style table with header repeat & zero vertical rules."""
    _add_p(doc, "IEEE Table Label", [roman_label])
    _add_p(doc, "IEEE Table Title", [title_text])

    num_rows = 1 + len(rows)
    num_cols = len(headers)
    tbl = doc.add_table(rows=num_rows, cols=num_cols)
    col_dxa = _configure_fixed_table(tbl, col_widths_in)

    tbl_pr = tbl._tbl.tblPr
    tbl_borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:left w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
        '<w:right w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="none" w:sz="0" w:space="0" w:color="auto"/>'
        "</w:tblBorders>"
    )
    tbl_pr.insert(tbl_pr.index(tbl_pr.find(qn("w:tblLayout"))), tbl_borders)

    all_rows_data = [headers] + list(rows)
    for r_idx, row_data in enumerate(all_rows_data):
        row = tbl.rows[r_idx]
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if r_idx == 0:
            tr_pr.append(OxmlElement("w:tblHeader"))

        data_idx = r_idx - 1  # index in `rows` for body rows
        is_header = r_idx == 0
        is_last = r_idx == (num_rows - 1)
        is_highlighted = (not is_header) and (data_idx in highlight_rows)
        is_bold_row = is_header or (data_idx in bold_rows)
        is_italic_row = (not is_header) and (data_idx in italic_rows)

        top_border = 12 if is_header else (4 if data_idx in midrule_rows else None)
        bot_border = 6 if is_header else (12 if is_last else None)

        for c_idx in range(num_cols):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_width_and_shading(
                cell,
                col_dxa[c_idx],
                fill_hex=HIGHLIGHT_FILL if is_highlighted else None,
                margins=(32, 32, lr_pad_dxa, lr_pad_dxa),
            )
            _set_cell_horizontal_border(cell, top_sz=top_border, bottom_sz=bot_border)

            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.line_spacing = 1.04
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.space_after = Pt(0.5)
            p.paragraph_format.keep_with_next = not is_last

            segs = row_data[c_idx]
            _emit_rich(p, segs, code_pt=font_pt - 0.5, inherit_size_pt=font_pt)
            for run in p.runs:
                run.font.size = Pt(font_pt)
                if is_bold_row:
                    run.bold = True
                if is_italic_row:
                    run.italic = True

    # Small breathing spacer after table
    sp = doc.add_paragraph()
    sp.paragraph_format.first_line_indent = Inches(0)
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4.0)
    sp.paragraph_format.line_spacing = 1.0
    sr = sp.add_run("")
    sr.font.size = Pt(2.0)


def _add_ieee_figure(doc: docx.Document, filename: str, fig_num: int, caption_segments: Sequence[Segment], width_in: float) -> None:
    """Embed figure preserving exact pixel aspect ratio and lock it to its justified IEEE caption."""
    img_path = os.path.join(FIG_DIR, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing figure asset: {img_path}")

    p_img = doc.add_paragraph(style="IEEE Figure Image")
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(width_in))  # omit height -> preserves natural aspect ratio

    cap_segs: List[Segment] = [f"Fig. {fig_num}.  "] + list(caption_segments)
    _add_p(doc, "IEEE Figure Caption", cap_segs, code_pt=7.5)


def _format_cell_segments(text: str) -> List[Segment]:
    return [text]


def build_docx() -> str:
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

    # Verified SSL vs Scratch gains directly from JSON
    g_bio = ours["bioavailability"]["roc_auc"] - scratch["bioavailability"]["roc_auc"]
    g_bbb = ours["bbb"]["roc_auc"] - scratch["bbb"]["roc_auc"]
    g_herg = ours["herg"]["roc_auc"] - scratch["herg"]["roc_auc"]
    g_macro = ours["aggregate"]["macro_roc_auc"] - scratch["aggregate"]["macro_roc_auc"]
    g_qed = ours["qed"]["r2"] - scratch["qed"]["r2"]
    g_sa = ours["sascore"]["r2"] - scratch["sascore"]["r2"]
    g_comp = ours["aggregate"]["composite_val_score"] - scratch["aggregate"]["composite_val_score"]
    g_gcn_macro = ours["aggregate"]["macro_roc_auc"] - gcn["aggregate"]["macro_roc_auc"]
    g_gcn_qed = ours["qed"]["r2"] - gcn["qed"]["r2"]

    doc = docx.Document()
    doc.core_properties.title = (
        "Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling, "
        "Explainable Safety Scoring, and Evolutionary Lead Optimization"
    )
    doc.core_properties.author = "Prathmesh Toke"
    doc.core_properties.subject = "IEEE Conference Research Paper — ADMET GraphGPS"
    doc.core_properties.keywords = (
        "ADMET prediction, GraphGPS, Graph Transformer, self-supervised learning, "
        "InfoNCE, Bemis-Murcko scaffold split, Pareto optimization, explainable AI, hERG cardiotoxicity"
    )

    _register_styles(doc)
    _enable_auto_hyphenation(doc)

    # =========================================================================
    # SECTION 1 (1-col): IEEE Title, Author, Affiliation, Email
    # =========================================================================
    _configure_section(doc.sections[0], cols=1)
    _add_p(
        doc,
        "IEEE Paper Title",
        [
            "Self-Supervised 2D+3D Hybrid GraphGPS for Multi-Objective ADMET Profiling, "
            "Explainable Safety Scoring, and Evolutionary Lead Optimization"
        ],
    )
    _add_p(doc, "IEEE Author Name", ["Prathmesh Toke"])
    _add_p(doc, "IEEE Author Affiliation", ["Department of Computer Science and Engineering"])
    _add_p(doc, "IEEE Author Affiliation", ["Computational Drug Discovery & Graph Representation Learning"])
    _add_p(doc, "IEEE Author Email", ["Email: tokeprathmesh54321@gmail.com"])

    # =========================================================================
    # SECTION 2 (2-col): Abstract, Keywords, I, II, III (Table I, Fig 1), IV intro
    # =========================================================================
    _add_continuous_section(doc, cols=2)

    _add_p(
        doc,
        "IEEE Abstract",
        [
            ("bi", "Abstract\u2014"),
            "Pre-clinical drug discovery exhibits over a 90% attrition rate, predominantly driven by poor absorption, "
            "distribution, metabolism, excretion, and toxicity (ADMET) profiles such as fatal hERG cardiotoxicity and "
            "low oral bioavailability. Training supervised graph neural networks (GNNs) on clinical assays is severely "
            "constrained by sparse, imbalanced, and incomplete endpoint tables, while standard message-passing GNNs suffer "
            "from 1-hop local myopia, oversmoothing, and insensitivity to three-dimensional (3D) conformation. Furthermore, "
            "naive random dataset splits leak near-identical chemical scaffolds across training and test sets, yielding inflated "
            "generalization estimates. To address these bottlenecks, we introduce an end-to-end Self-Supervised 2D+3D Hybrid "
            f"GraphGPS framework evaluated on {stats['unlabeled_ssl_count'] + stats['admet_total_count']:,} standardized compounds: "
            f"{stats['unlabeled_ssl_count']:,} unlabeled ZINC250k molecules for leakage-free contrastive pre-training and "
            f"{stats['admet_total_count']:,} unique clinical compounds across {stats['unique_scaffolds']:,} Bemis-Murcko "
            "scaffolds curated from Therapeutics Data Commons (TDC). Each molecular graph combines 29-dimensional chemical and "
            "3D conformer atom features, 8-dimensional bond descriptors, sign-canonicalized Laplacian positional encodings (LapPE), "
            "sign-invariant random-walk structural encodings (RWSE), and ETKDGv3 global 3D shape vectors. A parallel local covalent ",
            ("code", "TransformerConv"),
            " and global all-atom ",
            ("code", "MultiheadAttention"),
            " backbone is pre-trained via undirected-symmetric GraphCL InfoNCE loss "
            f"(converging from {ssl_hist['loss'][0]:.4f} to {min(ssl_hist['loss']):.4f}) and fine-tuned via ",
            ("code", "NaN"),
            "-masked, class-weighted multi-task heads. On a strictly unseen-scaffold held-out test set (",
            ("i", "n"),
            f" = {stats['splits']['test']['count']}), our SSL-GraphGPS achieves "
            f"{ours['aggregate']['macro_roc_auc']:.4f} Macro ROC-AUC across Oral Bioavailability "
            f"({ours['bioavailability']['roc_auc']:.4f}), BBB Permeability ({ours['bbb']['roc_auc']:.4f}), and hERG "
            f"Cardiotoxicity ({ours['herg']['roc_auc']:.4f}), plus ",
            ("isup", "R", "2"),
            f" = {ours['qed']['r2']:.4f} on QED and ",
            ("isup", "R", "2"),
            f" = {ours['sascore']['r2']:.4f} on SAScore\u2014outperforming Random Forest ECFP4 "
            f"({rf['aggregate']['macro_roc_auc']:.4f}), Standard-GCN ({gcn['aggregate']['macro_roc_auc']:.4f}), and "
            f"GraphGPS-Scratch ({scratch['aggregate']['macro_roc_auc']:.4f}) on every endpoint. Finally, pairing 23 medicinal "
            "chemistry SMIRKS rules, BRICS fragment recombination, PAINS/Lipinski filters, NSGA-II Pareto sorting, and "
            "Input \u00d7 Gradient atom saliency yields interpretable, synthesizable lead candidates.",
        ],
        code_pt=8.0,
    )

    _add_p(
        doc,
        "IEEE Index Terms",
        [
            ("bi", "Index Terms\u2014"),
            "ADMET prediction, GraphGPS, Graph Transformer, self-supervised learning, InfoNCE, "
            "Bemis-Murcko scaffold split, Pareto optimization, explainable AI, hERG cardiotoxicity.",
        ],
    )

    # I. INTRODUCTION
    _add_p(doc, "IEEE Heading 1", ["I.  Introduction"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Bringing a single new drug to market requires 10\u201315 years and over $2.6 billion in capitalized research "
            "and development expenditure [1], yet more than 90% of candidates fail during human clinical trials [2]. "
            "Retrospective pharmaceutical analyses demonstrate that poor pharmacokinetics and safety liabilities\u2014collectively "
            "termed ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity)\u2014account for a substantial "
            "fraction of late-stage attrition [2]. Specifically, low oral bioavailability (",
            ("i", "F"),
            " < 20%) prevents therapeutic systemic exposure [4], unintended Blood-Brain Barrier (BBB) penetration triggers "
            "central nervous system (CNS) side effects for peripheral drugs [5], and off-target inhibition of the cardiac hERG "
            "potassium channel induces fatal QT-interval prolongation [6], [7].",
        ],
    )
    _add_p(
        doc,
        "IEEE Body",
        [
            "Despite the urgency of computational ADMET screening, deep learning approaches face four coupled challenges:",
        ],
    )
    intro_challenges: List[List[Segment]] = [
        [
            "1)\t",
            ("i", "Clinical Data Scarcity and Missing Assays: "),
            "Public clinical ADMET tables rarely cover more than several hundred to a few thousand molecules [3], and "
            "experimental endpoints are sparsely observed across compound libraries (resulting in large numbers of missing ",
            ("code", "NaN"),
            " entries when combining assays).",
        ],
        [
            "2)\t",
            ("i", "Architectural Myopia and 2D Blindness: "),
            "Standard Graph Convolutional Networks (GCNs) [16] propagate messages strictly across 1-hop covalent bonds, "
            "suffer oversmoothing when stacked deeply, and omit bond attributes and 3D spatial conformation [13], [18].",
        ],
        [
            "3)\t",
            ("i", "Structural Data Leakage: "),
            "Conventional random splitting places structural analogues sharing identical core rings into both training and "
            "test folds, inflating evaluation metrics relative to true out-of-distribution scaffold generalization [10].",
        ],
        [
            "4)\t",
            ("i", "Single-Objective Lead Design Flaws: "),
            "Optimizing potency or absorption alone frequently degrades cardiac safety or synthetic feasibility, requiring "
            "explicit multi-objective trade-off resolution [26], [27] and chemically interpretable attribution [28].",
        ],
    ]
    for item in intro_challenges:
        _add_p(doc, "IEEE List Item", item)

    _add_p(
        doc,
        "IEEE Body",
        [
            "To resolve all four gaps in a reproducible framework, we make the following concrete contributions:",
        ],
    )
    intro_contribs: List[List[Segment]] = [
        [
            "\u2022\t",
            ("b", "Leakage-Free Real-World Multi-Task Benchmark: "),
            f"We standardize and merge four Therapeutics Data Commons (TDC) clinical datasets [3]\u2013[7] with exact "
            f"RDKit QED [11] and SAScore [12] descriptors into {stats['admet_total_count']:,} compounds across "
            f"{stats['unique_scaffolds']:,} Bemis-Murcko scaffolds [10], alongside {stats['unlabeled_ssl_count']:,} ZINC250k "
            "unlabeled molecules [8], [9] filtered for zero scaffold/SMILES overlap with validation and test sets.",
        ],
        [
            "\u2022\t",
            ("b", "2D+3D Hybrid Local+Global GraphGPS Architecture: "),
            "Extending the GraphGPS recipe [18] and edge-aware attention [17], our encoder combines local bond-aware ",
            ("code", "TransformerConv"),
            " with dense all-atom ",
            ("code", "MultiheadAttention"),
            ", sign-canonicalized Laplacian PE + RWSE [19], and fast ETKDGv3 3D conformer geometry [13].",
        ],
        [
            "\u2022\t",
            ("b", "Undirected-Safe GraphCL Pre-Training & Masked Multi-Task Learning: "),
            "We design undirected-symmetric edge dropping and subgraph node dropping [20], [22] for InfoNCE pre-training [21], "
            "coupled with ",
            ("code", "NaN"),
            "-masked class-weighted logits and normalized regression losses.",
        ],
        [
            "\u2022\t",
            ("b", "Evolutionary Pareto Lead Optimization & Atom Saliency XAI: "),
            "We combine 23 medicinal chemistry SMIRKS transformations, BRICS fragment recombination [23], PAINS/Lipinski "
            "filtering [24], [25], Derringer-Suich desirability [26], NSGA-II Pareto peeling [27], and Input \u00d7 Gradient "
            "attribution [28].",
        ],
    ]
    for item in intro_contribs:
        _add_p(doc, "IEEE List Item", item)

    # II. RELATED WORK
    _add_p(doc, "IEEE Heading 1", ["II.  Related Work"])

    _add_p(doc, "IEEE Heading 2", ["A.  Classical Chemoinformatics and QSAR Modeling"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Quantitative Structure-Activity Relationship (QSAR) baselines historically rely on fixed circular bit-vector "
            "encodings such as Extended-Connectivity Fingerprints (ECFP4) [14] paired with tree ensembles like Random Forests [15]. "
            "While computationally fast and strong at memorizing local functional groups, ECFP4 bit-vectors suffer hash "
            "collisions, omit long-range inter-atomic spatial relationships, discard 3D molecular shape, and train independent "
            "single-task models unable to share pharmacological representations across endpoints.",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["B.  Message-Passing GNNs and Graph Transformers"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Kipf and Welling [16] popularized spectral-inspired Graph Convolutional Networks (GCNs) operating on graph "
            "adjacency matrices. However, pure 1-hop message passing fails to capture non-bonded intramolecular electrostatic "
            "interactions and exhibits oversmoothing across deeper stacks. Shi ",
            ("i", "et al."),
            " [17] introduced masked graph attention incorporating edge embeddings (",
            ("code", "TransformerConv"),
            "), yet attention restricted solely to covalent edges remains locally bounded. Ramp\u00e1\u0161ek ",
            ("i", "et al."),
            " [18] demonstrated in GraphGPS that fusing parallel local message passing with global full-graph self-attention "
            "and structural/positional encodings (LSPE/RWSE [19]) achieves strong expressivity. We adapt GraphGPS specifically "
            "for drug discovery by integrating fast RDKit ETKDGv3 3D conformer descriptors [13] and canonicalizing eigenvector "
            "sign ambiguity.",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["C.  Self-Supervised Molecular Pre-Training"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Inspired by SimCLR [21] in computer vision, GraphCL [20] and MolCLR [22] established contrastive learning over "
            "molecular graphs via InfoNCE loss. A subtle implementation pitfall in molecular contrastive learning is breaking "
            "graph symmetry when dropping directed edge indices independently (~25% asymmetric one-way covalent edges) or "
            "omitting desalting before positional encoding. Our implementation enforces canonical undirected bond symmetry "
            "during edge perturbation and evaluates transfer strictly on unseen Bemis-Murcko scaffolds [10].",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["D.  Multi-Objective Lead Design and Explainability"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Automated molecular design frameworks [8] and fragment assembly toolkits such as BRICS [23] generate candidate "
            "analogues around seed leads, which must obey Lipinski oral drug-likeness constraints [25] and pass Pan-Assay "
            "Interference Compounds (PAINS) structural filters [24]. Balancing conflicting ADMET goals mathematically leverages "
            "Derringer-Suich geometric desirability [26] and NSGA-II non-dominated sorting [27], while Input \u00d7 Gradient "
            "attribution [28] verifies that neural predictions align with medicinal chemistry pharmacophores [6].",
        ],
    )

    # III. DATASET CURATION
    _add_p(doc, "IEEE Heading 1", ["III.  Dataset Curation and Zero-Leakage Protocol"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "To evaluate generalization under realistic medicinal chemistry deployment, all compounds are processed through RDKit ",
            ("code", "LargestFragmentChooser"),
            " desalting (removing counterions such as ",
            ("code", ".[Cl]"),
            "), sanitization, heavy-atom bounds filtering ([2, 80] heavy atoms with carbon presence), and canonical "
            "non-isomeric SMILES deduplication.",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["A.  Multi-Task Clinical ADMET and Unlabeled SSL Corpora"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Clinical binary endpoints are sourced from TDC [3]: ",
            ("code", "Bioavailability_Ma"),
            " [4] (human oral bioavailability ",
            ("i", "F"),
            " \u2265 20%), ",
            ("code", "BBB_Martins"),
            " [5] (blood-brain barrier permeability), ",
            ("code", "hERG"),
            " [6] (",
            ("i", "n"),
            " = 655), and a stratified balanced subsample (",
            ("i", "n"),
            " = 2,500) from ",
            ("code", "hERG_Karim"),
            " [7]. Duplicate canonical SMILES across sources are resolved via assay majority vote (dropping exact 50/50 "
            "conflicts). Continuous QED [11] and SAScore [12] ground-truth targets are computed for all "
            f"{stats['admet_total_count']:,} unique merged molecules (QED = {stats['tasks']['qed']['mean']:.3f} \u00b1 "
            f"{stats['tasks']['qed']['std']:.3f}, SAScore = {stats['tasks']['sascore']['mean']:.3f} \u00b1 "
            f"{stats['tasks']['sascore']['std']:.3f}). For contrastive pre-training, {stats['unlabeled_ssl_count']:,} "
            "drug-like molecules are sampled from ZINC250k [8], [9] after explicitly removing any molecule sharing a canonical "
            "SMILES or Bemis-Murcko scaffold [10] with ADMET validation or test splits.",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["B.  Disjoint Bemis-Murcko Scaffold Splitting"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            f"We partition the {stats['admet_total_count']:,} ADMET molecules into 80% training, 10% validation, and 10% test "
            "sets by grouping compounds on their Bemis-Murcko ring-and-linker scaffolds [10], guaranteeing every validation "
            "and test compound resides on a completely unseen core structure (Table I and Fig. 1). Automated assertions verify "
            "zero scaffold overlap and zero SMILES overlap across all splits while ensuring both binary classes appear in every split.",
        ],
    )

    sp_tr = stats["splits"]["train"]
    sp_va = stats["splits"]["val"]
    sp_te = stats["splits"]["test"]
    t_bio = stats["tasks"]["bioavailability"]
    t_bbb = stats["tasks"]["bbb"]
    t_herg = stats["tasks"]["herg"]

    _add_ieee_table(
        doc,
        roman_label="TABLE I",
        title_text="Curated Multi-Task ADMET Dataset and Disjoint Bemis-Murcko Scaffold Split Summary",
        headers=[["Split / Set"], ["Mols (Scaffolds)"], ["Bioavail. (+%)"], ["BBB (+%)"], ["hERG (+%)"]],
        rows=[
            [["SSL Unlabeled"], [f"{stats['unlabeled_ssl_count']:,} (ZINC250k)"], ["\u2014"], ["\u2014"], ["\u2014"]],
            [
                ["Train (80%)"],
                [f"{sp_tr['count']:,} ({sp_tr['unique_scaffolds']:,})"],
                [f"{sp_tr['bioavailability_count']} ({sp_tr['bioavailability_pos_rate']*100:.1f}%)"],
                [f"{sp_tr['bbb_count']:,} ({sp_tr['bbb_pos_rate']*100:.1f}%)"],
                [f"{sp_tr['herg_count']:,} ({sp_tr['herg_pos_rate']*100:.1f}%)"],
            ],
            [
                ["Val (10%)"],
                [f"{sp_va['count']} ({sp_va['unique_scaffolds']})"],
                [f"{sp_va['bioavailability_count']} ({sp_va['bioavailability_pos_rate']*100:.1f}%)"],
                [f"{sp_va['bbb_count']} ({sp_va['bbb_pos_rate']*100:.1f}%)"],
                [f"{sp_va['herg_count']} ({sp_va['herg_pos_rate']*100:.1f}%)"],
            ],
            [
                ["Test (10%)"],
                [f"{sp_te['count']} ({sp_te['unique_scaffolds']})"],
                [f"{sp_te['bioavailability_count']} ({sp_te['bioavailability_pos_rate']*100:.1f}%)"],
                [f"{sp_te['bbb_count']} ({sp_te['bbb_pos_rate']*100:.1f}%)"],
                [f"{sp_te['herg_count']} ({sp_te['herg_pos_rate']*100:.1f}%)"],
            ],
            [
                ["Total ADMET"],
                [f"{stats['admet_total_count']:,} ({stats['unique_scaffolds']:,})"],
                [f"{t_bio['count']} ({t_bio['pos_rate']*100:.1f}%)"],
                [f"{t_bbb['count']:,} ({t_bbb['pos_rate']*100:.1f}%)"],
                [f"{t_herg['count']:,} ({t_herg['pos_rate']*100:.1f}%)"],
            ],
        ],
        col_widths_in=[0.78, 0.80, 0.66, 0.63, 0.63],
        font_pt=7.5,
        lr_pad_dxa=30,
        midrule_rows=[4],
        bold_rows=[4],
    )

    _add_ieee_figure(
        doc,
        "01_dataset_eda.png",
        fig_num=1,
        caption_segments=[
            "Exploratory analysis of curated TDC clinical endpoints, disjoint Bemis-Murcko scaffold split proportions, "
            "and QED/SAScore distributions across Train, Validation, and Test splits.",
        ],
        width_in=3.42,
    )

    # IV. PROPOSED METHODOLOGY intro
    _add_p(doc, "IEEE Heading 1", ["IV.  Proposed Methodology"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Fig. 2 illustrates the complete 4-stage pipeline integrating contrastive pre-training, 2D+3D Hybrid GraphGPS "
            "encoding, masked multi-task inference, evolutionary Pareto optimization, and explainability.",
        ],
    )

    # =========================================================================
    # SECTION 3 (1-col): Fig. 2 Full-Width System Pipeline
    # =========================================================================
    _add_continuous_section(doc, cols=1)
    _add_ieee_figure(
        doc,
        "arch_01_system_pipeline.png",
        fig_num=2,
        caption_segments=[
            "End-to-end 4-stage computational drug discovery architecture integrating zero-leakage TDC/ZINC250k curation, "
            "GraphCL/InfoNCE self-supervised pre-training, 2D+3D Hybrid GraphGPS ADMET profiling, evolutionary Pareto lead "
            "optimization, and atom saliency XAI.",
        ],
        width_in=7.05,
    )

    # =========================================================================
    # SECTION 4 (2-col): IV-A..IV-E + Eqs (1)-(7) + Fig 3 + V-A, V-B
    # =========================================================================
    _add_continuous_section(doc, cols=2)

    _add_p(doc, "IEEE Heading 2", ["A.  2D+3D Graph Featurization and Structural Encodings"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Each desalted molecule is converted into a bidirectional graph ",
            ("i", "G"),
            " = (",
            ("i", "V"),
            ", ",
            ("i", "E"),
            "). Heavy atoms carry 29-dimensional vectors ",
            ("isub", "x", "i"),
            " \u2208 \u211d",
            ("sup", "29"),
            " encoding element one-hot (10), total degree (6), formal charge (5, essential for cationic hERG blockers and "
            "BBB ionization), hybridization (4), normalized hydrogen count (1), aromaticity (1), ring membership (1), and "
            "normalized 3D distance to the heavy-atom centroid (1). Covalent edges carry 8-dimensional vectors ",
            ("isub", "e", "ij"),
            " \u2208 \u211d",
            ("sup", "8"),
            " capturing bond order/aromaticity (5), conjugation (1), ring flag (1), and 3D bond length (1). Conformer "
            "coordinates and global 3D shape descriptors ",
            ("isub", "g", "3D", False),
            " = [NPR1, NPR2, Asphericity, RoG/10] \u2208 \u211d",
            ("sup", "4"),
            " are generated via RDKit ETKDGv3 [13] with deterministic zero fallbacks if embedding fails.",
        ],
    )
    _add_p(
        doc,
        "IEEE Body",
        [
            "To endow atoms with non-Euclidean topological positions [19], we compute the normalized graph Laplacian ",
            ("i", "L"),
            " = ",
            ("i", "I"),
            " \u2212 ",
            ("isup", "D", "\u22121/2"),
            ("i", "A"),
            ("isup", "D", "\u22121/2"),
            ", filter out trivial zero-eigenvalue modes (\u03bb > 10",
            ("sup", "\u22125"),
            "), and sign-canonicalize the smallest ",
            ("i", "k"),
            " = 4 eigenvectors by orienting each column\u2019s maximum-magnitude entry to be positive. We concatenate these "
            "with 4 sign-invariant Random-Walk Structural Encoding (RWSE) diagonals diag(",
            ("isup", "P", "1"),
            "), \u2026, diag(",
            ("isup", "P", "4"),
            ") where ",
            ("i", "P"),
            " = ",
            ("isup", "D", "\u22121"),
            ("i", "A"),
            ", yielding ",
            ("isub", "p", "i"),
            " \u2208 \u211d",
            ("sup", "8"),
            ".",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["B.  Hybrid Local+Global 2D+3D GraphGPS Backbone"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Fig. 3 depicts the internal structure of our backbone. Atom features and positional encodings are linearly projected to ",
            ("i", "H"),
            " = 128 and summed (",
            ("isup", "h", "(0)"),
            " = ",
            ("isub", "W", "x"),
            ("i", "x"),
            " + ",
            ("isub", "W", "p"),
            ("i", "p"),
            "). Across ",
            ("i", "L"),
            " = 3 layers (4 attention heads), each layer computes a covalent-edge local attention update [17] alongside a dense "
            "all-atom global self-attention update [18]:",
        ],
    )

    _add_equation(
        doc,
        r"h_{\mathrm{loc}}^{(l)} = \mathrm{LN}\!\left(h^{(l-1)} + \mathrm{Drop}\!\left(\mathrm{TransformerConv}(h^{(l-1)}, E, W_e e)\right)\right)",
        eq_num=1,
        font_pt=8.0,
    )
    _add_equation(
        doc,
        r"h_{\mathrm{glob}}^{(l)} = \mathrm{LN}\!\left(h^{(l-1)} + \mathrm{Drop}\!\left(\mathrm{MHA}_{\mathrm{dense}}(h^{(l-1)}, \mathrm{mask})\right)\right)",
        eq_num=2,
        font_pt=8.5,
    )
    _add_equation(
        doc,
        r"h^{(l)} = \mathrm{LN}\!\left(\bar{h}^{(l)} + \mathrm{FFN}(\bar{h}^{(l)})\right),\; \bar{h}^{(l)} = \tfrac{1}{2}\!\left(h_{\mathrm{loc}}^{(l)} + h_{\mathrm{glob}}^{(l)}\right)",
        eq_num=3,
        font_pt=8.5,
    )

    _add_p(
        doc,
        "IEEE Body First",
        [
            "A readout MLP fuses global mean pooling, global max pooling, and projected 3D shape descriptors into a 128-dimensional "
            "molecular representation ",
            ("isub", "h", "mol", False),
            ".",
        ],
    )

    _add_ieee_figure(
        doc,
        "arch_02_graphgps_layer.png",
        fig_num=3,
        caption_segments=[
            "Hybrid Local+Global GraphGPS layer combining edge-aware ",
            ("code", "TransformerConv"),
            ", dense all-atom ",
            ("code", "MultiheadAttention"),
            ", and global ETKDGv3 3D shape descriptors.",
        ],
        width_in=3.42,
    )

    _add_p(doc, "IEEE Heading 2", ["C.  Undirected-Safe Contrastive SSL Pre-Training"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Following GraphCL [20] and MolCLR [22], each molecule produces two distinct augmented views (",
            ("isub", "v", "i"),
            ", ",
            ("isub", "v", "j"),
            ") sampled from atom feature masking (15%), ",
            ("i", "undirected-symmetric"),
            " bond dropping (15%, sampling one keep decision per canonical edge (min(",
            ("i", "u"),
            ", ",
            ("i", "v"),
            "), max(",
            ("i", "u"),
            ", ",
            ("i", "v"),
            "))), subgraph node dropping (15% with edge remapping), and random LapPE sign flipping. A 2-layer MLP projector with ",
            ("code", "BatchNorm1d"),
            " maps ",
            ("isub", "h", "mol", False),
            " to L2-normalized vectors ",
            ("isub", "z", "i"),
            " \u2208 \u211d",
            ("sup", "64"),
            " trained via temperature-scaled InfoNCE loss (\u03c4 = 0.1) [21]:",
        ],
    )

    _add_equation(
        doc,
        r"\mathcal{L}_{\mathrm{InfoNCE}} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2B} \mathbf{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}",
        eq_num=4,
        font_pt=8.5,
    )

    _add_p(doc, "IEEE Heading 2", ["D.  NaN-Masked Multi-Task ADMET Heads and Bounded Losses"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Task-specific 2-layer MLP heads predict raw logits (",
            ("isub", "l", "bio", False),
            ", ",
            ("isub", "l", "bbb", False),
            ", ",
            ("isub", "l", "herg", False),
            ") mapped via \u03c3(\u00b7) to [0, 100]% oral bioavailability probability, [0, 1] BBB probability, and [0, 1] hERG "
            "hazard, alongside QED \u2208 [0, 1] and strictly bounded SAScore = 1 + 9\u03c3(",
            ("isub", "s", "raw", False),
            ") \u2208 [1, 10]. Given per-sample boolean mask ",
            ("isub", "M", "i,t"),
            " \u2208 {0, 1} indicating observed labels, the joint loss skips missing ",
            ("code", "NaN"),
            " targets and balances classification and normalized regression scales:",
        ],
    )

    _add_equation(
        doc,
        r"\mathcal{L}_{\mathrm{MT}} = \sum_{t} \mathcal{L}_{\mathrm{BCE}}^{\mathrm{pw}_t}(l_t, y_t; M_t) + 2\mathcal{L}_{\mathrm{SL1}}(\hat{q}, q) + 2\mathcal{L}_{\mathrm{SL1}}(\hat{\tilde{s}}, \tilde{s})",
        eq_num=5,
        font_pt=8.5,
    )

    _add_p(
        doc,
        "IEEE Body First",
        [
            "where ",
            ("i", "t"),
            " \u2208 {bio, bbb, herg}, ",
            ("i", "s\u0303"),
            " = (",
            ("i", "s"),
            " \u2212 1)/9 normalizes SAScore to [0, 1], and ",
            ("isub", "pw", "t"),
            " = clip(",
            ("isub", "N", "neg", False),
            "/",
            ("isub", "N", "pos", False),
            ", 0.5, 4.0) is computed strictly on the training split.",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["E.  Evolutionary Pareto Lead Optimization and Atom Saliency"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Starting from a seed lead, our mutator applies 23 warning-free medicinal chemistry reaction SMIRKS "
            "(aromatic/aliphatic substitutions, aza-scans, halogen swaps, carboxylic bioisosteres) plus RDKit BRICS fragment "
            "recombination [23], retaining only molecules passing PAINS alerts [24], Lipinski Ro5 (\u2264 1 violation) [25], and "
            "Tanimoto ECFP4 similarity ",
            ("i", "T"),
            " \u2265 0.35 [14]. Candidates are scored using the Derringer-Suich geometric desirability function [26]:",
        ],
    )

    _add_equation(
        doc,
        r"D(x) = \left( d_{\mathrm{bio}}^{1.0} \cdot d_{1-\mathrm{hERG}}^{1.5} \cdot d_{\mathrm{QED}}^{1.0} \cdot d_{\mathrm{SA}}^{0.8} \cdot d_{\mathrm{BBB}}^{0.4} \right)^{1/4.7}",
        eq_num=6,
        font_pt=8.5,
    )

    _add_p(
        doc,
        "IEEE Body First",
        [
            "with (",
            ("isub", "w", "1", False),
            ", \u2026, ",
            ("isub", "w", "5", False),
            ") = (1.0, 1.5, 1.0, 0.8, 0.4) and ",
            ("isub", "d", "BBB", False),
            " = 1 \u2212 BBB in ",
            ("code", "peripheral"),
            " mode (penalizing unwanted CNS exposure), and ranked into non-dominated fronts via NSGA-II peeling [27]. "
            "Finally, atom-level attribution [28] computes normalized sensitivity scores per atom ",
            ("i", "i"),
            " as:",
        ],
    )

    _add_equation(
        doc,
        r"A_i = \sqrt{\sum_{d=1}^{29} \left(x_{i,d} \cdot \frac{\partial l_{\mathrm{task}}}{\partial x_{i,d}}\right)^2}",
        eq_num=7,
        font_pt=8.5,
    )

    # V. EXPERIMENTAL SETUP AND ABLATION RESULTS
    _add_p(doc, "IEEE Heading 1", ["V.  Experimental Setup and Ablation Results"])

    _add_p(doc, "IEEE Heading 2", ["A.  Training Hyperparameters and Baselines"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "All experiments use deterministic seed ",
            ("code", "42"),
            ", Apple Silicon MPS acceleration, ",
            ("code", "AdamW"),
            " (wd = 10",
            ("sup", "\u22124"),
            "), cosine LR annealing (\u03b7",
            ("sub", "min"),
            " = 10",
            ("sup", "\u22125"),
            "), gradient norm clipping (1.0), and batch size 64. Stage 1 SSL pre-trains for 25 epochs (lr = 10",
            ("sup", "\u22123"),
            "). Stage 2 fine-tuning runs up to 40 epochs (lr = 5\u00d710",
            ("sup", "\u22124"),
            ", discriminative 0.5\u00d7 backbone LR for SSL initialization, ",
            ("code", "pos_weight"),
            " = 0.50, 0.50, 0.84) with validation early stopping (patience = 10) maximizing 0.7\u00b7Macro-ROC-AUC + 0.3\u00b7Mean-",
            ("isup", "R", "2"),
            ". We compare four models identically on the unseen-scaffold test split (",
            ("i", "n"),
            f" = {sp_te['count']}): (1) ",
            ("b", "RF-ECFP4"),
            " (300 balanced trees per task on 2048-bit Morgan ECFP4 [14], [15]), (2) ",
            ("b", "Standard-GCN"),
            " (3-layer ",
            ("code", "GCNConv"),
            " [16] with identical 5-task heads), (3) ",
            ("b", "GraphGPS-Scratch"),
            " (no SSL pre-training), and (4) ",
            ("b", "SSL-GraphGPS (Ours)"),
            ".",
        ],
    )

    _add_p(doc, "IEEE Heading 2", ["B.  Stage 1 Contrastive Convergence and Latent Organization"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "InfoNCE loss decreased monotonically from ",
            ("b", f"{ssl_hist['loss'][0]:.4f}"),
            " (epoch 1) to ",
            ("b", f"{ssl_hist['loss'][4]:.4f}"),
            " (epoch 5), ",
            ("b", f"{ssl_hist['loss'][9]:.4f}"),
            " (epoch 10), and ",
            ("b", f"{ssl_hist['loss'][-1]:.4f}"),
            " at epoch 25 (best checkpoint ",
            ("b", f"{min(ssl_hist['loss']):.4f}"),
            "). Fig. 4 shows smooth zero-shot latent clustering along QED drug-likeness and partial unsupervised "
            "separation of hERG-toxic scaffolds across validation and test compounds prior to seeing any clinical labels.",
        ],
    )

    # =========================================================================
    # SECTION 5 (1-col): Fig. 4 + Full-Width TABLE II
    # =========================================================================
    _add_continuous_section(doc, cols=1)

    _add_ieee_figure(
        doc,
        "02_ssl_pretraining.png",
        fig_num=4,
        caption_segments=[
            f"Stage 1 SSL InfoNCE convergence across 25 epochs on {stats['unlabeled_ssl_count']:,} ZINC250k molecules "
            "(left) and zero-shot t-SNE latent representations on unseen val/test scaffolds colored by QED (center) and "
            "hERG hazard (right).",
        ],
        width_in=7.05,
    )

    def _bm_row(name: str, m: dict) -> List[List[Segment]]:
        return [
            [name],
            [f"{m['bioavailability']['roc_auc']:.4f}"],
            [f"{m['bbb']['roc_auc']:.4f}"],
            [f"{m['herg']['roc_auc']:.4f}"],
            [f"{m['aggregate']['macro_roc_auc']:.4f}"],
            [f"{m['qed']['r2']:.4f} ({m['qed']['mae']:.3f})"],
            [f"{m['sascore']['r2']:.4f} ({m['sascore']['mae']:.3f})"],
            [f"{m['aggregate']['composite_val_score']:.4f}"],
        ]

    _add_ieee_table(
        doc,
        roman_label="TABLE II",
        title_text=f"Held-Out Unseen-Scaffold Test Benchmark (n = {sp_te['count']}, Zero Scaffold or SMILES Overlap with Training/SSL)",
        headers=[
            ["Model"],
            ["Bioavail. AUC"],
            ["BBB AUC"],
            ["hERG AUC"],
            ["Macro ROC-AUC"],
            ["QED ", ("isup", "R", "2"), " (MAE)"],
            ["SAScore ", ("isup", "R", "2"), " (MAE)"],
            ["Comp. Score"],
        ],
        rows=[
            _bm_row("RF-ECFP4 [14], [15]", rf),
            _bm_row("Standard-GCN [16]", gcn),
            _bm_row("GraphGPS-Scratch (Ablation)", scratch),
            _bm_row("SSL-GraphGPS (Ours)", ours),
            [
                ["SSL Gain vs. Scratch"],
                [f"+{g_bio:.4f}"],
                [f"+{g_bbb:.4f}"],
                [f"+{g_herg:.4f}"],
                [("b", f"+{g_macro:.4f}")],
                [f"+{g_qed:.4f}"],
                [f"+{g_sa:.4f}"],
                [f"+{g_comp:.4f}"],
            ],
        ],
        col_widths_in=[1.55, 0.78, 0.72, 0.74, 0.86, 0.90, 0.95, 0.75],
        font_pt=8.0,
        lr_pad_dxa=38,
        midrule_rows=[4],
        highlight_rows=[3],
        bold_rows=[3],
        italic_rows=[4],
    )

    # =========================================================================
    # SECTION 6 (2-col): V-C, Table III, Fig. 5
    # =========================================================================
    _add_continuous_section(doc, cols=2)

    _add_p(doc, "IEEE Heading 2", ["C.  Held-Out Unseen-Scaffold Benchmark and Ablation"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "Tables II\u2013III and Figs. 5\u20137 summarize the held-out unseen-scaffold evaluation. ",
            ("b", "SSL-GraphGPS"),
            " achieves the strongest performance on ",
            ("b", "every single endpoint"),
            ": ",
            ("b", f"{ours['aggregate']['macro_roc_auc']:.4f} Macro ROC-AUC"),
            f" (Bioavailability {ours['bioavailability']['roc_auc']:.4f} / PR-AUC {ours['bioavailability']['pr_auc']:.4f}; "
            f"BBB {ours['bbb']['roc_auc']:.4f} / PR-AUC {ours['bbb']['pr_auc']:.4f}; hERG {ours['herg']['roc_auc']:.4f} / "
            f"PR-AUC {ours['herg']['pr_auc']:.4f}), ",
            ("b", ""),
            ("isup", "R", "2"),
            ("b", f" = {ours['qed']['r2']:.4f}"),
            " on QED (MAE = ",
            f"{ours['qed']['mae']:.3f}, Pearson ",
            ("i", "r"),
            f" = {ours['qed']['pearson_r']:.4f}), and ",
            ("isup", "R", "2"),
            ("b", f" = {ours['sascore']['r2']:.4f}"),
            " on SAScore (MAE = ",
            f"{ours['sascore']['mae']:.3f}, Pearson ",
            ("i", "r"),
            f" = {ours['sascore']['pearson_r']:.4f}).",
        ],
    )
    _add_p(
        doc,
        "IEEE Body",
        [
            "Comparing ",
            ("b", "SSL-GraphGPS"),
            " against ",
            ("b", "GraphGPS-Scratch"),
            " quantifies the direct benefit of contrastive pre-training: ",
            ("b", f"+{g_macro*100:.2f} percentage points (pp) in Macro ROC-AUC"),
            ", led by a ",
            ("b", f"+{g_bio*100:.2f} pp"),
            f" gain on data-scarce Oral Bioavailability ({sp_tr['bioavailability_count']} train samples), "
            f"+{g_herg*100:.2f} pp on hERG, +{g_bbb*100:.2f} pp on BBB, +{g_qed:.3f} on QED ",
            ("isup", "R", "2"),
            f", and +{g_sa:.3f} on SAScore ",
            ("isup", "R", "2"),
            ". Comparing against ",
            ("b", "Standard-GCN"),
            " highlights the expressive advantage of hybrid 2D+3D local-global attention "
            f"(+{g_gcn_macro*100:.2f} pp Macro ROC-AUC and +{g_gcn_qed:.3f} QED ",
            ("isup", "R", "2"),
            ").",
        ],
    )

    def _sec_row(name: str, m: dict) -> List[List[Segment]]:
        return [
            [name],
            [f"{m['bioavailability']['pr_auc']:.3f} / {m['bioavailability']['f1']:.3f}"],
            [f"{m['bbb']['pr_auc']:.3f} / {m['bbb']['f1']:.3f}"],
            [f"{m['herg']['pr_auc']:.3f} / {m['herg']['f1']:.3f}"],
            [f"{m['qed']['pearson_r']:.3f} / {m['sascore']['pearson_r']:.3f}"],
        ]

    _add_ieee_table(
        doc,
        roman_label="TABLE III",
        title_text="Secondary Classification (PR-AUC / F1) and Regression (Pearson r)",
        headers=[["Model"], ["Bioavail. PR/F1"], ["BBB PR/F1"], ["hERG PR/F1"], ["QED / SA ", ("i", "r")]],
        rows=[
            _sec_row("RF-ECFP4", rf),
            _sec_row("Standard-GCN", gcn),
            _sec_row("GraphGPS-Scratch", scratch),
            _sec_row("SSL-GraphGPS", ours),
        ],
        col_widths_in=[0.86, 0.66, 0.66, 0.66, 0.66],
        font_pt=7.5,
        lr_pad_dxa=20,
        highlight_rows=[3],
        bold_rows=[3],
    )

    _add_ieee_figure(
        doc,
        "03_training_curves.png",
        fig_num=5,
        caption_segments=[
            "Training loss convergence and unseen-scaffold validation Macro ROC-AUC across SSL-GraphGPS, "
            "GraphGPS-Scratch, and Standard-GCN.",
        ],
        width_in=3.42,
    )

    # =========================================================================
    # SECTION 7 (1-col): Fig. 6 Full-Width ROC Comparison
    # =========================================================================
    _add_continuous_section(doc, cols=1)
    _add_ieee_figure(
        doc,
        "04_roc_comparison.png",
        fig_num=6,
        caption_segments=[
            f"Held-out unseen-scaffold ROC curve comparison across Oral Bioavailability (n = {sp_te['bioavailability_count']}), "
            f"BBB Permeability (n = {sp_te['bbb_count']}), and hERG Cardiotoxicity (n = {sp_te['herg_count']}).",
        ],
        width_in=7.05,
    )

    # =========================================================================
    # SECTION 8 (2-col): Fig. 7, Section VI + Table IV
    # =========================================================================
    _add_continuous_section(doc, cols=2)
    _add_ieee_figure(
        doc,
        "05_regression_parity.png",
        fig_num=7,
        caption_segments=[
            "Predicted vs. ground-truth test parity plots for SSL-GraphGPS on QED (",
            ("isup", "R", "2"),
            f" = {ours['qed']['r2']:.4f}) and SAScore (",
            ("isup", "R", "2"),
            f" = {ours['sascore']['r2']:.4f}).",
        ],
        width_in=3.42,
    )

    _add_p(doc, "IEEE Heading 1", ["VI.  Lead Optimization Case Study and Explainability"])
    top_cands = lead_opt["top_candidates"][:5]
    cand_names = ["#1 Di-Et", "#2 4-Et", "#3 4-Me", "#4 Me,Cl", "#5 4-Cl"]
    min_bbb = min(c["bbb"] for c in top_cands)
    max_bbb = max(c["bbb"] for c in top_cands)
    max_herg = max(c["herg"] for c in top_cands)

    _add_p(
        doc,
        "IEEE Body First",
        [
            "Applying two evolutionary generations around ",
            ("b", "Aspirin"),
            " (",
            ("code", "CC(=O)Oc1ccccc1C(=O)O"),
            ") in ",
            ("code", "peripheral"),
            f" mode explored {lead_opt['total_candidates']} PAINS-free, Lipinski-compliant analogues (",
            ("i", "T"),
            f" \u2265 0.35) and identified {lead_opt['pareto_front1_count']} non-dominated Pareto Front-1 solutions "
            "(Table IV and Fig. 8). Top Front-1 analogues increase composite desirability from 0.898 (seed) to ",
            ("b", f"{top_cands[0]['desirability']:.3f}"),
            f" while lowering predicted unwanted CNS exposure (BBB: 0.216 \u2192 {min_bbb:.3f}\u2013{max_bbb:.3f}), "
            f"preserving ultra-low hERG risk (\u2264 {max_herg:.3f}), and raising exact RDKit QED from 0.550 to "
            f"{top_cands[0]['qed_rdkit']:.3f}.",
        ],
    )

    lead_table_rows: List[List[List[Segment]]] = [
        [["Seed (Aspirin)"], ["99.9"], ["0.004"], ["0.216"], ["0.74/0.55"], ["2.06/1.58"], ["0.898"]],
    ]
    for cname, c in zip(cand_names, top_cands):
        lead_table_rows.append(
            [
                [("b", cname)],
                [f"{c['bioavailability']:.1f}"],
                [f"{c['herg']:.3f}"],
                [f"{c['bbb']:.3f}"],
                [f"{c['qed']:.2f}/{c['qed_rdkit']:.2f}"],
                [f"{c['sascore']:.2f}/{c['sascore_rdkit']:.2f}"],
                [("b", f"{c['desirability']:.3f}")],
            ]
        )

    _add_ieee_table(
        doc,
        roman_label="TABLE IV",
        title_text="Aspirin Multi-Generation Pareto Front-1 Lead Optimization",
        headers=[["Candidate"], ["Bio%"], ["hERG"], ["BBB"], ["QED(P/RD)"], ["SA(P/RD)"], ["Desir."]],
        rows=lead_table_rows,
        col_widths_in=[0.72, 0.38, 0.40, 0.40, 0.60, 0.60, 0.40],
        font_pt=7.2,
        lr_pad_dxa=24,
        highlight_rows=[1],
    )

    # =========================================================================
    # SECTION 9 (1-col): Fig. 8 Full-Width Pareto Plot
    # =========================================================================
    _add_continuous_section(doc, cols=1)
    _add_ieee_figure(
        doc,
        "06_pareto_optimization.png",
        fig_num=8,
        caption_segments=[
            "Bioavailability vs. hERG Pareto Front-1 scatter (left) and 2D structures of Aspirin seed vs. top "
            "Pareto-optimal analogues (right).",
        ],
        width_in=7.05,
    )

    # =========================================================================
    # SECTION 10 (2-col): XAI Narrative
    # =========================================================================
    _add_continuous_section(doc, cols=2)
    _add_p(
        doc,
        "IEEE Body",
        [
            "Fig. 9 visualizes atom-level gradient attribution across an unseen test-set hERG blocker (pred = 0.996), "
            "Aspirin (hERG = 0.004), and Caffeine (BBB = 0.941). On the unseen hERG blocker, saliency concentrates on the "
            "protonatable tertiary amine nitrogen and adjacent linker carbons\u2014consistent with classic hERG potassium "
            "channel binding pharmacophores [6].",
        ],
    )

    # =========================================================================
    # SECTION 11 (1-col): Fig. 9 Full-Width XAI Saliency Maps
    # =========================================================================
    _add_continuous_section(doc, cols=1)
    _add_ieee_figure(
        doc,
        "07_atom_saliency_xai.png",
        fig_num=9,
        caption_segments=[
            "Input \u00d7 Gradient atom attribution maps highlighting the tertiary amine hERG pharmacophore on an unseen "
            "test blocker (hERG = 0.996), safe Aspirin (hERG = 0.004), and CNS-penetrant Caffeine (BBB = 0.941).",
        ],
        width_in=7.05,
    )

    # =========================================================================
    # SECTION 12 (2-col): VII. Discussion, VIII. Conclusion, References [1]–[28]
    # =========================================================================
    _add_continuous_section(doc, cols=2)

    _add_p(doc, "IEEE Heading 1", ["VII.  Discussion, Honest Limitations, and Future Work"])
    _add_p(
        doc,
        "IEEE Body First",
        ["To uphold strict scientific transparency, several empirical boundaries must be noted:"],
    )
    limitations: List[List[Segment]] = [
        [
            "1)\t",
            ("i", "Single-Seed Evaluation: "),
            "Results reflect deterministic seed ",
            ("code", "42"),
            "; multi-seed confidence intervals remain to be quantified.",
        ],
        [
            "2)\t",
            ("i", "Small Bioavailability Test Split: "),
            "Because ",
            ("code", "Bioavailability_Ma"),
            f" contains {t_bio['count']} compounds total, the scaffold-disjoint test fold contains ",
            ("i", "n"),
            f" = {sp_te['bioavailability_count']} labeled samples, introducing moderate ROC-AUC variance relative to hERG (",
            ("i", "n"),
            f" = {sp_te['herg_count']}).",
        ],
        [
            "3)\t",
            ("i", "Binary vs. Continuous Bioavailability: "),
            "The bioavailability head predicts ",
            ("i", "P"),
            "(",
            ("i", "F"),
            " \u2265 20%) scaled to [0, 100]% rather than continuous human percentage bioavailability.",
        ],
        [
            "4)\t",
            ("i", "Calibrating Surrogate QED Against Exact Descriptors: "),
            "On Aspirin, the neural head predicts QED = 0.739 versus exact RDKit QED = 0.550 (despite overall test ",
            ("isup", "R", "2"),
            f" = {ours['qed']['r2']:.4f}), motivating our inclusion of exact RDKit QED/SAScore checks alongside neural "
            "ADMET predictions during Stage 3 optimization.",
        ],
        [
            "5)\t",
            ("i", "Static Single Conformer: "),
            "Each molecule uses a single ETKDGv3 conformer rather than Boltzmann-weighted multi-conformer ensembles or "
            "explicit-solvent molecular dynamics.",
        ],
    ]
    for item in limitations:
        _add_p(doc, "IEEE List Item", item)

    _add_p(
        doc,
        "IEEE Body",
        [
            "Future work will extend this framework to multi-conformer SE(3)-equivariant encoders, additional endpoints "
            "(CYP450 inhibition and Ames mutagenicity), and generative 3D scaffold diffusion.",
        ],
    )

    _add_p(doc, "IEEE Heading 1", ["VIII.  Conclusion"])
    _add_p(
        doc,
        "IEEE Body First",
        [
            "We presented an audited, zero-leakage Self-Supervised 2D+3D Hybrid GraphGPS framework for multi-objective ADMET "
            f"profiling and evolutionary lead optimization. Across {stats['admet_total_count']:,} standardized TDC clinical "
            "compounds evaluated strictly on unseen Bemis-Murcko scaffolds, contrastive pre-training on "
            f"{stats['unlabeled_ssl_count']:,} ZINC250k molecules boosted Macro ROC-AUC by +{g_macro*100:.2f} pp to ",
            ("b", f"{ours['aggregate']['macro_roc_auc']:.4f}"),
            ", outperforming ECFP4 Random Forests, Standard-GCN, and non-pretrained GraphGPS across every task while "
            "delivering interpretable atom-level saliency and PAINS-free Pareto-optimal analogues.",
        ],
    )

    # References [1]–[28]
    _add_p(doc, "IEEE Heading 1", ["References"])
    references: List[List[Segment]] = [
        ["[1]\tJ. A. DiMasi, H. G. Grabowski, and R. W. Hansen, \u201cInnovation in the pharmaceutical industry: New estimates of R&D costs,\u201d ", ("i", "J. Health Econ."), ", vol. 47, pp. 20\u201333, 2016."],
        ["[2]\tM. J. Waring ", ("i", "et al."), ", \u201cAn analysis of the attrition of drug candidates from four major pharmaceutical companies,\u201d ", ("i", "Nat. Rev. Drug Discov."), ", vol. 14, no. 7, pp. 475\u2013486, 2015."],
        ["[3]\tK. Huang ", ("i", "et al."), ", \u201cTherapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development,\u201d in ", ("i", "Proc. Neural Inf. Process. Syst. (NeurIPS) Track on Datasets & Benchmarks"), ", 2021."],
        ["[4]\tC.-Y. Ma, S.-Y. Yang, H. Zhang, M.-L. Xiang, Q. Huang, and Y.-Q. Wei, \u201cPrediction models of human plasma protein binding rate and oral bioavailability derived by using GA\u2013CG\u2013SVM method,\u201d ", ("i", "J. Pharm. Biomed. Anal."), ", vol. 47, no. 4\u20135, pp. 677\u2013682, 2008."],
        ["[5]\tI. F. Martins, A. L. Teixeira, L. Pinheiro, and A. O. Falc\u00e3o, \u201cA Bayesian approach to in silico blood-brain barrier penetration modeling,\u201d ", ("i", "J. Chem. Inf. Model."), ", vol. 52, no. 6, pp. 1686\u20131697, 2012."],
        ["[6]\tS. Wang, H. Sun, H. Liu, D. Li, Y. Li, and T. Hou, \u201cADMET evaluation in drug discovery. 16. Predicting hERG blockers by combining multiple pharmacophores and machine learning approaches,\u201d ", ("i", "Mol. Pharm."), ", vol. 13, no. 8, pp. 2855\u20132866, 2016."],
        ["[7]\tA. Karim, M. Lee, T. Balle, and A. Sattar, \u201cCardioTox net: A robust predictor for hERG channel blockade based on deep learning meta-feature ensembles,\u201d ", ("i", "J. Cheminform."), ", vol. 13, no. 1, Art. no. 60, 2021."],
        ["[8]\tR. G\u00f3mez-Bombarelli ", ("i", "et al."), ", \u201cAutomatic chemical design using a data-driven continuous representation of molecules,\u201d ", ("i", "ACS Cent. Sci."), ", vol. 4, no. 2, pp. 268\u2013276, 2018."],
        ["[9]\tT. Sterling and J. J. Irwin, \u201cZINC 15 \u2014 Ligand discovery for everyone,\u201d ", ("i", "J. Chem. Inf. Model."), ", vol. 55, no. 11, pp. 2324\u20132337, 2015."],
        ["[10]\tG. W. Bemis and M. A. Murcko, \u201cThe properties of known drugs. 1. Molecular frameworks,\u201d ", ("i", "J. Med. Chem."), ", vol. 39, no. 15, pp. 2887\u20132893, 1996."],
        ["[11]\tG. R. Bickerton, G. V. Paolini, J. Besnard, S. Muresan, and A. L. Hopkins, \u201cQuantifying the chemical beauty of drugs,\u201d ", ("i", "Nat. Chem."), ", vol. 4, no. 2, pp. 90\u201398, 2012."],
        ["[12]\tP. Ertl and A. Schuffenhauer, \u201cEstimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions,\u201d ", ("i", "J. Cheminform."), ", vol. 1, no. 1, Art. no. 8, 2009."],
        ["[13]\tS. Riniker and G. A. Landrum, \u201cBetter informed distance geometry: Using what we know to improve conformation generation,\u201d ", ("i", "J. Chem. Inf. Model."), ", vol. 55, no. 12, pp. 2562\u20132574, 2015."],
        ["[14]\tD. Rogers and M. Hahn, \u201cExtended-connectivity fingerprints,\u201d ", ("i", "J. Chem. Inf. Model."), ", vol. 50, no. 5, pp. 742\u2013754, 2010."],
        ["[15]\tL. Breiman, \u201cRandom forests,\u201d ", ("i", "Mach. Learn."), ", vol. 45, no. 1, pp. 5\u201332, 2001."],
        ["[16]\tT. N. Kipf and M. Welling, \u201cSemi-supervised classification with graph convolutional networks,\u201d in ", ("i", "Proc. Int. Conf. Learn. Represent. (ICLR)"), ", 2017."],
        ["[17]\tY. Shi, Z. Huang, S. Feng, H. Zhong, W. Wang, and Y. Sun, \u201cMasked label prediction: Unified message passing model for semi-supervised classification,\u201d in ", ("i", "Proc. Int. Joint Conf. Artif. Intell. (IJCAI)"), ", 2021, pp. 1548\u20131554."],
        ["[18]\tL. Ramp\u00e1\u0161ek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, and D. Beaini, \u201cRecipe for a general, powerful, scalable graph transformer,\u201d in ", ("i", "Adv. Neural Inf. Process. Syst. (NeurIPS)"), ", 2022."],
        ["[19]\tV. P. Dwivedi, A. T. Luu, T. Laurent, Y. Bengio, and X. Bresson, \u201cGraph neural networks with learnable structural and positional representations,\u201d in ", ("i", "Proc. Int. Conf. Learn. Represent. (ICLR)"), ", 2022."],
        ["[20]\tY. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, \u201cGraph contrastive learning with augmentations,\u201d in ", ("i", "Adv. Neural Inf. Process. Syst. (NeurIPS)"), ", 2020."],
        ["[21]\tT. Chen, S. Kornblith, M. Norouzi, and G. Hinton, \u201cA simple framework for contrastive learning of visual representations,\u201d in ", ("i", "Proc. 37th Int. Conf. Mach. Learn. (ICML)"), ", 2020, pp. 1597\u20131607."],
        ["[22]\tY. Wang, J. Wang, Z. Cao, and A. Barati Farimani, \u201cMolecular contrastive learning of representations via graph neural networks,\u201d ", ("i", "Nat. Mach. Intell."), ", vol. 4, no. 3, 2022."],
        ["[23]\tJ. Degen, C. Wegscheid-Gerlach, A. Zaliani, and M. Rarey, \u201cOn the art of compiling and using \u2018drug-like\u2019 chemical fragment spaces,\u201d ", ("i", "ChemMedChem"), ", vol. 3, no. 10, pp. 1503\u20131507, 2008."],
        ["[24]\tJ. B. Baell and G. A. Holloway, \u201cNew substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries and for their exclusion in bioassays,\u201d ", ("i", "J. Med. Chem."), ", vol. 53, no. 7, pp. 2719\u20132740, 2010."],
        ["[25]\tC. A. Lipinski, F. Lombardo, B. W. Dominy, and P. J. Feeney, \u201cExperimental and computational approaches to estimate solubility and permeability in drug discovery and development settings,\u201d ", ("i", "Adv. Drug Deliv. Rev."), ", vol. 46, no. 1\u20133, pp. 3\u201326, 2001."],
        ["[26]\tG. Derringer and R. Suich, \u201cSimultaneous optimization of several response variables,\u201d ", ("i", "J. Qual. Technol."), ", vol. 12, no. 4, pp. 214\u2013219, 1980."],
        ["[27]\tK. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, \u201cA fast and elitist multiobjective genetic algorithm: NSGA-II,\u201d ", ("i", "IEEE Trans. Evol. Comput."), ", vol. 6, no. 2, pp. 182\u2013197, 2002."],
        ["[28]\tA. Shrikumar, P. Greenside, and A. Kundaje, \u201cLearning important features through propagating activation differences,\u201d in ", ("i", "Proc. 34th Int. Conf. Mach. Learn. (ICML)"), ", PMLR, vol. 70, 2017, pp. 3145\u20133153."],
    ]
    for ref_segs in references:
        _add_p(doc, "IEEE Reference", ref_segs)

    # Trailing Continuous Section Break (1-col, 1 pt empty terminator) to trigger Word 2-col balancing
    _add_continuous_section(doc, cols=1)
    p_end = doc.add_paragraph()
    p_end.paragraph_format.first_line_indent = Inches(0)
    p_end.paragraph_format.space_before = Pt(0)
    p_end.paragraph_format.space_after = Pt(0)
    p_end.paragraph_format.line_spacing = 1.0
    r_end = p_end.add_run("")
    r_end.font.size = Pt(1.0)

    os.makedirs(PAPER_DIR, exist_ok=True)
    doc.save(OUT_DOCX)
    return OUT_DOCX


if __name__ == "__main__":
    out = build_docx()
    size_kb = os.path.getsize(out) / 1024.0
    print(f"Built editable IEEE Word (.docx) paper -> {out} ({size_kb:.1f} KB)")
