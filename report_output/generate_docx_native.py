"""DOCX authoring with python-docx from the report's page-oriented JSON."""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent / "_tools"
sys.path.append(str(TOOLS))

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

FONT_BODY = "\u65b0\u7d30\u660e\u9ad4"
FONT_HEADING = "\u6a19\u6977\u9ad4"
BLUE = "173A55"


def set_run_font(run, name=FONT_BODY, size=11, *, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), name)


def set_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=70, start=90, bottom=70, end=90):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_page_field(paragraph):
    run = paragraph.add_run()
    set_run_font(run, size=9, color="536779")
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    result = OxmlElement("w:t")
    result.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, result, end):
        run._r.append(element)


def add_hyperlink(paragraph, text, url):
    rid = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:eastAsia"), FONT_BODY)
    rpr.append(fonts)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    rpr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(underline)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "18")
    rpr.append(size)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    hyperlink.append(r)
    paragraph._p.append(hyperlink)


def add_table(doc, rows):
    if not rows:
        return
    cols = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    table.autofit = False
    width = Inches(6.45 / cols)
    for row_index, values in enumerate(rows):
        row = table.rows[row_index]
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))
        if row_index == 0:
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            tr_pr.append(repeat)
        for col_index in range(cols):
            cell = row.cells[col_index]
            cell.width = width
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if row_index == 0:
                set_shading(cell, "EAF0F5")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.05
            if col_index == 0 and row_index > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            value = str(values[col_index]) if col_index < len(values) else ""
            run = p.add_run(value)
            set_run_font(run, size=10, bold=(row_index == 0), color=BLUE if row_index == 0 else None)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.15)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.0
    for i, line in enumerate(text.splitlines()):
        if i:
            p.add_run().add_break()
        run = p.add_run(line)
        set_run_font(run, name="Consolas", size=9, color="283642")


def add_block(doc, item, base_dir):
    kind = item.get("type", "paragraph")
    text = item.get("text", "")
    if kind == "heading":
        level = int(item.get("level", 2))
        p = doc.add_paragraph(style=f"Heading {level}")
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        set_run_font(run, FONT_HEADING, 14, bold=True, color=BLUE)
    elif kind == "bullet":
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.left_indent = Cm(0.45)
        set_run_font(p.add_run("•　" + text), size=12)
    elif kind == "caption":
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(4)
        set_run_font(p.add_run(text), size=9, color="536779")
    elif kind == "code":
        add_code(doc, text)
    elif kind == "table":
        add_table(doc, item.get("rows", []))
    elif kind == "image":
        image_path = (base_dir / item["path"]).resolve()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.add_run().add_picture(str(image_path), width=Inches(float(item.get("width_in", 6.25))),
                                height=Inches(float(item.get("height_in", 2.5))))
    elif kind == "source" and item.get("url"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        add_hyperlink(p, text or item["url"], item["url"])
    else:
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.keep_together = True
        for i, line in enumerate(str(text).split("\n")):
            if i:
                p.add_run().add_break()
            set_run_font(p.add_run(line), size=12, color=item.get("color"))


def build(data, output: Path, base_dir: Path | None = None):
    pages = data.get("pages", [])
    if not pages:
        raise ValueError("input JSON must have a non-empty pages array")
    base_dir = base_dir or Path.cwd()
    doc = Document()
    if doc.paragraphs:
        initial = doc.paragraphs[0]
        initial._element.getparent().remove(initial._element)
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(2.1)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)

    normal = doc.styles["Normal"]
    normal.font.name = FONT_BODY
    normal.font.size = Pt(12)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.5
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[name]
        style.font.name = FONT_HEADING
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(BLUE)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(footer.add_run("— "), size=9, color="536779")
    add_page_field(footer)
    set_run_font(footer.add_run(" —"), size=9, color="536779")

    for page_number, page in enumerate(pages):
        title = page.get("title")
        if title:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(9)
            p.paragraph_format.keep_with_next = True
            for i, line in enumerate(title.split("\n")):
                if i:
                    p.add_run().add_break()
                set_run_font(p.add_run(line), FONT_HEADING, 16, bold=True, color=BLUE)
        for item in page.get("blocks", []):
            add_block(doc, item, base_dir)
        if page_number + 1 < len(pages):
            doc.add_page_break()

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = "台積電股價預測研究報告：LSTM與GRU正式實驗比較"
    doc.core_properties.language = "zh-TW"
    doc.save(output)


if __name__ == "__main__":
    import json
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    build(json.loads(source.read_text(encoding="utf-8")), output, source.resolve().parent)
