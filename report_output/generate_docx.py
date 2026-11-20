#!/usr/bin/env python
"""Build an explicitly paginated A4 DOCX from a JSON report specification.

This uses only the Python standard library. The input is intentionally content
agnostic; each item in `pages` becomes one designed page separated by a hard
page break. Word may still repaginate if a page overflows its usable area.
"""
from __future__ import annotations

import json
import mimetypes
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML = "http://www.w3.org/XML/1998/namespace"


def tag(name: str) -> str:
    return f"{{{W}}}{name}"


def run(text: str, *, bold=False, italic=False, font="新細明體", size=22, color=None):
    rpr = [f'<w:rFonts w:ascii="{escape(font)}" w:hAnsi="{escape(font)}" w:eastAsia="{escape(font)}"/>',
           f'<w:sz w:val="{size}"/>', f'<w:szCs w:val="{size}"/>']
    if bold: rpr.append('<w:b/>')
    if italic: rpr.append('<w:i/>')
    if color: rpr.append(f'<w:color w:val="{color}"/>')
    return '<w:r><w:rPr>' + ''.join(rpr) + '</w:rPr><w:t xml:space="preserve">' + escape(str(text)) + '</w:t></w:r>'


def paragraph(text="", *, style=None, bold=False, font="新細明體", size=22,
              align=None, before=0, after=100, color=None, keep=False, line=312):
    props = []
    if style: props.append(f'<w:pStyle w:val="{style}"/>')
    if align: props.append(f'<w:jc w:val="{align}"/>')
    props.append(f'<w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>')
    if keep: props.append('<w:keepNext/>')
    content = ''.join(run(line, bold=bold, font=font, size=size, color=color) if i == 0 else '<w:r><w:br/></w:r>' + run(line, font=font, size=size) for i, line in enumerate(str(text).split('\n')))
    return '<w:p><w:pPr>' + ''.join(props) + '</w:pPr>' + content + '</w:p>'


def table(rows):
    if not rows: return ''
    widths = [int(9000 / max(1, len(rows[0])))] * len(rows[0])
    out = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:tblBorders>'
           '<w:top w:val="single" w:sz="5" w:color="9EADBA"/><w:left w:val="single" w:sz="5" w:color="9EADBA"/>'
           '<w:bottom w:val="single" w:sz="5" w:color="9EADBA"/><w:right w:val="single" w:sz="5" w:color="9EADBA"/>'
           '<w:insideH w:val="single" w:sz="4" w:color="CFD8E0"/><w:insideV w:val="single" w:sz="4" w:color="CFD8E0"/>'
           '</w:tblBorders></w:tblPr>']
    for ri, row in enumerate(rows):
        out.append('<w:tr><w:trPr><w:cantSplit/>' + ('<w:tblHeader w:val="true"/>' if ri == 0 else '') + '</w:trPr>')
        for cell in row:
            shade = '<w:shd w:fill="EAF0F5"/>' if ri == 0 else ''
            out.append('<w:tc><w:tcPr>' + shade + '</w:tcPr>' + paragraph(str(cell), bold=(ri == 0), size=20, after=40) + '</w:tc>')
        out.append('</w:tr>')
    out.append('</w:tbl>')
    return ''.join(out)


@dataclass
class PackageAssets:
    base_dir: Path
    relationships: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    content_types: set = field(default_factory=set)

    def relationship(self, rel_type, target, target_mode=None):
        rid = f"rId{len(self.relationships) + 1}"
        mode = f' TargetMode="{target_mode}"' if target_mode else ""
        self.relationships.append(f'<Relationship Id="{rid}" Type="{rel_type}" Target="{escape(target)}"{mode}/>')
        return rid


def image_block(item, assets):
    path = (assets.base_dir / item["path"]).resolve()
    ext = path.suffix.lower().lstrip(".")
    if ext not in ("png", "jpg", "jpeg"):
        raise ValueError(f"unsupported image format: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(path)
    index = 1 + sum(1 for n in assets.files if n.startswith("media/image"))
    filename = f"image{index}.{ext}"
    target = f"media/{filename}"
    assets.files[f"word/{target}"] = path.read_bytes()
    assets.content_types.add(ext)
    rid = assets.relationship("http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", target)
    cx = int(float(item.get("width_in", 6.2)) * 914400)
    cy = int(float(item.get("height_in", 2.5)) * 914400)
    drawing = f'''<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="40" w:after="20" w:line="240" w:lineRule="exact"/></w:pPr><w:r><w:drawing><wp:inline xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{index}" name="{escape(filename)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{index}" name="{escape(filename)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}" xmlns:r="{R}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''
    return drawing


def block(item, assets):
    kind = item.get("type", "paragraph")
    text = item.get("text", "")
    if kind == "heading":
        level = int(item.get("level", 2))
        return paragraph(text, style=f"Heading{min(3, max(1, level))}", font="標楷體", size=28 if level == 1 else 24,
                         bold=True, color="173A55", before=120, after=90, keep=True)
    if kind == "bullet":
        return paragraph("•　" + text, after=50)
    if kind == "caption":
        return paragraph(text, align="center", size=18, color="536779", before=30, after=90)
    if kind == "code":
        return paragraph(text, font="Consolas", size=17, after=10, line=230)
    if kind == "table":
        return table(item.get("rows", []))
    if kind == "source":
        url = item.get("url")
        if url:
            rid = assets.relationship("http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", url, "External")
            link_run = run(text or url, font="新細明體", size=18, color="0563C1").replace("<w:rPr>", '<w:rPr><w:u w:val="single"/>', 1)
            return f'<w:p><w:pPr><w:spacing w:after="45" w:line="260" w:lineRule="auto"/></w:pPr><w:hyperlink r:id="{rid}" w:history="1">{link_run}</w:hyperlink></w:p>'
        return paragraph(text, size=18, color="536779", after=45, line=260)
    if kind == "image":
        return image_block(item, assets)
    return paragraph(text, after=100)


def page_xml(spec, assets):
    body = []
    title = spec.get("title")
    if title:
        body.append(paragraph(title, style="Heading1", font="標楷體", size=34, bold=True,
                              color="173A55", before=40, after=180, keep=True))
    body.extend(block(item, assets) for item in spec.get("blocks", []))
    return ''.join(body)


def build(data, output: Path, base_dir: Path | None = None):
    pages = data.get("pages", [])
    if not pages: raise ValueError("input JSON must contain a non-empty pages array")
    assets = PackageAssets(base_dir or Path.cwd())
    assets.relationship("http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer", "footer1.xml")
    assets.relationship("http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "styles.xml")
    assets.relationship("http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings", "settings.xml")
    body = []
    for i, page in enumerate(pages):
        body.append(page_xml(page, assets))
        if i + 1 < len(pages): body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    body.append('''<w:sectPr><w:footerReference w:type="default" r:id="rId1"/><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1247" w:right="1247" w:bottom="1247" w:left="1247" w:header="500" w:footer="500" w:gutter="0"/><w:cols w:space="425"/></w:sectPr>''')
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}" xmlns:r="{R}" xmlns:xml="{XML}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>{''.join(body)}</w:body></w:document>'''
    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="新細明體" w:hAnsi="新細明體" w:eastAsia="新細明體"/><w:lang w:val="zh-TW" w:eastAsia="zh-TW"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="312" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:qFormat/></w:style></w:styles>'''
    settings = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W}"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/><w:compat/></w:settings>'''
    core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>研究報告</dc:title><dc:language>zh-TW</dc:language><dc:creator>研究報告</dc:creator></cp:coreProperties>'''
    footer = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{W}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:rFonts w:eastAsia="新細明體"/><w:sz w:val="18"/></w:rPr><w:t>— </w:t></w:r><w:fldSimple w:instr="PAGE"><w:r><w:t>1</w:t></w:r></w:fldSimple><w:r><w:rPr><w:rFonts w:eastAsia="新細明體"/><w:sz w:val="18"/></w:rPr><w:t> —</w:t></w:r></w:p></w:ftr>'''
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(assets.relationships) + '</Relationships>'
    rootrels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>'''
    contenttypes = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>{''.join(f'<Default Extension="{ext}" ContentType="{mimetypes.types_map.get("." + ext, "application/octet-stream")}"/>' for ext in sorted(assets.content_types))}</Types>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in {
            "[Content_Types].xml": contenttypes,
            "_rels/.rels": rootrels,
            "docProps/core.xml": core,
            "word/document.xml": document,
            "word/styles.xml": styles,
            "word/settings.xml": settings,
            "word/footer1.xml": footer,
            "word/_rels/document.xml.rels": rels,
        }.items(): z.writestr(name, content.encode("utf-8"))
        for name, content in assets.files.items(): z.writestr(name, content)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python generate_docx.py input.json output.docx")
    source = Path(sys.argv[1])
    build(json.loads(source.read_text(encoding="utf-8")), Path(sys.argv[2]), source.resolve().parent)
    print(f"Created {sys.argv[2]}")
