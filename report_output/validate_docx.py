#!/usr/bin/env python
"""Structural DOCX checks and a small machine-readable manifest."""
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def validate(path: Path):
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure in {bad}")
        names = set(z.namelist())
        doc = ET.fromstring(z.read("word/document.xml"))
        body = doc.find("w:body", NS)
        paras = body.findall("w:p", NS)
        text = "\n".join("".join(node.itertext()) for node in paras)
        explicit_breaks = len(doc.findall('.//w:br[@w:type="page"]', NS))
        tables = len(body.findall("w:tbl", NS))
        drawings = len(doc.findall(".//w:drawing", NS))
        hyperlinks = len(doc.findall(".//w:hyperlink", NS))
        expected = ["一、資料蒐集與前處理", "二、AI 模型建構", "三、Python 程式實作",
                    "四、實際實驗與比較", "五、研究分析與改進"]
        chapter_presence = {heading: heading in text for heading in expected}
        report = {
            "docx": str(path),
            "zip_integrity": "ok",
            "xml_parse": "ok",
            "explicit_page_breaks": explicit_breaks,
            "explicit_pages_expected": explicit_breaks + 1,
            "chapters": chapter_presence,
            "tables": tables,
            "figures": drawings,
            "hyperlinks": hyperlinks,
            "word_text_characters": len(text),
            "word_parts": sorted(n for n in names if n.startswith("word/")),
            "word_rendered_pages": None,
            "pdf_pages": None,
        }
        missing_parts = {"[Content_Types].xml", "word/document.xml", "word/styles.xml"} - names
        if missing_parts:
            raise ValueError(f"Missing required package parts: {sorted(missing_parts)}")
        return report


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python validate_docx.py report.docx [manifest.json]")
    report = validate(Path(sys.argv[1]))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
