"""Render Word's native PDF and assemble a contact sheet for review."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_tools"))
import fitz
from PIL import Image, ImageDraw, ImageFont

pdf = ROOT / "LSTM_GRU_report_preview.pdf"
output = ROOT / "preview_pages"
output.mkdir(exist_ok=True)
doc = fitz.open(pdf)
if len(doc) != 8:
    raise SystemExit(f"Expected 8 Word-rendered PDF pages, found {len(doc)}")
thumbs = []
for index, page in enumerate(doc, 1):
    pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    path = output / f"page_{index:02d}.png"
    pix.save(path)
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((420, 594))
        thumbs.append(im.copy())

sheet = Image.new("RGB", (1680, 1188), "#e7ebee")
draw = ImageDraw.Draw(sheet)
try:
    label_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
except OSError:
    label_font = ImageFont.load_default()
for i, im in enumerate(thumbs):
    x = (i % 4) * 420
    y = (i // 4) * 594
    sheet.paste(im, (x, y))
    draw.rectangle((x, y, x + 419, y + 24), fill="white")
    draw.text((x + 8, y + 2), f"Page {i + 1}", fill="#263746", font=label_font)
contact = ROOT / "preview_contact_sheet.png"
sheet.save(contact)
print(f"pages={len(doc)}")
print(f"contact_sheet={contact}")
print(f"page_images={output}")
