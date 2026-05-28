from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parents[1] / "imgs" / "explanation.pdf"


W, H = 5.5 * inch, 3.42 * inch
M = 8

INK = colors.HexColor("#171717")
MUTED = colors.HexColor("#5f6368")
GRID = colors.HexColor("#d6d9df")
PANEL = colors.HexColor("#f8fafc")
HEADER = colors.HexColor("#eef2ff")
BLUE = colors.HexColor("#2f6fbb")
ORANGE = colors.HexColor("#d97706")
RED = colors.HexColor("#c7352b")
GREEN = colors.HexColor("#227a43")


def set_font(c, name="Helvetica", size=8.0, color=INK):
    c.setFont(name, size)
    c.setFillColor(color)


def text(c, x, y, s, size=8.0, font="Helvetica", color=INK, right=False, center=False):
    set_font(c, font, size, color)
    if right:
        c.drawRightString(x, y, s)
    elif center:
        c.drawCentredString(x, y, s)
    else:
        c.drawString(x, y, s)


def bold(c, x, y, s, size=8.0, color=INK, right=False, center=False):
    text(c, x, y, s, size=size, font="Helvetica-Bold", color=color, right=right, center=center)


def panel(c, x, y, w, h, title):
    c.setStrokeColor(GRID)
    c.setFillColor(PANEL)
    c.roundRect(x, y, w, h, 4, stroke=1, fill=1)
    bold(c, x + 7, y + h - 13, title, size=8.0)


def line(c, x1, y1, x2, y2, color=GRID, width=0.6):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)


def rect(c, x, y, w, h, fill, stroke=GRID, radius=3):
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def chip(c, x, y, label, fill, color=INK):
    pad_x = 4
    w = stringWidth(label, "Helvetica-Bold", 7.2) + 2 * pad_x
    rect(c, x, y, w, 13, fill, stroke=colors.HexColor("#d8dde7"), radius=3)
    bold(c, x + pad_x, y + 3.4, label, size=7.2, color=color)
    return w


def arrow(c, x1, y1, x2, y2, color=INK):
    from math import atan2, cos, sin

    line(c, x1, y1, x2, y2, color=color, width=0.8)
    ang = atan2(y2 - y1, x2 - x1)
    head = 5
    a1 = ang + 2.7
    a2 = ang - 2.7
    p1 = (x2 + head * cos(a1), y2 + head * sin(a1))
    p2 = (x2 + head * cos(a2), y2 + head * sin(a2))
    c.setFillColor(color)
    c.setStrokeColor(color)
    c.line(x2, y2, p1[0], p1[1])
    c.line(x2, y2, p2[0], p2[1])


def node(c, x, y, label, fill, stroke=BLUE, r=12):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1.2)
    c.circle(x, y, r, stroke=1, fill=1)
    bold(c, x, y - 2.5, label, size=9.0, color=INK, center=True)


def draw_table(c):
    x, y, w, h = M, H - M - 78, W - 2 * M, 78
    rect(c, x, y, w, h, colors.white, stroke=GRID, radius=4)
    bold(c, x + 8, y + h - 14, "Similar instances", size=8.1)

    cols = [
        ("Cand.",  27),
        ("skill",  28),
        ("lang.",  30),
        ("years",  32),
        ("edu.",   30),
        ("drive",  33),
        ("hours",  33),
        ("perm.",  32),
        ("gender", 35),
        ("deg.",   30),
        ("corr.",  50),
    ]
    total = sum(width for _, width in cols)
    start = x + (w - total) / 2
    top = y + h - 36
    row_h = 13
    c.setFillColor(HEADER)
    c.rect(start, top, total, row_h, stroke=0, fill=1)
    cx = start
    for name, cw in cols:
        bold(c, cx + cw / 2, top + 3.8, name, size=7.1, color=colors.HexColor("#26324d"), center=True)
        cx += cw

    rows = [
        ("A", "0", "0", "1", "1", "1", "0", "0", "0", "1", "-0.692", RED),
        ("B", "0", "0", "1", "1", "1", "0", "0", "1", "2", "+0.307", GREEN),
        ("C", "0", "0", "1", "1", "1", "0", "0", "0", "1", "-0.692", RED),
    ]
    for i, row in enumerate(rows):
        yy = top - (i + 1) * row_h
        if i % 2 == 1:
            c.setFillColor(colors.HexColor("#fbfcff"))
            c.rect(start, yy, total, row_h, stroke=0, fill=1)
        line(c, start, yy, start + total, yy, color=colors.HexColor("#e5e7eb"), width=0.4)
        cx = start
        for j, (_, cw) in enumerate(cols):
            value = row[j]
            color = row[-1] if j == len(cols) - 1 else INK
            font = "Helvetica-Bold" if j in (0, len(cols) - 1) else "Helvetica"
            text(c, cx + cw / 2, yy + 3.4, value, size=7.6, font=font, color=color, center=True)
            cx += cw

    line(c, start, top - 3 * row_h, start + total, top - 3 * row_h, color=GRID, width=0.5)
    cx = start
    for _, cw in cols:
        line(c, cx, top - 3 * row_h, cx, top + row_h, color=colors.HexColor("#edf0f5"), width=0.4)
        cx += cw
    line(c, start + total, top - 3 * row_h, start + total, top + row_h, color=colors.HexColor("#edf0f5"), width=0.4)


def draw_graph(c):
    x, y, w, h = M, M, 151, H - 2 * M - 86
    panel(c, x, y, w, h, "Neighborhood")

    q =  (x + 76,  y + 98)
    a =  (x + 33,  y + 54)
    b =  (x + 76,  y + 25)
    cc = (x + 120, y + 54)

    arrow(c, q[0] - 9, q[1] - 10, a[0] + 11, a[1] + 11, color=colors.HexColor("#30343b"))
    arrow(c, q[0], q[1] - 14, b[0], b[1] + 15, color=colors.HexColor("#30343b"))
    arrow(c, q[0] + 9, q[1] - 10, cc[0] - 11, cc[1] + 11, color=colors.HexColor("#30343b"))

    text(c, x + 33,  y + 77, "d=2.40",  size=6.8, color=MUTED)
    text(c, x + 70,  y + 55, "d=3.45",  size=6.8, color=MUTED)
    text(c, x + 104, y + 73, "d=2.323", size=6.8, color=MUTED)

    node(c, q[0], q[1], "x", fill=colors.HexColor("#e8f0fe"), stroke=BLUE, r=13)
    node(c, a[0], a[1], "A", fill=colors.white, stroke=colors.HexColor("#8fa7c7"), r=12)
    node(c, b[0], b[1], "B", fill=colors.HexColor("#fff7ed"), stroke=ORANGE, r=12)
    node(c, cc[0], cc[1], "C", fill=colors.white, stroke=colors.HexColor("#8fa7c7"), r=12)


def draw_correction(c):
    x, y, w, h = M + 159, M, W - 2 * M - 159, H - 2 * M - 86
    panel(c, x, y, w, h, "RuleTreeRank correction")

    top = y + h - 29
    bold(c, x + 9, top, "Initial score", size=7.8, color=BLUE)
    text(c, x + 78, top, "1.692", size=7.8)

    yy = top - 18
    bold(c, x + 9, yy + 0.8, "Tree path", size=7.7)
    xx = x + 62
    xx += chip(c, xx, yy - 3, "years exp <= 0.5", colors.HexColor("#eef2ff"), color=BLUE) + 4
    chip(c, xx, yy - 3, "permanent <= 0.5", colors.HexColor("#eef2ff"), color=BLUE)

    line(c, x + 8, yy - 12, x + w - 8, yy - 12, color=GRID, width=0.5)

    yy -= 36
    bold(c, x + 9, yy + 12, "Matched rules", size=7.7)
    bold(c, x + 14, yy - 2, "A", size=7.5, color=BLUE)
    text(c, x + 28, yy - 2, "degree <= 2.5", size=7.4)
    bold(c, x + 14, yy - 16, "B", size=7.5, color=ORANGE)
    text(c, x + 28, yy - 16, "degree <= 2.5 and hours <= 0.5", size=7.4)

    rect(c, x + w - 68, yy - 18, 58, 31, colors.white, stroke=colors.HexColor("#e5e7eb"), radius=4)
    bold(c, x + w - 39, yy + 1.5, "Distance", size=7.4, color=MUTED, center=True)
    bold(c, x + w - 39, yy - 10.5, "2.323", size=8.8, color=INK, center=True)

    line(c, x + 8, y + 43-3, x + w - 86, y + 43-3, color=GRID, width=0.5)
    bold(c, x + 9, y + 30, "Delta", size=7.7, color=ORANGE)
    text(c, x + 45, y + 30, "avg corrections = -0.358", size=7.6)
    text(c, x + 45, y + 18, "(-0.692 + 0.307 - 0.692) / 3", size=6.8, color=MUTED)

    rect(c, x + w - 68, y + 9, 58, 31, colors.HexColor("#fff7ed"), stroke=colors.HexColor("#fed7aa"), radius=4)
    bold(c, x + w - 43+5, y + 24.2+3, "Final score", size=7.2, color=ORANGE, center=True)
    bold(c, x + w - 43+5, y + 13.7+3, "1.334", size=10.0, color=INK, center=True)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("RuleTreeRank explanation figure")
    c.setAuthor("RuleTreeRank")
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    draw_table(c)
    draw_graph(c)
    draw_correction(c)
    c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
