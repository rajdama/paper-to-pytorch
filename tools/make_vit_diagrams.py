"""
Draws the Vision Transformer (after Figure 1 of "An Image is Worth 16x16 Words",
Dosovitskiy et al., 2020) as an SVG, plus one copy per ViT step with that step's
components highlighted.

Run from the repo root:   python tools/make_vit_diagrams.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H = 760, 560
INK, ACCENT = "#2b2b2b", "#e4572e"
MUTED_FILL, MUTED_TEXT, MUTED_STROKE = "#f1f1f1", "#a9a9a9", "#cfcfcf"
COL = {"proj": "#f7cfd9", "enc": "#fbe9d0", "head": "#cdeacb", "pos": "#d9d0f0", "tok": "#c4e3f3",
       "norm": "#f4f1b4", "attn": "#fbd9b5", "mlp": "#c4e3f3"}

N_TOK = 9                       # 3x3 patches drawn
TOK_W, TOK_GAP, TOK_X0 = 34, 10, 40
TOK_Y = 300                     # centre of the embedded-token row


def tok_x(i):                   # i = 0 is [CLS], 1..9 are patches
    return TOK_X0 + i * (TOK_W + TOK_GAP) + TOK_W / 2


def style(on, focus, kind):
    if focus and not on:
        return MUTED_FILL, MUTED_TEXT, MUTED_STROKE, 1.1
    return COL[kind], INK, (ACCENT if on else "#555"), (3 if on else 1.2)


def box(x, y, w, h, label, kind, on, focus, size=13, rx=6):
    fill, txt, stroke, sw = style(on, focus, kind)
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"/><text x="{x + w / 2}" y="{y + h / 2 + size * 0.36}" text-anchor="middle" '
            f'font-size="{size}" fill="{txt}">{label}</text>')


def arrow(x1, y1, x2, y2, color=INK):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5" '
            f'marker-end="url(#ah)"/>')


def render(hl=frozenset(), caption=None):
    focus = bool(hl)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="Helvetica, Arial, sans-serif">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W / 2}" y="30" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">'
         'Vision Transformer (ViT)</text>',
         f'<text x="{W / 2}" y="50" text-anchor="middle" font-size="12" fill="#777">after Figure 1 of '
         '“An Image is Worth 16x16 Words” (Dosovitskiy et al., 2020)</text>']

    # ---- input image cut into a 3x3 grid, and the flattened patch sequence ----
    on = "patches" in hl
    stroke = ACCENT if on else (MUTED_STROKE if focus else "#555")
    ix, iy, cell = 40, 450, 26
    shades = ["#e8e8e8", "#9a9a9a", "#e8e8e8", "#bdbdbd", "#6e6e6e", "#bdbdbd", "#e8e8e8", "#9a9a9a", "#e8e8e8"]
    for r in range(3):
        for c in range(3):
            o.append(f'<rect x="{ix + c * (cell + 3)}" y="{iy + r * (cell + 3)}" width="{cell}" height="{cell}" '
                     f'fill="{shades[r * 3 + c]}" stroke="{stroke}" stroke-width="{2 if on else 1}"/>')
    o.append(f'<text x="{ix + 41}" y="{iy - 8}" text-anchor="middle" font-size="11" fill="#666">image</text>')
    o.append(arrow(ix + 92, iy + 42, 150, iy + 42))
    for i in range(N_TOK):
        x = 160 + i * 38
        o.append(f'<rect x="{x}" y="{iy + 30}" width="26" height="26" fill="{shades[i]}" stroke="{stroke}" '
                 f'stroke-width="{2 if on else 1}"/>')
        o.append(arrow(x + 13, iy + 28, x + 13, 404))
    o.append(f'<text x="{160 + 4.5 * 38}" y="{iy + 74}" text-anchor="middle" font-size="11" fill="#666">'
             'split into patches, flatten each one</text>')

    # ---- linear projection ----
    o.append(box(150, 372, 345, 30, "Linear Projection of Flattened Patches", "proj", "proj" in hl, focus))

    # ---- embedded tokens: [CLS] + patches, each with a position embedding ----
    for i in range(N_TOK + 1):
        x = tok_x(i)
        if i > 0:
            o.append(arrow(173 + (i - 1) * 38, 372, x, TOK_Y + 14))
        cls_on, pos_on = "cls" in hl, "pos" in hl
        kind_on = cls_on if i == 0 else ("proj" in hl)
        fill, txt, stk, sw = style(kind_on, focus, "tok")
        o.append(f'<rect x="{x - TOK_W / 2}" y="{TOK_Y - 12}" width="{TOK_W}" height="24" rx="4" fill="{fill}" '
                 f'stroke="{stk}" stroke-width="{sw}"/>')
        o.append(f'<text x="{x + 6}" y="{TOK_Y + 4}" text-anchor="middle" font-size="11" fill="{txt}">'
                 f'{"*" if i == 0 else ""}</text>')
        pf, pt, ps, psw = style(pos_on, focus, "pos")
        o.append(f'<circle cx="{x - 9}" cy="{TOK_Y}" r="9" fill="{pf}" stroke="{ps}" stroke-width="{psw}"/>')
        o.append(f'<text x="{x - 9}" y="{TOK_Y + 4}" text-anchor="middle" font-size="10" fill="{pt}">{i}</text>')
        o.append(arrow(x, TOK_Y - 13, x, 252))
    lbl_col = INK if not focus or {"cls", "pos"} & hl else MUTED_TEXT
    o.append(f'<text x="160" y="166" font-size="10" fill="{lbl_col}">* = [CLS], an extra learned token</text>')
    o.append(f'<text x="160" y="181" font-size="10" fill="{lbl_col}">0–9 = learned position embeddings</text>')

    # ---- encoder and head ----
    o.append(box(30, 200, 430, 50, "Transformer Encoder  (Step 08 of paper 1, pre-norm)", "enc",
                 "encoder" in hl, focus, size=13))
    o.append(arrow(tok_x(0), 199, tok_x(0), 152))
    o.append(box(30, 120, 110, 30, "MLP Head", "head", "head" in hl, focus))
    o.append(arrow(85, 119, 85, 88))
    o.append(f'<text x="85" y="80" text-anchor="middle" font-size="12" fill="{INK}">class (0–9 / cat / dog …)</text>')
    o.append(f'<text x="160" y="140" font-size="10" fill="#777">only the [CLS] output is used</text>')

    # ---- encoder detail (right) ----
    dx, dw = 540, 170
    on = "encoder" in hl
    frame = ACCENT if on else ("#bdbdbd" if focus else "#9a9a9a")
    o.append(f'<rect x="{dx - 22}" y="92" width="{dw + 44}" height="330" rx="12" fill="{"#fff4ef" if on else "#fafafa"}" '
             f'stroke="{frame}" stroke-width="{3 if on else 1.3}"/>')
    o.append(f'<text x="{dx + dw + 30}" y="260" font-size="14" font-weight="700" fill="{INK}">L×</text>')
    rows = [(380, "Norm", "norm"), (330, "Multi-Head Attention", "attn"), (280, "+", None),
            (230, "Norm", "norm"), (180, "MLP", "mlp"), (130, "+", None)]
    o.append(arrow(dx + dw / 2, 445, dx + dw / 2, 396))
    o.append(f'<text x="{dx + dw / 2}" y="460" text-anchor="middle" font-size="11" fill="#666">embedded patches</text>')
    prev_top = None
    for y, label, kind in rows:
        if kind:
            o.append(box(dx, y - 14, dw, 28, label, kind, on, focus, size=12))
            top = y - 14
        else:
            o.append(f'<circle cx="{dx + dw / 2}" cy="{y}" r="10" fill="#fff" stroke="{INK}" stroke-width="1.3"/>'
                     f'<text x="{dx + dw / 2}" y="{y + 5}" text-anchor="middle" font-size="15" fill="{INK}">+</text>')
            top = y - 10
        if prev_top is not None:
            o.append(arrow(dx + dw / 2, prev_top, dx + dw / 2, (y + 14) if kind else (y + 10)))
        prev_top = top
    o.append(arrow(dx + dw / 2, 120, dx + dw / 2, 100))
    for start, end in ((402, 280), (252, 130)):            # residual connections
        o.append(f'<path d="M{dx + dw / 2},{start} L{dx - 12},{start} L{dx - 12},{end} L{dx + dw / 2 - 11},{end}" '
                 f'fill="none" stroke="{INK}" stroke-width="1.4" marker-end="url(#ah)"/>')

    if caption:
        o.append(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" font-size="13" font-weight="700" '
                 f'fill="{ACCENT}">{caption}</text>')
    o.append("</svg>")
    return "\n".join(o)


STEPS = {
    "step01_patch_embedding": ("ViT Step 01 · Patch Embedding", {"patches", "proj"}),
    "step02_vit_model": ("ViT Step 02 · [CLS] token, positions, encoder, head", {"cls", "pos", "encoder", "head"}),
    "step03_train_mnist": ("ViT Step 03 · Training on MNIST",
                           {"patches", "proj", "cls", "pos", "encoder", "head"}),
}

if __name__ == "__main__":
    (ROOT / "vit").mkdir(exist_ok=True)
    (ROOT / "vit" / "architecture.svg").write_text(render())
    for folder, (caption, ids) in STEPS.items():
        (ROOT / "vit" / folder / "diagram.svg").write_text(render(ids, caption))
    print(f"wrote vit/architecture.svg and {len(STEPS)} highlighted diagrams")
