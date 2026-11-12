"""
Draws the Transformer architecture (after Figure 1 of "Attention Is All You Need",
Vaswani et al., 2017) as an SVG, and writes one copy per step folder with the
component built in that step highlighted.

Run from the repo root:   python tools/make_diagrams.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

W, H = 600, 830
BOX_W, BOX_H = 170, 30
ENC_X, DEC_X = 175, 425          # column centres

# Colours loosely follow the paper's figure
COLORS = {
    "attn": "#fbd9b5",
    "norm": "#f4f1b4",
    "ff":   "#c4e3f3",
    "emb":  "#f7cfd9",
    "lin":  "#d9d0f0",
    "soft": "#cdeacb",
}
MUTED_FILL, MUTED_TEXT, MUTED_STROKE = "#f1f1f1", "#a9a9a9", "#cfcfcf"
ACCENT = "#e4572e"
INK = "#2b2b2b"

# id: (label, kind, x, y_centre)
BOXES = {
    # encoder side
    "in_emb":    ("Input Embedding", "emb", ENC_X, 735),
    "enc_mha":   ("Multi-Head Attention", "attn", ENC_X, 625),
    "enc_an1":   ("Add & Norm", "norm", ENC_X, 580),
    "enc_ff":    ("Feed Forward", "ff", ENC_X, 530),
    "enc_an2":   ("Add & Norm", "norm", ENC_X, 485),
    # decoder side
    "out_emb":   ("Output Embedding", "emb", DEC_X, 735),
    "dec_mmha":  ("Masked Multi-Head Attention", "attn", DEC_X, 625),
    "dec_an1":   ("Add & Norm", "norm", DEC_X, 580),
    "dec_mha":   ("Multi-Head Attention", "attn", DEC_X, 405),
    "dec_an2":   ("Add & Norm", "norm", DEC_X, 360),
    "dec_ff":    ("Feed Forward", "ff", DEC_X, 310),
    "dec_an3":   ("Add & Norm", "norm", DEC_X, 265),
    # head
    "linear":    ("Linear", "lin", DEC_X, 195),
    "softmax":   ("Softmax", "soft", DEC_X, 150),
}
PE = {"pe_enc": (ENC_X, 685), "pe_dec": (DEC_X, 685)}
FRAMES = {  # id: (x_centre, top, bottom)
    "encoder": (ENC_X, 460, 655),
    "decoder": (DEC_X, 240, 655),
}

ALL = set(BOXES) | set(PE) | set(FRAMES)
ATTN = {"enc_mha", "dec_mmha", "dec_mha"}
NORMS = {"enc_an1", "enc_an2", "dec_an1", "dec_an2", "dec_an3"}
ENC_PARTS = {"enc_mha", "enc_an1", "enc_ff", "enc_an2", "encoder"}
DEC_PARTS = {"dec_mmha", "dec_an1", "dec_mha", "dec_an2", "dec_ff", "dec_an3", "decoder"}

# step folder -> (caption, highlighted ids)
STEPS = {
    "step01_embeddings":           ("Step 01 · Embeddings", {"in_emb", "out_emb"}),
    "step02_positional_encoding":  ("Step 02 · Positional Encoding", {"pe_enc", "pe_dec"}),
    "step03_attention":            ("Step 03 · Scaled Dot-Product Attention (inside every attention box)", ATTN),
    "step04_masking":              ("Step 04 · Masking", {"dec_mmha", "enc_mha", "dec_mha"}),
    "step05_multi_head_attention": ("Step 05 · Multi-Head Attention", ATTN),
    "step06_feed_forward":         ("Step 06 · Position-wise Feed-Forward", {"enc_ff", "dec_ff"}),
    "step07_add_and_norm":         ("Step 07 · Residual + Layer Norm", NORMS),
    "step08_encoder":              ("Step 08 · Encoder (layer × N)", ENC_PARTS),
    "step09_decoder":              ("Step 09 · Decoder (layer × N)", DEC_PARTS),
    "step10_full_transformer":     ("Step 10 · The Full Transformer", ALL),
    "step11_training":             ("Step 11 · Training", ALL),
    "step12_inference":            ("Step 12 · Inference (decoding)", {"linear", "softmax", "out_emb"}),
    "bonus/step13_gpt_decoder_only": ("Bonus 13 · GPT keeps only these boxes",
                                      {"out_emb", "pe_dec", "dec_mmha", "dec_an1", "dec_ff", "dec_an3",
                                       "linear", "softmax"}),
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def arrow(x1, y1, x2, y2, color=INK):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.6" marker-end="url(#ah)"/>')


def path_arrow(points, color=INK):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'marker-end="url(#ah)"/>')


def top(bid):
    return BOXES[bid][3] - BOX_H / 2


def bottom(bid):
    return BOXES[bid][3] + BOX_H / 2


def residual(sub, norm, x):
    """Skip connection: branch below `sub`, run up the left side, enter `norm`."""
    left = x - BOX_W / 2
    y0 = bottom(sub) + 9
    return path_arrow([(x, y0), (left - 16, y0), (left - 16, BOXES[norm][3]), (left, BOXES[norm][3])])


def render(highlight=frozenset(), caption=None):
    hl = set(highlight)
    focus = bool(hl)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Helvetica, Arial, sans-serif">',
        '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{W/2}" y="34" text-anchor="middle" font-size="19" font-weight="700" fill="{INK}">'
        'The Transformer</text>',
        f'<text x="{W/2}" y="56" text-anchor="middle" font-size="12" fill="#777">'
        'after Figure 1 of “Attention Is All You Need” (Vaswani et al., 2017)</text>',
    ]

    # frames (encoder / decoder stacks)
    for fid, (x, t, b) in FRAMES.items():
        on = fid in hl
        stroke = ACCENT if on else ("#bdbdbd" if focus else "#9a9a9a")
        out.append(f'<rect x="{x - BOX_W/2 - 32}" y="{t}" width="{BOX_W + 64}" height="{b - t}" rx="14" '
                   f'fill="{"#fff4ef" if on else "#fafafa"}" stroke="{stroke}" stroke-width="{3 if on else 1.4}"/>')
    out.append(f'<text x="{ENC_X - BOX_W/2 - 44}" y="560" text-anchor="end" font-size="15" '
               f'font-weight="700" fill="{INK}">N×</text>')
    out.append(f'<text x="{DEC_X + BOX_W/2 + 44}" y="450" font-size="15" font-weight="700" '
               f'fill="{INK}">×N</text>')

    # arrows: main flow
    for x, emb, first in ((ENC_X, "in_emb", "enc_mha"), (DEC_X, "out_emb", "dec_mmha")):
        out.append(arrow(x, 785, x, bottom(emb)))
        out.append(arrow(x, top(emb), x, 696))
        out.append(arrow(x, 674, x, bottom(first)))
    for a, b in [("enc_mha", "enc_an1"), ("enc_an1", "enc_ff"), ("enc_ff", "enc_an2"),
                 ("dec_mmha", "dec_an1"), ("dec_an1", "dec_mha"), ("dec_mha", "dec_an2"),
                 ("dec_an2", "dec_ff"), ("dec_ff", "dec_an3"), ("dec_an3", "linear"),
                 ("linear", "softmax")]:
        out.append(arrow(BOXES[a][2], top(a), BOXES[b][2], bottom(b)))
    out.append(arrow(DEC_X, top("softmax"), DEC_X, 100))

    # residual connections
    for sub, norm, x in [("enc_mha", "enc_an1", ENC_X), ("enc_ff", "enc_an2", ENC_X),
                         ("dec_mmha", "dec_an1", DEC_X), ("dec_mha", "dec_an2", DEC_X),
                         ("dec_ff", "dec_an3", DEC_X)]:
        out.append(residual(sub, norm, x))

    # encoder output -> decoder cross-attention (keys & values)
    mha_left = DEC_X - BOX_W / 2
    kv_x = mha_left + 55
    out.append(path_arrow([(ENC_X, top("enc_an2")), (ENC_X, 446), (kv_x, 446),
                           (kv_x, bottom("dec_mha"))]))
    out.append(f'<text x="{kv_x - 6}" y="462" text-anchor="end" font-size="11" fill="#666">K, V</text>')
    out.append(f'<text x="{DEC_X + 6}" y="500" font-size="11" fill="#666">Q</text>')

    # boxes
    for bid, (label, kind, x, y) in BOXES.items():
        on = bid in hl
        if focus and not on:
            fill, txt, stroke, sw = MUTED_FILL, MUTED_TEXT, MUTED_STROKE, 1.2
        else:
            fill, txt, stroke, sw = COLORS[kind], INK, (ACCENT if on else "#555"), (3 if on else 1.2)
        out.append(f'<rect x="{x - BOX_W/2}" y="{y - BOX_H/2}" width="{BOX_W}" height="{BOX_H}" rx="6" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        size = 12 if len(label) > 22 else 13
        out.append(f'<text x="{x}" y="{y + 4.5}" text-anchor="middle" font-size="{size}" '
                   f'fill="{txt}">{esc(label)}</text>')

    # positional encoding: ⊕ plus a small sine wave
    for pid, (x, y) in PE.items():
        on = pid in hl
        stroke = ACCENT if on else (MUTED_STROKE if focus else "#555")
        out.append(f'<circle cx="{x}" cy="{y}" r="11" fill="#fff" stroke="{stroke}" stroke-width="{3 if on else 1.4}"/>')
        out.append(f'<path d="M{x-6},{y} L{x+6},{y} M{x},{y-6} L{x},{y+6}" stroke="{INK}" stroke-width="1.6"/>')
        wx = x - 72 if pid == "pe_enc" else x + 34
        wave_col = ACCENT if on else (MUTED_TEXT if focus else INK)
        out.append(f'<path d="M{wx},{y} q6,-12 12,0 t12,0 t12,0" fill="none" stroke="{wave_col}" stroke-width="1.6"/>')
        out.append(f'<line x1="{x - 38 if pid == "pe_enc" else x + 34}" y1="{y}" x2="{x - 12 if pid == "pe_enc" else x + 12}" '
                   f'y2="{y}" stroke="{wave_col}" stroke-width="1.4"/>')
        lx = x - 80 if pid == "pe_enc" else x + 80
        anchor = "end" if pid == "pe_enc" else "start"
        out.append(f'<text x="{lx}" y="{y - 2}" text-anchor="{anchor}" font-size="11" fill="{wave_col}">Positional</text>')
        out.append(f'<text x="{lx}" y="{y + 11}" text-anchor="{anchor}" font-size="11" fill="{wave_col}">Encoding</text>')

    # input / output labels
    out.append(f'<text x="{ENC_X}" y="803" text-anchor="middle" font-size="13" fill="{INK}">Inputs</text>')
    out.append(f'<text x="{DEC_X}" y="803" text-anchor="middle" font-size="13" fill="{INK}">Outputs (shifted right)</text>')
    out.append(f'<text x="{DEC_X}" y="90" text-anchor="middle" font-size="13" fill="{INK}">Output Probabilities</text>')

    if caption:
        out.append(f'<text x="{W/2}" y="824" text-anchor="middle" font-size="13" font-weight="700" '
                   f'fill="{ACCENT}">{esc(caption)}</text>')

    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    (ROOT / "assets").mkdir(exist_ok=True)
    (ROOT / "assets" / "architecture.svg").write_text(render())
    for folder, (caption, ids) in STEPS.items():
        d = ROOT / folder
        d.mkdir(exist_ok=True)
        (d / "diagram.svg").write_text(render(ids, caption))
    print(f"wrote assets/architecture.svg and {len(STEPS)} highlighted diagrams")
