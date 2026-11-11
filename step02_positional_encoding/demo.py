"""
Visualise the positional encoding table.
Run from the repo root:  python -m step02_positional_encoding.demo
Saves step02_positional_encoding/pe_heatmap.png
"""

from pathlib import Path
import matplotlib.pyplot as plt

from step02_positional_encoding.positional_encoding import sinusoidal_table

pe = sinusoidal_table(max_len=100, d_model=128)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

im = ax1.imshow(pe.numpy(), cmap="RdBu", aspect="auto")
ax1.set_title("Positional encoding table (each row = one position)")
ax1.set_xlabel("embedding dimension")
ax1.set_ylabel("position in sequence")
fig.colorbar(im, ax=ax1)

for dim in (0, 2, 10, 40):
    ax2.plot(pe[:, dim].numpy(), label=f"dim {dim}")
ax2.set_title("Low dims oscillate fast, high dims slowly")
ax2.set_xlabel("position")
ax2.legend()

out = Path(__file__).parent / "pe_heatmap.png"
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"saved {out}")
