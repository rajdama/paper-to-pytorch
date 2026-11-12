"""
Text data for the bonus steps.

Tiny Shakespeare (~1.1 MB, all of Shakespeare's plays concatenated; public domain),
as used in Karpathy's char-rnn. Downloaded once into data/ (which git ignores).

    python -m bonus.data          # download now
"""

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def tiny_shakespeare() -> str:
    """Return the Tiny Shakespeare text, downloading it the first time."""
    path = DATA_DIR / "tiny_shakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"downloading Tiny Shakespeare -> {path}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    text = tiny_shakespeare()
    print(f"{len(text):,} characters, {len(set(text))} unique")
    print(text[:300])
