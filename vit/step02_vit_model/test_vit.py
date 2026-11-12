"""Run from the repo root:  python -m vit.step02_vit_model.test_vit"""

import torch
import torch.nn.functional as F

from vit.step02_vit_model.vit import VisionTransformer, attention_rollout

torch.manual_seed(0)


def test_logits_shape():
    model = VisionTransformer()
    assert model(torch.randn(5, 1, 28, 28)).shape == (5, 10)


def test_sequence_has_cls_plus_patches():
    model = VisionTransformer(image_size=28, patch_size=7).eval()
    model(torch.randn(2, 1, 28, 28))
    maps = model.attention_maps()
    assert len(maps) == 4
    assert maps[0].shape == (2, 4, 17, 17)            # 16 patches + 1 [CLS]


def test_rgb_paper_config_size():
    """ViT-Base/16 (Table 1): 12 layers, d=768, MLP 3072, 12 heads -> about 86M parameters."""
    model = VisionTransformer(image_size=224, patch_size=16, in_channels=3, num_classes=1000,
                              d_model=768, num_heads=12, num_layers=12, d_ff=3072)
    n = sum(p.numel() for p in model.parameters())
    print(f"    ViT-B/16: {n / 1e6:.1f}M parameters")
    assert 85e6 < n < 88e6


def test_position_embeddings_make_patch_order_matter():
    """Without position embeddings the encoder can't tell where a patch is: shuffling
    the patches gives the SAME [CLS] output. With them, the output changes."""
    model = VisionTransformer(dropout=0.0).eval()
    x = torch.randn(1, 1, 28, 28)
    shuffled = torch.cat([x[..., 7:, :], x[..., :7, :]], dim=2)   # move the top row of patches to the bottom
    with torch.no_grad():
        model.pos_embed.zero_()
        assert torch.allclose(model(x), model(shuffled), atol=1e-5)
        model.pos_embed.normal_(0, 1.0)
        assert not torch.allclose(model(x), model(shuffled), atol=1e-3)


def test_rollout_is_a_distribution():
    model = VisionTransformer().eval()
    model(torch.randn(3, 1, 28, 28))
    r = attention_rollout(model.attention_maps())
    assert r.shape == (3, 16)
    assert torch.all(r >= 0) and torch.all(r.sum(-1) <= 1 + 1e-5)


def test_can_overfit_a_tiny_batch():
    model = VisionTransformer(dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y = torch.randn(16, 1, 28, 28), torch.randint(0, 10, (16,))
    for _ in range(150):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert (model(x).argmax(-1) == y).float().mean() == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("ViT Step 02: all tests passed")
