"""Run from the repo root:  python -m vit.step01_patch_embedding.test_patch_embedding"""

import torch

from vit.step01_patch_embedding.patch_embedding import PatchEmbedding, patchify

torch.manual_seed(0)


def test_patchify_shape():
    x = torch.randn(2, 3, 32, 32)
    assert patchify(x, 8).shape == (2, 16, 3 * 8 * 8)            # 4x4 grid of patches


def test_patchify_order_and_content():
    img = torch.arange(16.).reshape(1, 1, 4, 4)
    # [[ 0  1  2  3]
    #  [ 4  5  6  7]
    #  [ 8  9 10 11]
    #  [12 13 14 15]]  with 2x2 patches:
    p = patchify(img, 2)
    assert p[0].tolist() == [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]


def test_conv_equals_patchify_plus_linear():
    """The Conv2d trick computes exactly the paper's 'flatten patches, multiply by E'."""
    emb = PatchEmbedding(image_size=28, patch_size=7, in_channels=1, d_model=32)
    x = torch.randn(4, 1, 28, 28)
    via_conv = emb(x)
    via_linear = emb.as_linear()(patchify(x, 7))
    assert via_conv.shape == (4, 16, 32)
    assert torch.allclose(via_conv, via_linear, atol=1e-5)


def test_rgb_image_224_with_16x16_patches():
    """The paper's setting: 224x224 RGB, 16x16 patches -> 196 'words' of 768 numbers."""
    emb = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, d_model=768)
    assert emb.num_patches == 196
    assert emb(torch.randn(1, 3, 224, 224)).shape == (1, 196, 768)


def test_each_output_depends_only_on_its_patch():
    emb = PatchEmbedding(image_size=8, patch_size=4, in_channels=1, d_model=8)
    x = torch.randn(1, 1, 8, 8)
    out1 = emb(x)
    x2 = x.clone()
    x2[..., :4, :4] += 1.0                                     # change the top-left patch only
    out2 = emb(x2)
    assert not torch.allclose(out1[:, 0], out2[:, 0])
    assert torch.allclose(out1[:, 1:], out2[:, 1:])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("ViT Step 01: all tests passed")
