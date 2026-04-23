"""U-Net forward pass and parameter-count sanity checks (CPU only)."""

import torch

from models.unet import UNet


def test_forward_shape_small():
    model = UNet(in_channels=3, num_classes=6, base_filters=8)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 6, 128, 128)


def test_forward_shape_default():
    model = UNet(in_channels=3, num_classes=6, base_filters=8)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    assert y.shape == (2, 6, 256, 256)


def test_param_count_nonzero():
    model = UNet(in_channels=3, num_classes=6, base_filters=8)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total > 1_000, f"expected thousands of params, got {total}"


def test_backward_pass():
    """One gradient step should complete without NaNs."""
    model = UNet(in_channels=3, num_classes=6, base_filters=8)
    x = torch.randn(1, 3, 64, 64)
    target = torch.randint(0, 6, (1, 64, 64))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN in grad of {name}"
