"""
Slow integration test: U-Net should overfit a single synthetic batch in a
handful of gradient steps. Skipped by default; run with:

    pytest -m slow tests/test_overfit_batch.py
"""

import pytest
import torch
import torch.nn as nn

from models.unet import UNet


pytestmark = pytest.mark.slow


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = UNet(in_channels=3, num_classes=4, base_filters=8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, 4, (2, 64, 64))

    first_loss = None
    for step in range(30):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
    final = loss.item()

    assert final < first_loss * 0.8, (
        f"Model failed to overfit synthetic batch: "
        f"start={first_loss:.4f} final={final:.4f}"
    )
