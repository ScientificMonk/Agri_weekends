"""These tests fail until you finish the exercises in models.py. Make them pass!"""
import pytest
import torch

from p01_crop_type_mapping import models

BATCH, MONTHS, CHANNELS, CLASSES = 8, 12, 11, 4


@pytest.mark.parametrize("name", list(models.MODELS))
def test_output_shape(name):
    model = models.build(name, CHANNELS, MONTHS, CLASSES)
    out = model(torch.randn(BATCH, MONTHS, CHANNELS))
    assert out.shape == (BATCH, CLASSES)


@pytest.mark.parametrize("name", list(models.MODELS))
def test_gradients_flow(name):
    model = models.build(name, CHANNELS, MONTHS, CLASSES)
    model(torch.randn(BATCH, MONTHS, CHANNELS)).sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


@pytest.mark.parametrize("name", list(models.MODELS))
def test_can_overfit_tiny_batch(name):
    """A working model should memorise 16 samples. If not, something is wired wrong."""
    torch.manual_seed(0)
    model = models.build(name, CHANNELS, MONTHS, CLASSES)
    x, y = torch.randn(16, MONTHS, CHANNELS), torch.randint(0, CLASSES, (16,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    acc = (model(x).argmax(1) == y).float().mean().item()
    assert acc > 0.9
