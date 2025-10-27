"""Tests for probly.transformation.dropconnect.torch."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
from torch import nn

from probly.layers.torch import DropConnectLinear
from probly.transformation.dropconnect import torch as dc_torch

# fixPathProblem
ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.resolve()

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def simple_model():
    """A simple test model with two Linear Layers."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def test_replace_torch_dropconnect_returns_correct_type() -> None:
    """Test that replace_torch_dropconnect() wraps nn.Linear as DropConnectLinear."""
    layer = nn.Linear(10, 5)
    p = 0.4
    replaced = dc_torch.replace_torch_dropconnect(layer, p)

    assert isinstance(replaced, DropConnectLinear), "Should return a DropConnectLinear."
    assert pytest.approx(replaced.p, 1e-6) == p, "DropConnect probability should be set correctly."


def test_register_torch_linear(monkeypatch) -> None:
    """Test that nn.Linear is registered with dropconnect_traverser."""
    registered = []

    def mock_register(cls, traverser) -> None:
        registered.append((cls, traverser))

    common = importlib.import_module("probly.transformation.dropconnect.common")

    # replace common.register
    monkeypatch.setattr(common, "register", mock_register)

    # remove old module to retrigger register()
    sys.modules.pop("probly.transformation.dropconnect.torch", None)
    importlib.import_module("probly.transformation.dropconnect.torch")

    # Check whether nn.Linear is registered successfully
    assert any(cls == nn.Linear for cls, _ in registered), "nn.Linear should be registered."
