"""Tests for probly.transformation.dropconnect.torch."""

import pytest
import torch
from torch import nn


from probly.transformation.dropconnect import torch as dc_torch
from probly.layers.torch import DropConnectLinear


@pytest.fixture
def simple_model():
    """A simple test model with two Linear Layers."""
    return nn.Sequential(
        nn.Linear(4,8),
        nn.ReLU(),
        nn.Linear(8,2),
    )

def test_replace_torch_dropconnect_returns_correct_type():
    """Test that replace_torch_dropconnect() wraps nn.Linear as DropConnectLinear."""
    layer = nn.Linear(10,5)
    p = 0.4
    replaced = dc_torch.replace_torch_dropconnect(layer, p)

    assert isinstance(replaced, DropConnectLinear), """Should return a DropConnectLinear."""
    assert pytest.approx(replaced.p, 1e-6) == p, """DropConnect probability should be set correctly."""

"""tests/probly/transformation/dropconnect/test_torch.py::test_replace_torch_dropconnect_returns_correct_type PASSED          [100%]"""

