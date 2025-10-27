"""Tests for probly.transformation.dropconnect.torch."""

import pytest
import torch
from torch import nn


@pytest.fixture
def simple_model():
    """A simple test model with two Linear Layers."""
    return nn.Sequential(
        nn.Linear(4,8),
        nn.ReLU(),
        nn.Linear(8,2),
    )

