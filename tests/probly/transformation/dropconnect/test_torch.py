"""Simple test for torch dropconnect functionality."""

import torch
import torch.nn as nn
from probly.transformation.dropconnect import dropconnect


def test_dropconnect_torch():
    """Test that dropconnect works with PyTorch models."""
    # Create a simple PyTorch model
    model = nn.Linear(10, 2)

    # Apply dropconnect
    transformed_model = dropconnect(model, p=0.3)

    # Test forward pass
    x = torch.randn(4, 10)
    output = transformed_model(x)

    # Basic validation
    assert output is not None
    assert output.shape == (4, 2)
