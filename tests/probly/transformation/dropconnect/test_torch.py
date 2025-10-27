"""Tests for probly.transformation.dropconnect.torch."""
import os
import sys
import importlib

import pytest
import torch
from torch import nn

# fixPathProblem
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from probly.transformation.dropconnect import torch as dc_torch
from probly.layers.torch import DropConnectLinear


@pytest.fixture
def simple_model():
    """A simple test model with two Linear Layers."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def test_replace_torch_dropconnect_returns_correct_type():
    """Test that replace_torch_dropconnect() wraps nn.Linear as DropConnectLinear."""
    layer = nn.Linear(10, 5)
    p = 0.4
    replaced = dc_torch.replace_torch_dropconnect(layer, p)

    assert isinstance(replaced, DropConnectLinear), "Should return a DropConnectLinear."
    assert pytest.approx(replaced.p, 1e-6) == p, "DropConnect probability should be set correctly."


def test_register_torch_linear(monkeypatch):
    """Test that nn.Linear is registered with dropconnect_traverser."""
    registered = []

    def mock_register(cls, traverser):
        registered.append((cls, traverser))

    # importCommonModule
    common = importlib.import_module("probly.transformation.dropconnect.common")

    # replace common.register
    monkeypatch.setattr(common, "register", mock_register)

    # remove old module to retrigger register()
    sys.modules.pop("probly.transformation.dropconnect.torch", None)
    importlib.import_module("probly.transformation.dropconnect.torch")

    # Check whether nn.Linear is registered successfully
    assert any(cls == nn.Linear for cls, _ in registered), "nn.Linear should be registered."


    """
        tests/probly/transformation/dropconnect/test_common.py::test_register_adds_class_to_traverser PASSED                       [ 25%]
        tests/probly/transformation/dropconnect/test_common.py::test_dropconnect_function_runs PASSED                              [ 50%]
        tests/probly/transformation/dropconnect/test_torch.py::test_replace_torch_dropconnect_returns_correct_type PASSED          [ 75%]
        tests/probly/transformation/dropconnect/test_torch.py::test_register_torch_linear PASSED                                   [100%]

        ======================================================= 4 passed in 0.01s ========================================================
    """

    # all tests passed, final version
