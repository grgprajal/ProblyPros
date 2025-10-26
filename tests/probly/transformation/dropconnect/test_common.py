"""Tests for probly.transformation.dropconnect.common."""

import pytest
from probly.transformation.dropconnect import common

class DummyTraverser:
    """A dummy traverser for testing registration."""
    def __init__(self):
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True

def test_register_adds_class_to_traverser():
    """Test that register() correctly adds a class to the dropconnect traverser."""
    dummy_class = type("DummyLayer", (), {})
    dummy_traverser = DummyTraverser()

    # Before registration
    assert dummy_class not in common.dropconnect_traverser._registry

    # Perform registration
    common.register(dummy_class, dummy_traverser)

    # After registration
    assert dummy_class in common.dropconnect_traverser._registry, "Class should be registered"
    entry = common.dropconnect_traverser._registry[dummy_class]

    # Check that skip condition and variable are set correctly
    assert entry.vars.get("p") == common.P, "DropConnect probability variable not registered"
    assert callable(entry.skip_if), "skip_if must be a callable"

#Check that register() correctly registers the class to dropconnect_traverser

def test_dropconnect_function_runs(monkeypatch):
    """Test that dropconnect() calls traverse() and nn_compose() correctly."""
    called = {"traverse": False}

    # Mock traverse and nn_compose
    def mock_traverse(base, compose_fn, init):
        called["traverse"] = True
        assert init[common.P] == 0.25
        assert init["CLONE"] is True, "CLONE flag must be True"  # just ensure CLONE=True passes
        return "mock_result"

    monkeypatch.setattr(common, "traverse", mock_traverse)
    monkeypatch.setattr(common, "nn_compose", lambda x: x)

    result = common.dropconnect("dummy_model", p=0.25)
    assert called["traverse"], "dropconnect() should call traverse()"
    assert result == "mock_result"

# Check that dropconnect() calls traverse() correctly and passes the arguments.

