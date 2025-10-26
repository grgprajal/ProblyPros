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
    assert dummy_class not in common.dropconnect_traverser.registry

    # Perform registration
    common.register(dummy_class, dummy_traverser)

    # After registration
    assert dummy_class in common.dropconnect_traverser.registry, "Class should be registered"
    entry = common.dropconnect_traverser.registry[dummy_class]

    # Check that skip condition and variable are set correctly
    assert entry.vars.get("p") == common.P, "DropConnect probability variable not registered"
    assert callable(entry.skip_if), "skip_if must be a callable"

