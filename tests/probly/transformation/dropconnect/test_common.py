"""Tests for probly.transformation.dropconnect.common."""
from unittest.mock import MagicMock

import pytest
from probly.transformation.dropconnect import common

def test_register_adds_class_to_traverser(monkeypatch):
    """Test that register() correctly adds a class to the dropconnect traverser."""
    dummy_class = type("DummyLayer", (), {})
    mock_traverser = MagicMock()
    monkeypatch.setattr(common, "dropconnect_traverser", mock_traverser)

    # use register
    common.register(dummy_class, "dummy_traverser")

    #confirm dropconnect_traverser.register used
    mock_traverser.register.assert_called_once()

    #check all
    args, kwargs = mock_traverser.register.call_args
    assert kwargs["cls"] == dummy_class
    assert "traverser" in kwargs
    assert "skip_if" in kwargs
    assert "vars" in kwargs
    assert kwargs["vars"]["p"] == common.P

#Check that register() correctly registers the class to dropconnect_traverser

def test_dropconnect_function_runs(monkeypatch):
    """Test that dropconnect() calls traverse() and nn_compose() correctly."""
    called = {"traverse": False}

    # Mock traverse and nn_compose
    def mock_traverse(base, compose_fn, init):
        called["traverse"] = True
        assert init[common.P] == 0.25
        assert init[common.CLONE] is True, "CLONE flag must be True"  # just ensure CLONE=True passes
        return "mock_result"

    monkeypatch.setattr(common, "traverse", mock_traverse)
    monkeypatch.setattr(common, "nn_compose", lambda x: x)

    result = common.dropconnect("dummy_model", p=0.25)
    assert called["traverse"], "dropconnect() should call traverse()"
    assert result == "mock_result"

# Check that dropconnect() calls traverse() correctly and passes the arguments.

