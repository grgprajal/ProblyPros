"""Tests for probly.transformation.dropconnect.common."""

from __future__ import annotations

from unittest.mock import MagicMock
from typing import Any, Callable

import pytest

from probly.transformation.dropconnect import common


def test_register_adds_class_to_traverser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that register() correctly adds a class to the dropconnect traverser."""
    dummy_class = type("DummyLayer", (), {})
    mock_traverser = MagicMock()
    monkeypatch.setattr(common, "dropconnect_traverser", mock_traverser)

    def dummy_traverser_fn(*args: Any, **kwargs: Any) -> Any:
        return None
    """
    call register to add dummy_class to the dropconnect_traverser
    """
    common.register(dummy_class, dummy_traverser_fn)

    # confirm that dropconnect_traverser's register() method was called
    mock_traverser.register.assert_called_once()

    # check that the correct arguments were passed to register()
    args, kwargs = mock_traverser.register.call_args
    assert kwargs["cls"] == dummy_class
    assert "traverser" in kwargs
    assert "skip_if" in kwargs
    assert "vars" in kwargs
    assert kwargs["vars"]["p"] == common.P


# Check that register() correctly registers the class to dropconnect_traverser


def test_dropconnect_function_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that dropconnect() calls traverse() and nn_compose() correctly."""
    called: dict[str, bool] = {"traverse": False}

    # Mock traverse and nn_compose
    def mock_traverse(_base: Any, _compose_fn: Callable[[Any], Any], init: dict[str, Any]) -> Any:
        called["traverse"] = True
        assert init[common.P] == 0.25
        assert init.get(common.CLONE, True) is True, "CLONE flag must be True"
        return "mock_result"

    monkeypatch.setattr(common, "traverse", mock_traverse)
    monkeypatch.setattr(common, "nn_compose", lambda x: x)

    result = common.dropconnect("dummy_model", p=0.25)
    assert called["traverse"], "dropconnect() should call traverse()"
    assert result == "mock_result"
