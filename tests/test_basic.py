"""Basic tests for langfuse-mcp package."""

import sys

import logging
import os

import pytest


def test_configure_logging_console():
    """configure_logging should honor log level and console flag."""
    from langfuse_mcp.__main__ import configure_logging

    configure_logging("DEBUG", True)
    root_handlers = logging.getLogger().handlers
    assert logging.getLogger().level == logging.DEBUG
    assert any(isinstance(handler, logging.StreamHandler) for handler in root_handlers)


def test_cli_env_defaults(monkeypatch):
    """Environment variables should provide defaults for CLI flags."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-sk")
    monkeypatch.setenv("LANGFUSE_HOST", "https://env-host")
    monkeypatch.setenv("LANGFUSE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LANGFUSE_LOG_TO_CONSOLE", "true")

    from langfuse_mcp.__main__ import _build_arg_parser, _read_env_defaults

    parser = _build_arg_parser(_read_env_defaults())
    args = parser.parse_args([])

    assert args.public_key == "env-pk"
    assert args.secret_key == "env-sk"
    assert args.host == "https://env-host"
    assert args.log_level == "DEBUG"
    assert args.log_to_console is True


def test_cli_requires_keys_without_env(monkeypatch):
    """Missing env and args should keep credentials required."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LANGFUSE_LOG_TO_CONSOLE", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    from langfuse_mcp.__main__ import _build_arg_parser, _read_env_defaults

    parser = _build_arg_parser(_read_env_defaults())
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_package_importable():
    """Test that the package can be imported."""
    # This test verifies the package can be imported
    try:
        import langfuse_mcp

        assert langfuse_mcp.__version__ is not None
    except ImportError:
        pytest.fail("Failed to import langfuse_mcp package")


def test_main_module_importable():
    """Test that the main module can be imported."""
    # This test verifies the main module can be imported
    try:
        # Use pytest.importorskip instead of direct import
        pytest.importorskip("langfuse_mcp.__main__")
    except ImportError:
        pytest.fail("Failed to import langfuse_mcp.__main__ module")


def test_python_version():
    """Test that Python version is compatible."""
    # This test verifies we're running on a compatible Python version
    # (our package requires Python 3.10+)
    python_version = sys.version_info
    assert python_version.major == 3
    assert python_version.minor >= 10, f"Python version {python_version.major}.{python_version.minor} is not supported"
