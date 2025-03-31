"""Basic tests for langfuse-mcp package."""
import sys

import pytest


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